"""P&C for fully connected trunks, in PyTorch.

Mirrors :mod:`pnc.jax.dense` exactly, including the sequential multi-layer chain and
the batched per-member solve. See that module for the description of the
construction; the only differences here are the array library and that the whole
build runs under ``torch.no_grad`` -- P&C is post-hoc and never needs gradients.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from ..spec import CorrectionSpec
from .correct import augment, ridge_solve
from .perturb import PerturbationBasis

__all__ = ["DenseStack", "DensePnCEnsemble", "build_dense_pnc"]


def _affine(stream, weight, bias):
    """``stream @ weight + bias`` where either may carry a leading member axis.

    The weight and the bias are independently per-member or shared: a layer that has
    been perturbed but whose correction lands in the *next* layer carries a
    member-specific weight alongside the base model's shared bias.
    """
    product = (
        torch.einsum("mnd,mde->mne", stream, weight)  # per-member weights (M, d_in, d_out)
        if weight.ndim == 3
        else stream @ weight
    )
    return product + (bias[:, None, :] if bias.ndim == 2 else bias)


@dataclass
class DenseStack:
    """A plain MLP trunk with one or more affine output heads.

    ``weights[i]`` is ``(d_in, d_out)`` and the activation is applied after every one.
    Heads are applied to the final trunk post-activation and are never perturbed or
    corrected.
    """

    weights: list[torch.Tensor]
    biases: list[torch.Tensor]
    heads: dict[str, tuple[torch.Tensor, torch.Tensor]]
    activation: Callable = torch.relu

    def __post_init__(self) -> None:
        cast = lambda t: torch.as_tensor(t).detach().to(torch.float64)  # noqa: E731
        self.weights = [cast(w) for w in self.weights]
        self.biases = [cast(b) for b in self.biases]
        self.heads = {name: (cast(w), cast(b)) for name, (w, b) in self.heads.items()}

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    @property
    def device(self) -> torch.device:
        return self.weights[0].device

    def to(self, device) -> DenseStack:
        """A copy of the stack with every parameter on ``device``."""
        return DenseStack(
            weights=[w.to(device) for w in self.weights],
            biases=[b.to(device) for b in self.biases],
            heads={
                name: (w.to(device), b.to(device)) for name, (w, b) in self.heads.items()
            },
            activation=self.activation,
        )

    def as_input(self, inputs) -> torch.Tensor:
        """Cast to float64 **on the stack's device**.

        Callers legitimately pass host NumPy arrays to a stack whose weights live on
        an accelerator, so every entry point goes through here rather than assuming
        the caller already matched the device. JAX needs no equivalent because it
        places arrays implicitly.
        """
        return torch.as_tensor(inputs).detach().to(dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def activations(self, inputs) -> list[torch.Tensor]:
        """Inputs to each trunk layer: ``[x, h_1, ..., h_{L-1}]``."""
        current = self.as_input(inputs)
        collected = [current]
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            current = self.activation(current @ weight + bias)
            collected.append(current)
        return collected

    @torch.no_grad()
    def trunk(self, inputs) -> torch.Tensor:
        current = self.as_input(inputs)
        for weight, bias in zip(self.weights, self.biases, strict=True):
            current = self.activation(current @ weight + bias)
        return current

    def apply_heads(self, features) -> dict[str, torch.Tensor]:
        return {name: features @ weight + bias for name, (weight, bias) in self.heads.items()}

    @torch.no_grad()
    def predict(self, inputs) -> dict[str, torch.Tensor]:
        """Base-model prediction; ``var`` passes through softplus if present."""
        outputs = self.apply_heads(self.trunk(inputs))
        if "var" in outputs:
            outputs["var"] = torch.nn.functional.softplus(outputs["var"]) + 1e-6
        return outputs


@dataclass
class DensePnCEnsemble:
    """Members stored as deltas and corrections against a shared base stack."""

    stack: DenseStack
    deltas: dict[int, torch.Tensor]
    corrections: dict[int, tuple[torch.Tensor, torch.Tensor]]
    perturb_layers: tuple[int, ...]
    n_members: int = field(default=0)

    def __len__(self) -> int:
        return self.n_members

    def _forward(self, inputs) -> torch.Tensor:
        current = inputs.unsqueeze(0).expand(self.n_members, *inputs.shape)
        for index in range(self.stack.n_layers):
            weight, bias = self.stack.weights[index], self.stack.biases[index]
            if index in self.deltas:
                weight = weight.unsqueeze(0) + self.deltas[index]
            if index in self.corrections:
                weight, bias = self.corrections[index]
            current = self.stack.activation(_affine(current, weight, bias))
        return current

    @torch.no_grad()
    def predict(self, inputs, batch_size: int | None = 4096) -> dict[str, torch.Tensor]:
        """Stacked member predictions, each ``(n_members, n_inputs, n_outputs)``.

        Inputs are chunked because the member axis multiplies activation memory by
        ``M``; ``batch_size=None`` does it in one go.
        """
        inputs = self.stack.as_input(inputs)
        step = batch_size or inputs.shape[0]
        chunks = [
            self.stack.apply_heads(self._forward(inputs[start : start + step]))
            for start in range(0, inputs.shape[0], step)
        ]
        stacked = {
            name: torch.cat([c[name] for c in chunks], dim=1) for name in self.stack.heads
        }
        if "var" in stacked:
            stacked["var"] = torch.nn.functional.softplus(stacked["var"]) + 1e-6
        return stacked


@torch.no_grad()
def build_dense_pnc(
    stack: DenseStack,
    calibration_inputs,
    *,
    perturb_layers: Sequence[int],
    rank: int,
    n_members: int,
    scale: float,
    spec: CorrectionSpec,
    seed: int,
    correct: bool = True,
) -> DensePnCEnsemble:
    """Construct a P&C ensemble from a trained dense stack.

    ``perturb_layers`` are zero-based trunk layers, in network order and at least 2
    apart; each needs a following trunk layer to correct in, so output heads stay
    untouched.
    ``correct=False`` gives the matched uncorrected ablation with identical
    perturbations.
    """
    perturb_layers = list(perturb_layers)
    for layer in perturb_layers:
        if not 0 <= layer < stack.n_layers - 1:
            raise ValueError(
                f"layer {layer} has no following trunk layer to correct in "
                f"(trunk has {stack.n_layers} layers, and heads are never corrected)"
            )
    if sorted(perturb_layers) != perturb_layers:
        raise ValueError("perturb_layers must be in network order")
    # Each perturbation is repaired in the layer immediately after it, so two targets
    # one apart would have the second perturb a layer the first has already corrected
    # -- the fitted correction would not match the design it was fit on. Rejected
    # rather than silently producing an inconsistent ensemble.
    for previous, current in zip(perturb_layers, perturb_layers[1:], strict=False):
        if current < previous + 2:
            raise ValueError(
                f"perturb_layers {previous} and {current} overlap: layer {previous}'s "
                f"correction lands in layer {previous + 1}, so consecutive targets must "
                f"differ by at least 2"
            )

    device = stack.device
    calibration_inputs = stack.as_input(calibration_inputs)
    n_examples = int(calibration_inputs.shape[0])

    bases = {
        layer: PerturbationBasis.random(
            tuple(stack.weights[layer].shape), rank, seed=seed + 101 * layer, device=device
        )
        for layer in perturb_layers
    }
    all_deltas = {
        layer: bases[layer].deltas(
            bases[layer].coefficients(n_members, seed=seed + 17 + 101 * layer), scale
        )
        for layer in perturb_layers
    }

    base_activations = stack.activations(calibration_inputs)
    subsets = spec.subsets(n_examples, n_members)
    gather = None if subsets is None else torch.as_tensor(np.stack(subsets), device=device)

    def per_member(array):
        """``(n, d)`` base rows -> ``(M, |S|, d)``, honouring the bootstrap subsets."""
        if gather is None:
            return array.unsqueeze(0).expand(n_members, *array.shape)
        return array[gather]

    stream = per_member(calibration_inputs)
    corrections: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    cursor = 0

    for layer in perturb_layers:
        # Advance the already-corrected stream up to the layer being perturbed.
        while cursor < layer:
            weight, bias = stack.weights[cursor], stack.biases[cursor]
            if cursor in corrections:
                weight, bias = corrections[cursor]
            stream = stack.activation(_affine(stream, weight, bias))
            cursor += 1

        perturbed = stack.activation(
            _affine(
                stream,
                stack.weights[layer].unsqueeze(0) + all_deltas[layer],
                stack.biases[layer],
            )
        )

        correction_index = layer + 1
        original = torch.cat(
            [stack.biases[correction_index].unsqueeze(0), stack.weights[correction_index]], dim=0
        )
        if correct:
            # Target: the base model's own pre-activation at this interface.
            reference = per_member(base_activations[correction_index])
            target = (
                reference @ stack.weights[correction_index] + stack.biases[correction_index]
            )
            solution = ridge_solve(
                augment(perturbed), target, spec.ridge, prior=spec.use_prior(original)
            )
        else:
            solution = original.unsqueeze(0).expand(n_members, *original.shape)
        corrections[correction_index] = (solution[:, 1:], solution[:, 0])

        stream = stack.activation(_affine(perturbed, solution[:, 1:], solution[:, 0]))
        cursor = correction_index + 1

    return DensePnCEnsemble(
        stack=stack,
        deltas=all_deltas,
        corrections=corrections,
        perturb_layers=tuple(perturb_layers),
        n_members=n_members,
    )
