"""P&C for fully connected trunks, in JAX (the MuJoCo dynamics models).

Section 3's construction with no architectural special-casing: perturb a hidden
affine layer's weight, refit the immediately following affine layer on calibration
activations.

Multi-layer P&C applies the step *sequentially*. After perturbing and correcting at
layer ``l_j``, the activations feeding the next stage are the ones the corrected
member produces, so the construction is a chain of local repairs rather than one
end-of-network correction. The regression target at every stage is still the *base*
model's pre-activation at that interface, so members are pulled back toward the
original network at each step rather than drifting.

All members are solved in a single batched call: their correction subsets have equal
size by construction, so the designs stack into one ``(M, |S|, p)`` tensor and the
ridge solves into one batched ``jnp.linalg.solve``. Peak memory scales as
``M * |S| * width``; bootstrapping shrinks ``|S|`` and with it that cost.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

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
        jnp.einsum("mnd,mde->mne", stream, weight)  # per-member weights (M, d_in, d_out)
        if weight.ndim == 3
        else stream @ weight
    )
    return product + (bias[:, None, :] if bias.ndim == 2 else bias)


@dataclass
class DenseStack:
    """A plain MLP trunk with one or more affine output heads.

    ``weights[i]`` is ``(d_in, d_out)`` and the activation is applied after every one.
    ``heads`` are applied to the final trunk post-activation -- the MuJoCo models use
    ``{"mean": ..., "var": ...}`` with a softplus on the variance head. Heads are
    never perturbed or corrected.
    """

    weights: list[jnp.ndarray]
    biases: list[jnp.ndarray]
    heads: dict[str, tuple[jnp.ndarray, jnp.ndarray]]
    activation: Callable = jax.nn.relu

    def __post_init__(self) -> None:
        self.weights = [jnp.asarray(w, dtype=jnp.float64) for w in self.weights]
        self.biases = [jnp.asarray(b, dtype=jnp.float64) for b in self.biases]
        self.heads = {
            name: (jnp.asarray(w, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64))
            for name, (w, b) in self.heads.items()
        }

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    def activations(self, inputs) -> list[jnp.ndarray]:
        """Inputs to each trunk layer: ``[x, h_1, ..., h_{L-1}]``."""
        current = jnp.asarray(inputs, dtype=jnp.float64)
        collected = [current]
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            current = self.activation(current @ weight + bias)
            collected.append(current)
        return collected

    def trunk(self, inputs) -> jnp.ndarray:
        current = jnp.asarray(inputs, dtype=jnp.float64)
        for weight, bias in zip(self.weights, self.biases, strict=True):
            current = self.activation(current @ weight + bias)
        return current

    def apply_heads(self, features) -> dict[str, jnp.ndarray]:
        return {name: features @ weight + bias for name, (weight, bias) in self.heads.items()}

    def predict(self, inputs) -> dict[str, jnp.ndarray]:
        """Base-model prediction; ``var`` passes through softplus if present."""
        outputs = self.apply_heads(self.trunk(inputs))
        if "var" in outputs:
            outputs["var"] = jax.nn.softplus(outputs["var"]) + 1e-6
        return outputs


@dataclass
class DensePnCEnsemble:
    """Members stored as deltas and corrections against a shared base stack."""

    stack: DenseStack
    deltas: dict[int, jnp.ndarray]
    corrections: dict[int, tuple[jnp.ndarray, jnp.ndarray]]
    perturb_layers: tuple[int, ...]
    n_members: int = field(default=0)

    def __len__(self) -> int:
        return self.n_members

    def _forward(self, inputs) -> jnp.ndarray:
        current = jnp.broadcast_to(
            jnp.asarray(inputs, dtype=jnp.float64), (self.n_members, *jnp.shape(inputs))
        )
        for index in range(self.stack.n_layers):
            weight, bias = self.stack.weights[index], self.stack.biases[index]
            if index in self.deltas:
                weight = weight[None] + self.deltas[index]
            if index in self.corrections:
                weight, bias = self.corrections[index]
            current = self.stack.activation(_affine(current, weight, bias))
        return current

    def predict(self, inputs, batch_size: int | None = 4096) -> dict[str, jnp.ndarray]:
        """Stacked member predictions, each ``(n_members, n_inputs, n_outputs)``.

        Inputs are chunked because the member axis multiplies activation memory by
        ``M``; ``batch_size=None`` does it in one go.
        """
        inputs = jnp.asarray(inputs, dtype=jnp.float64)
        step = batch_size or inputs.shape[0]
        chunks = [
            self.stack.apply_heads(self._forward(inputs[start : start + step]))
            for start in range(0, inputs.shape[0], step)
        ]
        stacked = {
            name: jnp.concatenate([c[name] for c in chunks], axis=1) for name in self.stack.heads
        }
        if "var" in stacked:
            stacked["var"] = jax.nn.softplus(stacked["var"]) + 1e-6
        return stacked


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

    Parameters
    ----------
    perturb_layers:
        Zero-based trunk layers to perturb, in network order, each at least 2 apart.
        Each needs a following trunk layer to correct in, so output heads stay
        untouched; the paper's multi-block MuJoCo setting is ``[0, 2]``.
    scale:
        ``sigma`` in Section 3, used directly (the absolute convention).
    correct:
        ``False`` skips the repair, giving the "uncorrected random perturbation"
        ablation of Figures 2 and 3(A). The perturbations are identical either way.
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

    calibration_inputs = jnp.asarray(calibration_inputs, dtype=jnp.float64)
    n_examples = int(calibration_inputs.shape[0])

    bases = {
        layer: PerturbationBasis.random(
            tuple(stack.weights[layer].shape), rank, seed=seed + 101 * layer
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
    gather = None if subsets is None else jnp.asarray(np.stack(subsets))

    def per_member(array):
        """``(n, d)`` base rows -> ``(M, |S|, d)``, honouring the bootstrap subsets."""
        if gather is None:
            return jnp.broadcast_to(array, (n_members, *array.shape))
        return array[gather]

    stream = per_member(calibration_inputs)
    corrections: dict[int, tuple[jnp.ndarray, jnp.ndarray]] = {}
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
            _affine(stream, stack.weights[layer][None] + all_deltas[layer], stack.biases[layer])
        )

        correction_index = layer + 1
        original = jnp.concatenate(
            [stack.biases[correction_index][None, :], stack.weights[correction_index]], axis=0
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
            solution = jnp.broadcast_to(original, (n_members, *original.shape))
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
