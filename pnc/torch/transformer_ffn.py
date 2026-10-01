"""P&C inside a transformer feed-forward block, in PyTorch.

Mirrors :mod:`pnc.jax.transformer_ffn`; see that module for why CLS rows alone suffice
for a final block and why attention, LayerNorm, and the head are never touched.

This is the backend the ImageNet ViT-B/16 and DistilBERT Banking77 tracks use, since
their pretrained checkpoints are PyTorch.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..spec import CorrectionSpec
from .correct import NormalEquations
from .perturb import PerturbationBasis

__all__ = ["FeedForwardBlock", "FFNMember", "gelu", "build_ffn_pnc"]


def gelu(x) -> torch.Tensor:
    """Exact GELU, matching ``approximate=False`` in JAX and PyTorch."""
    return torch.nn.functional.gelu(torch.as_tensor(x).to(torch.float64), approximate="none")


@dataclass
class FeedForwardBlock:
    """The two affine maps of one FFN, in ``x @ W`` layout.

    ``w1`` is ``(d_model, d_hidden)`` and ``w2`` is ``(d_hidden, d_model)``.
    ``torch.nn.Linear`` stores the transpose of both, so transpose on the way in and
    out -- :meth:`from_linear` does that for you.
    """

    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    activation: Callable = gelu

    def __post_init__(self) -> None:
        cast = lambda t: torch.as_tensor(t).detach().to(torch.float64)  # noqa: E731
        self.w1, self.b1 = cast(self.w1), cast(self.b1)
        self.w2, self.b2 = cast(self.w2), cast(self.b2)

    @classmethod
    def from_linear(cls, first, second, activation=gelu) -> FeedForwardBlock:
        """Build from the two ``nn.Linear`` modules of an FFN, handling the transpose."""
        return cls(
            w1=first.weight.detach().T, b1=first.bias.detach(),
            w2=second.weight.detach().T, b2=second.bias.detach(),
            activation=activation,
        )

    @property
    def d_model(self) -> int:
        return int(self.w1.shape[0])

    @property
    def d_hidden(self) -> int:
        return int(self.w1.shape[1])

    def original_theta(self) -> torch.Tensor:
        """The second affine map as ``[b2; W2]``, the layout the solve returns."""
        return torch.cat([self.b2.unsqueeze(0), self.w2], dim=0)

    @torch.no_grad()
    def forward(self, hidden, w1=None, w2=None, b2=None) -> torch.Tensor:
        """FFN output for the given rows, optionally with member parameters."""
        cast = lambda t: torch.as_tensor(t).detach().to(torch.float64)  # noqa: E731
        hidden = cast(hidden)
        activated = self.activation(hidden @ (self.w1 if w1 is None else cast(w1)) + self.b1)
        return activated @ (self.w2 if w2 is None else cast(w2)) + (
            self.b2 if b2 is None else cast(b2)
        )


@dataclass
class FFNMember:
    """One member: its coefficient vector and the corrected second affine map."""

    coefficient: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    relative_magnitude: float


@torch.no_grad()
def build_ffn_pnc(
    block: FeedForwardBlock,
    hidden,
    targets,
    *,
    rank: int,
    n_members: int,
    relative_magnitude: float,
    spec: CorrectionSpec,
    seed: int,
    chunk_size: int = 8192,
    correct: bool = True,
) -> tuple[list[FFNMember], PerturbationBasis, float]:
    """Construct FFN members from cached calibration activations.

    ``hidden`` and ``targets`` are ``(n_rows, d_model)`` FFN inputs at the chosen
    token positions and the base-model FFN outputs at the same rows.
    ``relative_magnitude`` is ``r = ||dW1||_F / ||W1||_F`` for the median member.

    Returns ``(members, basis, scale)``; keep ``scale`` with the members, since a
    coefficient alone does not determine the perturbation.
    """
    hidden = torch.as_tensor(hidden).detach().to(torch.float64)
    targets = torch.as_tensor(targets).detach().to(torch.float64)
    if hidden.shape[0] != targets.shape[0]:
        raise ValueError("hidden and targets must have the same number of rows")
    if hidden.shape[1] != block.d_model:
        raise ValueError(f"hidden has width {hidden.shape[1]}, expected {block.d_model}")

    device = hidden.device
    # Independent unit-norm rows rather than a QR: at d_model*d_hidden ~ 2.4e6 these
    # are near-orthogonal by concentration and avoid a huge factorisation.
    basis = PerturbationBasis.random(
        tuple(block.w1.shape), rank, seed=seed, orthonormal=False, device=device
    )
    coefficients = basis.coefficients(n_members, seed=seed + 1)
    scale = basis.scale_for_relative_magnitude(block.w1, coefficients, relative_magnitude)

    block_w1 = block.w1.to(device)
    original = block.original_theta().to(device)
    subsets = spec.subsets(int(hidden.shape[0]), n_members)
    weight_norm = float(torch.linalg.norm(block_w1))

    members: list[FFNMember] = []
    for member_index in range(n_members):
        delta = basis.delta(coefficients[member_index], scale)
        achieved = float(torch.linalg.norm(delta) / (weight_norm + 1e-12))

        if not correct:
            members.append(
                FFNMember(coefficients[member_index], block.w2, block.b2, achieved)
            )
            continue

        if subsets is None:
            member_hidden, member_targets = hidden, targets
        else:
            index = torch.as_tensor(subsets[member_index], device=device)
            member_hidden, member_targets = hidden[index], targets[index]

        perturbed_w1 = block_w1 + delta
        system = NormalEquations(block.d_hidden + 1, block.d_model, device=device)
        for start in range(0, member_hidden.shape[0], chunk_size):
            rows = member_hidden[start : start + chunk_size]
            activated = block.activation(rows @ perturbed_w1 + block.b1.to(device))
            ones = torch.ones((activated.shape[0], 1), dtype=torch.float64, device=device)
            system.update(
                torch.cat([ones, activated], dim=1), member_targets[start : start + chunk_size]
            )

        solution = system.solve(spec.ridge, prior=spec.use_prior(original))
        members.append(
            FFNMember(coefficients[member_index], solution[1:], solution[0], achieved)
        )

    return members, basis, scale
