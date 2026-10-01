"""P&C inside a transformer feed-forward block, in JAX.

Writing the tokenwise feed-forward map as ``u = phi(W1 h + b1)``, ``z = W2 u + b2``,
P&C perturbs ``W1`` and refits ``(W2, b2)`` to reproduce the base model's
feed-forward output on ID calibration examples (Appendix B.7). That is the same
affine ridge problem as Eq. (1) with token representations as the design rows -- no
new machinery, only a different source of rows.

**Which token rows to fit on.** For the *final* block the correction may be fit on CLS
rows alone: the only operations left before classification are tokenwise
normalisation, CLS selection, and the head, so no later attention can mix patch-token
changes back into CLS. For an earlier block, later attention does mix positions, and
the relevant token rows must be passed instead.

Attention, LayerNorm, and the classifier head are never touched.

The construction runs on *cached* activations, which is what makes it cheap at scale.
Members loop rather than batch: the Gram is ``(d_hidden + 1)^2``, so stacking would
cost ``M`` times a 75 MB matrix.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..spec import CorrectionSpec
from .correct import NormalEquations
from .perturb import PerturbationBasis

__all__ = ["FeedForwardBlock", "FFNMember", "gelu", "build_ffn_pnc"]


def gelu(x):
    """Exact GELU, matching ``approximate=False`` in JAX and PyTorch."""
    return jax.nn.gelu(jnp.asarray(x, dtype=jnp.float64), approximate=False)


@dataclass
class FeedForwardBlock:
    """The two affine maps of one FFN, in ``x @ W`` layout.

    ``w1`` is ``(d_model, d_hidden)`` and ``w2`` is ``(d_hidden, d_model)``. PyTorch
    ``nn.Linear`` stores the transpose of both, so transpose on the way in and out.
    """

    w1: jnp.ndarray
    b1: jnp.ndarray
    w2: jnp.ndarray
    b2: jnp.ndarray
    activation: Callable = gelu

    def __post_init__(self) -> None:
        self.w1 = jnp.asarray(self.w1, dtype=jnp.float64)
        self.b1 = jnp.asarray(self.b1, dtype=jnp.float64)
        self.w2 = jnp.asarray(self.w2, dtype=jnp.float64)
        self.b2 = jnp.asarray(self.b2, dtype=jnp.float64)

    @property
    def d_model(self) -> int:
        return int(self.w1.shape[0])

    @property
    def d_hidden(self) -> int:
        return int(self.w1.shape[1])

    def original_theta(self) -> jnp.ndarray:
        """The second affine map as ``[b2; W2]``, the layout the solve returns."""
        return jnp.concatenate([self.b2[None, :], self.w2], axis=0)

    def forward(self, hidden, w1=None, w2=None, b2=None) -> jnp.ndarray:
        """FFN output for the given rows, optionally with member parameters."""
        activated = self.activation(jnp.asarray(hidden, dtype=jnp.float64) @ (
            self.w1 if w1 is None else w1
        ) + self.b1)
        return activated @ (self.w2 if w2 is None else w2) + (self.b2 if b2 is None else b2)


@dataclass
class FFNMember:
    """One member: its coefficient vector and the corrected second affine map.

    ``coefficient`` reconstructs ``W1 + dW1`` against the shared basis on demand, so
    the per-member cost is ``K`` floats plus the corrected ``(W2, b2)``.
    """

    coefficient: jnp.ndarray
    w2: jnp.ndarray
    b2: jnp.ndarray
    relative_magnitude: float


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

    Parameters
    ----------
    hidden, targets:
        ``(n_rows, d_model)`` FFN inputs at the chosen token positions, and the
        base-model FFN outputs at the same rows.
    relative_magnitude:
        ``r = ||dW1||_F / ||W1||_F`` for the median member.
    spec:
        ``ridge`` is against the *summed* objective of Eq. (1).
    correct:
        ``False`` reuses the identical perturbations with no repair.

    Returns
    -------
    ``(members, basis, scale)``. Keep ``scale`` with the members: a coefficient alone
    does not determine the perturbation.
    """
    hidden = jnp.asarray(hidden, dtype=jnp.float64)
    targets = jnp.asarray(targets, dtype=jnp.float64)
    if hidden.shape[0] != targets.shape[0]:
        raise ValueError("hidden and targets must have the same number of rows")
    if hidden.shape[1] != block.d_model:
        raise ValueError(f"hidden has width {hidden.shape[1]}, expected {block.d_model}")

    # Independent unit-norm rows rather than a QR: at d_model*d_hidden ~ 2.4e6 these
    # are near-orthogonal by concentration and avoid a huge factorisation.
    basis = PerturbationBasis.random(tuple(block.w1.shape), rank, seed=seed, orthonormal=False)
    coefficients = basis.coefficients(n_members, seed=seed + 1)
    scale = basis.scale_for_relative_magnitude(block.w1, coefficients, relative_magnitude)

    original = block.original_theta()
    subsets = spec.subsets(int(hidden.shape[0]), n_members)
    weight_norm = float(jnp.linalg.norm(block.w1))

    members: list[FFNMember] = []
    for member_index in range(n_members):
        delta = basis.delta(coefficients[member_index], scale)
        achieved = float(jnp.linalg.norm(delta) / (weight_norm + 1e-12))

        if not correct:
            members.append(
                FFNMember(coefficients[member_index], block.w2, block.b2, achieved)
            )
            continue

        if subsets is None:
            member_hidden, member_targets = hidden, targets
        else:
            index = jnp.asarray(subsets[member_index])
            member_hidden, member_targets = hidden[index], targets[index]

        perturbed_w1 = block.w1 + delta
        system = NormalEquations(block.d_hidden + 1, block.d_model)
        for start in range(0, member_hidden.shape[0], chunk_size):
            rows = member_hidden[start : start + chunk_size]
            activated = block.activation(rows @ perturbed_w1 + block.b1)
            design = jnp.concatenate(
                [jnp.ones((activated.shape[0], 1), dtype=jnp.float64), activated], axis=1
            )
            system.update(design, member_targets[start : start + chunk_size])

        solution = system.solve(spec.ridge, prior=spec.use_prior(original))
        members.append(
            FFNMember(coefficients[member_index], solution[1:], solution[0], achieved)
        )

    return members, basis, scale
