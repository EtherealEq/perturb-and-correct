"""Framework-free pieces of the construction: sampling, subsets, and conventions.

Everything here is NumPy and shared by both the JAX and PyTorch implementations.
That is deliberate for the random draws: the perturbation basis, the member
coefficients, and the bootstrap subsets are all sampled here, so **the same seed
produces the same ensemble in either framework**. Only the linear algebra differs
between backends, which is what makes :mod:`tests.test_equivalence` able to assert
that the two implementations agree to float64 precision.

The draws happen once at construction, so keeping them on the host costs nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CorrectionSpec",
    "bootstrap_indices",
    "sample_directions",
    "sample_coefficients",
    "scale_for_relative_magnitude",
    "mean_to_sum_ridge",
    "sum_to_mean_ridge",
]


# -- ridge conventions ------------------------------------------------------
# `ridge` throughout this package multiplies an *unnormalised* sum-of-squares data
# term, matching Eq. (1). Implementations that divide the data term by the row count
# need these conversions to compare like with like.
def mean_to_sum_ridge(ridge_mean: float, n_rows: int) -> float:
    """Convert a mean-objective ridge value to the summed convention used here."""
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    return float(ridge_mean) * float(n_rows)


def sum_to_mean_ridge(ridge_sum: float, n_rows: int) -> float:
    """Convert this package's ridge value to its mean-objective equivalent."""
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    return float(ridge_sum) / float(n_rows)


# -- perturbation sampling --------------------------------------------------
def sample_directions(
    param_shape: tuple[int, ...],
    rank: int,
    seed: int,
    *,
    orthonormal: bool = True,
) -> np.ndarray:
    """The shared rank-``K`` basis: ``(rank, prod(param_shape))`` float64 rows.

    ``orthonormal=True`` (what Section 3 specifies) QR-factorises a Gaussian matrix.
    ``orthonormal=False`` uses independent unit-norm rows, the cheaper convention for
    the transformer weight matrices, where ``D ~ 2.4e6`` makes those rows
    near-orthogonal by concentration anyway and a QR would be wasteful.
    """
    dim = int(np.prod(param_shape))
    if rank > dim:
        raise ValueError(f"rank {rank} exceeds parameter dimension {dim}")
    rng = np.random.RandomState(seed)
    if orthonormal:
        basis, _ = np.linalg.qr(rng.standard_normal((dim, rank)).astype(np.float64))
        return np.ascontiguousarray(basis.T)
    rows = rng.standard_normal((rank, dim)).astype(np.float64)
    return rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)


def sample_coefficients(n_members: int, rank: int, seed: int) -> np.ndarray:
    """``(M, K)`` standard-Gaussian coefficients, one row per member."""
    return np.random.RandomState(seed).standard_normal((n_members, rank)).astype(np.float64)


def scale_for_relative_magnitude(
    directions: np.ndarray,
    weight: np.ndarray,
    coefficients: np.ndarray,
    target_relative: float,
) -> float:
    """Scale making the *median* member satisfy ``||dW||_F / ||W||_F == target``.

    The median rather than the mean, so one heavy-tailed draw cannot move the
    operating point -- this is how ``r`` is reported for the ImageNet and Banking77
    experiments.
    """
    weight_norm = float(np.linalg.norm(np.asarray(weight, dtype=np.float64)))
    if weight_norm == 0.0:
        raise ValueError("cannot calibrate a relative magnitude against a zero weight")
    directions = np.asarray(directions, dtype=np.float64)
    relatives = [
        float(np.linalg.norm(z @ directions)) / weight_norm
        for z in np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    ]
    return float(target_relative / (np.median(relatives) + 1e-12))


# -- correction subsets -----------------------------------------------------
def bootstrap_indices(
    n_examples: int,
    fraction: float,
    seed: int,
    n_members: int,
) -> list[np.ndarray] | None:
    """Per-member correction subsets ``S_m``, drawn with replacement.

    Bootstrapped correction is P&C's second diversity source: members that solve
    Eq. (1) on different empirical correction geometries decorrelate their correction
    failures.

    ``fraction`` is the subset size as a fraction of the calibration pool, and
    sampling is always *with replacement*, so ``1.0`` is the textbook bootstrap --
    same size, resampled -- not a no-op. ``0.0`` disables bootstrapping and every
    member shares the full pool, which is signalled by returning ``None``.

    All subsets have the same size, which is what lets the JAX dense backend map the
    per-member solves in one batched call.

    Shrinking the subset trades diversity against conditioning: as it approaches the
    affine design dimension the solve nears its interpolation threshold and can
    destabilise, which is why the paper treats bootstrap fraction as controlling both.
    """
    if fraction <= 0.0:
        return None
    if fraction > 1.0:
        raise ValueError(f"bootstrap fraction must be in [0, 1], got {fraction}")
    rng = np.random.RandomState(seed)
    size = max(8, int(round(fraction * n_examples)))
    return [rng.randint(0, n_examples, size) for _ in range(n_members)]


@dataclass(frozen=True)
class CorrectionSpec:
    """Settings for the correction half of the construction.

    Attributes
    ----------
    ridge:
        ``lambda`` in Eq. (1), against the summed data term.
    bootstrap_fraction:
        Size of each member's correction subset as a fraction of the calibration
        pool, resampled with replacement. ``0.0`` shares the full pool across
        members; ``1.0`` is a same-size bootstrap.
    center_on_original:
        Shrink toward the pretrained affine map -- the paper's convention, and what
        makes ``lambda -> inf`` recover the base model rather than the zero map.
        ``False`` shrinks toward zero and exists only for ablations.
    seed:
        Seed for the bootstrap resampling.
    """

    ridge: float = 1e-3
    bootstrap_fraction: float = 0.0
    center_on_original: bool = True
    seed: int = 0

    def subsets(self, n_examples: int, n_members: int) -> list[np.ndarray] | None:
        return bootstrap_indices(n_examples, self.bootstrap_fraction, self.seed, n_members)

    def use_prior(self, original):
        """The ridge centre, or ``None`` when shrinking toward zero."""
        return original if self.center_on_original else None
