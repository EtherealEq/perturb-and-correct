"""Random low-dimensional perturbations of a hidden-layer parameter (JAX)."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..spec import sample_coefficients, sample_directions, scale_for_relative_magnitude

__all__ = ["PerturbationBasis"]


@dataclass(frozen=True)
class PerturbationBasis:
    """A shared rank-``K`` perturbation subspace for one parameter tensor.

    One basis is drawn and shared across members; members differ only in their
    Gaussian coefficient vector ``z``, so an ensemble costs ``K * D`` floats for the
    basis plus ``K`` floats per member rather than ``M`` copies of the parameter.

        dW = sigma * reshape(z @ U),   z ~ N(0, I_K)
    """

    directions: jnp.ndarray
    param_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.directions.ndim != 2:
            raise ValueError("directions must be a (rank, dim) matrix")
        dim = int(np.prod(self.param_shape))
        if self.directions.shape[1] != dim:
            raise ValueError(
                f"directions have dim {self.directions.shape[1]} but param_shape "
                f"{self.param_shape} flattens to {dim}"
            )

    @classmethod
    def random(
        cls,
        param_shape: tuple[int, ...],
        rank: int,
        seed: int,
        *,
        orthonormal: bool = True,
    ) -> PerturbationBasis:
        """Draw the shared basis. Sampling is on the host, so this matches PyTorch."""
        rows = sample_directions(tuple(param_shape), rank, seed, orthonormal=orthonormal)
        return cls(directions=jnp.asarray(rows), param_shape=tuple(param_shape))

    @classmethod
    def from_array(cls, directions, param_shape: tuple[int, ...]) -> PerturbationBasis:
        """Wrap an explicit basis -- used to hold the basis fixed across backends."""
        return cls(directions=jnp.asarray(directions, dtype=jnp.float64),
                   param_shape=tuple(param_shape))

    @property
    def rank(self) -> int:
        return int(self.directions.shape[0])

    @property
    def dim(self) -> int:
        return int(self.directions.shape[1])

    def coefficients(self, n_members: int, seed: int) -> jnp.ndarray:
        """``(M, K)`` standard-Gaussian coefficients, one row per member."""
        return jnp.asarray(sample_coefficients(n_members, self.rank, seed))

    def delta(self, coefficient, scale: float) -> jnp.ndarray:
        """``scale * reshape(z @ U)`` for one member, materialised on demand."""
        flat = scale * (jnp.asarray(coefficient, dtype=jnp.float64) @ self.directions)
        return flat.reshape(self.param_shape)

    def deltas(self, coefficients, scale: float) -> jnp.ndarray:
        """All members at once: ``(M, *param_shape)``."""
        flat = scale * (jnp.asarray(coefficients, dtype=jnp.float64) @ self.directions)
        return flat.reshape((-1, *self.param_shape))

    def perturb(self, weight, coefficient, scale: float) -> jnp.ndarray:
        """``W + dW`` for one member, preserving ``weight``'s dtype."""
        weight = jnp.asarray(weight)
        return (weight + self.delta(coefficient, scale).reshape(weight.shape)).astype(weight.dtype)

    def scale_for_relative_magnitude(self, weight, coefficients, target_relative: float) -> float:
        """Scale making the median member satisfy ``||dW||_F/||W||_F == target``."""
        return scale_for_relative_magnitude(
            np.asarray(self.directions), np.asarray(weight), np.asarray(coefficients),
            target_relative,
        )
