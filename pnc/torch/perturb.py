"""Random low-dimensional perturbations of a hidden-layer parameter (PyTorch)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..spec import sample_coefficients, sample_directions, scale_for_relative_magnitude

__all__ = ["PerturbationBasis"]


@dataclass(frozen=True)
class PerturbationBasis:
    """A shared rank-``K`` perturbation subspace for one parameter tensor.

    One basis is drawn and shared across members; members differ only in their
    Gaussian coefficient vector ``z``, so an ensemble costs ``K * D`` floats for the
    basis plus ``K`` floats per member rather than ``M`` copies of the parameter.

        dW = sigma * reshape(z @ U),   z ~ N(0, I_K)

    Draws come from :mod:`pnc.spec`, so a given seed yields exactly the basis the JAX
    implementation produces.
    """

    directions: torch.Tensor
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
        device=None,
    ) -> PerturbationBasis:
        """Draw the shared basis. Sampling is on the host, so this matches JAX."""
        rows = sample_directions(tuple(param_shape), rank, seed, orthonormal=orthonormal)
        return cls(
            directions=torch.as_tensor(rows, dtype=torch.float64, device=device),
            param_shape=tuple(param_shape),
        )

    @classmethod
    def from_array(cls, directions, param_shape, device=None) -> PerturbationBasis:
        """Wrap an explicit basis -- used to hold the basis fixed across backends."""
        return cls(
            directions=torch.as_tensor(
                np.asarray(directions), dtype=torch.float64, device=device
            ),
            param_shape=tuple(param_shape),
        )

    @property
    def rank(self) -> int:
        return int(self.directions.shape[0])

    @property
    def dim(self) -> int:
        return int(self.directions.shape[1])

    def coefficients(self, n_members: int, seed: int) -> torch.Tensor:
        """``(M, K)`` standard-Gaussian coefficients, one row per member."""
        return torch.as_tensor(
            sample_coefficients(n_members, self.rank, seed),
            dtype=torch.float64,
            device=self.directions.device,
        )

    @torch.no_grad()
    def delta(self, coefficient, scale: float) -> torch.Tensor:
        """``scale * reshape(z @ U)`` for one member, materialised on demand."""
        coefficient = torch.as_tensor(
            coefficient, dtype=torch.float64, device=self.directions.device
        )
        return (scale * (coefficient @ self.directions)).reshape(self.param_shape)

    @torch.no_grad()
    def deltas(self, coefficients, scale: float) -> torch.Tensor:
        """All members at once: ``(M, *param_shape)``."""
        coefficients = torch.as_tensor(
            coefficients, dtype=torch.float64, device=self.directions.device
        )
        return (scale * (coefficients @ self.directions)).reshape((-1, *self.param_shape))

    @torch.no_grad()
    def perturb(self, weight, coefficient, scale: float) -> torch.Tensor:
        """``W + dW`` for one member, preserving ``weight``'s dtype."""
        weight = torch.as_tensor(weight)
        delta = self.delta(coefficient, scale).reshape(weight.shape)
        return (weight.to(torch.float64) + delta).to(weight.dtype)

    def scale_for_relative_magnitude(self, weight, coefficients, target_relative: float) -> float:
        """Scale making the median member satisfy ``||dW||_F/||W||_F == target``."""
        return scale_for_relative_magnitude(
            self.directions.cpu().numpy(),
            torch.as_tensor(weight).detach().cpu().numpy(),
            torch.as_tensor(coefficients).detach().cpu().numpy(),
            target_relative,
        )
