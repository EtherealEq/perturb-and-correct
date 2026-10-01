"""The least-squares correction, in PyTorch.

Mathematically identical to :mod:`pnc.jax.correct`; see that module for the full
statement of Eq. (1)/(3) and why the target is the base model's own pre-activation
and the ridge is centred on the original affine map rather than on zero.

Everything is computed in ``torch.float64`` regardless of the model's dtype: the Gram
matrix reaches 4,609 x 4,609 for the CIFAR blocks and 3,073 x 3,073 for the
transformer FFNs, and in float32 those are badly enough conditioned to change the
answer. Unlike JAX, no global flag is involved -- the dtype is requested explicitly.
"""
from __future__ import annotations

import torch

__all__ = ["augment", "ridge_solve", "NormalEquations"]


def _f64(a) -> torch.Tensor:
    return torch.as_tensor(a).detach().to(torch.float64)


def augment(rows) -> torch.Tensor:
    """Prepend the constant column: ``(n, d) -> (n, 1 + d)``.

    Bias first, so a solution has the ``[b; W]`` layout of Eq. (2). That constant
    column is what lets the repair absorb perturbation-induced mean shifts, which is
    why a bias is fit even for a bias-free base convolution.
    """
    rows = torch.as_tensor(rows)
    ones = torch.ones((*rows.shape[:-1], 1), dtype=rows.dtype, device=rows.device)
    return torch.cat([ones, rows], dim=-1)


@torch.no_grad()
def ridge_solve(design, target, ridge: float, prior=None) -> torch.Tensor:
    """Solve ``design @ Theta ~= target`` with a ridge centred on ``prior``.

    ``ridge > 0``::

        Theta = (X^T X + lambda I)^-1 (X^T target + lambda * prior)

    ``ridge == 0`` returns the minimum-norm least-squares solution via the
    pseudoinverse, which stays defined when the design is rank deficient -- as
    post-ReLU designs routinely are, and unlike ``torch.linalg.lstsq``'s default
    driver, which assumes full rank. The unregularised solve is the numerically
    fragile case in any event: the CIFAR-10 sweep finds it singular or unstable, while
    any ``lambda >= 1e-3`` behaves near-identically.

    Leading batch dimensions are allowed on ``design`` and ``target`` and are solved
    together, which is how the dense backend does all members in one call.
    """
    design, target = _f64(design), _f64(target)
    if design.shape[-2] != target.shape[-2]:
        raise ValueError(
            f"design has {design.shape[-2]} rows but target has {target.shape[-2]}"
        )
    if ridge < 0.0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")
    if ridge == 0.0:
        return torch.linalg.pinv(design) @ target

    swap = design.transpose(-1, -2)
    eye = torch.eye(design.shape[-1], dtype=torch.float64, device=design.device)
    gram = swap @ design + ridge * eye
    rhs = swap @ target
    if prior is not None:
        rhs = rhs + ridge * _f64(prior).to(rhs.device)
    return torch.linalg.solve(gram, rhs)


class NormalEquations:
    """Streaming accumulator for ``G = X^T X`` and ``C = X^T target``.

    The unfolded design for a ResNet block has ``|S| * H' * W'`` rows, and a
    197-token ViT batch has ``|S| * 197``; neither needs to be held, since the ridge
    solution depends on the data only through these two statistics. Chunk size
    therefore changes peak memory and nothing else -- the accumulated system, and so
    the solution, is identical (Appendix B.4).
    """

    def __init__(self, n_features: int, n_outputs: int, device=None) -> None:
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        options = {"dtype": torch.float64, "device": device}
        self.gram = torch.zeros((self.n_features, self.n_features), **options)
        self.cross = torch.zeros((self.n_features, self.n_outputs), **options)
        self.target_sq = 0.0
        self.n_rows = 0

    @torch.no_grad()
    def update(self, design_block, target_block) -> NormalEquations:
        """Fold one chunk of rows into the accumulated statistics."""
        design_block, target_block = _f64(design_block), _f64(target_block)
        if design_block.shape[1] != self.n_features:
            raise ValueError(
                f"design block has {design_block.shape[1]} features, expected {self.n_features}"
            )
        if target_block.shape[1] != self.n_outputs:
            raise ValueError(
                f"target block has {target_block.shape[1]} outputs, expected {self.n_outputs}"
            )
        self.gram = self.gram.to(design_block.device)
        self.cross = self.cross.to(design_block.device)
        self.gram += design_block.T @ design_block
        self.cross += design_block.T @ target_block
        self.target_sq += float((target_block ** 2).sum())
        self.n_rows += int(design_block.shape[0])
        return self

    @torch.no_grad()
    def solve(self, ridge: float, prior=None) -> torch.Tensor:
        """Solve the accumulated system; identical to :func:`ridge_solve`."""
        if ridge < 0.0:
            raise ValueError(f"ridge must be non-negative, got {ridge}")
        eye = torch.eye(self.n_features, dtype=torch.float64, device=self.gram.device)
        matrix = self.gram + ridge * eye
        rhs = self.cross if prior is None else self.cross + ridge * _f64(prior).to(self.gram.device)
        return torch.linalg.solve(matrix, rhs)

    @torch.no_grad()
    def relative_residual(self, solution) -> float:
        """``||X Theta - target||_F / ||target||_F`` from the statistics alone.

        Uses ``||X Th - T||^2 = tr(Th^T G Th) - 2 tr(Th^T C) + ||T||^2``, so the
        design is still never materialised.

        That identity subtracts nearly-equal large quantities, so it loses roughly
        half the available digits as the residual approaches zero: on a perfectly
        reproducible target it bottoms out around ``sqrt(eps)`` rather than ``eps``.
        It is a diagnostic for how well a correction fits, not a way to detect an
        exact fit.
        """
        solution = _f64(solution).to(self.gram.device)
        squared = float(
            torch.trace(solution.T @ self.gram @ solution)
            - 2.0 * torch.trace(solution.T @ self.cross)
            + self.target_sq
        )
        return float(max(squared, 0.0) ** 0.5 / (self.target_sq ** 0.5 + 1e-30))
