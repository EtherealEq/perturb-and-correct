"""The least-squares correction, in JAX.

After a hidden-layer perturbation changes a representation ``y -> y_v``, the
*following* affine layer is refit on ID calibration data so the corrected predictor
reproduces the base model's next-layer pre-activations there. Writing that layer in
the augmented form of Eq. (2), ``z(x) = Theta ybar(x)`` with ``ybar = [1; y]`` and
``Theta = [b, W]``, the correction solves Eq. (1)/(3):

    Theta_hat(v) = argmin || Y_v Theta'^T - Y Theta^T ||_F^2 + lambda || Theta' - Theta ||_F^2

Two things are load-bearing and easy to get wrong:

1.  **The target is the base model's own pre-activation**, ``Y Theta^T`` -- never a
    label. The correction is label-free.
2.  **The ridge is centred on the original affine map**, so ``lambda -> inf`` recovers
    the base model rather than the zero map.

Everything runs in float64 (see :mod:`pnc.jax`). For designs too large to materialise
-- the unfolded conv patch matrix, or many-token transformer batches -- accumulate
with :class:`NormalEquations`, which solves the identical system from sufficient
statistics.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["augment", "ridge_solve", "NormalEquations", "require_x64"]


def require_x64() -> None:
    """Fail loudly rather than silently solving an ill-conditioned system in f32."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "pnc.jax requires JAX x64 mode. Importing pnc.jax enables it, so this "
            "means it was disabled afterwards. Re-enable with "
            'jax.config.update("jax_enable_x64", True) before creating arrays.'
        )


def _f64(a) -> jnp.ndarray:
    return jnp.asarray(a, dtype=jnp.float64)


def augment(rows) -> jnp.ndarray:
    """Prepend the constant column: ``(n, d) -> (n, 1 + d)``.

    Bias first, so a solution has the ``[b; W]`` layout of Eq. (2). That constant
    column is what lets the repair absorb perturbation-induced mean shifts, which is
    why a bias is fit even for a bias-free base convolution.
    """
    rows = jnp.asarray(rows)
    return jnp.concatenate([jnp.ones((*rows.shape[:-1], 1), rows.dtype), rows], axis=-1)


@jax.jit
def _solve_ridge(gram, rhs):
    return jnp.linalg.solve(gram, rhs)


def ridge_solve(design, target, ridge: float, prior=None) -> jnp.ndarray:
    """Solve ``design @ Theta ~= target`` with a ridge centred on ``prior``.

    ``ridge > 0``::

        Theta = (X^T X + lambda I)^-1 (X^T target + lambda * prior)

    ``ridge == 0`` falls back to minimum-norm least squares, which stays defined when
    the design is rank deficient -- as post-ReLU designs routinely are. The
    unregularised solve is the numerically fragile case: the CIFAR-10 sweep finds it
    singular or unstable, while any ``lambda >= 1e-3`` behaves near-identically.

    Leading batch dimensions are allowed on ``design`` and ``target`` and are solved
    together, which is how the dense backend does all members in one call.
    """
    require_x64()
    design, target = _f64(design), _f64(target)
    if design.shape[-2] != target.shape[-2]:
        raise ValueError(
            f"design has {design.shape[-2]} rows but target has {target.shape[-2]}"
        )
    if ridge < 0.0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")
    if ridge == 0.0:
        if design.ndim > 2:
            return jnp.stack(
                [jnp.linalg.lstsq(d, t, rcond=None)[0] for d, t in zip(design, target, strict=True)]
            )
        return jnp.linalg.lstsq(design, target, rcond=None)[0]

    swap = jnp.swapaxes(design, -1, -2)
    gram = swap @ design + ridge * jnp.eye(design.shape[-1], dtype=jnp.float64)
    rhs = swap @ target
    if prior is not None:
        rhs = rhs + ridge * _f64(prior)
    return _solve_ridge(gram, rhs)


class NormalEquations:
    """Streaming accumulator for ``G = X^T X`` and ``C = X^T target``.

    The unfolded design for a ResNet block has ``|S| * H' * W'`` rows, and a
    197-token ViT batch has ``|S| * 197``; neither needs to be held, since the ridge
    solution depends on the data only through these two statistics. Chunk size
    therefore changes peak memory and nothing else -- the accumulated system, and so
    the solution, is identical (Appendix B.4).
    """

    def __init__(self, n_features: int, n_outputs: int) -> None:
        require_x64()
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.gram = jnp.zeros((self.n_features, self.n_features), dtype=jnp.float64)
        self.cross = jnp.zeros((self.n_features, self.n_outputs), dtype=jnp.float64)
        self.target_sq = 0.0
        self.n_rows = 0

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
        self.gram, self.cross, contribution = _accumulate(
            self.gram, self.cross, design_block, target_block
        )
        self.target_sq += float(contribution)
        self.n_rows += int(design_block.shape[0])
        return self

    def solve(self, ridge: float, prior=None) -> jnp.ndarray:
        """Solve the accumulated system; identical to :func:`ridge_solve`."""
        if ridge < 0.0:
            raise ValueError(f"ridge must be non-negative, got {ridge}")
        matrix = self.gram + ridge * jnp.eye(self.n_features, dtype=jnp.float64)
        rhs = self.cross if prior is None else self.cross + ridge * _f64(prior)
        return _solve_ridge(matrix, rhs)

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
        solution = _f64(solution)
        squared = float(
            jnp.trace(solution.T @ self.gram @ solution)
            - 2.0 * jnp.trace(solution.T @ self.cross)
            + self.target_sq
        )
        return float(jnp.sqrt(jnp.maximum(squared, 0.0)) / (jnp.sqrt(self.target_sq) + 1e-30))


@jax.jit
def _accumulate(gram, cross, design, target):
    return (
        gram + design.T @ design,
        cross + design.T @ target,
        (target ** 2).sum(),
    )
