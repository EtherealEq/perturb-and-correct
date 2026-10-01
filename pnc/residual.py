"""Exact post-correction residual identities (Section 4).

These are the quantities the theory is stated in.  They are kept here mainly so the
correction solver can be checked against a closed form rather than against itself:
:func:`residual_direct` evaluates the residual by actually applying the fitted
correction, :func:`residual_identity` reconstructs the same number from Lemma 1, and
the two agree to float64 precision.  ``tests/test_residual.py`` asserts that.

Notation follows the paper, with ``S`` the correction subset and ``v`` the applied
perturbation:

    r_S(x; v) = Theta_hat_S(v) ybar_v(x) - Theta ybar(x)          (Eq. 4)
              = Theta [ dybar_v(x) - dY_v,S^T alpha_v,S(x) ]      (Eq. 7)
              = Theta g_S(x; v)                                   (Eq. 8)

All arrays are row-major: designs are ``(n, p)`` with the bias column first, and
``Theta`` is ``(p, d)`` so that pre-activations are ``design @ Theta``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "augment",
    "ridge_solve",
    "residual_direct",
    "residual_identity",
    "transfer_defect",
    "reconstruction_weights",
    "ridge_leverage",
    "residual_bound",
    "disagreement",
]


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(np.atleast_2d(np.asarray(a, dtype=np.float64)))


def ridge_solve(design, target, ridge: float, prior=None) -> np.ndarray:
    """NumPy reference for the correction solve of Eq. (1).

    Deliberately independent of :mod:`pnc.jax` and :mod:`pnc.torch` -- this module is
    the specification those two are checked against, so it must not share their code.
    It does enforce the same preconditions, though: a reference that accepted inputs
    the implementations reject could not catch a convention slip.
    """
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if design.shape[0] != target.shape[0]:
        raise ValueError(
            f"design has {design.shape[0]} rows but target has {target.shape[0]}"
        )
    if ridge < 0.0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")
    if ridge == 0.0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    rhs = design.T @ target
    if prior is not None:
        rhs = rhs + ridge * np.asarray(prior, dtype=np.float64)
    return np.linalg.solve(gram, rhs)


def augment(rows) -> np.ndarray:
    """Prepend the constant column, bias first, as in Eq. (2)."""
    rows = np.asarray(rows, dtype=np.float64)
    return np.concatenate([np.ones((rows.shape[0], 1)), rows], axis=1)


def _fit(design_v, design, theta, ridge, center_on_original):
    prior = theta if center_on_original else None
    return ridge_solve(design_v, design @ theta, ridge, prior=prior)


def residual_direct(
    design: np.ndarray,
    design_v: np.ndarray,
    theta: np.ndarray,
    test: np.ndarray,
    test_v: np.ndarray,
    ridge: float,
    *,
    center_on_original: bool = True,
) -> np.ndarray:
    """``r_S(x; v)`` by applying the fitted correction (Eq. 4). Ground truth."""
    theta = np.asarray(theta, dtype=np.float64)
    fitted = _fit(_f64(design_v), _f64(design), theta, ridge, center_on_original)
    return _f64(test_v) @ fitted - _f64(test) @ theta


def reconstruction_weights(
    design_v: np.ndarray,
    test_v: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """``alpha_v,S(x) = Y_v G_v,S^-1 ybar_v(x)`` (Eq. 6), shape ``(n_test, n_cal)``.

    How the least-squares fit reconstructs the test point's perturbation effect as a
    combination of the effects observed on the correction set.
    """
    design_v = _f64(design_v)
    gram = design_v.T @ design_v + ridge * np.eye(design_v.shape[1])
    return _f64(test_v) @ np.linalg.solve(gram, design_v.T)


def transfer_defect(
    design: np.ndarray,
    design_v: np.ndarray,
    test: np.ndarray,
    test_v: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """``g_S(x; v)`` (Eq. 8): the part of the representation change the correction
    set fails to reconstruct.

    Small where calibration examples predict the perturbation effect at ``x`` -- so
    the repair keeps suppressing the perturbation -- and large where that
    relationship does not transfer.  An OOD point with a small defect legitimately
    produces little disagreement; the paper does not claim otherwise.
    """
    delta_design = _f64(design_v) - _f64(design)
    delta_test = _f64(test_v) - _f64(test)
    alpha = reconstruction_weights(design_v, test_v, ridge)
    return delta_test - alpha @ delta_design


def residual_identity(
    design: np.ndarray,
    design_v: np.ndarray,
    theta: np.ndarray,
    test: np.ndarray,
    test_v: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """``r_S(x; v) = Theta g_S(x; v)`` (Eq. 7).

    Exact for the *applied* perturbation: no Taylor expansion in ``v``, no assumption
    that activations stay in a fixed linear region.  Valid for the original-centred
    ridge with ``lambda > 0``, which is the algorithm the paper reports.
    """
    return transfer_defect(design, design_v, test, test_v, ridge) @ np.asarray(
        theta, dtype=np.float64
    )


def ridge_leverage(design_v: np.ndarray, test_v: np.ndarray, ridge: float) -> np.ndarray:
    """``h^lambda_v,S(x) = ybar_v(x)^T G_v,S^-1 ybar_v(x)`` (Eq. 9), shape ``(n_test,)``.

    How well the correction design covers the perturbed test representation.
    """
    design_v = _f64(design_v)
    test_v = _f64(test_v)
    gram = design_v.T @ design_v + ridge * np.eye(design_v.shape[1])
    return np.einsum("ij,ij->i", test_v, np.linalg.solve(gram, test_v.T).T)


def residual_bound(
    design: np.ndarray,
    design_v: np.ndarray,
    theta: np.ndarray,
    test: np.ndarray,
    test_v: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """The Theorem 2 upper bound on ``||r_S(x; v)||_2``.

        ||Theta||_op * ( delta_x(v) + Delta_S(v) * sqrt(h^lambda_v,S(x)) )

    Pointwise in the applied perturbation, with no Taylor remainder.
    """
    theta_op = float(np.linalg.norm(np.asarray(theta, dtype=np.float64), ord=2))
    design_change = float(np.linalg.norm(_f64(design_v) - _f64(design)))
    test_change = np.linalg.norm(_f64(test_v) - _f64(test), axis=1)
    leverage = ridge_leverage(design_v, test_v, ridge)
    return theta_op * (test_change + design_change * np.sqrt(np.maximum(leverage, 0.0)))


def disagreement(member_outputs: np.ndarray) -> np.ndarray:
    """Expected squared pairwise disagreement per input, ``2 tr K_S(x, x)`` (Eq. 14).

    Equal to twice the diagonal mass of the covariance kernel of Theorem 4, estimated
    from ``member_outputs`` of shape ``(n_members, n_inputs, n_outputs)``.  The
    unbiased (``ddof=1``) member covariance is used, matching Theorem 4's statement
    that the sample covariance is unbiased for ``K_S``.
    """
    outputs = np.asarray(member_outputs, dtype=np.float64)
    if outputs.ndim != 3:
        raise ValueError("member_outputs must be (n_members, n_inputs, n_outputs)")
    if outputs.shape[0] < 2:
        raise ValueError("pairwise disagreement needs at least two members")
    centered = outputs - outputs.mean(axis=0, keepdims=True)
    trace_cov = (centered ** 2).sum(axis=(0, 2)) / (outputs.shape[0] - 1)
    return 2.0 * trace_cov
