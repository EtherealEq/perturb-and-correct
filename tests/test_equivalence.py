"""The two implementations must agree.

`pnc.jax` and `pnc.torch` are separate code, so nothing structural stops them from
drifting apart. What keeps them honest is that both draw their randomness from
`pnc.spec` on the host: a seed means the same thing in either, so for identical inputs
they must produce *identical members*, not merely similar ones.

Both are also checked against `pnc.residual`, the NumPy reference implementation of
Section 4, which shares no code with either.

Tolerances are ~1e-10 rather than exact: the two libraries order floating-point
reductions differently, so bit-identity is not on offer, but anything looser than this
would mean a genuine divergence.
"""
import numpy as np
import pytest

import pnc.jax as pj
import pnc.torch as pt
from pnc.residual import augment as np_augment
from pnc.residual import residual_direct, residual_identity
from pnc.residual import ridge_solve as np_ridge_solve
from pnc.spec import CorrectionSpec

from .conftest import asnp, calibration, make_conv_case, make_ffn, make_stack

SPEC = CorrectionSpec(ridge=1e-3, bootstrap_fraction=0.1, seed=0)
TOL = 1e-10


def test_ridge_solve_agrees_across_backends_and_with_numpy():
    rng = np.random.RandomState(0)
    design = np_augment(rng.standard_normal((150, 10)))
    target = rng.standard_normal((150, 4))
    prior = rng.standard_normal((11, 4))
    reference = np_ridge_solve(design, target, 0.3, prior=prior)
    assert np.allclose(asnp(pj.ridge_solve(design, target, 0.3, prior=prior)), reference, atol=TOL)
    assert np.allclose(asnp(pt.ridge_solve(design, target, 0.3, prior=prior)), reference, atol=TOL)


def test_augment_agrees_across_backends():
    rows = np.random.RandomState(1).standard_normal((20, 6))
    assert np.allclose(asnp(pj.augment(rows)), np_augment(rows))
    assert np.allclose(asnp(pt.augment(rows)), np_augment(rows))


def test_perturbation_basis_is_identical_for_a_given_seed():
    """A seed names the same subspace in either implementation."""
    for orthonormal in (True, False):
        a = pj.PerturbationBasis.random((12, 9), rank=4, seed=5, orthonormal=orthonormal)
        b = pt.PerturbationBasis.random((12, 9), rank=4, seed=5, orthonormal=orthonormal)
        assert np.allclose(asnp(a.directions), asnp(b.directions))
        assert np.allclose(asnp(a.coefficients(7, seed=6)), asnp(b.coefficients(7, seed=6)))
        assert np.allclose(
            asnp(a.deltas(a.coefficients(7, seed=6), 2.0)),
            asnp(b.deltas(b.coefficients(7, seed=6), 2.0)),
        )


@pytest.mark.parametrize("perturb_layers", [[0], [0, 2]])
def test_dense_ensembles_are_identical(perturb_layers):
    """Same seed, same data -> the same members, including the sequential chain."""
    inputs = calibration(n=400)
    kwargs = dict(perturb_layers=perturb_layers, rank=8, n_members=6, scale=1.0,
                  spec=SPEC, seed=0)
    jax_ensemble = pj.build_dense_pnc(make_stack(pj), inputs, **kwargs)
    torch_ensemble = pt.build_dense_pnc(make_stack(pt), inputs, **kwargs)

    for layer in perturb_layers:
        assert np.allclose(asnp(jax_ensemble.deltas[layer]), asnp(torch_ensemble.deltas[layer]))
        jax_weight, jax_bias = jax_ensemble.corrections[layer + 1]
        torch_weight, torch_bias = torch_ensemble.corrections[layer + 1]
        assert np.allclose(asnp(jax_weight), asnp(torch_weight), atol=TOL)
        assert np.allclose(asnp(jax_bias), asnp(torch_bias), atol=TOL)

    assert np.allclose(
        asnp(jax_ensemble.predict(inputs)["mean"]),
        asnp(torch_ensemble.predict(inputs)["mean"]),
        atol=TOL,
    )


def test_dense_uncorrected_ablation_is_identical():
    inputs = calibration(n=300)
    kwargs = dict(perturb_layers=[0], rank=6, n_members=4, scale=1.5, spec=SPEC,
                  seed=3, correct=False)
    assert np.allclose(
        asnp(pj.build_dense_pnc(make_stack(pj), inputs, **kwargs).predict(inputs)["mean"]),
        asnp(pt.build_dense_pnc(make_stack(pt), inputs, **kwargs).predict(inputs)["mean"]),
        atol=TOL,
    )


def test_ffn_members_are_identical():
    hidden = np.random.RandomState(2).standard_normal((500, 16))
    targets = asnp(make_ffn(pj).forward(hidden))
    kwargs = dict(rank=8, n_members=4, relative_magnitude=0.5, spec=SPEC, seed=0)
    jax_members, jax_basis, jax_scale = pj.build_ffn_pnc(make_ffn(pj), hidden, targets, **kwargs)
    torch_members, torch_basis, torch_scale = pt.build_ffn_pnc(
        make_ffn(pt), hidden, targets, **kwargs
    )

    assert jax_scale == pytest.approx(torch_scale, rel=1e-12)
    assert np.allclose(asnp(jax_basis.directions), asnp(torch_basis.directions))
    for jax_member, torch_member in zip(jax_members, torch_members, strict=True):
        assert np.allclose(asnp(jax_member.coefficient), asnp(torch_member.coefficient))
        assert np.allclose(asnp(jax_member.w2), asnp(torch_member.w2), atol=TOL)
        assert np.allclose(asnp(jax_member.b2), asnp(torch_member.b2), atol=TOL)
        assert jax_member.relative_magnitude == pytest.approx(
            torch_member.relative_magnitude, rel=1e-12
        )


def test_conv_members_are_identical_up_to_kernel_layout():
    """The backends take different kernel layouts by design; the members they imply
    are the same, so the block-output error must match."""
    jax_case = make_conv_case(pj)
    torch_case = make_conv_case(pt)
    kwargs = dict(rank=6, n_members=3, scale=1.0, spec=SPEC, seed=0)
    jax_members = pj.build_conv_pnc(
        jax_case["block"], jax_case["chunks"], jax_case["targets"], **kwargs
    )
    torch_members = pt.build_conv_pnc(
        torch_case["block"], torch_case["chunks"], torch_case["targets"], **kwargs
    )
    for jax_member, torch_member in zip(jax_members, torch_members, strict=True):
        assert np.allclose(asnp(jax_member.b2), asnp(torch_member.b2), atol=TOL)
    assert jax_case["block_error"](jax_members) == pytest.approx(
        torch_case["block_error"](torch_members), rel=1e-8
    )


def test_both_backends_satisfy_the_exact_residual_identity():
    """Eq. (7) against the NumPy reference, for the corrections each backend fits.

    This ties the two implementations to the theory rather than only to each other.
    """
    rng = np.random.RandomState(4)
    width = 10
    direction = rng.standard_normal((width, width)) * 0.4
    features = rng.standard_normal((120, width))
    test_features = rng.standard_normal((25, width)) * 1.7

    design = np_augment(np.maximum(features, 0.0))
    design_v = np_augment(np.maximum(features @ (np.eye(width) + direction), 0.0))
    test = np_augment(np.maximum(test_features, 0.0))
    test_v = np_augment(np.maximum(test_features @ (np.eye(width) + direction), 0.0))
    theta = rng.standard_normal((width + 1, 4))
    ridge = 1.0

    expected = residual_identity(design, design_v, theta, test, test_v, ridge)
    assert np.allclose(
        residual_direct(design, design_v, theta, test, test_v, ridge), expected, atol=1e-10
    )
    for backend in (pj, pt):
        fitted = asnp(backend.ridge_solve(design_v, design @ theta, ridge, prior=theta))
        assert np.allclose(test_v @ fitted - test @ theta, expected, atol=1e-9)
