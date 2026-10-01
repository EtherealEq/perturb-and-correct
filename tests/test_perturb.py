"""Perturbation sampling: shared basis, member coefficients, magnitude conventions."""
import numpy as np
import pytest

from pnc.spec import sample_directions

from .conftest import asnp


def test_orthonormal_basis_is_orthonormal(backend):
    basis = backend.PerturbationBasis.random((8, 6), rank=5, seed=0)
    assert np.allclose(asnp(basis.directions) @ asnp(basis.directions).T, np.eye(5), atol=1e-12)


def test_unit_row_basis_has_unit_rows(backend):
    basis = backend.PerturbationBasis.random((64, 64), rank=10, seed=0, orthonormal=False)
    assert np.allclose(np.linalg.norm(asnp(basis.directions), axis=1), 1.0)


def test_basis_is_shared_and_only_coefficients_vary(backend):
    """The storage argument: one K x D basis plus K floats per member."""
    basis = backend.PerturbationBasis.random((10, 10), rank=4, seed=1)
    coefficients = basis.coefficients(6, seed=2)
    assert tuple(coefficients.shape) == (6, 4)
    stacked = asnp(basis.deltas(coefficients, 1.0)).reshape(6, -1)
    assert np.linalg.matrix_rank(stacked, tol=1e-9) == 4  # all members in one subspace


def test_deltas_agrees_with_per_member_delta(backend):
    basis = backend.PerturbationBasis.random((5, 5), rank=3, seed=3)
    coefficients = basis.coefficients(4, seed=4)
    batched = asnp(basis.deltas(coefficients, 2.5))
    for index in range(4):
        assert np.allclose(batched[index], asnp(basis.delta(coefficients[index], 2.5)))


def test_delta_scales_linearly(backend):
    basis = backend.PerturbationBasis.random((5, 5), rank=3, seed=3)
    z = basis.coefficients(1, seed=4)[0]
    assert np.allclose(asnp(basis.delta(z, 2.5)), 2.5 * asnp(basis.delta(z, 1.0)))


def test_perturb_preserves_shape_and_dtype(backend):
    basis = backend.PerturbationBasis.random((3, 4), rank=2, seed=5)
    weight = np.ones((3, 4), dtype=np.float32)
    perturbed = basis.perturb(weight, basis.coefficients(1, seed=6)[0], 0.1)
    assert tuple(perturbed.shape) == (3, 4)
    assert asnp(perturbed).dtype == np.float32


def test_relative_magnitude_calibration_hits_the_target_at_the_median(backend):
    basis = backend.PerturbationBasis.random((40, 40), rank=20, seed=7, orthonormal=False)
    weight = np.random.RandomState(8).standard_normal((40, 40))
    coefficients = basis.coefficients(64, seed=9)
    scale = basis.scale_for_relative_magnitude(weight, coefficients, target_relative=2.0)
    achieved = np.linalg.norm(asnp(basis.deltas(coefficients, scale)).reshape(64, -1), axis=1)
    assert np.median(achieved / np.linalg.norm(weight)) == pytest.approx(2.0, rel=1e-9)


def test_rank_cannot_exceed_parameter_dimension(backend):
    with pytest.raises(ValueError):
        backend.PerturbationBasis.random((2, 2), rank=5, seed=0)


def test_shape_mismatch_is_rejected(backend):
    with pytest.raises(ValueError):
        backend.PerturbationBasis.from_array(np.zeros((2, 7)), (3, 3))


def test_basis_is_reproducible_from_the_seed(backend):
    a = backend.PerturbationBasis.random((6, 6), rank=3, seed=11)
    b = backend.PerturbationBasis.random((6, 6), rank=3, seed=11)
    assert np.array_equal(asnp(a.directions), asnp(b.directions))
    assert np.array_equal(asnp(a.coefficients(4, seed=12)), asnp(b.coefficients(4, seed=12)))


def test_sampling_lives_on_the_host_so_backends_agree(backend):
    """Draws come from pnc.spec, not from jax.random or torch.Generator, which is
    what makes a seed mean the same thing in both implementations."""
    reference = sample_directions((6, 6), 3, seed=11)
    assert np.allclose(asnp(backend.PerturbationBasis.random((6, 6), 3, seed=11).directions),
                       reference)
