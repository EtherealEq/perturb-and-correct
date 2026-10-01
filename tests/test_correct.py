"""The correction solve: conventions, streaming equivalence, limiting behaviour.

Runs against both implementations.
"""
import numpy as np
import pytest

from pnc.spec import CorrectionSpec, bootstrap_indices, mean_to_sum_ridge, sum_to_mean_ridge

from .conftest import asnp


@pytest.fixture
def problem(backend):
    rng = np.random.RandomState(0)
    design = asnp(backend.augment(rng.standard_normal((200, 12))))
    theta = rng.standard_normal((13, 5))
    return design, design @ theta, theta


def test_augment_puts_bias_first(backend):
    rows = np.arange(6.0).reshape(3, 2)
    augmented = asnp(backend.augment(rows))
    assert np.allclose(augmented[:, 0], 1.0)
    assert np.allclose(augmented[:, 1:], rows)


def test_ridge_solve_matches_normal_equations(backend, problem):
    design, target, theta = problem
    direct = asnp(backend.ridge_solve(design, target, ridge=0.7, prior=theta))
    system = backend.NormalEquations(design.shape[1], target.shape[1])
    for start in range(0, design.shape[0], 37):  # deliberately ragged chunks
        system.update(design[start : start + 37], target[start : start + 37])
    assert np.allclose(direct, asnp(system.solve(ridge=0.7, prior=theta)), atol=1e-11)


def test_chunking_does_not_change_the_system(backend, problem):
    design, target, _ = problem
    solutions = []
    for chunk in (13, 64, 200):
        system = backend.NormalEquations(design.shape[1], target.shape[1])
        for start in range(0, design.shape[0], chunk):
            system.update(design[start : start + chunk], target[start : start + chunk])
        solutions.append(asnp(system.solve(ridge=1e-3)))
    assert np.allclose(solutions[0], solutions[1], atol=1e-12)
    assert np.allclose(solutions[0], solutions[2], atol=1e-12)


def test_exact_design_recovers_the_original_map(backend, problem):
    """With no perturbation the repair is a no-op at any ridge: the original map
    already attains zero loss and is also the ridge centre."""
    design, target, theta = problem
    for ridge in (1e-6, 1.0, 1e4):
        solution = asnp(backend.ridge_solve(design, target, ridge, prior=theta))
        assert np.allclose(solution, theta, atol=1e-8)


def test_large_ridge_shrinks_toward_the_original_not_zero(backend):
    rng = np.random.RandomState(1)
    design = asnp(backend.augment(rng.standard_normal((80, 6))))
    theta = rng.standard_normal((7, 3))
    target = rng.standard_normal((80, 3))  # unreachable, so the trade-off is real
    toward_original = asnp(backend.ridge_solve(design, target, 1e9, prior=theta))
    toward_zero = asnp(backend.ridge_solve(design, target, 1e9, prior=None))
    assert np.allclose(toward_original, theta, atol=1e-4)
    assert np.allclose(toward_zero, 0.0, atol=1e-4)


def test_zero_ridge_is_minimum_norm_least_squares(backend):
    rng = np.random.RandomState(2)
    design = rng.standard_normal((10, 25))  # underdetermined, rank deficient
    target = rng.standard_normal((10, 4))
    solution = asnp(backend.ridge_solve(design, target, 0.0))
    assert np.allclose(design @ solution, target, atol=1e-8)
    reference, *_ = np.linalg.lstsq(design, target, rcond=None)
    assert np.linalg.norm(solution) <= np.linalg.norm(reference) + 1e-9


def test_batched_solve_matches_looping(backend):
    """The dense backend solves every member in one call; that must equal a loop."""
    rng = np.random.RandomState(7)
    designs = asnp(backend.augment(rng.standard_normal((5, 60, 8))))
    targets = rng.standard_normal((5, 60, 3))
    prior = rng.standard_normal((9, 3))
    batched = asnp(backend.ridge_solve(designs, targets, 0.5, prior=prior))
    for index in range(5):
        single = asnp(backend.ridge_solve(designs[index], targets[index], 0.5, prior=prior))
        assert np.allclose(batched[index], single, atol=1e-10)


def test_relative_residual_needs_no_design(backend, problem):
    """The residual is recoverable from the sufficient statistics alone."""
    design, _, theta = problem
    rng = np.random.RandomState(21)
    target = rng.standard_normal((design.shape[0], theta.shape[1]))  # not reproducible
    system = backend.NormalEquations(design.shape[1], target.shape[1])
    system.update(design, target)
    solution = asnp(system.solve(ridge=1e-2, prior=theta))
    expected = np.linalg.norm(design @ solution - target) / np.linalg.norm(target)
    assert system.relative_residual(solution) == pytest.approx(expected, rel=1e-9)


def test_relative_residual_loses_precision_near_an_exact_fit(backend, problem):
    """The statistics form cancels large terms, so it bottoms out near sqrt(eps).

    Documented rather than worked around: it is a fit diagnostic, not a way to detect
    an exact fit. Anything above ~1e-6 is a genuine residual.
    """
    design, target, theta = problem  # target = design @ theta, so the fit is exact
    system = backend.NormalEquations(design.shape[1], target.shape[1])
    system.update(design, target)
    reported = system.relative_residual(asnp(system.solve(ridge=1e-9, prior=theta)))
    true_residual = np.linalg.norm(design @ theta - target) / np.linalg.norm(target)
    assert true_residual < 1e-14
    assert reported < 1e-6  # small, but nowhere near the true value


def test_rejects_bad_shapes(backend, problem):
    design, target, _ = problem
    with pytest.raises(ValueError):
        backend.ridge_solve(design, target[:10], ridge=1.0)
    with pytest.raises(ValueError):
        backend.ridge_solve(design, target, ridge=-1.0)


# -- framework-free conventions --------------------------------------------
def test_ridge_convention_roundtrip():
    assert mean_to_sum_ridge(0.0305, 32768) == pytest.approx(999.4, rel=1e-3)
    assert sum_to_mean_ridge(1000.0, 32768) == pytest.approx(0.0305, rel=1e-3)


def test_bootstrap_subsets_differ_across_members_and_are_reproducible():
    first = bootstrap_indices(1024, 0.05, seed=3, n_members=4)
    again = bootstrap_indices(1024, 0.05, seed=3, n_members=4)
    assert all(len(s) == 51 for s in first)
    assert not np.array_equal(first[0], first[1])
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))


def test_bootstrap_disabled_shares_the_full_pool():
    assert bootstrap_indices(1024, 0.0, seed=0, n_members=3) is None


def test_full_bootstrap_resamples_at_the_same_size_with_replacement():
    """fraction=1.0 is the textbook bootstrap, not a no-op: same size, resampled."""
    subsets = bootstrap_indices(1024, 1.0, seed=0, n_members=3)
    assert all(len(s) == 1024 for s in subsets)
    assert all(len(np.unique(s)) < 1024 for s in subsets)  # replacement means repeats
    assert not np.array_equal(subsets[0], subsets[1])


def test_bootstrap_fraction_above_one_is_rejected():
    with pytest.raises(ValueError):
        bootstrap_indices(1024, 1.5, seed=0, n_members=2)


def test_all_subsets_share_a_size():
    """Equal sizes are what let the dense backend batch every member's solve."""
    subsets = bootstrap_indices(1000, 0.3, seed=0, n_members=8)
    assert len({len(s) for s in subsets}) == 1


def test_correction_spec_defaults_to_the_papers_convention():
    spec = CorrectionSpec()
    assert spec.center_on_original
    theta = np.eye(3)
    assert spec.use_prior(theta) is theta
    assert CorrectionSpec(center_on_original=False).use_prior(theta) is None
