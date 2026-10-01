"""Section 4's identities, checked against the solver that implements them.

The paper reports the Eq. (7) identity holding to ~1e-13 relative error in float64
on real networks; these tests assert the same agreement on synthetic problems, which
is where a convention slip in the solver would show up first.
"""
import numpy as np
import pytest

from pnc.residual import (
    augment,
    disagreement,
    reconstruction_weights,
    residual_bound,
    residual_direct,
    residual_identity,
    ridge_leverage,
    ridge_solve,
    transfer_defect,
)


def _setup(seed=0, n_cal=120, width=10, out=4, n_test=25, perturbation=0.4):
    """A correction problem with a genuinely nonlinear representation change."""
    rng = np.random.RandomState(seed)
    features = rng.standard_normal((n_cal, width))
    test_features = rng.standard_normal((n_test, width)) * 1.7  # deliberately shifted
    direction = rng.standard_normal((width, width)) * perturbation

    def represent(x, perturbed):
        # ReLU makes the change nonlinear in the perturbation, as in a real network.
        return np.maximum(x @ (np.eye(width) + direction) if perturbed else x, 0.0)

    design = augment(represent(features, False))
    design_v = augment(represent(features, True))
    test = augment(represent(test_features, False))
    test_v = augment(represent(test_features, True))
    theta = rng.standard_normal((width + 1, out))
    return design, design_v, theta, test, test_v


@pytest.mark.parametrize("ridge", [1e-2, 1.0, 100.0])
def test_exact_residual_identity(ridge):
    """Eq. (7): applying the fitted correction and reconstructing it agree exactly."""
    design, design_v, theta, test, test_v = _setup()
    direct = residual_direct(design, design_v, theta, test, test_v, ridge)
    identity = residual_identity(design, design_v, theta, test, test_v, ridge)
    relative = np.linalg.norm(direct - identity) / np.linalg.norm(direct)
    assert relative < 1e-12, f"relative error {relative:.2e}"


def test_identity_holds_at_large_perturbation():
    """The identity uses the actual representation change, so it does not degrade as
    the perturbation grows -- unlike a derivative-at-zero approximation."""
    for magnitude in (0.1, 1.0, 5.0):
        design, design_v, theta, test, test_v = _setup(perturbation=magnitude)
        direct = residual_direct(design, design_v, theta, test, test_v, 1.0)
        identity = residual_identity(design, design_v, theta, test, test_v, 1.0)
        assert np.linalg.norm(direct - identity) / np.linalg.norm(direct) < 1e-12


def test_residual_is_small_on_the_calibration_set_itself():
    """Points the correction was fit on are where the repair works best."""
    design, design_v, theta, test, test_v = _setup()
    on_calibration = residual_direct(design, design_v, theta, design, design_v, 1e-3)
    off_calibration = residual_direct(design, design_v, theta, test, test_v, 1e-3)
    assert np.linalg.norm(on_calibration, axis=1).mean() < np.linalg.norm(
        off_calibration, axis=1
    ).mean()


def test_transfer_defect_is_smaller_where_calibration_data_predict_it():
    """Eq. (8) is the mechanism claim: the correction set reconstructs the
    perturbation effect well on rows like the ones it was fit on, and less well on
    shifted rows, so the defect -- and hence the residual -- grows under shift."""
    design, design_v, _, test, test_v = _setup()
    on_calibration = transfer_defect(design, design_v, design, design_v, ridge=1e-3)
    off_calibration = transfer_defect(design, design_v, test, test_v, ridge=1e-3)
    assert np.linalg.norm(on_calibration, axis=1).mean() < np.linalg.norm(
        off_calibration, axis=1
    ).mean()


def test_correction_interpolates_when_the_design_is_underdetermined():
    """With fewer correction rows than affine design dimensions the ridge solve
    approaches interpolation and the calibration residual collapses -- the
    conditioning regime the bootstrap-fraction sweeps run into (Appendix D.4)."""
    design, design_v, theta, _, _ = _setup(n_cal=8, width=20)
    residual = residual_direct(
        design, design_v, theta, design, design_v, ridge=1e-10
    )
    assert np.linalg.norm(residual) < 1e-4


def test_theorem_2_bound_holds():
    design, design_v, theta, test, test_v = _setup()
    for ridge in (1e-2, 1.0, 100.0):
        residual = residual_direct(design, design_v, theta, test, test_v, ridge)
        actual = np.linalg.norm(residual, axis=1)
        bound = residual_bound(design, design_v, theta, test, test_v, ridge)
        assert np.all(actual <= bound + 1e-9)


def test_reconstruction_weight_norm_is_bounded_by_leverage():
    """Eq. (11): ||alpha||^2 <= h^lambda."""
    design, design_v, _, test, test_v = _setup()
    for ridge in (1e-2, 1.0, 100.0):
        alpha = reconstruction_weights(design_v, test_v, ridge)
        leverage = ridge_leverage(design_v, test_v, ridge)
        assert np.all((alpha ** 2).sum(axis=1) <= leverage + 1e-9)


def test_leverage_is_larger_off_the_calibration_geometry():
    design, design_v, _, test, test_v = _setup()
    off_calibration = ridge_leverage(design_v, test_v, 1.0).mean()
    on_calibration = ridge_leverage(design_v, design_v, 1.0).mean()
    assert off_calibration > on_calibration


def test_disagreement_is_twice_the_covariance_trace():
    """Eq. (14): E||r_V(x) - r_V'(x)||^2 = 2 tr K_S(x, x)."""
    rng = np.random.RandomState(4)
    outputs = rng.standard_normal((200, 7, 3))
    reported = disagreement(outputs)
    manual = np.array(
        [
            2.0 * np.trace(np.cov(outputs[:, i, :], rowvar=False, ddof=1))
            for i in range(outputs.shape[1])
        ]
    )
    assert np.allclose(reported, manual)


def test_disagreement_requires_two_members():
    with pytest.raises(ValueError):
        disagreement(np.zeros((1, 4, 2)))


def test_reference_solver_enforces_the_same_guards_as_the_backends():
    """Regression: this module is the specification `pnc.jax` and `pnc.torch` are
    checked against, so it must reject what they reject -- a reference that accepted a
    negative ridge could not catch a convention slip."""
    with pytest.raises(ValueError):
        ridge_solve(np.zeros((5, 3)), np.zeros((4, 2)), 1.0)
    with pytest.raises(ValueError):
        ridge_solve(np.zeros((5, 3)), np.zeros((5, 2)), -1.0)
