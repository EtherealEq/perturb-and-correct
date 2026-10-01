"""Member aggregation and ID-only temperature calibration."""
import warnings

import numpy as np
import pytest

from pnc.ensemble import (
    average_probabilities,
    fit_temperature,
    moment_match_gaussian,
    predictive_entropy,
    softmax,
    variance_decomposition,
)


def test_moment_matching_matches_the_mixture_moments():
    rng = np.random.RandomState(0)
    means = rng.standard_normal((30, 5, 2))
    variances = rng.gamma(2.0, size=(30, 5, 2))
    mean, variance = moment_match_gaussian(means, variances)
    assert np.allclose(mean, means.mean(axis=0))
    assert np.allclose(variance, variances.mean(axis=0) + means.var(axis=0))


def test_identical_members_carry_no_epistemic_variance():
    means = np.tile(np.array([[[1.0, 2.0]]]), (8, 4, 1))
    variances = np.full_like(means, 0.25)
    parts = variance_decomposition(means, variances)
    assert np.allclose(parts["epistemic"], 0.0, atol=1e-12)
    assert np.allclose(parts["aleatoric"], 0.25)
    assert np.allclose(parts["total"], 0.25)


def test_disagreeing_members_raise_total_variance():
    rng = np.random.RandomState(1)
    variances = np.full((10, 6, 3), 0.1)
    agreeing = np.zeros((10, 6, 3))
    disagreeing = rng.standard_normal((10, 6, 3))
    assert (
        variance_decomposition(disagreeing, variances)["total"].mean()
        > variance_decomposition(agreeing, variances)["total"].mean()
    )


def test_average_probabilities_are_normalised():
    rng = np.random.RandomState(2)
    logits = rng.standard_normal((7, 20, 4))
    probabilities = average_probabilities(logits, temperature=1.3)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_predictive_entropy_extremes():
    confident = np.array([[1.0, 0.0, 0.0]])
    uniform = np.full((1, 3), 1.0 / 3.0)
    assert predictive_entropy(confident)[0] == pytest.approx(0.0, abs=1e-8)
    assert predictive_entropy(uniform)[0] == pytest.approx(np.log(3.0), abs=1e-8)


def test_ensemble_averaging_is_not_logit_averaging():
    """Averaging probabilities, not logits, is what makes disagreement show up as
    entropy -- two confident members that disagree average to an uncertain mixture."""
    logits = np.array([[[10.0, -10.0]], [[-10.0, 10.0]]])
    averaged = average_probabilities(logits)
    assert np.allclose(averaged, 0.5, atol=1e-4)
    assert predictive_entropy(averaged)[0] > predictive_entropy(softmax(logits.mean(0)))[0] - 1e-9


def test_temperature_tracks_a_known_rescaling_of_the_logits():
    """Temperature scaling is scale-equivariant: sharpening the logits by a factor c
    must move the fitted temperature by the same factor."""
    rng = np.random.RandomState(3)
    labels = rng.randint(0, 5, size=800)
    logits = rng.standard_normal((800, 5))
    logits[np.arange(800), labels] += 3.0
    baseline = fit_temperature(logits[None], labels)
    for factor in (0.5, 2.5):
        assert fit_temperature((logits * factor)[None], labels) == pytest.approx(
            factor * baseline, rel=1e-3
        )


def test_temperature_improves_id_nll():
    """The fitted temperature is chosen on ID data and must not make ID NLL worse."""
    rng = np.random.RandomState(13)
    labels = rng.randint(0, 5, size=600)
    logits = rng.standard_normal((4, 600, 5)) * 3.0
    logits[:, np.arange(600), labels] += 4.0
    fitted = fit_temperature(logits, labels)
    from pnc.metrics import categorical_nll

    assert categorical_nll(average_probabilities(logits, fitted), labels) <= categorical_nll(
        average_probabilities(logits, 1.0), labels
    ) + 1e-9


def test_temperature_is_positive_and_defaults_sanely():
    logits = np.zeros((2, 0, 3))
    assert fit_temperature(logits, np.zeros(0, dtype=int)) == 1.0


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((4, 3)), np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        average_probabilities(np.zeros((2, 3, 4)), temperature=0.0)


def test_temperature_warns_when_it_saturates_the_bracket():
    """Regression: an untrained model's near-uniform logits drove the golden-section
    search to exactly the bracket edge (100.0), and it was returned silently. A
    boundary value means the bracket is wrong, so it must be surfaced."""
    rng = np.random.RandomState(7)
    labels = rng.randint(0, 10, size=400)
    logits = (rng.standard_normal((2, 400, 10)) * 0.01)[:]  # essentially uninformative
    with pytest.warns(RuntimeWarning, match="edge of the search bracket"):
        fit_temperature(logits, labels)


def test_temperature_does_not_warn_on_a_well_posed_problem():
    """An interior optimum must stay silent. Members that agree give one -- with
    independent per-member noise the ensemble optimum really does run below the
    default bracket, which is the case the warning above is for."""
    rng = np.random.RandomState(3)
    labels = rng.randint(0, 5, size=800)
    logits = rng.standard_normal((800, 5))
    logits[np.arange(800), labels] += 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        temperature = fit_temperature(np.repeat(logits[None], 3, axis=0), labels)
    assert 0.02 < temperature < 50.0


def test_temperature_rejects_a_degenerate_bracket():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((2, 4, 3)), np.zeros(4, dtype=int), bounds=(1.0, 0.5))
