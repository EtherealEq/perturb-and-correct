"""Metric definitions, pinned against hand-computable cases."""
import numpy as np
import pytest

from pnc.metrics import (
    accuracy,
    auroc,
    categorical_nll,
    expected_calibration_error,
    fpr_at_tpr,
    gaussian_nll,
    rmse,
    spearman,
)


def test_perfect_and_inverted_separation():
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0


def test_ties_score_one_half():
    assert auroc(np.ones(10), np.ones(10)) == pytest.approx(0.5)


def test_auroc_matches_the_rank_sum_definition():
    rng = np.random.RandomState(0)
    id_scores, ood_scores = rng.standard_normal(200), rng.standard_normal(150) + 1.0
    greater = ood_scores[:, None] > id_scores[None, :]
    tied = ood_scores[:, None] == id_scores[None, :]
    brute = np.mean(greater + 0.5 * tied)
    assert auroc(id_scores, ood_scores) == pytest.approx(brute, abs=1e-12)


def test_fpr95_is_zero_under_perfect_separation():
    assert fpr_at_tpr(np.zeros(100), np.ones(100)) == 0.0


def test_fpr95_is_about_the_threshold_that_catches_95_percent_of_ood():
    id_scores = np.linspace(0.0, 1.0, 1000)
    ood_scores = np.linspace(0.5, 1.5, 1000)
    # 95% of OOD sit at or above the 5th percentile of OOD, i.e. ~0.55.
    assert fpr_at_tpr(id_scores, ood_scores) == pytest.approx(0.45, abs=0.02)


def test_rmse_and_gaussian_nll():
    assert rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(np.sqrt(2.0))
    # Unit Gaussian at the mean: 0.5 * log(2 pi) per output dimension.
    value = gaussian_nll(np.zeros((1, 3)), np.ones((1, 3)), np.zeros((1, 3)))
    assert value == pytest.approx(1.5 * np.log(2.0 * np.pi))


def test_gaussian_nll_penalises_overconfidence():
    tight = gaussian_nll(np.zeros((1, 1)), np.full((1, 1), 1e-3), np.ones((1, 1)))
    loose = gaussian_nll(np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
    assert tight > loose


def test_classification_metrics():
    probabilities = np.array([[0.7, 0.3], [0.2, 0.8]])
    labels = np.array([0, 1])
    assert accuracy(probabilities, labels) == 1.0
    assert categorical_nll(probabilities, labels) == pytest.approx(
        -0.5 * (np.log(0.7) + np.log(0.8))
    )


def test_ece_is_zero_for_a_perfectly_calibrated_split():
    # Ten examples at confidence 0.8, exactly eight correct.
    probabilities = np.tile(np.array([[0.8, 0.2]]), (10, 1))
    labels = np.array([0] * 8 + [1] * 2)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0, abs=1e-12)


def test_ece_detects_overconfidence():
    probabilities = np.tile(np.array([[0.99, 0.01]]), (10, 1))
    labels = np.array([0] * 5 + [1] * 5)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.49, abs=1e-6)


def test_spearman_is_rank_based():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, np.exp(x)) == pytest.approx(1.0)  # monotone but nonlinear
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    assert spearman(np.array([1.0, 1.0, 2.0]), np.array([5.0, 5.0, 9.0])) == pytest.approx(1.0)
