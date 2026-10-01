"""Metrics used in the paper's tables.

Definitions follow the reported protocol: FPR95 is the false-positive rate at the
first threshold reaching at least 95% true-positive rate, and Near/Far aggregates
are arithmetic means over the datasets in their OpenOOD groups.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rmse",
    "gaussian_nll",
    "categorical_nll",
    "accuracy",
    "expected_calibration_error",
    "auroc",
    "fpr_at_tpr",
    "spearman",
]


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root mean squared error over all elements."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def gaussian_nll(
    mean: np.ndarray,
    variance: np.ndarray,
    targets: np.ndarray,
    *,
    reduce: bool = True,
) -> float | np.ndarray:
    """Diagonal-Gaussian negative log-likelihood, summed over output dimensions."""
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-12)
    targets = np.asarray(targets, dtype=np.float64)
    per_example = 0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + (targets - mean) ** 2 / variance, axis=-1
    )
    return float(np.mean(per_example)) if reduce else per_example


def categorical_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    return float(-np.mean(np.log(probabilities[np.arange(labels.shape[0]), labels] + 1e-12)))


def accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy."""
    return float(np.mean(np.argmax(np.asarray(probabilities), axis=-1) == np.asarray(labels)))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Equal-width-binned ECE over the top-1 confidence."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    confidence = probabilities.max(axis=-1)
    correct = (np.argmax(probabilities, axis=-1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lower) & (confidence <= upper)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC for detecting OOD, with *higher score = more OOD*.

    Computed from the rank-sum statistic, so ties contribute 0.5 as they should.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64).ravel()
    ood_scores = np.asarray(ood_scores, dtype=np.float64).ravel()
    if id_scores.size == 0 or ood_scores.size == 0:
        raise ValueError("both score arrays must be non-empty")
    combined = np.concatenate([ood_scores, id_scores])
    ranks = _average_ranks(combined)
    positive_rank_sum = ranks[: ood_scores.size].sum()
    n_pos, n_neg = float(ood_scores.size), float(id_scores.size)
    return float((positive_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, tpr_level: float = 0.95) -> float:
    """FPR at the first threshold reaching at least ``tpr_level`` TPR.

    Higher score = more OOD, so OOD examples are the positives and the FPR is the
    fraction of ID examples scoring at or above that threshold.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64).ravel()
    ood_scores = np.asarray(ood_scores, dtype=np.float64).ravel()
    threshold = float(np.percentile(ood_scores, 100.0 * (1.0 - tpr_level)))
    return float(np.mean(id_scores >= threshold))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    rank_x = _average_ranks(np.asarray(x, dtype=np.float64).ravel())
    rank_y = _average_ranks(np.asarray(y, dtype=np.float64).ravel())
    rank_x = rank_x - rank_x.mean()
    rank_y = rank_y - rank_y.mean()
    denominator = np.sqrt((rank_x ** 2).sum() * (rank_y ** 2).sum())
    if denominator == 0.0:
        return 0.0
    return float((rank_x * rank_y).sum() / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties receiving their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks
