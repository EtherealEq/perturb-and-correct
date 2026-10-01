"""Combining P&C members into a predictive distribution.

The paper uses two aggregation rules, one per task type, both following
Lakshminarayanan et al. (2017):

*Regression (MuJoCo).*  Members emit diagonal Gaussians.  They are combined by
moment matching into a single diagonal Gaussian, and the mean total predictive
variance is the OOD score.  The total variance splits into the average member
variance (aleatoric, learned by the base model's variance head) and the variance of
the member means (epistemic, which is what the perturbation exposes).

*Classification (CIFAR-10, ImageNet, Banking77).*  A single scalar temperature is
fit on ID data, member probabilities are averaged under it, and predictive entropy
of the average is the OOD score.

The temperature is fit on in-distribution data only -- as is every other
hyperparameter and post-hoc calibration choice in the paper.  OOD data never enters
construction, selection, or calibration.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "moment_match_gaussian",
    "variance_decomposition",
    "average_probabilities",
    "predictive_entropy",
    "softmax",
    "fit_temperature",
]


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def moment_match_gaussian(
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse member diagonal Gaussians into one by matching moments.

    Parameters
    ----------
    means, variances:
        ``(n_members, n_inputs, n_outputs)``.

    Returns
    -------
    ``(mixture_mean, mixture_variance)``, each ``(n_inputs, n_outputs)``, with

        mu = E_m[mu_m],   sigma^2 = E_m[sigma_m^2] + E_m[mu_m^2] - mu^2.
    """
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if means.shape != variances.shape:
        raise ValueError("means and variances must have the same shape")
    mixture_mean = means.mean(axis=0)
    mixture_variance = variances.mean(axis=0) + (means ** 2).mean(axis=0) - mixture_mean ** 2
    return mixture_mean, np.maximum(mixture_variance, 0.0)


def variance_decomposition(
    means: np.ndarray,
    variances: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split the mixture variance into aleatoric and epistemic parts.

    Returns per-input mean-over-output-dimensions arrays for ``total``,
    ``aleatoric`` (average member variance) and ``epistemic`` (variance of member
    means).  ``total`` is the MuJoCo OOD ranking score.
    """
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    _, total = moment_match_gaussian(means, variances)
    aleatoric = variances.mean(axis=0)
    return {
        "total": total.mean(axis=-1),
        "aleatoric": aleatoric.mean(axis=-1),
        "epistemic": (total - aleatoric).mean(axis=-1),
    }


def average_probabilities(member_logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Mean member probability under a shared temperature.

    ``member_logits`` is ``(n_members, n_inputs, n_classes)``; the result is
    ``(n_inputs, n_classes)``.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return softmax(np.asarray(member_logits, dtype=np.float64) / temperature, axis=-1).mean(axis=0)


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """``H[pbar]`` per input -- the P&C OOD score for classification."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return -np.sum(probabilities * np.log(probabilities + 1e-12), axis=-1)


def fit_temperature(
    member_logits: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (1e-2, 1e2),
    max_iters: int = 40,
) -> float:
    """Scalar temperature minimising ID negative log-likelihood of the *ensemble*.

    The objective is the NLL of the averaged member probabilities, not of any single
    member, so the temperature calibrates the object actually used at test time.
    Golden-section search over ``log T`` keeps the temperature positive; the
    objective is unimodal in practice over the search bracket.
    """
    member_logits = np.asarray(member_logits, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if member_logits.ndim != 3:
        raise ValueError("member_logits must be (n_members, n_inputs, n_classes)")
    if member_logits.shape[1] == 0:
        return 1.0
    rows = np.arange(labels.shape[0])

    def objective(log_temperature: float) -> float:
        mean_probabilities = average_probabilities(member_logits, float(np.exp(log_temperature)))
        return float(-np.mean(np.log(mean_probabilities[rows, labels] + 1e-8)))

    if not 0.0 < bounds[0] < bounds[1]:
        raise ValueError(f"bounds must satisfy 0 < low < high, got {bounds}")
    left, right = float(np.log(bounds[0])), float(np.log(bounds[1]))
    inv_phi = 2.0 / (1.0 + np.sqrt(5.0))
    probe_low = right - (right - left) * inv_phi
    probe_high = left + (right - left) * inv_phi
    value_low, value_high = objective(probe_low), objective(probe_high)
    for _ in range(max_iters):
        if value_low <= value_high:
            right, probe_high, value_high = probe_high, probe_low, value_low
            probe_low = right - (right - left) * inv_phi
            value_low = objective(probe_low)
        else:
            left, probe_low, value_low = probe_low, probe_high, value_high
            probe_high = left + (right - left) * inv_phi
            value_high = objective(probe_high)

    temperature = float(np.exp(0.5 * (left + right)))
    # Golden-section search converges to a bracket edge when the minimum lies outside
    # the bracket. That means the bracket is wrong, not that the data wants an extreme
    # temperature, so say so rather than returning a boundary value silently.
    if not bounds[0] * 1.01 < temperature < bounds[1] * 0.99:
        warnings.warn(
            f"fitted temperature {temperature:.4g} sits at the edge of the search "
            f"bracket {bounds}; the optimum is probably outside it. Widen `bounds`, or "
            f"check that the logits and labels line up.",
            RuntimeWarning,
            stacklevel=2,
        )
    return temperature
