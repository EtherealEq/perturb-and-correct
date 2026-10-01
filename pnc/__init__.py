"""Perturb-and-Correct: post-hoc ensembles from a single pretrained network.

Reference implementation for *Perturb and Correct: Post-Hoc Ensembles using Affine
Redundancy* (NeurIPS 2026).

P&C turns one completed checkpoint into an epistemically diverse ensemble in two
steps. Sample a random low-rank perturbation of a hidden-layer weight, then refit the
*following* affine layer by least squares so the perturbed member reproduces the base
model's next-layer pre-activations on ID calibration data. Members agree where
calibration data constrain behaviour and are free to disagree where that repair no
longer transfers.

The construction ships in **two native implementations with the same API**:

``pnc.jax``
    JAX/float64, with all members solved in one batched call for dense trunks.
    Used by the MuJoCo and CIFAR-10 tracks.
``pnc.torch``
    PyTorch/float64, under ``no_grad``. Used by the ImageNet ViT and DistilBERT
    tracks, whose pretrained checkpoints are PyTorch.

Both draw their perturbations from the shared NumPy RNG in :mod:`pnc.spec`, so a given
seed produces the *same ensemble* in either -- which is what lets
``tests/test_equivalence.py`` assert the two agree to float64 precision.

This top-level module holds only framework-free pieces and imports neither JAX nor
PyTorch, so it stays cheap to import::

    from pnc import CorrectionSpec          # shared settings
    from pnc.jax import build_dense_pnc     # or: from pnc.torch import build_dense_pnc

    ensemble = build_dense_pnc(
        stack, calibration_inputs,
        perturb_layers=[0, 2], rank=20, n_members=50, scale=16.0,
        spec=CorrectionSpec(ridge=1e-2, bootstrap_fraction=0.1, seed=0), seed=0,
    )
"""
from .ensemble import (
    average_probabilities,
    fit_temperature,
    moment_match_gaussian,
    predictive_entropy,
    variance_decomposition,
)
from .spec import (
    CorrectionSpec,
    bootstrap_indices,
    mean_to_sum_ridge,
    sum_to_mean_ridge,
)

__version__ = "1.1.0"

__all__ = [
    "CorrectionSpec",
    "average_probabilities",
    "bootstrap_indices",
    "fit_temperature",
    "mean_to_sum_ridge",
    "moment_match_gaussian",
    "predictive_entropy",
    "sum_to_mean_ridge",
    "variance_decomposition",
    "__version__",
]
