"""JAX-native Perturb-and-Correct.

Same API as :mod:`pnc.torch`, and -- because both draw from the shared NumPy RNG in
:mod:`pnc.spec` -- the same seed gives the same ensemble in either.

**This module enables JAX's x64 mode on import.** The correction solves a Gram matrix
that reaches 4,609 x 4,609 for the CIFAR blocks and 3,073 x 3,073 for the transformer
FFNs; in float32 those are badly enough conditioned to change the answer, and the
paper's corrections are float64 throughout. JAX only honours the flag if it is set
before arrays are created, so it is set here rather than left to the caller.
Importing :mod:`pnc` alone does not touch it -- only importing this subpackage does.
"""
import jax

jax.config.update("jax_enable_x64", True)

from .conv import (  # noqa: E402
    ConvBlockTarget,
    ConvMember,
    build_conv_pnc,
    flatten_kernel_to_patch_order,
    unflatten_kernel_from_patch_order,
)
from .correct import NormalEquations, augment, ridge_solve  # noqa: E402
from .dense import DensePnCEnsemble, DenseStack, build_dense_pnc  # noqa: E402
from .perturb import PerturbationBasis  # noqa: E402
from .transformer_ffn import FeedForwardBlock, FFNMember, build_ffn_pnc, gelu  # noqa: E402

__all__ = [
    "ConvBlockTarget",
    "ConvMember",
    "DensePnCEnsemble",
    "DenseStack",
    "FFNMember",
    "FeedForwardBlock",
    "NormalEquations",
    "PerturbationBasis",
    "augment",
    "build_conv_pnc",
    "build_dense_pnc",
    "build_ffn_pnc",
    "flatten_kernel_to_patch_order",
    "gelu",
    "ridge_solve",
    "unflatten_kernel_from_patch_order",
]
