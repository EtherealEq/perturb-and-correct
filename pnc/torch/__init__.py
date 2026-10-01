"""PyTorch-native Perturb-and-Correct.

Same API as :mod:`pnc.jax`, and -- because both draw from the shared NumPy RNG in
:mod:`pnc.spec` -- the same seed gives the same ensemble in either.

All linear algebra runs in ``torch.float64`` regardless of the model's dtype, and the
whole construction runs under ``torch.no_grad``: P&C is post-hoc and never needs
gradients. No global configuration is changed on import.

One convention differs from the JAX module by design: :mod:`pnc.torch.conv` takes
**OIHW** kernels and NCHW feature maps, matching PyTorch, where the JAX module takes
HWIO and NHWC.
"""
from .conv import (
    ConvBlockTarget,
    ConvMember,
    build_conv_pnc,
    flatten_kernel_to_patch_order,
    unflatten_kernel_from_patch_order,
    unfold_nchw,
)
from .correct import NormalEquations, augment, ridge_solve
from .dense import DensePnCEnsemble, DenseStack, build_dense_pnc
from .perturb import PerturbationBasis
from .transformer_ffn import FeedForwardBlock, FFNMember, build_ffn_pnc, gelu

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
    "unfold_nchw",
    "unflatten_kernel_from_patch_order",
]
