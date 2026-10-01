"""P&C for convolutional residual blocks, in PyTorch.

Mirrors :mod:`pnc.jax.conv`; see that module for the patchwise formulation and why
the shortcut and BatchNorm statistics stay frozen.

One convention difference matters. PyTorch stores convolution kernels in OIHW order
and ``torch.nn.functional.unfold`` emits patch rows in ``(Cin, kh, kw)`` order, so the
flattening helpers here take an **OIHW** kernel, not the HWIO one the JAX module uses.
Both produce the same column order; only the caller's kernel layout differs.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..spec import CorrectionSpec
from .correct import NormalEquations, augment
from .perturb import PerturbationBasis

__all__ = [
    "ConvBlockTarget",
    "ConvMember",
    "flatten_kernel_to_patch_order",
    "unflatten_kernel_from_patch_order",
    "unfold_nchw",
    "build_conv_pnc",
]


def flatten_kernel_to_patch_order(kernel) -> torch.Tensor:
    """OIHW kernel ``(Cout, Cin, kh, kw)`` -> ``(Cin*kh*kw, Cout)``.

    ``Cin`` is the slowest-varying axis, matching ``torch.nn.functional.unfold``.
    """
    kernel = torch.as_tensor(kernel)
    return kernel.reshape(kernel.shape[0], -1).T


def unflatten_kernel_from_patch_order(flat, kernel_shape) -> torch.Tensor:
    """Inverse of :func:`flatten_kernel_to_patch_order`, returning an OIHW kernel."""
    return torch.as_tensor(flat).T.reshape(kernel_shape)


def unfold_nchw(features, kernel_size: int, stride: int = 1, padding: int | None = None):
    """NCHW feature maps -> ``(N*H'*W', Cin*k*k)`` patch rows.

    ``padding`` defaults to ``kernel_size // 2``, i.e. 'same' for odd kernels.
    """
    padding = kernel_size // 2 if padding is None else padding
    patches = torch.nn.functional.unfold(
        features, kernel_size=kernel_size, stride=stride, padding=padding
    )  # (N, Cin*k*k, L)
    return patches.transpose(1, 2).reshape(-1, patches.shape[1])


@dataclass
class ConvBlockTarget:
    """Everything P&C needs about one residual block.

    ``w1`` (perturbed) and ``w2`` (corrected) are OIHW kernels. ``design_fn`` maps a
    Conv1 kernel and an input chunk to the unfolded patches entering Conv2, computed
    with the frozen BN2/ReLU in between; its rows must line up with the target rows.
    """

    w1: torch.Tensor
    w2: torch.Tensor
    design_fn: Callable

    @property
    def n_features(self) -> int:
        """Bias-augmented correction dimension ``D_M = 1 + k2^2 * Cin2``."""
        return 1 + int(np.prod(tuple(self.w2.shape)[1:]))

    @property
    def n_outputs(self) -> int:
        return int(self.w2.shape[0])


@dataclass
class ConvMember:
    """One member's block parameters: perturbed Conv1, corrected Conv2 and its bias."""

    w1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor


@torch.no_grad()
def build_conv_pnc(
    block: ConvBlockTarget,
    calibration_chunks: Sequence,
    target_chunks: Sequence,
    *,
    rank: int,
    n_members: int,
    scale: float,
    spec: CorrectionSpec,
    seed: int,
    correct: bool = True,
) -> list[ConvMember]:
    """Construct the members for one target block.

    ``target_chunks`` are the base model's Conv2 outputs, ``(n, Cout2, H', W')`` per
    chunk in NCHW. ``correct=False`` keeps the perturbation and skips the repair.

    Bootstrapping resamples at the *image* level and then re-chunks, so a member's
    correction subset is a set of images rather than a set of patch rows.

    The perturbation subspace is drawn in the **patch-order** coordinates
    ``(Cin1*k1*k1, Cout1)`` -- the same coordinates the correction is solved in --
    rather than in kernel coordinates. Kernel layout is a storage convention (OIHW
    here, HWIO in JAX) and reshaping one flat draw into two layouts would give two
    different perturbations; drawing in patch order makes a seed name the same
    subspace regardless of layout.
    """
    if len(calibration_chunks) != len(target_chunks):
        raise ValueError("calibration and target chunk lists must have equal length")
    if not calibration_chunks:
        raise ValueError("at least one calibration chunk is required")

    w1 = torch.as_tensor(block.w1).detach()
    w2 = torch.as_tensor(block.w2).detach()
    device = w1.device
    patch_shape = (int(np.prod(tuple(w1.shape)[1:])), int(w1.shape[0]))
    basis = PerturbationBasis.random(patch_shape, rank, seed=seed, device=device)
    coefficients = basis.coefficients(n_members, seed=seed + 17)

    original = torch.cat(
        [
            torch.zeros((1, block.n_outputs), dtype=torch.float64, device=device),
            flatten_kernel_to_patch_order(w2).to(torch.float64),
        ],
        dim=0,
    )

    flat_inputs = torch.cat([torch.as_tensor(c) for c in calibration_chunks], dim=0)
    flat_targets = torch.cat([torch.as_tensor(t) for t in target_chunks], dim=0)
    chunk_size = int(torch.as_tensor(calibration_chunks[0]).shape[0])
    subsets = spec.subsets(int(flat_inputs.shape[0]), n_members)

    members: list[ConvMember] = []
    for member_index in range(n_members):
        w1_perturbed = w1 + unflatten_kernel_from_patch_order(
            basis.delta(coefficients[member_index], scale), tuple(w1.shape)
        ).to(w1.dtype)

        if not correct:
            members.append(
                ConvMember(
                    w1=w1_perturbed,
                    w2=w2,
                    b2=torch.zeros(block.n_outputs, dtype=w2.dtype, device=device),
                )
            )
            continue

        if subsets is None:
            inputs, targets = calibration_chunks, target_chunks
        else:
            index = torch.as_tensor(subsets[member_index], device=flat_inputs.device)
            resampled_inputs, resampled_targets = flat_inputs[index], flat_targets[index]
            n_chunks = max(1, int(np.ceil(resampled_inputs.shape[0] / chunk_size)))
            inputs = [
                resampled_inputs[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)
            ]
            targets = [
                resampled_targets[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)
            ]

        system = NormalEquations(block.n_features, block.n_outputs, device=device)
        for input_chunk, target_chunk in zip(inputs, targets, strict=True):
            design = augment(torch.as_tensor(block.design_fn(w1_perturbed, input_chunk)))
            # NCHW targets: move channels last so rows match the unfolded patch order.
            target = torch.as_tensor(target_chunk)
            target = target.permute(0, 2, 3, 1).reshape(-1, block.n_outputs)
            system.update(design, target)

        solution = system.solve(spec.ridge, prior=spec.use_prior(original))
        members.append(
            ConvMember(
                w1=w1_perturbed,
                w2=unflatten_kernel_from_patch_order(solution[1:], tuple(w2.shape)).to(w2.dtype),
                b2=solution[0].to(w2.dtype),
            )
        )

    return members
