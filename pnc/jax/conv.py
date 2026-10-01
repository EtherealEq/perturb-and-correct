"""P&C for convolutional residual blocks, in JAX (the CIFAR-10 PreAct ResNet-18).

A convolution is a linear map on unfolded patches (Appendix B.1):

    flat(Conv(X, W)) = P(X) @ mat(W),

so the correction is the same ridge problem as the dense case with patch rows in
place of representation rows. For a pre-activation basic block

    y = Conv1(ReLU(BN1(x))),  u = ReLU(BN2(y)),  z = Conv2(u),  Block(x) = z + Shortcut(x),

P&C perturbs ``W1`` and refits ``W2`` plus a *new* Conv2 bias. The shortcut is
untouched, so matching the original Conv2 output is exactly matching the block's
residual-branch contribution. BatchNorm keeps its trained running statistics and stays
frozen; no labels are used, only the base model's own Conv2 outputs.

Unlike the dense backend this loops over members rather than batching them: the Gram
matrix is 4,609 x 4,609 in float64, so stacking 50 members would need ~8.5 GB. The
per-member accumulation is jitted instead, and the design is never materialised --
sufficient statistics accumulate chunk by chunk (Appendix B.4), so chunk size changes
peak memory only, not the solution.

The caller supplies ``design_fn``, which is where the framework-specific convolution
lives.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..spec import CorrectionSpec
from .correct import NormalEquations, augment
from .perturb import PerturbationBasis

__all__ = [
    "ConvBlockTarget",
    "ConvMember",
    "flatten_kernel_to_patch_order",
    "unflatten_kernel_from_patch_order",
    "build_conv_pnc",
]


def flatten_kernel_to_patch_order(kernel) -> jnp.ndarray:
    """HWIO kernel ``(kh, kw, Cin, Cout)`` -> ``(Cin*kh*kw, Cout)``.

    The column order must match the unfolded patch layout.
    ``jax.lax.conv_general_dilated_patches`` with
    ``dimension_numbers=('NHWC','OIHW','NHWC')`` lays each patch out with the input
    channel as the slowest-varying axis, i.e. ``(Cin, kh, kw)`` -- hence the
    transpose. A backend unfolding in a different order needs its own pair.
    """
    kernel = jnp.asarray(kernel)
    return kernel.transpose(2, 0, 1, 3).reshape(-1, kernel.shape[-1])


def unflatten_kernel_from_patch_order(flat, kernel_shape) -> jnp.ndarray:
    """Inverse of :func:`flatten_kernel_to_patch_order`, returning an HWIO kernel."""
    kh, kw, c_in, _ = kernel_shape
    return jnp.asarray(flat).reshape(c_in, kh, kw, -1).transpose(1, 2, 0, 3)


@dataclass
class ConvBlockTarget:
    """Everything P&C needs about one residual block.

    ``w1`` (perturbed) and ``w2`` (corrected) are HWIO kernels. ``design_fn`` maps a
    Conv1 kernel and an input chunk to the unfolded patches entering Conv2, computed
    with the frozen BN2/ReLU in between; its rows must line up with the target rows.
    """

    w1: jnp.ndarray
    w2: jnp.ndarray
    design_fn: Callable

    @property
    def n_features(self) -> int:
        """Bias-augmented correction dimension ``D_M = 1 + k2^2 * Cin2``."""
        return 1 + int(np.prod(self.w2.shape[:3]))

    @property
    def n_outputs(self) -> int:
        return int(self.w2.shape[-1])


@dataclass
class ConvMember:
    """One member's block parameters: perturbed Conv1, corrected Conv2 and its bias."""

    w1: jnp.ndarray
    w2: jnp.ndarray
    b2: jnp.ndarray


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

    Parameters
    ----------
    calibration_chunks:
        Inputs to the block, chunked to fit memory. The paper uses a 1,024-image
        correction pool drawn from the training-fit split.
    target_chunks:
        Matching base-model Conv2 outputs, ``(n, H', W', Cout2)`` per chunk.
    scale:
        ``sigma`` in Section 3; the CIFAR grid is ``{25, 50, 100}`` and the
        ID-validation optimum is 25. Larger values cross the ID-stability boundary.
    correct:
        ``False`` keeps the perturbation and skips the repair.

    Notes
    -----
    Bootstrapping resamples at the *image* level and then re-chunks, so a member's
    correction subset is a set of images rather than a set of patch rows. The same
    member index uses the same images at every block, keeping multi-block members
    comparable.

    The perturbation subspace is drawn in the **patch-order** coordinates
    ``(Cin1*k1*k1, Cout1)`` -- the same coordinates the correction is solved in --
    rather than in kernel coordinates. Kernel layout is a storage convention (HWIO
    here, OIHW in PyTorch) and reshaping one flat draw into two layouts would give two
    different perturbations; drawing in patch order makes a seed name the same
    subspace regardless of layout.
    """
    if len(calibration_chunks) != len(target_chunks):
        raise ValueError("calibration and target chunk lists must have equal length")
    if not calibration_chunks:
        raise ValueError("at least one calibration chunk is required")

    w1 = jnp.asarray(block.w1)
    w2 = jnp.asarray(block.w2)
    patch_shape = (int(np.prod(w1.shape[:3])), int(w1.shape[-1]))
    basis = PerturbationBasis.random(patch_shape, rank, seed=seed)
    coefficients = basis.coefficients(n_members, seed=seed + 17)

    original = jnp.concatenate(
        [
            jnp.zeros((1, block.n_outputs), dtype=jnp.float64),  # base Conv2 is bias-free
            flatten_kernel_to_patch_order(w2).astype(jnp.float64),
        ],
        axis=0,
    )

    flat_inputs = jnp.concatenate([jnp.asarray(c) for c in calibration_chunks], axis=0)
    flat_targets = jnp.concatenate([jnp.asarray(t) for t in target_chunks], axis=0)
    chunk_size = int(jnp.asarray(calibration_chunks[0]).shape[0])
    subsets = spec.subsets(int(flat_inputs.shape[0]), n_members)

    members: list[ConvMember] = []
    for member_index in range(n_members):
        w1_perturbed = w1 + unflatten_kernel_from_patch_order(
            basis.delta(coefficients[member_index], scale), w1.shape
        ).astype(w1.dtype)

        if not correct:
            members.append(
                ConvMember(
                    w1=w1_perturbed,
                    w2=w2,
                    b2=jnp.zeros(block.n_outputs, dtype=w2.dtype),
                )
            )
            continue

        if subsets is None:
            inputs, targets = calibration_chunks, target_chunks
        else:
            index = jnp.asarray(subsets[member_index])
            resampled_inputs, resampled_targets = flat_inputs[index], flat_targets[index]
            n_chunks = max(1, int(np.ceil(resampled_inputs.shape[0] / chunk_size)))
            inputs = [
                resampled_inputs[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)
            ]
            targets = [
                resampled_targets[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)
            ]

        system = NormalEquations(block.n_features, block.n_outputs)
        for input_chunk, target_chunk in zip(inputs, targets, strict=True):
            design = augment(jnp.asarray(block.design_fn(w1_perturbed, input_chunk)))
            target = jnp.asarray(target_chunk).reshape(-1, block.n_outputs)
            system.update(design, target)

        solution = system.solve(spec.ridge, prior=spec.use_prior(original))
        members.append(
            ConvMember(
                w1=w1_perturbed,
                w2=unflatten_kernel_from_patch_order(solution[1:], w2.shape).astype(w2.dtype),
                b2=solution[0].astype(w2.dtype),
            )
        )

    return members
