"""CIFAR-adapted pre-activation ResNet-18 (Appendix B.2).

A 3x3 stride-1 stem with no max-pool, then four stages of two basic blocks at widths
64/128/256/512.  A basic block is

    y = Conv1(ReLU(BN1(x))),   u = ReLU(BN2(y)),   z = Conv2(u),
    Block(x) = z + Shortcut(x),

which is exactly a perturb-correct unit: P&C perturbs ``Conv1`` and refits ``Conv2``
plus a new Conv2 bias.  The shortcut is untouched, so reproducing the original Conv2
output reproduces the block's contribution to its output.

BatchNorm uses the trained running statistics and stays frozen for the whole
construction -- members differ only in the two convolutions of one block.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = ["PreActResNet18", "PreActBasicBlock", "BLOCK_INDICES"]

# (stage, block) for all eight residual blocks, in network order.
BLOCK_INDICES = [(stage, block) for stage in range(4) for block in range(2)]


class PreActBasicBlock(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, strides: int, rngs: nnx.Rngs):
        conv = dict(kernel_size=(3, 3), padding="SAME", use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=in_channels, momentum=0.9, rngs=rngs)
        self.conv1 = nnx.Conv(in_channels, out_channels, strides=(strides, strides), **conv)
        self.bn2 = nnx.BatchNorm(num_features=out_channels, momentum=0.9, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, strides=(1, 1), **conv)
        self.downsample = (
            nnx.Conv(
                in_channels, out_channels, kernel_size=(1, 1),
                strides=(strides, strides), padding="SAME", use_bias=False, rngs=rngs,
            )
            if strides != 1 or in_channels != out_channels
            else None
        )

    def preactivation(self, x, frozen_stats: bool = True):
        """``ReLU(BN1(x))`` -- shared by the residual branch and the shortcut.

        Evaluated once per block and passed to both. Calling BN1 separately for each
        would fold the same batch statistics into its running average twice whenever a
        downsample is present, quietly changing its effective momentum.
        """
        return jax.nn.relu(self.bn1(x, use_running_average=frozen_stats))

    def branch(self, x, w1=None, w2=None, b2=None, *, frozen_stats: bool = True,
               activated=None):
        """The residual branch, with optionally substituted member parameters.

        Returns ``(pre_conv2_activation, conv2_output)`` so callers can reuse the
        activation as the correction design without a second forward pass. Pass
        ``activated`` to reuse a pre-activation already computed for this block.
        """
        if activated is None:
            activated = self.preactivation(x, frozen_stats)
        kernel = self.conv1.kernel.get_value() if w1 is None else w1
        hidden = jax.lax.conv_general_dilated(
            lhs=activated,
            rhs=kernel.transpose(3, 2, 0, 1),
            window_strides=tuple(self.conv1.strides),
            padding="SAME",
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )
        hidden = jax.nn.relu(self.bn2(hidden, use_running_average=frozen_stats))
        if w2 is None:
            return hidden, self.conv2(hidden)
        output = jax.lax.conv_general_dilated(
            lhs=hidden,
            rhs=w2.transpose(3, 2, 0, 1),
            window_strides=tuple(self.conv2.strides),
            padding="SAME",
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )
        if b2 is not None:
            output = output + b2
        return hidden, output

    def shortcut(self, x, activated):
        """The identity path, projected when the block changes shape."""
        if self.downsample is None:
            return x
        return self.downsample(activated)

    def __call__(self, x, *, w1=None, w2=None, b2=None, frozen_stats: bool = True):
        activated = self.preactivation(x, frozen_stats)
        _, branch = self.branch(
            x, w1, w2, b2, frozen_stats=frozen_stats, activated=activated
        )
        return branch + self.shortcut(x, activated)


def _stage(in_channels, out_channels, n_blocks, strides, rngs):
    blocks = []
    for index in range(n_blocks):
        blocks.append(
            PreActBasicBlock(
                in_channels if index == 0 else out_channels,
                out_channels,
                strides if index == 0 else 1,
                rngs,
            )
        )
    return nnx.List(blocks)


class PreActResNet18(nnx.Module):
    def __init__(self, n_classes: int = 10, rngs: nnx.Rngs = None):
        self.stem = nnx.Conv(
            3, 64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False, rngs=rngs
        )
        self.stages = nnx.List(
            [
                _stage(64, 64, 2, 1, rngs),
                _stage(64, 128, 2, 2, rngs),
                _stage(128, 256, 2, 2, rngs),
                _stage(256, 512, 2, 2, rngs),
            ]
        )
        self.final_bn = nnx.BatchNorm(num_features=512, momentum=0.9, rngs=rngs)
        self.fc = nnx.Linear(512, n_classes, rngs=rngs)

    def block(self, stage: int, block: int) -> PreActBasicBlock:
        return self.stages[stage][block]

    def __call__(self, x, *, frozen_stats: bool = False, member=None):
        """Forward pass, optionally substituting one member's block parameters.

        ``member`` is ``(stage, block, w1, w2, b2)``; everything outside that block is
        shared with the base model, which is what makes the ensemble compact.
        """
        h = self.stem(x)
        for stage_index, stage in enumerate(self.stages):
            for block_index, block in enumerate(stage):
                substitute = (
                    member is not None
                    and (stage_index, block_index) == (member[0], member[1])
                )
                if substitute:
                    h = block(
                        h, w1=member[2], w2=member[3], b2=member[4], frozen_stats=frozen_stats
                    )
                else:
                    h = block(h, frozen_stats=frozen_stats)
        h = jax.nn.relu(self.final_bn(h, use_running_average=frozen_stats))
        return self.fc(jnp.mean(h, axis=(1, 2)))
