"""The CIFAR base model, where P&C's assumptions about the block structure live.

Only the properties the construction depends on: BatchNorm must behave like a normal
BatchNorm during training, and the member-substitution path must actually swap
parameters. Everything else about the architecture is standard.
"""
import numpy as np
import pytest

pytest.importorskip("flax")

import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from experiments.cifar10.model import PreActBasicBlock, PreActResNet18  # noqa: E402

MOMENTUM = 0.9


def _batch(n=32, size=8, channels=8, seed=0):
    return jnp.asarray(
        np.random.RandomState(seed).standard_normal((n, size, size, channels)),
        dtype=jnp.float32,
    )


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "strides", "has_downsample"),
    [(8, 8, 1, False), (8, 16, 2, True)],
)
def test_bn1_updates_its_running_stats_exactly_once_per_forward(
    in_channels, out_channels, strides, has_downsample
):
    """Regression: the residual branch and the shortcut each used to evaluate BN1, so
    a block with a downsample folded the same batch statistics in twice -- effective
    momentum 0.81 instead of 0.9, and only in the blocks that project, so the
    inconsistency was silent and uneven across the network.
    """
    block = PreActBasicBlock(in_channels, out_channels, strides, nnx.Rngs(0))
    assert (block.downsample is not None) == has_downsample

    inputs = _batch(channels=in_channels)
    block(inputs, frozen_stats=False)

    # flax BatchNorm: running = momentum * running + (1 - momentum) * batch, from zero.
    expected = (1.0 - MOMENTUM) * np.asarray(inputs.mean(axis=(0, 1, 2)))
    assert np.allclose(np.asarray(block.bn1.mean.get_value()), expected, atol=1e-6)


def test_frozen_stats_leaves_running_averages_untouched():
    """The whole construction runs with frozen BatchNorm, so this must hold."""
    block = PreActBasicBlock(8, 16, 2, nnx.Rngs(0))
    before = np.asarray(block.bn1.mean.get_value()).copy()
    block(_batch(), frozen_stats=True)
    assert np.array_equal(np.asarray(block.bn1.mean.get_value()), before)


def test_preactivation_is_what_both_paths_consume():
    """`branch` given a precomputed pre-activation must match computing it itself."""
    block = PreActBasicBlock(8, 8, 1, nnx.Rngs(0))
    inputs = _batch()
    activated = block.preactivation(inputs, frozen_stats=True)
    _, direct = block.branch(inputs, frozen_stats=True)
    _, reused = block.branch(inputs, frozen_stats=True, activated=activated)
    assert np.allclose(np.asarray(direct), np.asarray(reused), atol=1e-6)


def test_member_substitution_changes_only_the_target_block():
    """P&C swaps one block's parameters and shares the rest of the network."""
    model = PreActResNet18(n_classes=10, rngs=nnx.Rngs(0))
    images = jnp.asarray(
        np.random.RandomState(1).standard_normal((4, 32, 32, 3)), dtype=jnp.float32
    )
    base = np.asarray(model(images, frozen_stats=True))

    target = model.block(3, 0)
    w1 = target.conv1.kernel.get_value()
    w2 = target.conv2.kernel.get_value()
    unchanged = model(
        images, frozen_stats=True,
        member=(3, 0, w1, w2, jnp.zeros(w2.shape[-1], dtype=w2.dtype)),
    )
    assert np.allclose(base, np.asarray(unchanged), atol=1e-5)

    perturbed = model(
        images, frozen_stats=True,
        member=(3, 0, w1 * 1.5, w2, jnp.zeros(w2.shape[-1], dtype=w2.dtype)),
    )
    assert not np.allclose(base, np.asarray(perturbed), atol=1e-3)
