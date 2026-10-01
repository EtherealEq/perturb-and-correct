"""The base dynamics model: a 4x200 ReLU MLP with a diagonal-Gaussian head.

Deliberately ordinary.  P&C is a post-hoc construction, so the base model is just a
well-trained network -- nothing about the training is aware that an ensemble will be
built from it later.

Trained with a heteroscedastic Gaussian NLL so each member emits its own aleatoric
variance; ensemble disagreement then adds the epistemic part on top.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from pnc.jax import DenseStack

__all__ = ["ProbabilisticMLP", "train", "to_dense_stack"]

HIDDEN_WIDTH = 200
HIDDEN_DEPTH = 4


class ProbabilisticMLP(nnx.Module):
    """``depth`` hidden ReLU layers feeding separate mean and log-variance heads."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        rngs: nnx.Rngs,
        width: int = HIDDEN_WIDTH,
        depth: int = HIDDEN_DEPTH,
    ):
        dims = [n_inputs] + [width] * depth
        self.layers = nnx.List(
            [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(depth)]
        )
        self.mean_head = nnx.Linear(width, n_outputs, rngs=rngs)
        self.var_head = nnx.Linear(width, n_outputs, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return self.mean_head(x), jax.nn.softplus(self.var_head(x)) + 1e-6


def _gaussian_nll(mean, variance, targets):
    return jnp.mean(
        0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * variance) + (targets - mean) ** 2 / variance, axis=-1)
    )


@dataclass
class TrainConfig:
    """The paper's MuJoCo recipe (Appendix C.2)."""

    steps: int = 5_000
    batch_size: int = 64
    learning_rate: float = 1e-3


def train(
    model: ProbabilisticMLP,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: TrainConfig | None = None,
    seed: int = 0,
) -> ProbabilisticMLP:
    """Fit the base model by minimising Gaussian NLL."""
    config = config or TrainConfig()
    optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate), wrt=nnx.Param)
    inputs = jnp.asarray(inputs)
    targets = jnp.asarray(targets)

    @nnx.jit
    def step(model, optimizer, batch_inputs, batch_targets):
        def loss_fn(m):
            mean, variance = m(batch_inputs)
            return _gaussian_nll(mean, variance, batch_targets)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    rng = np.random.RandomState(seed)
    for index in range(config.steps):
        batch = rng.choice(inputs.shape[0], config.batch_size, replace=False)
        loss = step(model, optimizer, inputs[batch], targets[batch])
        if index % 1000 == 0 or index == config.steps - 1:
            print(f"  step {index:5d}  nll {float(loss):.4f}")
    return model


def to_dense_stack(model: ProbabilisticMLP) -> DenseStack:
    """Extract the trained parameters into the plain array form P&C consumes.

    The base model is frozen from here on; ``DenseStack`` holds nothing but weights,
    so the construction never touches the nnx module again. ``pnc.jax`` upcasts to
    float64 on the way in.
    """
    return DenseStack(
        weights=[layer.kernel.get_value() for layer in model.layers],
        biases=[layer.bias.get_value() for layer in model.layers],
        heads={
            "mean": (model.mean_head.kernel.get_value(), model.mean_head.bias.get_value()),
            "var": (model.var_head.kernel.get_value(), model.var_head.bias.get_value()),
        },
    )
