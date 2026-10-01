"""Train the base PreAct ResNet-18 on CIFAR-10.

    python -m experiments.cifar10.train --seed 0

The recipe is the one in Appendix D.1 and has nothing to do with P&C: SGD with
Nesterov momentum 0.9, initial learning rate 0.1 under cosine decay with a five-epoch
warmup, weight decay 5e-4, batch size 128, 300 epochs, random crop and horizontal
flip plus Cutout(8), and no label smoothing.

This is the only expensive step in the CIFAR track -- roughly 8,010 s on the paper's
GPU.  P&C then adds at most ~100 s on top, against ~400,500 s to train a matched
50-member Deep Ensemble by repeated training.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from . import data as cifar_data
from .model import PreActResNet18

__all__ = ["save_checkpoint", "load_checkpoint", "augment_batch"]


def save_checkpoint(model: PreActResNet18, path: Path) -> None:
    _, state = nnx.split(model)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(
            {"leaves": [np.asarray(leaf) for leaf in leaves], "treedef": treedef}, handle
        )


def load_checkpoint(path: Path, n_classes: int = 10) -> PreActResNet18:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    model = PreActResNet18(n_classes=n_classes, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(model)
    state = jax.tree_util.tree_unflatten(
        payload["treedef"], [jnp.asarray(leaf) for leaf in payload["leaves"]]
    )
    return nnx.merge(graphdef, state)


def augment_batch(images: np.ndarray, rng: np.random.RandomState, cutout: int = 8) -> np.ndarray:
    """Random crop with 4-pixel reflect padding, horizontal flip, and Cutout."""
    n, height, width, _ = images.shape
    padded = np.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    offsets_y = rng.randint(0, 9, size=n)
    offsets_x = rng.randint(0, 9, size=n)
    cropped = np.stack(
        [
            padded[i, y : y + height, x : x + width]
            for i, (y, x) in enumerate(zip(offsets_y, offsets_x, strict=True))
        ]
    )
    flip = rng.rand(n) < 0.5
    cropped[flip] = cropped[flip, :, ::-1, :]

    if cutout > 0:
        centers_y = rng.randint(0, height, size=n)
        centers_x = rng.randint(0, width, size=n)
        half = cutout // 2
        for i in range(n):
            y0, y1 = max(0, centers_y[i] - half), min(height, centers_y[i] + half)
            x0, x1 = max(0, centers_x[i] - half), min(width, centers_x[i] + half)
            cropped[i, y0:y1, x0:x1, :] = 0.0
    return cropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/cifar10"))
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = cifar_data.load_cifar10(args.data_root)
    splits = cifar_data.protocol_splits(train_x, train_y)
    fit_x, fit_y = splits["train_fit"]
    print(f"train_fit {len(fit_x)}  val_select {splits['n_val_select']}  test {len(test_x)}")

    model = PreActResNet18(n_classes=10, rngs=nnx.Rngs(args.seed))
    steps_per_epoch = len(fit_x) // args.batch_size
    # optax counts decay_steps from step 0, warmup included, so a short run must also
    # shorten its warmup or the cosine phase gets a negative length.
    warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))
    if warmup_epochs != args.warmup_epochs:
        print(
            f"note: warmup shortened from {args.warmup_epochs} to {warmup_epochs} epochs "
            f"because --epochs {args.epochs} leaves no room for the default warmup"
        )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup_epochs * steps_per_epoch,
        decay_steps=args.epochs * steps_per_epoch,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.add_decayed_weights(args.weight_decay),
            optax.sgd(schedule, momentum=0.9, nesterov=True),
        ),
        wrt=nnx.Param,
    )

    @nnx.jit
    def step(model, optimizer, images, labels):
        def loss_fn(m):
            logits = m(images, frozen_stats=False)
            return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def predict(model, images):
        return model(images, frozen_stats=True)

    rng = np.random.RandomState(args.seed)
    # Monotonic, not time.time(): wall-clock jumps (NTP, host suspend) otherwise
    # show up as absurd training durations.
    started = time.monotonic()
    for epoch in range(args.epochs):
        order = rng.permutation(len(fit_x))
        losses = []
        for start in range(0, steps_per_epoch * args.batch_size, args.batch_size):
            batch = order[start : start + args.batch_size]
            images = augment_batch(np.asarray(fit_x[batch]), rng)
            losses.append(
                float(step(model, optimizer, jnp.asarray(images), jnp.asarray(fit_y[batch])))
            )
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            correct = 0
            for start in range(0, len(test_x), 500):
                logits = np.asarray(predict(model, jnp.asarray(test_x[start : start + 500])))
                correct += int((logits.argmax(-1) == test_y[start : start + 500]).sum())
            elapsed = time.monotonic() - started
            print(
                f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  "
                f"test acc {100.0 * correct / len(test_x):.2f}%  ({elapsed:.0f}s)"
            )

    destination = args.out / f"resnet18_seed{args.seed}.pkl"
    save_checkpoint(model, destination)
    print(f"base training took {time.monotonic() - started:.0f}s; wrote {destination}")


if __name__ == "__main__":
    main()
