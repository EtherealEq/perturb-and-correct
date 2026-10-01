"""Build P&C on a trained CIFAR-10 checkpoint and evaluate OOD detection.

    python -m experiments.cifar10.run --checkpoint checkpoints/cifar10/resnet18_seed0.pkl \\
        --openood-root data/openood

Reproduces the construction behind Table 2: single-block P&C on the first block of
the final ResNet stage, K=20 directions, M=50 members, sigma=25, ridge 1e-3,
bootstrap fraction 0.05, and a 1,024-image correction pool from ``train_fit``.  Those
values are the ID-validation optimum of the joint block x scale x bootstrap grid
(Appendix D.3); they are defaults here, not re-selected.

The matched uncorrected ablation reuses the identical perturbations with the repair
switched off.

Ordering matters and is enforced by the script: the temperature is fit on ID
validation data, and OOD datasets are not touched until after that.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pnc import CorrectionSpec
from pnc.ensemble import average_probabilities, fit_temperature, predictive_entropy
from pnc.jax import ConvBlockTarget, build_conv_pnc
from pnc.metrics import accuracy, auroc, categorical_nll, expected_calibration_error, fpr_at_tpr

from . import data as cifar_data
from .train import load_checkpoint

# ID-validation optimum from the 162-run joint grid: first block of the final stage.
TARGET_BLOCK = (3, 0)


def _extract_patches(features: jnp.ndarray, kernel: int, stride: int) -> jnp.ndarray:
    """Unfold NHWC feature maps into (N*H'*W', Cin*k*k) rows, channel-slowest."""
    patches = jax.lax.conv_general_dilated_patches(
        lhs=features,
        filter_shape=(kernel, kernel),
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    return patches.reshape(-1, patches.shape[-1])


def _block_inputs_and_targets(model, images, stage, block, chunk_size):
    """Run the frozen prefix once: block inputs, and the base model's Conv2 outputs."""
    target = model.block(stage, block)
    inputs_chunks, target_chunks = [], []
    for start in range(0, len(images), chunk_size):
        h = model.stem(jnp.asarray(images[start : start + chunk_size]))
        for stage_index, stage_blocks in enumerate(model.stages):
            for block_index, current in enumerate(stage_blocks):
                if (stage_index, block_index) == (stage, block):
                    inputs_chunks.append(np.asarray(h))
                    target_chunks.append(np.asarray(target.branch(h)[1]))
                    break
                h = current(h, frozen_stats=True)
            else:
                continue
            break
    return inputs_chunks, target_chunks


def _member_logits(model, members, images, stage, block, batch_size=500):
    """(M, N, C) logits, evaluating each member's substituted block in turn."""
    stacked = []
    for member in members:
        per_member = [
            np.asarray(
                model(
                    jnp.asarray(images[start : start + batch_size]),
                    frozen_stats=True,
                    member=(
                        stage, block,
                        jnp.asarray(member.w1, dtype=jnp.float32),
                        jnp.asarray(member.w2, dtype=jnp.float32),
                        jnp.asarray(member.b2, dtype=jnp.float32),
                    ),
                )
            )
            for start in range(0, len(images), batch_size)
        ]
        stacked.append(np.concatenate(per_member))
    return np.stack(stacked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openood-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--members", type=int, default=50)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--scale", type=float, default=25.0, help="sigma; grid {25, 50, 100}")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--bootstrap", type=float, default=0.05, help="grid {0.05, 0.10, 0.20}")
    parser.add_argument("--correction-pool", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="images per accumulation chunk; affects memory only")
    parser.add_argument("--stage", type=int, default=TARGET_BLOCK[0])
    parser.add_argument("--block", type=int, default=TARGET_BLOCK[1])
    parser.add_argument("--out", type=Path, default=Path("results/cifar10"))
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = cifar_data.load_cifar10(args.data_root)
    splits = cifar_data.protocol_splits(train_x, train_y)
    fit_x, _ = splits["train_fit"]
    val_x, val_y = splits["val_select"]

    model = load_checkpoint(args.checkpoint)
    target = model.block(args.stage, args.block)
    w1 = np.asarray(target.conv1.kernel.get_value())
    w2 = np.asarray(target.conv2.kernel.get_value())
    kernel2, stride2 = int(target.conv2.kernel_size[0]), int(target.conv2.strides[0])

    # Correction pool: 1,024 images drawn from train_fit, never from val or test.
    rng = np.random.RandomState(args.seed)
    pool = fit_x[rng.choice(len(fit_x), min(args.correction_pool, len(fit_x)), replace=False)]

    print(f"caching block ({args.stage},{args.block}) activations for {len(pool)} images...")
    inputs_chunks, target_chunks = _block_inputs_and_targets(
        model, pool, args.stage, args.block, args.chunk_size
    )

    def design_fn(kernel, chunk):
        # Conv1 -> BN2 -> ReLU with the member's kernel, then unfold: the rows entering
        # Conv2. Stays in JAX, so the accumulation never round-trips through the host.
        hidden, _ = target.branch(jnp.asarray(chunk), w1=jnp.asarray(kernel, dtype=jnp.float32))
        return _extract_patches(hidden, kernel2, stride2)

    block_target = ConvBlockTarget(w1=w1, w2=w2, design_fn=design_fn)
    print(f"correction design dimension D_M = {block_target.n_features}")

    spec = CorrectionSpec(ridge=args.ridge, bootstrap_fraction=args.bootstrap, seed=args.seed)
    build = dict(
        rank=args.rank, n_members=args.members, scale=args.scale, spec=spec, seed=args.seed
    )

    started = time.monotonic()
    ensembles = {
        "pnc": build_conv_pnc(block_target, inputs_chunks, target_chunks, correct=True, **build),
        "uncorrected": build_conv_pnc(
            block_target, inputs_chunks, target_chunks, correct=False, **build
        ),
    }
    construction_seconds = time.monotonic() - started
    print(f"construction took {construction_seconds:.1f}s for both ensembles")

    # ID-only calibration, before any OOD data is read from disk.
    temperatures = {}
    for name, members in ensembles.items():
        validation_logits = _member_logits(model, members, val_x, args.stage, args.block)
        temperatures[name] = fit_temperature(validation_logits, val_y)
        print(f"{name}: ID-validation temperature {temperatures[name]:.4f}")

    print("loading OpenOOD splits...")
    ood = cifar_data.load_openood(args.openood_root)

    results: dict[str, dict] = {}
    for name, members in ensembles.items():
        temperature = temperatures[name]
        test_logits = _member_logits(model, members, test_x, args.stage, args.block)
        test_probabilities = average_probabilities(test_logits, temperature)
        id_scores = predictive_entropy(test_probabilities)

        summary = {
            "accuracy": 100.0 * accuracy(test_probabilities, test_y),
            "nll": categorical_nll(test_probabilities, test_y),
            "ece": expected_calibration_error(test_probabilities, test_y),
            "temperature": temperature,
        }
        for family, datasets in ood.items():
            aurocs, fprs = [], []
            for dataset_name, images in datasets.items():
                scores = predictive_entropy(
                    average_probabilities(
                        _member_logits(model, members, images, args.stage, args.block), temperature
                    )
                )
                aurocs.append(100.0 * auroc(id_scores, scores))
                fprs.append(100.0 * fpr_at_tpr(id_scores, scores))
                summary[f"{dataset_name}_auroc"] = aurocs[-1]
            summary[f"{family}_auroc"] = float(np.mean(aurocs))
            summary[f"{family}_fpr95"] = float(np.mean(fprs))
        results[name] = summary
        print(f"\n{name}:")
        for key in ("accuracy", "near_auroc", "near_fpr95", "far_auroc", "far_fpr95"):
            print(f"  {key:12s} {summary[key]:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"pnc_seed{args.seed}_s{args.stage}b{args.block}.json"
    destination.write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "construction_seconds": construction_seconds,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
