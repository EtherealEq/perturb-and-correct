"""Train a MuJoCo dynamics model and evaluate P&C on it end to end.

    python -m experiments.mujoco.run --env Ant-v5

Reproduces the construction behind Table 1: multi-block P&C (hidden layers 1 and 3,
corrected in the layers that follow them) with K=20 directions and M=50 members, and
the matched uncorrected ablation.

Everything that selects or calibrates uses ID data only.  Shifted tiers are loaded
after construction is finished and are used only to report numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flax import nnx

from pnc import CorrectionSpec
from pnc.ensemble import moment_match_gaussian, variance_decomposition
from pnc.jax import build_dense_pnc
from pnc.metrics import auroc, gaussian_nll, rmse, spearman

from . import data as mujoco_data
from .model import ProbabilisticMLP, TrainConfig, to_dense_stack, train

# Hidden layers to perturb (zero-based). The correction for each lands in the layer
# immediately after it, so layers 1 and 3 are perturbed and 2 and 4 are refit.
PERTURB_LAYERS = (0, 2)


def _evaluate(ensemble, base_stack, inputs, targets):
    """Predictive metrics plus the per-example uncertainty used for OOD ranking."""
    members = {k: np.asarray(v) for k, v in ensemble.predict(inputs).items()}
    mean, variance = moment_match_gaussian(members["mean"], members["var"])
    parts = variance_decomposition(members["mean"], members["var"])
    per_example_error = np.sqrt(((mean - targets) ** 2).mean(axis=-1))
    return {
        "rmse": rmse(mean, targets),
        "nll": gaussian_nll(mean, variance, targets),
        "base_rmse": rmse(np.asarray(base_stack.predict(inputs)["mean"]), targets),
        # Mean total predictive variance is the OOD score (Section 5.1).
        "uncertainty": parts["total"],
        "epistemic": float(parts["epistemic"].mean()),
        "aleatoric": float(parts["aleatoric"].mean()),
        "error": per_example_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Ant-v5", choices=sorted(mujoco_data.ENV_SLUGS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=20_000,
                        help="transitions drawn from the expert pool for base training")
    parser.add_argument("--eval-steps", type=int, default=10_000, help="per-tier evaluation size")
    parser.add_argument("--correction-pool", type=int, default=4_096,
                        help="ID calibration examples for the least-squares repair")
    parser.add_argument("--members", type=int, default=50)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--scale", type=float, default=16.0, help="sigma; grid {8, 16, 32}")
    parser.add_argument("--ridge", type=float, default=1e-2, help="lambda; grid {1e-4, 1e-2}")
    parser.add_argument("--bootstrap", type=float, default=0.1,
                        help="per-member correction fraction; grid {0.05, 0.1, 0.2, 0.3}")
    parser.add_argument("--out", type=Path, default=Path("results/mujoco"))
    args = parser.parse_args()

    print(f"=== {args.env} (seed {args.seed}) ===")
    inputs, targets = mujoco_data.load_transitions(args.env, "id", args.train_steps, args.seed)
    split = int(0.9 * len(inputs))
    train_inputs, validation_inputs = inputs[:split], inputs[split:]
    train_targets, validation_targets = targets[:split], targets[split:]

    print("training base model...")
    model = ProbabilisticMLP(
        inputs.shape[1], targets.shape[1], rngs=nnx.Rngs(args.seed)
    )
    model = train(model, train_inputs, train_targets, TrainConfig(), seed=args.seed)
    stack = to_dense_stack(model)

    # The correction pool is drawn from the training transitions the base model saw.
    rng = np.random.RandomState(args.seed)
    pool_size = min(args.correction_pool, len(train_inputs))
    calibration = train_inputs[rng.choice(len(train_inputs), pool_size, replace=False)]

    spec = CorrectionSpec(ridge=args.ridge, bootstrap_fraction=args.bootstrap, seed=args.seed)
    build = dict(
        perturb_layers=PERTURB_LAYERS, rank=args.rank, n_members=args.members,
        scale=args.scale, spec=spec, seed=args.seed,
    )
    print(f"constructing P&C (M={args.members}, K={args.rank}, sigma={args.scale})...")
    ensembles = {
        "pnc": build_dense_pnc(stack, calibration, correct=True, **build),
        "uncorrected": build_dense_pnc(stack, calibration, correct=False, **build),
    }

    # Shifted data are only loaded now that construction and calibration are done.
    tiers = {"id": (validation_inputs, validation_targets)}
    for tier in ("near", "mid", "far"):
        if not mujoco_data.tier_available(args.env, tier):
            print(f"  {tier}: no dataset for this environment, skipping")
            continue
        tiers[tier] = mujoco_data.load_transitions(args.env, tier, args.eval_steps, args.seed)

    results: dict[str, dict] = {}
    for name, ensemble in ensembles.items():
        per_tier = {t: _evaluate(ensemble, stack, x, y) for t, (x, y) in tiers.items()}
        summary = {
            "id_rmse": per_tier["id"]["rmse"],
            "base_id_rmse": per_tier["id"]["base_rmse"],
            # Does uncertainty track error on ID data?  (Table 1's Spearman column.)
            "id_uncertainty_error_spearman": spearman(
                per_tier["id"]["uncertainty"], per_tier["id"]["error"]
            ),
        }
        for tier in ("near", "mid", "far"):
            if tier not in per_tier:
                continue
            summary[f"{tier}_nll"] = per_tier[tier]["nll"]
            summary[f"{tier}_auroc"] = auroc(
                per_tier["id"]["uncertainty"], per_tier[tier]["uncertainty"]
            )
        results[name] = summary
        print(f"\n{name}:")
        for key, value in summary.items():
            print(f"  {key:34s} {value:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.env}_seed{args.seed}.json"
    destination.write_text(
        json.dumps(
            {"env": args.env, "seed": args.seed, "config": vars(args) | {"out": str(args.out)},
             "results": results},
            indent=2, default=str,
        )
    )
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
