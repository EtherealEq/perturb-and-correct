"""P&C on the final FFN of a pretrained ViT-B/16, with ID-preservation selection.

    python -m experiments.imagenet_vit.run

Reproduces Table 3.  Two stages, in this order:

1.  **Selection under a preservation budget, using ID data only.**  Perturbation
    magnitude ``r`` is increased until the corrected ensemble would cost more than
    ``--budget`` percentage points of top-1 accuracy on a held-out *training* ID pool,
    judged by a one-sided 95% lower confidence bound from a paired bootstrap.  The
    largest ``r`` that passes is selected, with the ridge chosen at each candidate by
    ID NLL.  The paper's primary 0.50 pp budget selects ``r = 2`` and ``lambda = 1000``.

2.  **Evaluation**, which is the first time the official 50,000-image validation set
    is read.

The matched uncorrected ablation reuses the identical perturbations with no repair.
It is an ablation and never enters selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pnc import CorrectionSpec, sum_to_mean_ridge
from pnc.ensemble import average_probabilities, fit_temperature, predictive_entropy, softmax
from pnc.metrics import accuracy, auroc, categorical_nll, expected_calibration_error
from pnc.torch import FeedForwardBlock, build_ffn_pnc

from . import data as vit_data
from .features import ViTAdapter, cache_features, load_model

RIDGE_GRID = (100.0, 1000.0, 10000.0)


@torch.inference_mode()
def _ensemble_probabilities(adapter, block, members, basis, scale, features, temperature,
                            chunk: int = 8192):
    """Mean member probabilities, accumulated over input chunks.

    The obvious form -- stack every member's logits, then average -- needs an
    ``(M, N, 1000)`` float64 tensor, which is 8 GB for M=20 over the 50,000-image
    validation set and larger still for SSB-hard. Chunking over inputs keeps the peak at
    ``M * chunk * 1000`` while giving exactly the same answer, since the average is taken
    per input.
    """
    hidden = torch.from_numpy(features["hidden"]).to(torch.float64)
    residual = torch.from_numpy(features["residual"])
    perturbed = [block.w1 + basis.delta(m.coefficient, scale) for m in members]
    out = np.empty((hidden.shape[0], adapter.model.heads.head.out_features), dtype=np.float64)
    for start in range(0, hidden.shape[0], chunk):
        stop = min(start + chunk, hidden.shape[0])
        member_logits = np.stack([
            adapter.tail(
                residual[start:stop],
                block.forward(hidden[start:stop], w1=w1, w2=m.w2, b2=m.b2).float(),
            ).float().cpu().numpy()
            for w1, m in zip(perturbed, members, strict=True)
        ])
        out[start:stop] = average_probabilities(member_logits, temperature)
    return out


@torch.inference_mode()
def _member_logits(adapter, block, members, basis, scale, features, batch=2048):
    """(M, N, 1000) logits from cached rows -- the base prefix is never recomputed."""
    hidden = torch.from_numpy(features["hidden"]).to(torch.float64)
    residual = torch.from_numpy(features["residual"])
    stacked = []
    for member in members:
        perturbed = block.w1 + basis.delta(member.coefficient, scale)
        per_batch = []
        for start in range(0, len(hidden), batch):
            ffn = block.forward(
                hidden[start : start + batch], w1=perturbed, w2=member.w2, b2=member.b2
            )
            per_batch.append(
                adapter.tail(residual[start : start + batch], ffn.float())
                .float().cpu().numpy()
            )
        stacked.append(np.concatenate(per_batch))
    return np.stack(stacked)


def _base_logits(adapter, features, batch=2048):
    """Base-model logits from cached rows: the cached target *is* the base FFN output."""
    residual = torch.from_numpy(features["residual"])
    target = torch.from_numpy(features["target"])
    return np.concatenate(
        [
            adapter.tail(residual[start : start + batch], target[start : start + batch])
            .float().cpu().numpy()
            for start in range(0, len(residual), batch)
        ]
    )


def _preservation_lower_bound(member_correct, base_correct, n_boot=1000, seed=0):
    """One-sided 95% lower bound on (ensemble - base) top-1, paired bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(base_correct)
    differences = [
        float((member_correct[idx] - base_correct[idx]).mean())
        for idx in (rng.randint(0, n, n) for _ in range(n_boot))
    ]
    return float(np.percentile(differences, 5.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", type=int, default=None,
                        help="ImageNet train parquet shards to index (default: 28)")
    parser.add_argument("--max-eval", type=int, default=None,
                        help="cap the validation set and each OOD set; for smoke runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--members", type=int, default=20)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--correction-rows", type=int, default=32768,
                        help="ImageNet training CLS rows used for the correction fit")
    parser.add_argument("--selection-rows", type=int, default=16384,
                        help="held-out ID training rows for preservation selection")
    parser.add_argument("--budget", type=float, default=0.50,
                        help="top-1 preservation budget in percentage points")
    parser.add_argument("--r-grid", type=float, nargs="+",
                        default=[0.5, 1.0, 1.25, 1.5, 2.0, 2.125, 3.0, 4.0])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/imagenet_vit"))
    args = parser.parse_args()

    model, transform = load_model(args.device)
    adapter = ViTAdapter(model, args.device)
    parameters = adapter.weights()
    block = FeedForwardBlock(**parameters)

    print("indexing ImageNet shards...")
    train = vit_data.ImageNetShards(
        vit_data.shard_paths("train", args.train_shards), transform, args.workers
    )
    correction_idx, selection_idx = train.split(args.correction_rows, args.selection_rows)
    print(f"  train rows {len(train)}; correction {len(correction_idx)}, "
          f"selection {len(selection_idx)} (disjoint)")

    print("caching ID training features (correction + selection pools)...")
    started = time.monotonic()
    correction = cache_features(adapter, train.batches(correction_idx, args.batch_size))
    selection = cache_features(adapter, train.batches(selection_idx, args.batch_size))
    cache_seconds = time.monotonic() - started
    print(f"feature cache: {cache_seconds:.1f}s")

    # Base top-1 on the selection pool, the reference the budget is measured against.
    base_logits = _base_logits(adapter, selection)
    base_correct = (base_logits.argmax(-1) == selection["labels"]).astype(np.float64)
    print(f"base top-1 on selection pool: {100.0 * base_correct.mean():.3f}%")

    # --- stage 1: ID-only selection under the preservation budget ----------
    selected, trace = None, []
    for r in sorted(args.r_grid):
        best = None
        for ridge in RIDGE_GRID:
            spec = CorrectionSpec(ridge=ridge, bootstrap_fraction=0.0, seed=args.seed)
            members, basis, scale = build_ffn_pnc(
                block, correction["hidden"], correction["target"],
                rank=args.rank, n_members=args.members, relative_magnitude=r,
                spec=spec, seed=args.seed,
            )
            logits = _member_logits(adapter, block, members, basis, scale, selection)
            temperature = fit_temperature(logits, selection["labels"])
            probabilities = average_probabilities(logits, temperature)
            candidate = {
                "ridge": ridge,
                "nll": categorical_nll(probabilities, selection["labels"]),
                "correct": (probabilities.argmax(-1) == selection["labels"]).astype(np.float64),
                "members": members, "basis": basis, "scale": scale,
                "temperature": temperature,
            }
            if best is None or candidate["nll"] < best["nll"]:
                best = candidate

        bound = 100.0 * _preservation_lower_bound(best["correct"], base_correct, seed=args.seed)
        passes = bound >= -args.budget
        trace.append({"r": r, "ridge": best["ridge"], "id_nll": best["nll"],
                      "preservation_lower_bound_pp": bound, "passes": passes})
        print(f"  r={r:<6.3f} lambda={best['ridge']:<8g} ID NLL {best['nll']:.4f}  "
              f"preservation LCB {bound:+.3f} pp  {'pass' if passes else 'FAIL'}")
        if passes:
            selected = {"r": r, **best}
        else:
            break  # preservation is monotone in r; stop at the first failure

    if selected is None:
        raise SystemExit("no candidate met the preservation budget; widen --r-grid downward")
    mean_ridge = sum_to_mean_ridge(selected["ridge"], args.correction_rows)
    print(
        f"\nselected r={selected['r']}, lambda={selected['ridge']:g} "
        f"(mean-objective equivalent {mean_ridge:.4g})"
    )

    # --- stage 2: evaluation. The official validation set is read only now. ---
    spec = CorrectionSpec(ridge=selected["ridge"], bootstrap_fraction=0.0, seed=args.seed)
    build = dict(rank=args.rank, n_members=args.members,
                 relative_magnitude=selected["r"], spec=spec, seed=args.seed)
    started = time.monotonic()
    corrected, basis, scale = build_ffn_pnc(
        block, correction["hidden"], correction["target"], correct=True, **build
    )
    uncorrected, _, _ = build_ffn_pnc(
        block, correction["hidden"], correction["target"], correct=False, **build
    )
    print(f"construction after caching: {time.monotonic() - started:.1f}s")

    # Cached features do not depend on which ensemble scores them, so cache the
    # evaluation sets once here rather than inside the per-ensemble loop below --
    # otherwise every one of these ~136k images is pushed through the prefix twice.
    print("caching ImageNet validation features...")
    validation_shards = vit_data.ImageNetShards(
        vit_data.shard_paths("validation"), transform, args.workers
    )
    validation_idx = np.arange(
        len(validation_shards) if args.max_eval is None
        else min(args.max_eval, len(validation_shards))
    )
    validation = cache_features(
        adapter, validation_shards.batches(validation_idx, args.batch_size)
    )

    ood_features = {}
    for dataset_name in vit_data.NEAR_OOD + vit_data.FAR_OOD:
        dataset = vit_data.OODZip(dataset_name, transform, args.workers)
        count = len(dataset) if args.max_eval is None else min(args.max_eval, len(dataset))
        print(f"caching {dataset_name} features ({count} images)...")
        ood_features[dataset_name] = cache_features(
            adapter, dataset.batches(args.batch_size, limit=args.max_eval), verbose=False
        )

    # The base checkpoint on the *validation* set. Without this the preservation claim
    # is unverifiable from the output: the budget is enforced against a training-data
    # pool, so the reported P&C top-1 has nothing to be compared with.
    base_probabilities = softmax(_base_logits(adapter, validation))
    results = {
        "base": {
            "top1": 100.0 * accuracy(base_probabilities, validation["labels"]),
            "nll": categorical_nll(base_probabilities, validation["labels"]),
            "ece": expected_calibration_error(base_probabilities, validation["labels"]),
            "temperature": 1.0,
        }
    }
    print(
        f"base (val): top-1 {results['base']['top1']:.3f}%  "
        f"NLL {results['base']['nll']:.4f}  ECE {results['base']['ece']:.4f}"
    )
    temperatures = {}
    for name, members in (("pnc", corrected), ("uncorrected", uncorrected)):
        # Each ensemble is calibrated against itself, on the ID selection pool. Reusing
        # the corrected ensemble's temperature for the ablation would report its NLL and
        # ECE under another model's calibration, and the official validation set must
        # not be fit on at all. Matches the CIFAR-10 and Banking77 tracks.
        temperatures[name] = fit_temperature(
            _member_logits(adapter, block, members, basis, scale, selection),
            selection["labels"],
        )
        print(f"{name}: ID-selection temperature {temperatures[name]:.4f}")

        temperature = temperatures[name]
        probabilities = _ensemble_probabilities(
            adapter, block, members, basis, scale, validation, temperature
        )
        id_scores = predictive_entropy(probabilities)
        summary = {
            "top1": 100.0 * accuracy(probabilities, validation["labels"]),
            "nll": categorical_nll(probabilities, validation["labels"]),
            "ece": expected_calibration_error(probabilities, validation["labels"]),
            "temperature": temperature,
        }
        for family, names in (("near", vit_data.NEAR_OOD), ("far", vit_data.FAR_OOD)):
            scores = []
            for dataset_name in names:
                ood_scores = predictive_entropy(
                    _ensemble_probabilities(
                        adapter, block, members, basis, scale, ood_features[dataset_name],
                        temperature,
                    )
                )
                scores.append(100.0 * auroc(id_scores, ood_scores))
                summary[f"{dataset_name}_auroc"] = scores[-1]
            summary[f"{family}_auroc"] = float(np.mean(scores))
        summary["top1_cost_pp"] = results["base"]["top1"] - summary["top1"]
        results[name] = summary
        print(f"\n{name}: top-1 {summary['top1']:.3f}% "
              f"({summary['top1_cost_pp']:+.3f} pp vs base)  NLL {summary['nll']:.4f}  "
              f"ECE {summary['ece']:.4f}  Near {summary['near_auroc']:.2f}  "
              f"Far {summary['far_auroc']:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"vit_seed{args.seed}_budget{args.budget}.json"
    destination.write_text(json.dumps({
        "selection": {"trace": trace, "r": selected["r"], "ridge": selected["ridge"],
                      "temperature": selected["temperature"]},
        "temperatures": temperatures,
        "feature_cache_seconds": cache_seconds,
        "results": results,
    }, indent=2))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
