"""Fine-tune DistilBERT on Banking77 and evaluate P&C under semantic shift.

    python -m experiments.banking77.run --seed 0

Reproduces the construction behind Table 14 (Appendix F): P&C inside the final FFN of
a 67M-parameter DistilBERT classifier, with the matched uncorrected ablation.  This
is a modality-transfer check -- can the same post-hoc construction operate inside a
different model family -- not a comprehensive language OOD benchmark.

The correction here uses a full bootstrap (fraction 1.0, drawn with replacement),
which resamples the calibration set per member without shrinking it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pnc import CorrectionSpec
from pnc.ensemble import average_probabilities, fit_temperature, predictive_entropy
from pnc.metrics import accuracy, auroc, categorical_nll, expected_calibration_error
from pnc.torch import FeedForwardBlock, build_ffn_pnc

from . import data as text_data
from .model import DistilBertAdapter, finetune


def _member_logits(adapter, block, members, basis, scale, features):
    """(M, N, 77) logits, recomputed from cached rows rather than re-running DistilBERT."""
    hidden = torch.from_numpy(features["hidden"]).to(torch.float64)
    stacked = []
    for member in members:
        perturbed = block.w1 + basis.delta(member.coefficient, scale)
        ffn = block.forward(hidden, w1=perturbed, w2=member.w2, b2=member.b2)
        stacked.append(adapter.tail(features["residual"], ffn.float().numpy()))
    return np.stack(stacked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--members", type=int, default=20)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--relative-magnitude", type=float, default=2.0,
                        help="r = ||dW1||_F / ||W1||_F for the median member; below ~2 the "
                             "perturbation is too small to exercise the correction (see README)")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="reuse a fine-tuned checkpoint instead of training one")
    parser.add_argument("--out", type=Path, default=Path("results/banking77"))
    args = parser.parse_args()

    (fit_texts, fit_labels), (val_texts, val_labels), (test_texts, test_labels), n_labels = (
        text_data.load_banking77(seed=args.seed)
    )

    if args.checkpoint and args.checkpoint.exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(args.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    else:
        print("fine-tuning DistilBERT on Banking77...")
        model, tokenizer = finetune(
            fit_texts, fit_labels, n_labels,
            epochs=args.epochs, seed=args.seed, device=args.device,
        )

    adapter = DistilBertAdapter(model, tokenizer, args.device)
    if args.checkpoint and not args.checkpoint.exists():
        adapter.save(args.checkpoint)

    block = FeedForwardBlock(**adapter.weights())

    # Calibration activations come from the ID training split the model was fit on.
    print("caching calibration activations...")
    calibration = adapter.prefix(fit_texts)

    # Full bootstrap: each member solves Eq. (1) on its own same-size resample.
    spec = CorrectionSpec(ridge=args.ridge, bootstrap_fraction=1.0, seed=args.seed)
    build = dict(
        rank=args.rank, n_members=args.members,
        relative_magnitude=args.relative_magnitude, spec=spec, seed=args.seed,
    )
    corrected, basis, scale = build_ffn_pnc(
        block, calibration["hidden"], calibration["target"], correct=True, **build
    )
    uncorrected, _, _ = build_ffn_pnc(
        block, calibration["hidden"], calibration["target"], correct=False, **build
    )

    print("caching evaluation activations...")
    validation = adapter.prefix(val_texts)
    test = adapter.prefix(test_texts)
    ood = {tier: adapter.prefix(texts) for tier, texts in text_data.load_clinc_tiers().items()}

    results = {}
    for name, members in (("pnc", corrected), ("uncorrected", uncorrected)):
        temperature = fit_temperature(
            _member_logits(adapter, block, members, basis, scale, validation), val_labels
        )
        probabilities = average_probabilities(
            _member_logits(adapter, block, members, basis, scale, test), temperature
        )
        id_scores = predictive_entropy(probabilities)
        summary = {
            "accuracy": accuracy(probabilities, test_labels),
            "nll": categorical_nll(probabilities, test_labels),
            "ece": expected_calibration_error(probabilities, test_labels),
            "temperature": temperature,
        }
        for tier, features in ood.items():
            scores = predictive_entropy(
                average_probabilities(
                    _member_logits(adapter, block, members, basis, scale, features), temperature
                )
            )
            summary[f"{tier}_auroc"] = auroc(id_scores, scores)
        results[name] = summary
        print(
            f"\n{name}: acc {summary['accuracy']:.3f}  NLL {summary['nll']:.3f}  "
            f"ECE {summary['ece']:.3f}  near {summary['near_auroc']:.3f}  "
            f"cross {summary['cross_auroc']:.3f}  far {summary['far_auroc']:.3f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"banking77_seed{args.seed}.json"
    destination.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
