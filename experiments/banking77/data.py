"""Banking77 (ID) and CLINC-OOS (OOD) text data.

Banking77 is 77 fine-grained banking intents.  CLINC-OOS supplies three graded OOD
tiers, from a *taxonomy* of the CLINC150 intent set rather than from model behaviour:

``near``
    CLINC in-scope intents in the ``banking`` and ``credit_cards`` domains -- same
    subject matter as Banking77, different label set.
``cross``
    CLINC in-scope intents from the other eight domains.
``far``
    the CLINC ``oos`` split: utterances deliberately outside every in-scope intent.

Both datasets come from the HuggingFace hub.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["load_banking77", "load_clinc_tiers", "DOMAINS"]

DOMAINS = json.loads((Path(__file__).parent / "clinc_domains.json").read_text())


def load_banking77(seed: int = 0, val_fraction: float = 0.1):
    """Return ``(train, val, test)`` as ``(texts, labels)`` pairs, plus the label count.

    The validation split is carved deterministically out of train and is the only
    thing used for calibration and any post-hoc choice.
    """
    from datasets import load_dataset

    dataset = load_dataset("PolyAI/banking77")
    train_texts = np.array(dataset["train"]["text"], dtype=object)
    train_labels = np.array(dataset["train"]["label"], dtype=np.int64)
    order = np.random.RandomState(seed).permutation(len(train_texts))
    n_val = int(val_fraction * len(order))
    val_idx, fit_idx = order[:n_val], order[n_val:]
    return (
        (train_texts[fit_idx], train_labels[fit_idx]),
        (train_texts[val_idx], train_labels[val_idx]),
        (
            np.array(dataset["test"]["text"], dtype=object),
            np.array(dataset["test"]["label"], dtype=np.int64),
        ),
        77,
    )


def load_clinc_tiers() -> dict[str, np.ndarray]:
    """CLINC-OOS test utterances grouped into the near / cross / far tiers."""
    from datasets import load_dataset

    dataset = load_dataset("clinc_oos", "plus")["test"]
    names = dataset.features["intent"].names
    texts = np.array(dataset["text"], dtype=object)
    intents = np.array(dataset["intent"], dtype=np.int64)

    near_intents = set(DOMAINS["banking"]) | set(DOMAINS["credit_cards"])
    oos_label = DOMAINS["oos_label"]
    tier_of = np.array(
        [
            "far" if names[i] == oos_label else ("near" if names[i] in near_intents else "cross")
            for i in intents
        ]
    )
    return {tier: texts[tier_of == tier] for tier in ("near", "cross", "far")}
