# Banking77: P&C inside a fine-tuned DistilBERT

The same post-hoc construction inside a different model family — self-attention,
residual connections, LayerNorm, and categorical text prediction — with the checkpoint
frozen. Corresponds to Table 14 (Appendix F).

This is a **modality-transfer check**, not a comprehensive language OOD benchmark.

## Data

Both datasets come from the HuggingFace hub and download on first use.

- **ID:** [Banking77][b77] — 77 fine-grained banking intents.
- **OOD:** [CLINC-OOS][clinc] (`plus` configuration), split into three tiers by the
  published CLINC150 taxonomy rather than by model behaviour:

| Tier | Contents |
| --- | --- |
| `near` | in-scope intents in the `banking` and `credit_cards` domains |
| `cross` | in-scope intents from the other eight domains |
| `far` | the `oos` split — utterances outside every in-scope intent |

The mapping lives in `clinc_domains.json`.

## Run

```bash
pip install -e ".[text]"
python -m experiments.banking77.run --seed 0
```

Fine-tunes `distilbert-base-uncased` (67M parameters) on Banking77, then builds P&C and
the matched uncorrected ablation inside the final FFN. Pass `--checkpoint <dir>` to save
and reuse a fine-tuned model across runs.

## Construction

Perturb `ffn.lin1`, refit `(ffn.lin2.weight, ffn.lin2.bias)` against the base model's
own FFN output on ID calibration activations. Attention, both LayerNorms, and the
classifier head stay fixed.

DistilBERT applies `output_layer_norm` *after* the FFN residual, so a member's CLS
output is `output_layer_norm(residual + ffn_member(hidden))`, then the pre-classifier
and classifier. As with the ViT, this is the final block, so caching CLS rows alone is
sound.

The correction uses a **full bootstrap** (`bootstrap_fraction=1.0`): each member solves
Eq. (1) on its own same-size resample of the calibration set, drawn with replacement.

## Perturbation magnitude

Appendix F does not state `r` for this experiment, so the default here is **selected, not
taken from the paper**. It matters more than it looks: a random rank-20 perturbation is poorly
aligned with the data, so a surprisingly large `r` is needed before it affects the output at
all, and below that the uncorrected ablation is indistinguishable from P&C and the experiment
demonstrates nothing.

Measured on a 2-epoch CPU checkpoint (ID accuracy 0.890), M=20, K=20, full bootstrap:

| r | P&C acc / NLL / ECE | uncorrected acc / NLL / ECE |
| --- | --- | --- |
| 0.5 | 0.890 / 0.367 / 0.014 | 0.890 / 0.366 / 0.014 |
| 1.0 | 0.891 / 0.367 / 0.015 | 0.889 / 0.366 / 0.016 |
| **2.0** | **0.890 / 0.366 / 0.015** | **0.889 / 0.396 / 0.015** |
| 4.0 | 0.889 / 0.366 / 0.015 | 0.838 / 0.809 / 0.190 |
| 8.0 | 0.889 / 0.365 / 0.015 | 0.375 / 2.342 / 0.113 |

Two things to read off this. P&C is **flat across a 16x range** -- accuracy 0.889-0.891, NLL
0.365-0.367 -- so the correction absorbs the perturbation completely even at `r = 8`. And the
uncorrected ablation separates from it only from `r = 2` upward, matching the scale dependence
Appendix E.1 reports for the ViT, where `r = 1` leaves corrected and uncorrected nearly
indistinguishable.

`r = 2` is the default: it is where the ablation starts to separate while ID accuracy is still
preserved, which is the signature Table 14 reports, and it coincides with the ImageNet track's
ID-selected operating point. `r = 4` reproduces the *magnitude* of Table 14's NLL degradation
but also costs the uncorrected ensemble 5 points of accuracy, which Table 14 does not show --
plausibly because the checkpoint above is undertrained relative to the paper's.

## What to expect

P&C preserves ID behaviour where the uncorrected ablation does not — 92.5% accuracy with
NLL 0.301 and ECE 0.031, against the same perturbations uncorrected at 92.1% accuracy but
NLL 1.079 and ECE 0.525. That gap is the point of the experiment: the repair remains
necessary in this architecture.

Reaching those absolute numbers needs the full five-epoch fine-tune; a short run lands near
89% and scales the whole table down with it. Check the *gap* between the two rows rather than
the absolute values, and make sure `r` is large enough for there to be a gap at all.

On CLINC-OOS, P&C is competitive with MC Dropout while a plain Energy score remains the
strongest detector in this setting.

[b77]: https://huggingface.co/datasets/PolyAI/banking77
[clinc]: https://huggingface.co/datasets/clinc_oos
