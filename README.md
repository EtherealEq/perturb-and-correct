# Perturb and Correct

Reference implementation of **[Perturb and Correct: Post-Hoc Ensembles using Affine Redundancy][paper]** (NeurIPS 2026).

P&C turns a single completed checkpoint into an epistemically diverse ensemble, with no retraining.
Sample a random low-rank perturbation of a hidden-layer weight, then refit the *following* affine
layer by least squares so the perturbed member reproduces the base model's next-layer
pre-activations on in-distribution calibration data. The repair suppresses whatever the calibration
geometry can account for, so members agree where data constrain them and are free to disagree where
that repair no longer transfers.

```
                    ┌─ perturb W_ℓ ─┐      ┌─ refit W_{ℓ+1} on ID calibration data ─┐
    x ──► … ──►  h_ℓ = φ(W_ℓ h + b) ──►  z = W_{ℓ+1} h_ℓ + b_{ℓ+1}  ──► … ──► ŷ
                 random, rank-K            least squares, ridge-centred on W_{ℓ+1}
```

Formally, the correction solves Eq. (1) of the paper:

```
Θ̂(v) = argmin_Θ' ‖ Y_v Θ'ᵀ − Y Θᵀ ‖²_F + λ ‖ Θ' − Θ ‖²_F
```

Two details are load-bearing and easy to get wrong:

1. **The regression target is the base model's own pre-activation**, `Y Θᵀ` — not a ground-truth
   label. The correction never sees labels.
2. **The ridge is centred on the original affine map**, not on zero, so `λ → ∞` recovers the base
   model rather than the zero map.

## Two implementations

The construction ships **natively in both JAX and PyTorch**, with the same API:

```python
from pnc.jax   import build_dense_pnc, build_conv_pnc, build_ffn_pnc
from pnc.torch import build_dense_pnc, build_conv_pnc, build_ffn_pnc
```

Both draw their perturbations from the same host-side RNG in `pnc/spec.py`, so **a given seed
produces the same ensemble in either** — which is what lets `tests/test_equivalence.py` assert the
two agree member-for-member to ~1e-10, and check both against a third, independent NumPy reference
of the Section 4 identities.

Each experiment track uses whichever fits its checkpoint: JAX for MuJoCo and CIFAR-10, PyTorch for
the pretrained ViT and DistilBERT.

Two deliberate differences:

- `pnc.jax` **enables JAX's x64 mode on import.** The Gram matrix reaches 4,609 × 4,609 for the
  CIFAR blocks, and float32 is not enough to solve it faithfully. `pnc.torch` asks for
  `torch.float64` explicitly and changes no global state. Importing `pnc` alone touches neither.
- `pnc.jax.conv` takes **HWIO** kernels and NHWC maps; `pnc.torch.conv` takes **OIHW** and NCHW,
  matching each framework. The perturbation itself is drawn in layout-independent patch coordinates,
  so a seed still means the same subspace in both.

## Install

```bash
pip install -e ".[dev]"        # both implementations + tests
pip install -e ".[jax]"        # JAX implementation only
pip install -e ".[torch]"      # PyTorch implementation only
pip install -e ".[mujoco]"     # + Gymnasium, Minari, flax
pip install -e ".[cifar10]"    # + flax, optax
pip install -e ".[imagenet]"   # + torchvision, pyarrow, huggingface-hub
pip install -e ".[text]"       # + transformers, datasets
pytest                         # 148 tests, no GPU, ~30s
```

The top-level `pnc` package imports neither JAX nor PyTorch, so `from pnc import CorrectionSpec`
stays cheap.

## Quickstart

```python
from pnc import CorrectionSpec
from pnc.jax import DenseStack, build_dense_pnc     # or: from pnc.torch import ...

stack = DenseStack(weights=..., biases=..., heads={"mean": ..., "var": ...})

ensemble = build_dense_pnc(
    stack, calibration_inputs,
    perturb_layers=[0, 2],   # hidden layers to perturb; each is corrected in the next layer
    rank=20, n_members=50, scale=16.0,
    spec=CorrectionSpec(ridge=1e-2, bootstrap_fraction=0.1, seed=0),
    seed=0,
)
predictions = ensemble.predict(test_inputs)   # {"mean": (M, N, D), "var": (M, N, D)}
```

The matched ablation the paper reports throughout — identical perturbations, no repair — is
`correct=False` on the same call.

## Layout

| Path | What it is |
| --- | --- |
| `pnc/spec.py` | `CorrectionSpec`, the shared RNG, bootstrap subsets, ridge conventions |
| `pnc/ensemble.py` | Gaussian moment matching, probability averaging, ID-only temperature fitting |
| `pnc/metrics.py` | RMSE, NLL, ECE, AUROC, FPR95, Spearman |
| `pnc/residual.py` | Section 4's exact identities — an independent NumPy reference for the tests |
| `pnc/jax/`, `pnc/torch/` | `perturb.py`, `correct.py`, and the three backends, in each framework |
| `…/dense.py` | Fully connected trunks (MuJoCo) — solves every member in one batched call |
| `…/conv.py` | Convolutional residual blocks (CIFAR-10) — chunked normal equations |
| `…/transformer_ffn.py` | Transformer feed-forward blocks (ViT, DistilBERT) |
| `experiments/` | One runnable track per setting, each with its own README |

Each backend answers the same two questions for its architecture: which parameter gets the
perturbation, and which rows form the design and target of the following layer's repair. The
sampling, the subsets, and the conventions are shared.

## Experiments

| Track | Setting | Uses | Paper |
| --- | --- | --- | --- |
| [`experiments/mujoco`](experiments/mujoco) | Dynamics prediction under behavioural shift, 11 environments | `pnc.jax` | Table 1 |
| [`experiments/cifar10`](experiments/cifar10) | OOD detection, PreAct ResNet-18 under OpenOOD v1.5 | `pnc.jax` | Table 2 |
| [`experiments/imagenet_vit`](experiments/imagenet_vit) | OOD detection, pretrained 86.6M ViT-B/16 | `pnc.torch` | Table 3 |
| [`experiments/banking77`](experiments/banking77) | Semantic shift, fine-tuned DistilBERT | `pnc.torch` | Table 14 |

Each track ships **P&C and its matched uncorrected ablation**. The other baselines in the paper's
tables (Deep Ensembles, MC Dropout, SWAG, Laplace, Subspace Inference, SCOD, LLLA, Epinet,
Mahalanobis, ReAct+Energy, MSP) are not reimplemented here; they came from their authors' released
code and, for the post-hoc detectors, from [OpenOOD v1.5][openood].

Scripts run **one configuration at a time**, with `--seed` as a plain flag. The paper's multi-seed
aggregations are not orchestrated here — run the script per seed and aggregate however you like.

## Selected configurations

These are the ID-validation optima reported in the paper, and the defaults in each script. They are
*not* re-selected at runtime.

| | MuJoCo | CIFAR-10 | ImageNet ViT | Banking77 |
| --- | --- | --- | --- | --- |
| Target | hidden layers 1, 3 | stage 3, block 0 | final FFN | final FFN |
| Members `M` | 50 | 50 | 20 | 20 |
| Rank `K` | 20 | 20 | 20 | 20 |
| Magnitude | σ ∈ {8, 16, 32} | σ = 25 | r = 2 | r = 2 (selected here) |
| Ridge `λ` | {1e-4, 1e-2} | 1e-3 | 1000 | 1e-3 |
| Bootstrap fraction | {0.05, 0.1, 0.2, 0.3} | 0.05 | — | 1.0 |
| Correction pool | 4,096 transitions | 1,024 images | 32,768 CLS rows | ID train split |

Sensitivity around these points is mild in rank (flat for `K ≥ 5`) and ridge (flat for `λ ≥ 1e-3`,
singular at `λ = 0`), and sharp in perturbation scale, which has a clear ID-stability boundary.
Bootstrap fraction is conditioning-sensitive rather than monotone. See Appendices C and D.

## What to expect

The construction is cheap, and that is the point of the comparison — not inference, which stays
`O(M)` for both P&C and a Deep Ensemble:

| | Base training | P&C construction | Matched 50-member Deep Ensemble |
| --- | --- | --- | --- |
| CIFAR-10 ResNet-18 | 8,010 s | ≤ ~100 s | ~400,500 s (derived) |
| MuJoCo MLP | — | +1.5–1.8 s | 46.8–61.7× more |
| ImageNet ViT-B/16 (M=20) | pretrained | 284.7 s (276.3 s of it feature caching) | 6.61 GiB of checkpoints vs ~510 MiB |

P&C does not improve the base model's ID prediction. It preserves that prediction while introducing
model variation that becomes useful under shift.

## Limitations

Worth reading before applying this to something new. P&C depends on the calibration set covering the
intended ID behaviour, on the correction design staying well conditioned, and on perturbations
staying inside an ID-stable range. Arbitrary distribution shift need not increase disagreement — an
OOD point whose perturbation effects remain predictable from calibration data will legitimately show
little. Dedicated representation-space scores can be stronger for pure OOD ranking: Mahalanobis on
the final CLS representation beats P&C on ImageNet, which Appendix E.3 traces to the two methods
weighting opposite ends of the representation's covariance spectrum. Results are also uneven across
environments — Humanoid remains difficult.

## Citation

```bibtex
@inproceedings{quint2026perturb,
  title     = {Perturb and Correct: Post-Hoc Ensembles using Affine Redundancy},
  author    = {Quint, Eleanor},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

[paper]: https://arxiv.org/search/?searchtype=all&query=Perturb+and+Correct+Post-Hoc+Ensembles+Affine+Redundancy
[openood]: https://github.com/Jingkang50/OpenOOD
