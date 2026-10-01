# CIFAR-10: OOD detection under OpenOOD v1.5

Single-block P&C on a CIFAR-adapted pre-activation ResNet-18. Corresponds to Table 2.

## Data

CIFAR-10 downloads automatically to `data/`. The OOD splits need
[OpenOOD v1.5][openood] data, which the script expects in the standard layout:

```
<openood-root>/
  benchmark_imglist/cifar10/test_{cifar100,tin,mnist,svhn,texture,places365}.txt
  images_classic/...
```

Follow OpenOOD's own download instructions (`scripts/download/download.py`), then pass
`--openood-root`. Near OOD is CIFAR-100 and TinyImageNet; Far OOD is MNIST, SVHN,
Textures, and Places365. The protocol defines no Mid tier for CIFAR.

## Run

```bash
pip install -e ".[cifar10]"
python -m experiments.cifar10.train --seed 0
python -m experiments.cifar10.run \
    --checkpoint checkpoints/cifar10/resnet18_seed0.pkl \
    --openood-root data/openood
```

Training is the expensive step (~8,010 s on the paper's GPU): SGD with Nesterov
momentum 0.9, lr 0.1 under cosine decay with a five-epoch warmup, weight decay 5e-4,
batch 128, 300 epochs, flip/crop plus Cutout(8), no label smoothing. P&C then adds at
most ~100 s.

## Construction

The block is a perturb-correct unit as it stands (Appendix B.2):

```
y = Conv1(ReLU(BN1(x)));  u = ReLU(BN2(y));  z = Conv2(u);  Block(x) = z + Shortcut(x)
```

P&C perturbs `Conv1` and refits `Conv2` plus a **new Conv2 bias** — the base
convolution is bias-free, and the augmented constant column is what lets the repair
absorb perturbation-induced mean shifts. The shortcut is untouched, so matching the
original `Conv2` output is exactly matching the block's contribution. BatchNorm keeps
its trained running statistics and stays frozen throughout. No labels are used.

A convolution is a linear map on unfolded patches, so this is the same ridge problem as
the dense case with patch rows as the design. The unfolded design for the selected
block has `|S| · H' · W'` rows against `1 + 9·512 = 4,609` columns, so it is never
materialised: sufficient statistics accumulate in chunks and `--chunk-size` changes
peak memory only, not the solution.

## Defaults

`--stage 3 --block 0 --scale 25 --bootstrap 0.05 --ridge 1e-3 --rank 20 --members 50`,
with a 1,024-image correction pool from `train_fit`. That is the ID-validation optimum
of the complete 6 blocks × 3 scales × 3 bootstrap fractions grid (162 runs across three
checkpoints); forward and reverse coordinate-wise selection recover the same point.

Sensitivity (Appendix D.4): Near AUROC is flat for `K ∈ [5, 40]`; `λ = 0` is singular
and `λ ≥ 1e-3` is nearly flat; scale has a sharp ID-stability boundary, with mean
accuracy collapsing to ~35% at σ=50 and ~10% at σ=100. Early, middle, and late blocks
are all viable — the final-stage block wins on ID-validation NLL, not by much.

## Protocol

The script enforces the ordering: the temperature is fit on the ID validation split,
and OOD datasets are not read from disk until after that. Splits are deterministic —
`train_fit` (base training + correction pool) and `val_select` (the only thing any
choice is made on) are disjoint, and the test set is reported once.

[openood]: https://github.com/Jingkang50/OpenOOD
