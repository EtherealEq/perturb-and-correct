# ImageNet-1K: P&C on a pretrained ViT-B/16

P&C applied to the final feed-forward block of an 86.6M-parameter ViT-B/16, with no
retraining of any kind. Corresponds to Table 3.

## Data

Everything resolves from the HuggingFace hub through `hf_hub_download`, which serves
files from the local cache when present and downloads them otherwise. Nothing is
extracted; parquet shards and zip archives are read in place. Set `HF_HUB_OFFLINE=1` to
force cache-only.

| Source | What |
| --- | --- |
| `ILSVRC/imagenet-1k` | 28 train shards (correction + selection pools), all 14 validation shards |
| `torch-uncertainty/{SSB_hard,Ninco,inaturalist,Texture,Openimage-O}` | the five OOD archives |
| `torch-uncertainty/ood-datasets-splits` | OpenOOD's canonical imglists |

`ILSVRC/imagenet-1k` is **gated** — you need an accepted licence on your HF account and
`huggingface-cli login`. It is used rather than a resized mirror on purpose: re-encoding
costs 1.85 pp of top-1 and leaves ID images more compressed than the OOD sets, which ship
as originals, so a detector could separate ID from OOD on encoding artefacts rather than
content.

Split membership comes from OpenOOD's imglists, not from archive contents — the
OpenImage-O zip holds 17,632 images of which the test split is 15,869. Canonical sizes,
all of which the loader asserts: SSB-hard 49,000 and NINCO 5,879 (Near); iNaturalist
10,000, Textures 5,160, OpenImage-O 15,869 (Far); 85,908 total.

The checkpoint is `torchvision`'s `ViT_B_16_Weights.IMAGENET1K_V1` (81.07% top-1),
downloaded automatically.

## Run

```bash
pip install -e ".[imagenet]"
python -m experiments.imagenet_vit.run                    # full protocol
python -m experiments.imagenet_vit.run --max-eval 256 \
    --correction-rows 4096 --selection-rows 1024 --members 4   # quick smoke
```

Keep `--correction-rows` above the correction design dimension (`d_hidden + 1 = 3073`),
or the ridge solve sits at its interpolation threshold.

The whole run is ~185k forward passes through the frozen prefix (49k train, 50k
validation, 86k OOD), so wall-clock is set by how fast your hardware runs ViT-B/16.

> On older GPUs, check `torch.cuda.get_arch_list()`. Recent CUDA-13 wheels ship no
> `sm_6x` kernels, so Pascal cards fail with `CUBLAS_STATUS_ARCH_MISMATCH` even though
> `torch.cuda.is_available()` returns True. A `cu121` build (arch list `sm_50`–`sm_90`)
> works, since CUDA binaries are forward-compatible within a major capability family.

## Why this is cheap

Members differ only inside the final FFN, so the entire transformer prefix runs once.
One pass caches, per image at the CLS position: the FFN input `ln_2(x)`, the residual
`x` it is added back onto, and the base model's FFN output. Construction is then pure
linear algebra on those rows.

For the reported `M = 20` run: 284.7 s total from a loaded checkpoint, of which 276.3 s
is the one-time feature pass and **8.4 s** is the perturbations and corrections. Peak
GPU memory 1.49 GiB. The base checkpoint plus the compact P&C state is ~510 MiB against
6.61 GiB for 20 full ViT checkpoints. At inference the shared prefix means 98.4 img/s
against the base model's 100.4.

**CLS rows alone are sufficient here only because this is the final block** — after it
just tokenwise normalisation, CLS selection, and the head remain, so no later attention
can mix patch-token changes back into CLS (Appendix B.7). For an earlier block this
would be wrong; pass all relevant token rows instead.

## Selection under a preservation budget

Unlike the other tracks, the operating point is chosen by an explicit **ID-preservation
budget**, using ID data only. Perturbation magnitude `r = ‖ΔW₁‖_F / ‖W₁‖_F` is increased
until the corrected ensemble would cost more than `--budget` percentage points of top-1
accuracy on a held-out *training* ID pool, judged by a one-sided 95% lower confidence
bound from a paired bootstrap. The largest passing `r` is selected; the ridge is chosen
at each candidate by ID NLL.

The primary 0.50 pp budget selects `r = 2` and `λ = 1000`. The ensemble passes at
`r = 2.0` and fails at `r = 2.125`. Budgets of 0.25 and 1.00 pp select `r = 1.25` and
`r = 3.0` with the same ridge.

`λ = 1000` is against the **summed** objective of Eq. (1); over 32,768 correction rows
its mean-objective equivalent is ≈0.0305. Use `pnc.sum_to_mean_ridge` to convert.

The matched uncorrected ablation never enters selection.

> The preservation-budget protocol was developed after an initial feasibility study
> that had inspected OOD behaviour at several perturbation scales, so the paper treats
> this as an ID-selected diagnostic rather than a preregistered OOD study.

## What to expect

Correction matters more as the perturbation grows: at `r = 1` corrected and uncorrected
ID accuracy are nearly indistinguishable, while at `r = 4` the corrected ensemble loses
0.960 pp and the uncorrected one loses 10.754 pp.

A class-conditional Mahalanobis detector on the final normalised CLS representation
remains a stronger pure OOD ranker here (Near/Far 78.82/92.55 against P&C's
76.49/87.89). Appendix E.3 traces this to spectrum: Mahalanobis separation concentrates
in *low*-variance ID directions, while P&C's induced CLS response has the opposite
weighting (Spearman 0.993 with the ID eigenvalue). The two are not estimating the same
object — Mahalanobis leaves the predictive distribution unchanged.

[openood]: https://github.com/Jingkang50/OpenOOD
