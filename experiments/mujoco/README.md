# MuJoCo: uncertainty under behavioural shift

Probabilistic one-step dynamics prediction — predict `next_state` from
`concat(state, action)` — where the *policy* generating the data shifts while the
environment stays fixed. Corresponds to Table 1 of the paper.

## Data

Transitions come from [Minari][minari] MuJoCo datasets, downloaded on first use.
Nothing is rolled out from a policy checkpoint.

| Tier | Source | Notes |
| --- | --- | --- |
| `id` | `mujoco/<env>/expert-v0` | base training and the correction pool |
| `near` | `mujoco/<env>/medium-v0` | |
| `mid` | `mujoco/<env>/simple-v0` | absent for several environments; skipped automatically |
| `far` | uniform-random actions | collected live; no Minari dataset exists |

Shifted tiers are loaded only after construction and calibration are finished.

## Run

```bash
pip install -e ".[mujoco]"
python -m experiments.mujoco.run --env Ant-v5 --seed 0
```

Trains the base model (4×200 ReLU MLP, heteroscedastic Gaussian head, 5,000 steps at
batch 64), builds multi-block P&C and the matched uncorrected ablation, and writes a
JSON summary to `results/mujoco/`.

Useful flags:

```
--scale 16.0        # sigma; the paper's grid is {8, 16, 32}
--ridge 1e-2        # lambda; grid {1e-4, 1e-2}
--bootstrap 0.1     # per-member correction fraction; grid {0.05, 0.1, 0.2, 0.3}
--members 50 --rank 20
```

For a per-environment sweep, run the script once per `(env, seed)` and aggregate the
JSON yourself. Every method in Table 1 was selected independently per environment by
ID-validation NLL.

## Construction

Multi-block: hidden layers 1 and 3 are perturbed (zero-based `[0, 2]`), each corrected
in the layer immediately after it. Corrections are applied *sequentially* — the second
stage sees activations produced by the first stage's corrected member — but the
regression target at each interface is always the base model's own pre-activation, so
members are pulled back toward the original network at every step rather than drifting.

Output heads are never perturbed or corrected.

## Metrics

Members emit diagonal Gaussians, combined by moment matching. **Mean total predictive
variance** is the OOD score. Reported: ID RMSE, per-tier NLL, per-tier AUROC against
the ID tier, and the Spearman correlation between ID uncertainty and ID error.

Multi-block correction is cheap for MLPs this small, so there is no reason to restrict
to a single block here — unlike the convolutional setting, where compounding patchwise
solves is expensive and one block is selected by ID-validation NLL.

[minari]: https://minari.farama.org/
