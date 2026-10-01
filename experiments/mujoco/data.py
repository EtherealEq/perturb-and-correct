"""MuJoCo dynamics data under graded behavioural shift.

One-step transition prediction: the model maps ``concat(state, action)`` to
``next_state``.  Shift is *behavioural* -- the environment is unchanged and only the
policy that generated the transitions varies, so the distribution moves without the
task changing underneath it.

Tiers, and where they come from:

==========  ====================  =========================================
tier        source                meaning
==========  ====================  =========================================
``id``      Minari ``expert``     the pool the base model is trained on
``near``    Minari ``medium``     a competent but non-expert policy
``mid``     Minari ``simple``     a weaker policy; absent for some envs
``far``     uniform-random actions  no Minari dataset exists for this
==========  ====================  =========================================

Shifted data are used for *evaluation only*.  They never enter training, the
correction pool, hyperparameter selection, or any post-hoc calibration.
"""
from __future__ import annotations

import contextlib
from functools import lru_cache

import numpy as np

__all__ = ["TIERS", "load_transitions", "tier_available", "ENV_SLUGS"]

# Environment name -> Minari dataset slug.
ENV_SLUGS = {
    "Ant-v5": "ant",
    "HalfCheetah-v5": "halfcheetah",
    "Hopper-v5": "hopper",
    "Humanoid-v5": "humanoid",
    "HumanoidStandup-v5": "humanoidstandup",
    "InvertedDoublePendulum-v5": "inverteddoublependulum",
    "InvertedPendulum-v5": "invertedpendulum",
    "Pusher-v5": "pusher",
    "Reacher-v5": "reacher",
    "Swimmer-v5": "swimmer",
    "Walker2d-v5": "walker2d",
}

# Our tier names -> Minari quality level.
TIERS = {"id": "expert", "near": "medium", "mid": "simple"}


@lru_cache(maxsize=1)
def _available_dataset_ids() -> frozenset[str]:
    """Dataset ids Minari can serve, local cache first then the remote index."""
    try:
        import minari
    except ImportError:  # pragma: no cover - environment-dependent
        return frozenset()
    ids: set[str] = set()
    for lister in ("list_local_datasets", "list_remote_datasets"):
        # Offline or unreachable: the other source may still work.
        with contextlib.suppress(Exception):
            ids |= set(getattr(minari, lister)())
    return frozenset(ids)


def tier_available(env: str, tier: str) -> bool:
    """Whether ``tier`` can be sourced for ``env``.

    Not every environment ships every quality level -- several have no ``simple``
    dataset, so their ``mid`` tier simply does not exist and asking for it raises
    rather than degrading.  ``far`` is always available because it is collected live.
    """
    if tier == "far":
        return True
    slug, level = ENV_SLUGS.get(env), TIERS.get(tier)
    if slug is None or level is None:
        return False
    ids = _available_dataset_ids()
    if not ids:  # no index reachable: do not prune, let the load itself speak
        return True
    return f"mujoco/{slug}/{level}-v0" in ids


def _random_action_transitions(env_name: str, n_steps: int, seed: int):
    """Far-OOD: roll out uniform-random actions against the live environment."""
    import gymnasium as gym

    env = gym.make(env_name)
    inputs, targets = [], []
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    while len(inputs) < n_steps:
        action = env.action_space.sample()
        next_observation, _, terminated, truncated, _ = env.step(action)
        inputs.append(np.concatenate([observation, action]))
        targets.append(next_observation)
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()
    return (
        np.stack(inputs[:n_steps]).astype(np.float32),
        np.stack(targets[:n_steps]).astype(np.float32),
    )


def load_transitions(
    env: str,
    tier: str,
    n_steps: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``n_steps`` transitions for one tier.

    Returns ``(inputs, targets)`` with ``inputs[i] = concat(state_i, action_i)`` and
    ``targets[i] = next_state_i``.  ``seed`` selects which episodes are drawn, so
    different seeds give deterministic sub-samples of the same dataset.
    """
    if tier == "far":
        return _random_action_transitions(env, n_steps, seed)
    if env not in ENV_SLUGS:
        raise ValueError(f"unknown environment {env!r}; expected one of {sorted(ENV_SLUGS)}")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)} or 'far'")

    import minari

    dataset_id = f"mujoco/{ENV_SLUGS[env]}/{TIERS[tier]}-v0"
    dataset = minari.load_dataset(dataset_id, download=True)

    order = np.random.default_rng(seed).permutation(len(dataset))
    input_blocks, target_blocks, collected = [], [], 0
    for episode_index in order:
        episode = next(iter(dataset.iterate_episodes(episode_indices=[int(episode_index)])))
        observations = np.asarray(episode.observations, dtype=np.float32)
        actions = np.asarray(episode.actions, dtype=np.float32)
        # observations has length T+1; transition i is (obs[i], act[i]) -> obs[i+1].
        n_transitions = len(actions)
        take = min(n_transitions, n_steps - collected)
        input_blocks.append(
            np.concatenate([observations[:n_transitions], actions], axis=-1)[:take]
        )
        target_blocks.append(observations[1 : n_transitions + 1][:take])
        collected += take
        if collected >= n_steps:
            break

    if collected < n_steps:
        print(f"warning: {dataset_id} yielded only {collected} of {n_steps} requested transitions")
    return (
        np.concatenate(input_blocks, axis=0).astype(np.float32),
        np.concatenate(target_blocks, axis=0).astype(np.float32),
    )
