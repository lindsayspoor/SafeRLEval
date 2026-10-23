#!/usr/bin/env python3
"""Run stochastic final-policy evaluation on saved model checkpoints.

Reads runs.csv, finds rows whose test_episode_costs column is NaN,
loads the corresponding checkpoint from models_dir, runs stochastic
episodes (deterministic=False), and fills in the test_* columns.

Writes the updated runs.csv in place (backup saved as runs.csv.bak).

Usage:
    uv run python experiments/eval_stoch.py \\
        --runs_csv experiments/data/aggregate/runs.csv \\
        --models_dir models/models \\
        --episodes 100 \\
        --level 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CRAX_DIR = Path(__file__).parent / "crax"
sys.path.insert(0, str(CRAX_DIR))

from brax import envs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

# ---------------------------------------------------------------------------
# Parsing model directory names
# ---------------------------------------------------------------------------

_DIR_RE = re.compile(
    r"^(?P<env>.+)_Level_(?P<level>\d+)_(?P<algo>.+)_bound(?P<bound>[\d.]+)"
    r"_seed(?P<seed>\d+)_(?P<ts>\d{8}_\d{6}_\d{6})$"
)

# Known algos that contain underscores — order matters (longest first).
_KNOWN_ALGOS = ["ppo_lag", "ppo_pid", "ppo_saute", "ppo_cost", "ppo",
                "p3o", "focops", "sac_lag", "sac_pid"]


def _parse_dir(name: str) -> Optional[dict]:
    m = _DIR_RE.match(name)
    if m is None:
        return None
    d = m.groupdict()
    algo = d["algo"]
    if algo not in _KNOWN_ALGOS:
        return None
    return {
        "env":   d["env"],
        "algo":  algo,
        "bound": float(d["bound"]),
        "seed":  int(d["seed"]),
        "level": int(d["level"]),
        "ts":    d["ts"],
    }


def _build_model_index(models_dir: Path) -> dict:
    """Return dict mapping (env, algo, bound, seed) -> sorted list of (ts, path)."""
    index: dict = {}
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = _parse_dir(p.name)
        if parsed is None:
            print(f"  [skip] could not parse: {p.name}")
            continue
        key = (parsed["env"], parsed["algo"], parsed["bound"], parsed["seed"])
        index.setdefault(key, []).append((parsed["ts"], p, parsed["level"]))
    # Sort each list so the most recent run is last.
    for key in index:
        index[key].sort(key=lambda x: x[0])
    return index


def _latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Return the path of the highest-step checkpoint inside run_dir."""
    ckpts = [d for d in run_dir.iterdir()
             if d.is_dir() and d.name.isdigit()]
    if not ckpts:
        return None
    return max(ckpts, key=lambda d: int(d.name))


# ---------------------------------------------------------------------------
# Stochastic rollout (mirrors train.py's run_det_episodes with deterministic=False)
# ---------------------------------------------------------------------------

def run_stoch_episodes(policy_fn, env_name: str, safety_bound: float,
                        num_episodes: int, episode_length: int,
                        level: int, seed: int) -> dict:
    env = envs.get_environment(env_name, level=level)

    def run_one(ep_key):
        state = env.reset(ep_key)

        def step_fn(carry, _):
            state, rng, cum_cost, cum_reward, done = carry
            rng, act_rng = jax.random.split(rng)
            action, _ = policy_fn(state.obs, act_rng)
            next_state = env.step(state, action)
            cost   = next_state.metrics.get("cost",   jnp.zeros(()))
            reward = next_state.metrics.get("reward", jnp.zeros(()))
            cum_cost   = cum_cost   + jnp.where(done, 0.0, cost)
            cum_reward = cum_reward + jnp.where(done, 0.0, reward)
            new_done = done | (next_state.done > 0)
            return (next_state, rng, cum_cost, cum_reward, new_done), None

        init = (state, ep_key,
                jnp.zeros(()), jnp.zeros(()),
                jnp.bool_(False))
        (_, _, cum_cost, cum_reward, _), _ = jax.lax.scan(
            step_fn, init, None, length=episode_length)
        return cum_cost, cum_reward

    ep_keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    cum_costs, cum_rewards = jax.jit(jax.vmap(run_one))(ep_keys)

    costs   = np.asarray(cum_costs)
    rewards = np.asarray(cum_rewards)
    viol    = costs[costs > safety_bound]
    return {
        "test_reward":              float(np.mean(rewards)),
        "test_cost":                float(np.mean(costs)),
        "test_violation_rate":      float(np.mean(costs > safety_bound)),
        "test_cost_mean_violating": float(np.mean(viol)) if len(viol) > 0 else 0.0,
        "test_episode_costs":       json.dumps(costs.tolist()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_csv",   default="experiments/data/aggregate/runs.csv")
    p.add_argument("--models_dir", default="models/models")
    p.add_argument("--episodes",   type=int, default=100)
    p.add_argument("--episode_length", type=int, default=None,
                   help="Episode length for rollouts.")
    p.add_argument("--level",      type=int, default=1,
                   help="Env level to pass to envs.get_environment (default 1).")
    p.add_argument("--algos", nargs="+", default=None,
                   help="Only process these algorithms, e.g. --algos ppo_lag focops.")
    p.add_argument("--envs", nargs="+", default=None,
                   help="Only process these environments, e.g. --envs safe_goal_point.")
    p.add_argument("--bounds", nargs="+", type=float, default=None,
                   help="Only process these safety bounds, e.g. --bounds 15 25.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if test_episode_costs already populated.")
    p.add_argument("--update_wandb", action="store_true",
                   help="Also push results to wandb run summaries.")
    p.add_argument("--wandb_project", default=None,
                   help="Wandb project name, required when --update_wandb is set.")
    args = p.parse_args()

    if args.update_wandb and not args.wandb_project:
        p.error("--wandb_project is required when using --update_wandb")

    runs_csv   = Path(args.runs_csv)
    models_dir = Path(args.models_dir)

    df = pd.read_csv(runs_csv)
    if "test_episode_costs" not in df.columns:
        df["test_episode_costs"] = pd.NA
    df["test_episode_costs"] = df["test_episode_costs"].astype(object)

    # Rows that still need stochastic eval.
    todo_mask = df["test_episode_costs"].isna() if not args.force \
                else pd.Series([True] * len(df))
    if args.algos:
        todo_mask &= df["algo"].isin(args.algos)
    if args.envs:
        todo_mask &= df["env"].isin(args.envs)
    if args.bounds:
        todo_mask &= df["bound"].isin(args.bounds)

    n_todo = todo_mask.sum()
    print(f"Rows needing stochastic eval: {n_todo} / {len(df)}")
    if n_todo == 0:
        print("Nothing to do.")
        return

    index = _build_model_index(models_dir)
    print(f"Model dirs indexed: {len(index)} unique (env, algo, bound, seed) combos.")

    # Back up runs.csv before modifying.
    bak = runs_csv.with_suffix(".csv.bak")
    df.to_csv(bak, index=False)
    print(f"Backup saved: {bak}")

    done, skipped = 0, 0
    for row_idx in df.index[todo_mask]:
        row  = df.loc[row_idx]
        env  = str(row["env"])
        algo = str(row["algo"])
        bound = float(row["bound"])
        seed  = int(row["seed"])

        key = (env, algo, bound, seed)
        if key not in index:
            print(f"  [miss] no model for {key}")
            skipped += 1
            continue

        # Pick the most recent run dir for this (env, algo, bound, seed).
        _, run_dir, level = index[key][-1]
        ckpt = _latest_checkpoint(run_dir)
        if ckpt is None:
            print(f"  [miss] no checkpoint in {run_dir}")
            skipped += 1
            continue

        print(f"  [{done+1}/{n_todo}] {env} {algo} b={bound} s={seed}  ckpt={ckpt.name}")

        try:
            policy_fn = ppo_checkpoint.load_policy(ckpt, deterministic=False)
        except Exception as exc:
            print(f"    load error: {exc}")
            skipped += 1
            continue

        # Use CLI override, else infer from environment (default is usually 1000).
        if args.episode_length is not None:
            ep_len = args.episode_length
        else:
            try:
                tmp_env = envs.get_environment(env, level=level)
                ep_len  = getattr(tmp_env, "episode_length", None) or 1000
            except Exception:
                ep_len = 1000

        try:
            result = run_stoch_episodes(
                policy_fn  = policy_fn,
                env_name   = env,
                safety_bound = bound,
                num_episodes = args.episodes,
                episode_length = ep_len,
                level      = level,
                seed       = seed,
            )
        except Exception as exc:
            print(f"    rollout error: {exc}")
            skipped += 1
            continue

        for col, val in result.items():
            df.at[row_idx, col] = val

        # Save after every row so progress survives interruption.
        df.to_csv(runs_csv, index=False)
        done += 1
        print(f"    test_cost={result['test_cost']:.2f}  "
              f"test_V={result['test_violation_rate']:.2%}")

        if args.update_wandb:
            try:
                import wandb as _wandb
                api = _wandb.Api()
                wb_run = api.run(f"{args.wandb_project}/{run_dir.name}")
                wb_run.summary.update({
                    "test/reward_mean":         result["test_reward"],
                    "test/cost_mean":           result["test_cost"],
                    "test/cost_violation_rate": result["test_violation_rate"],
                    "test/cost_mean_violating": result["test_cost_mean_violating"],
                    "test/episode_costs":       json.loads(result["test_episode_costs"]),
                })
                wb_run.update()
                print(f"    wandb summary updated: {run_dir.name}")
            except Exception as exc:
                print(f"    wandb update failed: {exc}")

    print(f"\nDone: {done} rows updated, {skipped} skipped.")
    print(f"Updated runs.csv saved: {runs_csv}")


if __name__ == "__main__":
    main()
