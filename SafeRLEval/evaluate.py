#!/usr/bin/env python3
"""Compute safety metrics for evaluated safe RL algorithms.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import wandb

from saferleval.metrics import (
    build_summary,
    compute_aggregate_new_metrics,
    compute_iqm_bootstrap_ci,
    compute_new_metrics,
    mean_over_seeds,
    print_aggregate_new_metrics,
    print_iqm_bootstrap_table,
    print_per_cell_new_metrics,
    save_data,
    std_over_seeds,
)

_NEEDED_COLS = [
    "train_reward",              "train_cost",
    "train_violation_rate",      "train_cost_mean_violating",
    "test_reward",               "test_cost",
    "test_violation_rate",       "test_cost_mean_violating",
    "det_test_reward",           "det_test_cost",
    "det_test_violation_rate",   "det_test_cost_mean_violating",
]



def _load_df(source: str) -> pd.DataFrame:
    path = Path(source)
    if path.is_dir():
        files = sorted(path.glob("*_summary.csv"))
        if not files:
            sys.exit(f"No *_summary.csv files found in {path}")
        print(f"Combining {len(files)} summary CSV(s) from {path}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return pd.read_csv(path)


def _build_per_point(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, List[dict]]:
    available = [c for c in _NEEDED_COLS if c in df.columns]
    per_point: Dict[str, List[dict]] = {algo: [] for algo in args.algos}

    for algo in args.algos:
        algo_df = df[df["algo"] == algo]
        for env in args.envs:
            for bound in args.safety_bounds:
                rows = algo_df[
                    (algo_df["env"] == env) & (algo_df["bound"] == bound)
                ].to_dict("records")
                if not rows:
                    print(f"no data for algo={algo} env={env} bound={bound}")
                    continue

                point: dict = {
                    "env": env,
                    "bound": bound,
                    "means":{c: mean_over_seeds(rows, c) for c in available},
                    "stds":{c: std_over_seeds(rows, c) for c in available},
                }

                for col, cell_key in [("det_test_episode_costs", "episode_costs_per_seed"),
                                      ("test_episode_costs",     "test_episode_costs_per_seed")]:
                    if col not in df.columns:
                        continue
                    ep_arrs = []
                    for r in rows:
                        raw = r.get(col)
                        if raw is None:
                            continue
                        try:
                            costs = json.loads(raw) if isinstance(raw, str) else raw
                            ep_arrs.append(np.array(costs, dtype=float))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                    if ep_arrs:
                        point[cell_key] = ep_arrs

                per_point[algo].append(point)

    return per_point



def _print_tables(new_metrics, agg_metrics, iqm_ci, algos) -> None:
    """Print two table sets: (1) training + det_test, (2) stochastic test + det_test."""
    for splits, heading in [
        (["train",       "final_greedy"], "TABLE SET 1: TRAINING  &  FINAL POLICY (GREEDY)"),
        (["final_expl",  "final_greedy"], "TABLE SET 2: FINAL POLICY (EXPLORATION)  &  FINAL POLICY (GREEDY)"),
    ]:
        print(f"\n{'#' * 80}\n# {heading}\n{'#' * 80}")
        print_per_cell_new_metrics(new_metrics, algos, splits=splits)
        print_aggregate_new_metrics(agg_metrics, algos, splits=splits)
        print_iqm_bootstrap_table(agg_metrics, iqm_ci, algos, splits=splits)


def _run_local(args: argparse.Namespace) -> None:
    df = _load_df(args.source)

    if args.algos is None:
        args.algos = df["algo"].unique().tolist()
    else:
        df = df[df["algo"].isin(args.algos)]

    if args.envs is None:
        args.envs = df["env"].unique().tolist()
    else:
        df = df[df["env"].isin(args.envs)]

    if args.safety_bounds is None:
        args.safety_bounds = sorted(df["bound"].dropna().unique().tolist())

    per_point = _build_per_point(df, args)
    new_metrics = compute_new_metrics(per_point, args.algos)
    agg_metrics = compute_aggregate_new_metrics(new_metrics, args.algos)
    cache_rows = df.to_dict("records")
    iqm_ci = compute_iqm_bootstrap_ci(cache_rows, args.algos)
    _print_tables(new_metrics, agg_metrics, iqm_ci, args.algos)
    save_data(cache_rows, new_metrics, agg_metrics, args, iqm_ci=iqm_ci)


def _run_wandb(args: argparse.Namespace) -> None:
    api = wandb.Api()
    per_algo_cells, cache_rows = build_summary(api, args)
    new_metrics = compute_new_metrics(per_algo_cells, args.algos)
    agg_metrics = compute_aggregate_new_metrics(new_metrics, args.algos)
    iqm_ci = compute_iqm_bootstrap_ci(cache_rows, args.algos)
    _print_tables(new_metrics, agg_metrics, iqm_ci, args.algos)
    save_data(cache_rows, new_metrics, agg_metrics, args, iqm_ci=iqm_ci)



def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("source", nargs="?", default=None,
                   help="Local CSV file or directory of *_summary.csv files (local).")
    p.add_argument("--project", type=str, default=None,
                   help="WandB project name (wandb).")

    p.add_argument("--algos", nargs="+", default=None,
                   help="Algorithms to include.")
    p.add_argument("--envs", nargs="+", default=None,
                   help="Environments to include.")
    p.add_argument("--safety_bounds", nargs="+", type=float, default=None,
                   help="Safety bounds to include.")

    p.add_argument("--output_data_dir", type=str, default="experiments/data/aggregate",
                   help="Directory for output CSVs.")

    # Wandb
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--seeds",type=int, nargs="+", default=None)
    p.add_argument("--max_seeds", type=int, default=None)
    p.add_argument("--state", type=str, default="finished",
                   choices=["finished", "running", "any"])
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--train_window_frac", type=float, default=1.0)
    p.add_argument("--deterministic_test", action="store_true")

    args = p.parse_args()

    if args.source is not None and args.project is not None:
        p.error("Pass either a source path (local) or --project (wandb).")
    if args.source is None and args.project is None:
        p.error("Provide a source path for local logging, or --project <name> for wandb.")

    if args.source is not None:
        _run_local(args)
    else:
        if args.envs is None or args.algos is None or args.safety_bounds is None:
            p.error("Wandb mode requires --envs, --algos, and --safety_bounds")
        _run_wandb(args)


if __name__ == "__main__":
    main()
