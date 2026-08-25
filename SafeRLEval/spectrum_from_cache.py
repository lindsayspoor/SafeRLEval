#!/usr/bin/env python3
"""Re-run safety_spectrum tables from a cached _raw.csv file.

No wandb connection needed. Produces:
  - Per-condition table: r̄, c̄, V, C_norm, D+_norm per (algo, env, bound)
  - Aggregate table: mean + IQM of V, C_norm, D+_norm across all (env, bound) pairs

Usage:
    python scripts/spectrum_from_cache.py saved_data/safety_spectrum_raw.csv \\
        --algos ppo ppo_lag focops p3o ppo_cost

    # Restrict to a subset of envs or bounds:
    python scripts/spectrum_from_cache.py saved_data/safety_spectrum_raw.csv \\
        --algos ppo_lag focops --envs safe_goal_point --safety_bounds 25 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# saferleval package — no sys.path manipulation needed
from saferleval.safety_spectrum import (
    compute_aggregate_new_metrics,
    compute_new_metrics,
    mean_over_seeds,
    print_aggregate_new_metrics,
    print_per_cell_new_metrics,
    std_over_seeds,
)

# Scalar columns used to compute summary metrics (mean/std over seeds).
_NEEDED_COLS = [
    "train_reward",              "test_reward",
    "train_cost",                "test_cost",
    "train_violation_rate",      "test_violation_rate",
    "train_cost_mean_violating", "test_cost_mean_violating",
]

# Non-scalar column: per-episode det-test costs stored as a JSON string.
# Collected separately in build_per_point as a list of arrays (one per seed).
_EP_COSTS_COL = "det_test_episode_costs"


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def build_per_point(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, List[dict]]:
    """Build per-algo per-(env, bound) cell dicts from the raw CSV.

    Returns algo -> list of {
        "env", "bound",
        "means": {col: scalar},   # mean over seeds for each scalar column
        "stds":  {col: scalar},   # std  over seeds for each scalar column
        "episode_costs_per_seed": List[np.ndarray],  # present when det_test_episode_costs exists
    }.
    """
    available = [c for c in _NEEDED_COLS if c in df.columns]
    has_ep_costs = _EP_COSTS_COL in df.columns
    per_point: Dict[str, List[dict]] = {algo: [] for algo in args.algos}

    for algo in args.algos:
        algo_df = df[df["algo"] == algo]
        for env in args.envs:
            for bound in args.safety_bounds:
                rows = algo_df[
                    (algo_df["env"] == env) & (algo_df["bound"] == bound)
                ].to_dict("records")
                if not rows:
                    print(f"  [no data] algo={algo} env={env} bound={bound}")
                    continue

                point = {
                    "env":   env,
                    "bound": bound,
                    "means": {c: mean_over_seeds(rows, c) for c in available},
                    "stds":  {c: std_over_seeds(rows, c)  for c in available},
                }

                # Collect per-seed episode cost arrays (non-scalar; kept as list).
                if has_ep_costs:
                    ep_arrs = []
                    for r in rows:
                        raw = r.get(_EP_COSTS_COL)
                        if raw is None:
                            continue
                        try:
                            costs = json.loads(raw) if isinstance(raw, str) else raw
                            ep_arrs.append(np.array(costs, dtype=float))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                    if ep_arrs:
                        point["episode_costs_per_seed"] = ep_arrs

                per_point[algo].append(point)

    return per_point


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Re-run spectrum tables from cached _raw.csv — no wandb needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("raw_csv", help="Path to the _raw.csv saved by safety_spectrum.py.")
    p.add_argument("--algos",         nargs="+", default=None,
                   help="Algorithms to include (default: all found in CSV).")
    p.add_argument("--envs",          nargs="+", default=None,
                   help="Environments to include (default: all found in CSV).")
    p.add_argument("--safety_bounds", nargs="+", type=float, default=None,
                   help="Safety bounds to include (default: all found in CSV).")
    p.add_argument("--output_data_dir", type=str, default="saved_data")
    p.add_argument("--out_name",        type=str, default="safety_spectrum_recached")
    args = p.parse_args()

    df = pd.read_csv(args.raw_csv)

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

    per_point = build_per_point(df, args)

    new_metrics = compute_new_metrics(per_point, args.algos)
    agg_metrics = compute_aggregate_new_metrics(new_metrics, args.algos)
    print_per_cell_new_metrics(new_metrics, args.algos)
    print_aggregate_new_metrics(agg_metrics, args.algos)

    data_dir = Path(args.output_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if new_metrics:
        per_cell_rows = [{"algo": algo, **c}
                         for algo, cells in new_metrics.items() for c in cells]
        per_cell_path = data_dir / f"{args.out_name}_per_cell_normalized.csv"
        pd.DataFrame(per_cell_rows).to_csv(per_cell_path, index=False)
        print(f"Saved per-condition normalized metrics: {per_cell_path}")

    if agg_metrics:
        agg_path = data_dir / f"{args.out_name}_aggregate_normalized.csv"
        pd.DataFrame(agg_metrics).T.to_csv(agg_path)
        print(f"Saved aggregate normalized metrics: {agg_path}")


if __name__ == "__main__":
    main()
