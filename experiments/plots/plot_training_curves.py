#!/usr/bin/env python3
"""Plot training curves averaged over seeds, fetched directly from wandb.

Built for SLURM array jobs (scripts/slurm_train.sh): each array task trains one
seed of the same env/algo/level combo and logs to its own wandb run. This script
queries wandb for all runs matching env/algo/level, groups them by seed *and* by
the 'safety_bound' (cost limit) each run was trained with, and plots mean +/- std
bands per metric, per (algo, cost limit) group. The cost subplot also gets a red
dashed line at each distinct safety bound.

Reuses results/common.py from the vendored CRAX/ checkout (already on sys.path
via its editable install), same as scripts/train.py.

Example (matching scripts/slurm_train.sh, run from the project root):
    uv run python scripts/plot_seed_curves.py \\
        --project crax --envs safe_goal_point --algos ppo_lag --level 1 \\
        --metrics reward cost training/lambda_lagr
"""

import argparse
import colorsys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from results.common import (
    DEFAULT_METRIC_COLS,
    DEFAULT_REWARD_METRIC,
    REWARD_METRIC_MAP,
    BASELINES_COLORS,
    TRANSLATIONS,
    align_and_stack,
    get_series,
    moving_average,
    nice_grid,
    set_mpl_style,
)

# Per-metric y-axis label overrides, applied before falling back to TRANSLATIONS.
LABEL_OVERRIDES = {
    "training/lambda_lagr": r"$\lambda$",
}


def metric_label(metric: str) -> str:
    return LABEL_OVERRIDES.get(metric, TRANSLATIONS.get(metric, metric))


def color_for(algo: str, bound_idx: int, n_bounds: int):
    """Color for an (algo, cost-limit) group: shades of the algo's base color.

    With a single cost-limit group per algo, this is just BASELINES_COLORS[algo]
    (unchanged behavior). With multiple groups, each gets a distinct lightness so
    curves stay visually tied to their algorithm while remaining distinguishable.
    """
    base = BASELINES_COLORS.get(algo)
    if base is None or n_bounds <= 1:
        return base
    h, l, s = colorsys.rgb_to_hls(*base[:3])
    offset = (bound_idx - (n_bounds - 1) / 2) * 0.16
    l = min(0.82, max(0.22, l + offset))
    return colorsys.hls_to_rgb(h, l, s)


def resolve_wandb_keys(metrics: List[str], algos: List[str], env: str) -> List[str]:
    """Map shorthand metric names ('reward', 'cost') to literal wandb history columns.

    Any metric not in {'reward', 'cost'} is treated as a literal wandb key
    (e.g. 'training/lambda_lagr', 'training/mean_cost').
    """
    keys = set()
    for metric in metrics:
        if metric == "reward":
            keys.add(REWARD_METRIC_MAP.get(env, DEFAULT_REWARD_METRIC))
            if "ppo_cost" in algos:
                # ppo_cost logs reward with cost already subtracted; get_series
                # reconstructs the true reward by adding the cost column back.
                keys.add(DEFAULT_METRIC_COLS["cost"])
        elif metric == "cost":
            keys.add(DEFAULT_METRIC_COLS["cost"])
        else:
            keys.add(metric)
    return sorted(keys)


def extract_series(df: pd.DataFrame, metric: str, algo: str, env: str) -> Optional[pd.Series]:
    if metric in ("reward", "cost"):
        return get_series(df, algo=algo, metric=metric, metric_cols=DEFAULT_METRIC_COLS, env_name=env)
    if metric not in df.columns:
        return None
    return df[metric].astype(np.float32)


def build_filters(env: str, algo: str, level: int, seeds: Optional[List[int]], state: str,
                   wandb_tags: List[str], safety_bounds: Optional[List[float]] = None) -> dict:
    f = {"config.alg": algo, "config.env_name": env, "config.difficulty": level}
    if seeds:
        f["config.seed"] = {"$in": seeds}
    if safety_bounds:
        f["config.safety_bound"] = {"$in": safety_bounds}
    if state != "any":
        f["state"] = state
    if wandb_tags:
        f["tags"] = {"$in": wandb_tags}
    return f


def fetch_run_history(run, metric_keys: List[str], samples: int) -> Optional[pd.DataFrame]:
    """Fetch each metric key as its own wandb history call and merge on '_step'.

    wandb's Run.history(keys=[...]) issues a single server-side query that
    requires all requested keys to co-occur; if even one key was never logged
    for this run (e.g. a Lagrangian-specific metric on a non-Lagrangian algo),
    it silently returns zero rows for the *whole* request. Fetching per-key
    avoids one missing metric blanking out metrics that the run does have.
    """
    merged = None
    for key in metric_keys:
        h = run.history(keys=["_step", key], samples=samples, x_axis="_step", pandas=True)
        if h is None or h.empty or key not in h.columns:
            continue
        h = h[["_step", key]].dropna()
        if h.empty:
            continue
        merged = h if merged is None else merged.merge(h, on="_step", how="outer")
    if merged is None or merged.empty:
        return None
    return merged.sort_values("_step", kind="mergesort").reset_index(drop=True)


def fetch_grouped_runs(api: wandb.Api, args: argparse.Namespace, env: str, algo: str,
                        metric_keys: List[str]) -> Dict[Optional[float], Dict[int, pd.DataFrame]]:
    """Return dict[safety_bound] -> dict[seed] -> history DataFrame.

    Runs are grouped by their 'safety_bound' (cost limit) wandb config value so
    that sweeps over different cost limits for the same env/algo/level are kept
    as separate series instead of being averaged together. Within a group, if
    multiple runs match the same seed, keeps the most recently created one.
    """
    filters = build_filters(env, algo, args.level, args.seeds, args.state, args.wandb_tags, args.safety_bounds)
    runs = api.runs(args.project, filters=filters, order="-created_at", per_page=200)

    grouped: Dict[Optional[float], Dict[int, pd.DataFrame]] = {}
    for run in runs:
        seed = run.config.get("seed")
        if seed is None:
            continue
        bound = run.config.get("safety_bound")
        seed_map = grouped.setdefault(bound, {})
        if seed in seed_map:
            continue
        if args.max_seeds and len(seed_map) >= args.max_seeds:
            continue
        hist = fetch_run_history(run, metric_keys, args.samples)
        if hist is None:
            print(f"No history for run {run.id} (env={env} algo={algo} seed={seed} bound={bound})")
            continue
        seed_map[seed] = hist
    return grouped


def plot(args: argparse.Namespace) -> None:
    set_mpl_style()
    api = wandb.Api()

    envs = args.envs
    metrics = args.metrics
    n_env = len(envs)
    nrows_env, ncols_env = nice_grid(n_env, max_cols=args.max_cols)
    m = len(metrics)
    total_rows = nrows_env * m
    total_cols = ncols_env

    fig, axs = plt.subplots(total_rows, total_cols, figsize=(args.panel_w * total_cols, args.panel_h * total_rows),
                             squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.12, wspace=0.35, hspace=0.55)

    def get_ax(env_i: int, metric_i: int):
        gr = env_i // ncols_env
        gc = env_i % ncols_env
        return axs[gr * m + metric_i, gc]

    legend_handles: Dict[str, plt.Line2D] = {}
    drawn_thresholds: Dict[int, set] = {}
    cache_frames: List[pd.DataFrame] = []

    for env_i, env in enumerate(envs):
        metric_keys = resolve_wandb_keys(metrics, args.algos, env)

        for algo in args.algos:
            grouped = fetch_grouped_runs(api, args, env, algo, metric_keys)
            if not grouped:
                print(f"No matching runs for env={env} algo={algo} level={args.level}")
                continue

            bounds_sorted = sorted(grouped.keys(), key=lambda b: (b is None, b if b is not None else 0.0))
            n_bounds = len(bounds_sorted)

            for bound_idx, bound in enumerate(bounds_sorted):
                seed_runs = grouped[bound]
                print(f"{env}/{algo} (bound={bound}): found seeds {sorted(seed_runs.keys())}")
                color = color_for(algo, bound_idx, n_bounds)
                bound_suffix = f", bound={bound:g}" if n_bounds > 1 and bound is not None else ""
                legend_key = f"{algo}__{bound}"

                for metric_i, metric in enumerate(metrics):
                    ax = get_ax(env_i, metric_i)

                    dfs = []
                    for seed, hist in seed_runs.items():
                        series = extract_series(hist, metric, algo, env)
                        if series is None:
                            continue
                        d = pd.DataFrame({
                            "_step": hist["_step"].astype(np.int64),
                            "value": series.astype(np.float32),
                        }).dropna()
                        if not d.empty:
                            dfs.append(d)
                            cache_frames.append(d.rename(columns={"_step": "step"}).assign(
                                env=env, algo=algo, bound=bound, metric=metric, seed=seed))
                    if not dfs:
                        continue

                    steps, vals = align_and_stack(dfs)
                    if vals.size == 0:
                        continue

                    mean = vals.mean(axis=0)
                    std = vals.std(axis=0)
                    if args.smoothing_window > 1:
                        mean = moving_average(mean, args.smoothing_window)
                        std = moving_average(std, args.smoothing_window)

                    label = f"{TRANSLATIONS.get(algo, algo)}{bound_suffix} (n={vals.shape[0]})"
                    line, = ax.plot(steps, mean, color=color, label=label)
                    ax.fill_between(steps, mean - args.std_mult * std, mean + args.std_mult * std,
                                     color=line.get_color(), alpha=0.25)
                    if legend_key not in legend_handles:
                        legend_handles[legend_key] = line

                    ax.set_xlabel("Env steps")
                    ax.set_ylabel(metric_label(metric))
                    if args.x_max is not None:
                        ax.set_xlim(0.0, args.x_max)
                    if metric == "cost":
                        # CLI --threshold overrides the per-group safety_bound logged to wandb.
                        threshold = args.threshold if args.threshold is not None else bound
                        if threshold is not None:
                            seen = drawn_thresholds.setdefault(id(ax), set())
                            if threshold not in seen:
                                thr_line = ax.axhline(threshold, linestyle="--", color="red", linewidth=1.8,
                                                       label="Threshold")
                                seen.add(threshold)
                                if "Threshold" not in legend_handles:
                                    legend_handles["Threshold"] = thr_line
                    if args.grid:
                        ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)

        # Title above the top subplot of this env's column: env name, then level.
        top_bbox = get_ax(env_i, 0).get_position()
        col_x = 0.5 * (top_bbox.x0 + top_bbox.x1)
        fig.text(col_x, top_bbox.y1 + 0.045, TRANSLATIONS.get(env, env),
                  ha="center", va="bottom", fontsize=14, fontweight="bold")
        fig.text(col_x, top_bbox.y1 + 0.01, f"level {args.level}",
                  ha="center", va="bottom", fontsize=11)

    # Hide any unused env cells.
    for env_i in range(n_env, nrows_env * ncols_env):
        for metric_i in range(m):
            get_ax(env_i, metric_i).axis("off")

    if legend_handles:
        handles = list(legend_handles.values())
        labels = [h.get_label() for h in handles]
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
                   ncol=min(len(labels), 10), fancybox=True, shadow=True)

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_part = "_".join(envs)
    algo_part = "_".join(args.algos)
    out_path = out_dir / f"{args.out_name}_{env_part}_level_{args.level}_{algo_part}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

    if cache_frames:
        data_dir = Path(args.output_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        all_cache = pd.concat(cache_frames, ignore_index=True)
        bounds_in_data = sorted(b for b in all_cache["bound"].dropna().unique())
        bound_part = "_bound" + "_".join(f"{b:g}" for b in bounds_in_data) if bounds_in_data else ""
        csv_name = f"{args.out_name}_{env_part}_level_{args.level}_{algo_part}{bound_part}.csv"
        cache_path = data_dir / csv_name
        all_cache[["env", "algo", "bound", "metric", "seed", "step", "value"]].to_csv(cache_path, index=False)
        print(f"Saved underlying data points (for restyling without re-fetching from wandb): {cache_path}")


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot mean +/- std training curves over seeds, from wandb.")
    p.add_argument("--project", type=str, required=True, help="WandB project name (e.g. 'crax').")
    p.add_argument("--envs", type=str, nargs="+", default=["safe_goal_point"], help="Environment name(s).")
    p.add_argument("--algos", type=str, nargs="+", default=["ppo_lag"], help="Algorithm name(s).")
    p.add_argument("--level", type=int, default=1, help="Difficulty level.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Specific seed values to include (default: all found).")
    p.add_argument("--max_seeds", type=int, default=None,
                   help="Cap the number of seeds per (algo, bound) group (default: all found).")
    p.add_argument("--safety_bounds", type=float, nargs="+", default=None,
                   help="Cost limit(s)/safety_bound value(s) to include (default: all found, each "
                        "plotted as its own group so different sweeps aren't averaged together).")
    p.add_argument("--metrics", type=str, nargs="+", default=["reward", "cost"],
                   help="Metrics to plot. 'reward'/'cost' are shorthands; any other value is treated "
                        "as a literal wandb key (e.g. training/lambda_lagr, training/mean_cost, "
                        "training/policy_loss, training/v_loss, training/cost_v_loss, "
                        "training/entropy_loss, training/total_loss, training/sps).")
    p.add_argument("--state", type=str, default="finished", choices=["finished", "running", "any"],
                   help="Only include wandb runs in this state.")
    p.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="Filter runs by wandb tags.")
    p.add_argument("--samples", type=int, default=10000, help="Max history rows to fetch per run from wandb.")
    p.add_argument("--smoothing_window", type=int, default=1, help="Moving average window size for smoothing.")
    p.add_argument("--std_mult", type=float, default=1.0, help="Number of std devs to shade around the mean.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Cost limit / safety bound line on cost subplots. Default: read each "
                        "matched run's 'safety_bound' wandb config value; pass this to override it.")
    p.add_argument("--x_max", type=float, default=None, help="If set, clip the x-axis to (0, x_max).")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--max_cols", type=int, default=2, help="Max env columns in grid.")
    p.add_argument("--panel_w", type=float, default=3.1, help="Width per metric subplot.")
    p.add_argument("--panel_h", type=float, default=3.0, help="Height per env row.")
    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--output_data_dir", type=str, default="saved_data")
    p.add_argument("--out_name", type=str, default="training_curves")
    return p


if __name__ == "__main__":
    plot(build_args().parse_args())
