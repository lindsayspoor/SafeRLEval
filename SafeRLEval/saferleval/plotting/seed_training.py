#!/usr/bin/env python3
"""Training curves for a given environment, averaged over seeds.

One panel per safety bound, each showing the mean episodic cost (± std across
seeds) over training steps for every algorithm.  Style mirrors plot_paper.py.

Data sources (in priority order):
  1. Wandb run history  — pass --project <name>
  2. Local *_history.csv files — pass --data_dir <dir>

Usage (wandb):
    uv run python SafeRLEval/saferleval/plotting/seed_training.py \\
        --project crax \\
        --env safe_goal_point \\
        --algos ppo ppo_lag p3o focops \\
        --bounds 15 25 50 \\
        --out figures/training_goal.pdf

Usage (local CSV):
    uv run python SafeRLEval/saferleval/plotting/seed_training.py \\
        --data_dir experiments/data/raw \\
        --env safe_goal_point \\
        --algos ppo ppo_lag p3o focops \\
        --bounds 15 25 50 \\
        --out figures/training_goal.pdf
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# STYLE — mirrors plot_paper.py exactly
# =============================================================================
FONT_FAMILY  = "times new roman"
FONT_SIZE    = 24
LABEL_SIZE   = 22
TITLE_SIZE   = 24
LEGEND_SIZE  = 22
TICK_SIZE    = 22

LINE_WIDTH  = 1.8
FILL_ALPHA  = 0.18
SMOOTHING   = 2

THRESHOLD_COLOR = "black"
THRESHOLD_STYLE = "--"
THRESHOLD_WIDTH = 1.4

ALGO_COLORS = {
    "ppo_lag":   "royalblue",
    "ppo_pid":   "#ff7f0e",
    "focops":    "orangered",
    "p3o":       "darkmagenta",
    "ppo_saute": "#8c564b",
    "ppo":       "#7f7f7f",
    "sac_lag":   "#17becf",
    "sac_pid":   "#bcbd22",
}

ALGO_LABELS = {
    "ppo_lag":   "PPO-Lag",
    "ppo_pid":   "PPO-PID",
    "focops":    "FOCOPS",
    "p3o":       "P3O",
    "ppo_saute": "PPO-Saute",
    "ppo":       "PPO",
    "sac_lag":   "SAC-Lag",
    "sac_pid":   "SAC-PID",
}

ALGO_DISPLAY_ORDER = ["ppo", "ppo_lag", "ppo_pid", "focops", "p3o",
                      "ppo_saute", "sac_lag", "sac_pid"]

FIG_WIDTH = 4.5
PANEL_H   = 3.0
# =============================================================================


def _apply_style() -> None:
    mpl.rcParams.update({
        "font.family":       FONT_FAMILY,
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":         FONT_SIZE,
        "axes.labelsize":    LABEL_SIZE,
        "axes.titlesize":    TITLE_SIZE,
        "legend.fontsize":   LEGEND_SIZE,
        "xtick.labelsize":   TICK_SIZE,
        "ytick.labelsize":   TICK_SIZE,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    pad = w // 2
    x_padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_padded, np.ones(w) / w, mode="valid")[:len(x)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DIR_PATTERN = re.compile(
    r"^(?P<env>.+)_Level_(?P<level>\d+)_(?P<algo>.+)_bound(?P<bound>[\d.]+)"
    r"_seed(?P<seed>\d+)_\d{8}_\d{6}_\d{6}$"
)


def _load_from_wandb(project: str, env: str, algos: list[str],
                     bounds: list[float], level: int = 1,
                     ) -> pd.DataFrame:
    """Return DataFrame with columns [algo, bound, seed, step, value]."""
    import wandb
    api  = wandb.Api()
    rows = []
    for algo in algos:
        for bound in bounds:
            filters = {
                "config.alg":          algo,
                "config.env_name":     env,
                "config.difficulty":   level,
                "config.safety_bound": bound,
                "state":               "finished",
            }
            runs = list(api.runs(project, filters=filters,
                                 order="-created_at", per_page=50))
            print(f"  {algo} b={bound:g}: {len(runs)} runs found")
            for run in runs:
                seed = run.config.get("seed", None)
                h = run.history(keys=["_step", "episodic/cost"],
                                samples=10000, x_axis="_step", pandas=True)
                if h is None or h.empty or "episodic/cost" not in h.columns:
                    continue
                h = h[["_step", "episodic/cost"]].dropna().sort_values("_step")
                for _, r in h.iterrows():
                    rows.append({"algo": algo, "bound": bound, "seed": seed,
                                 "step": r["_step"], "value": r["episodic/cost"]})
    return pd.DataFrame(rows)


def _load_from_csv(data_dir: str, env: str, algos: list[str],
                   bounds: list[float]) -> pd.DataFrame:
    """Return DataFrame with columns [algo, bound, seed, step, value]."""
    rows = []
    for path in Path(data_dir).glob("*_history.csv"):
        stem = path.stem.replace("_history", "")
        m = _DIR_PATTERN.match(stem)
        if m is None:
            continue
        if m["env"] != env:
            continue
        algo  = m["algo"]
        bound = float(m["bound"])
        seed  = int(m["seed"])
        if algos and algo not in algos:
            continue
        if bounds and bound not in bounds:
            continue
        df = pd.read_csv(path)
        cost_df = df[df["metric"].isin(["episodic/cost", "eval/episode_cost"])].copy()
        if cost_df.empty:
            continue
        for _, r in cost_df.iterrows():
            rows.append({"algo": algo, "bound": bound, "seed": seed,
                         "step": r["step"], "value": r["value"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_panel(ax, df: pd.DataFrame, bound: float,
                smoothing: int, x_max: float | None) -> dict:
    """Draw one panel (one bound). Returns legend handles dict."""
    handles = {}
    sorted_algos = sorted(
        df["algo"].unique(),
        key=lambda a: (ALGO_DISPLAY_ORDER.index(a)
                       if a in ALGO_DISPLAY_ORDER else len(ALGO_DISPLAY_ORDER)),
    )
    for algo in sorted_algos:
        adf = df[df["algo"] == algo]
        pivot = adf.pivot_table(index="step", columns="seed",
                                values="value", aggfunc="mean")
        steps = pivot.index.to_numpy()
        vals  = pivot.to_numpy(dtype=float)
        mean  = np.nanmean(vals, axis=1)
        std   = np.nanstd(vals,  axis=1)
        if smoothing > 1:
            mean = _moving_avg(mean, smoothing)
            std  = _moving_avg(std,  smoothing)

        color = ALGO_COLORS.get(algo, "#333333")
        label = ALGO_LABELS.get(algo, algo)
        line, = ax.plot(steps, mean, color=color, linewidth=LINE_WIDTH, label=label)
        ax.fill_between(steps, mean - std, mean + std,
                        color=color, alpha=FILL_ALPHA)
        handles[algo] = line

    thr = ax.axhline(bound, color=THRESHOLD_COLOR, linestyle=THRESHOLD_STYLE,
                     linewidth=THRESHOLD_WIDTH, label=f"$d={bound:g}$", zorder=10)
    handles["threshold"] = thr

    ax.set_xlabel("Env steps")
    ax.set_ylabel("Episodic cost")
    ax.set_title(f"$d = {bound:g}$", fontsize=TITLE_SIZE)
    if x_max is not None:
        ax.set_xlim(0, x_max)
    return handles


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def plot(env: str, algos: list[str], bounds: list[float], out_path: str,
         project: str = None, data_dir: str = None, level: int = 1,
         smoothing: int = SMOOTHING, x_max: float = None,
         fig_width: float = FIG_WIDTH, panel_h: float = PANEL_H) -> None:

    _apply_style()

    # Load data
    df = pd.DataFrame()
    if project:
        print("Loading from wandb …")
        df = _load_from_wandb(project, env, algos, bounds, level)
    if df.empty and data_dir:
        print("Loading from local CSV …")
        df = _load_from_csv(data_dir, env, algos, bounds)
    if df.empty:
        print("No training data found.")
        return

    n_panels = len(bounds)
    fig, axs = plt.subplots(1, n_panels,
                             figsize=(fig_width * n_panels, panel_h),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.45)

    all_handles: dict = {}
    for col, bound in enumerate(bounds):
        bdf = df[df["bound"] == bound]
        ax  = axs[0, col]
        if bdf.empty:
            ax.text(0.5, 0.5, f"no data\n(d={bound:g})",
                    transform=ax.transAxes, ha="center", va="center", color="grey")
            ax.set_title(f"$d = {bound:g}$", fontsize=TITLE_SIZE)
            continue
        h = _draw_panel(ax, bdf, bound, smoothing, x_max)
        all_handles.update(h)
        if col > 0:
            ax.set_ylabel("")

    env_nice = env.replace("_", " ").title()
    fig.suptitle(env_nice, fontsize=TITLE_SIZE, y=1.02)

    # Legend: algos first, threshold last
    order = ([a for a in ALGO_DISPLAY_ORDER if a in all_handles]
             + [k for k in all_handles if k not in ALGO_DISPLAY_ORDER])
    handles = [all_handles[k] for k in order if k in all_handles]
    labels  = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=len(handles),
               frameon=False, fontsize=LEGEND_SIZE)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project",   default=None,
                   help="Wandb project name (e.g. crax).")
    p.add_argument("--data_dir",  default=None,
                   help="Directory with *_history.csv files (local fallback).")
    p.add_argument("--env",       required=True,
                   help="Environment name, e.g. safe_goal_point.")
    p.add_argument("--algos",     nargs="+", required=True,
                   help="Algorithms to plot, e.g. ppo ppo_lag p3o focops.")
    p.add_argument("--bounds",    nargs="+", type=float, required=True,
                   help="Safety bounds d, e.g. 15 25 50.")
    p.add_argument("--level",     type=int, default=1,
                   help="Environment difficulty level (default 1).")
    p.add_argument("--out",       default="figures/training_curves.pdf",
                   help="Output PDF path.")
    p.add_argument("--smoothing", type=int, default=SMOOTHING,
                   help=f"Moving-average window (default {SMOOTHING}; 1 = off).")
    p.add_argument("--x_max",     type=float, default=None,
                   help="Clip x-axis to (0, x_max).")
    p.add_argument("--fig_width", type=float, default=FIG_WIDTH)
    p.add_argument("--panel_h",   type=float, default=PANEL_H)
    a = p.parse_args()
    plot(
        env       = a.env,
        algos     = a.algos,
        bounds    = a.bounds,
        out_path  = a.out,
        project   = a.project,
        data_dir  = a.data_dir,
        level     = a.level,
        smoothing = a.smoothing,
        x_max     = a.x_max,
        fig_width = a.fig_width,
        panel_h   = a.panel_h,
    )


if __name__ == "__main__":
    main()
