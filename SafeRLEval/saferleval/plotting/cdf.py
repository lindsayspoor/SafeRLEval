"""Aggregate CDF plot of D_norm.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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

LINE_WIDTH = 1.8
FILL_ALPHA = 0.18

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

ALGO_ORDER = ["ppo", "ppo_lag", "focops", "p3o"]
# =============================================================================

TRAIN_COST_KEYS = {"cost", "episodic/cost"}


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


def _color_for(algo: str) -> str:
    return ALGO_COLORS.get(algo, "#333333")


def _build_test_norm_costs(raw_csv_path: str,
                            algos: Optional[List[str]] = None,
                            envs:  Optional[List[str]] = None,
                            bounds: Optional[List[float]] = None,
                            col: str = "det_test_episode_costs") -> Dict[str, List[np.ndarray]]:

    df = pd.read_csv(raw_csv_path)
    if col not in df.columns:
        print(f"Warning: '{col}' not found in raw CSV.")
        return {}
    if algos:
        df = df[df["algo"].isin(algos)]
    if envs:
        df = df[df["env"].isin(envs)]
    if bounds:
        df = df[df["bound"].isin(bounds)]

    norm_costs: dict = defaultdict(list)
    for _, row in df.iterrows():
        algo  = row.get("algo")
        bound = row.get("bound")
        if algo is None or bound is None:
            continue
        try:
            bound = float(bound)
        except (ValueError, TypeError):
            continue
        if abs(bound) < 1e-10:
            continue
        raw = row[col]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        try:
            ep  = json.loads(raw) if isinstance(raw, str) else raw
            arr = np.array(ep, dtype=float) / bound - 1.0
            if len(arr) > 0:
                norm_costs[algo].append(arr)
        except (ValueError, TypeError):
            continue
    return dict(norm_costs)


def _build_train_norm_costs(train_df: Optional[pd.DataFrame],
                             raw_csv_path: Optional[str] = None,
                             algos: Optional[List[str]] = None,
                             envs:  Optional[List[str]] = None,
                             bounds: Optional[List[float]] = None) -> Dict[str, List[np.ndarray]]:

    norm_costs: dict = defaultdict(list)

    if train_df is not None:
        cost_df = train_df[train_df["metric"].isin(TRAIN_COST_KEYS)].copy()
        if algos:
            cost_df = cost_df[cost_df["algo"].isin(algos)]
        if envs:
            cost_df = cost_df[cost_df["env"].isin(envs)]
        if not cost_df.empty:
            for (algo, bound, seed), grp in cost_df.groupby(["algo", "bound", "seed"],
                                                              sort=False):
                try:
                    bound = float(bound)
                except (ValueError, TypeError):
                    continue
                if abs(bound) < 1e-10:
                    continue
                vals = pd.to_numeric(grp["value"], errors="coerce").dropna().to_numpy()
                if len(vals) > 0:
                    norm_costs[algo].append(vals / bound - 1.0)
            return dict(norm_costs)

    if raw_csv_path is None:
        return {}
    df = pd.read_csv(raw_csv_path)
    if "train_cost" not in df.columns:
        return {}
    if algos:
        df = df[df["algo"].isin(algos)]
    if envs and "env" in df.columns:
        df = df[df["env"].isin(envs)]
    if bounds and "bound" in df.columns:
        df = df[df["bound"].isin(bounds)]
    for _, row in df.iterrows():
        algo = row.get("algo")
        bound = row.get("bound")
        val = row.get("train_cost")
        if algo is None or bound is None or val is None:
            continue
        try:
            bound, val = float(bound), float(val)
        except (ValueError, TypeError):
            continue
        if abs(bound) < 1e-10 or np.isnan(val):
            continue
        norm_costs[algo].append(np.array([val / bound - 1.0]))
    return dict(norm_costs)



def _bootstrap_cdf_ci(seed_arrays: List[np.ndarray], x_grid: np.ndarray,
                       n_bootstrap: int = 2000, alpha: float = 0.05):
    n_seeds  = len(seed_arrays)
    rng = np.random.default_rng(0)
    boot_cdfs = np.empty((n_bootstrap, len(x_grid)))
    for b in range(n_bootstrap):
        idx= rng.integers(0, n_seeds, size=n_seeds)
        pooled = np.sort(np.concatenate([seed_arrays[i] for i in idx]))
        boot_cdfs[b] = np.searchsorted(pooled, x_grid, side="right") / len(pooled)
    lower = np.percentile(boot_cdfs, 100 * alpha / 2, axis=0)
    upper = np.percentile(boot_cdfs, 100 * (1 - alpha / 2), axis=0)
    return lower, upper



def _draw_cdf_panel(ax, data: Dict[str, List[np.ndarray]],
                    x_min: float, x_max: float, title: str, show_legend: bool,
                    n_bootstrap: int = 2000) -> None:
    x_grid = np.linspace(x_min, x_max, 500)
    sorted_algos = sorted(data.keys(),
                          key=lambda a: (ALGO_ORDER.index(a)
                                         if a in ALGO_ORDER else len(ALGO_ORDER)))
    for algo in sorted_algos:
        seed_arrs = data[algo]
        pooled = np.sort(np.concatenate(seed_arrs))
        n = len(pooled)
        color = _color_for(algo)
        label = ALGO_LABELS.get(algo, algo)

        mean_cdf = np.searchsorted(pooled, x_grid, side="right") / n
        ax.plot(x_grid, mean_cdf, color=color, linewidth=LINE_WIDTH, label=label)

        if n_bootstrap > 0 and len(seed_arrs) >= 2:
            lo, hi = _bootstrap_cdf_ci(seed_arrs, x_grid, n_bootstrap)
            ax.fill_between(x_grid, lo, hi, color=color, alpha=FILL_ALPHA)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.4,
               label=r"$D_{\mathrm{norm}}=0$", zorder=10)
    ax.set_xlabel(r"$D_{\mathrm{norm}}$")
    ax.set_ylabel("CDF")
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.05)
    if show_legend:
        ax.legend(loc="upper left", frameon=False,
                  bbox_to_anchor=(1.01, 1.0), fontsize=LEGEND_SIZE)


def plot_agg_cdf_figure(raw_csv_path: str, out_path: str | Path,
                         algos: Optional[List[str]] = None,
                         envs:  Optional[List[str]] = None,
                         bounds: Optional[List[float]] = None,
                         train_df: Optional[pd.DataFrame] = None,
                         x_min: float = -1.0,
                         x_max: float = 3.0,
                         n_bootstrap: int = 2000) -> None:
    """Always plots up to three panels: Training | Stochastic test | Deterministic test.

    Panels are only included when data is available for them.
    """
    _apply_style()
    out_path = Path(out_path)

    train_data = _build_train_norm_costs(train_df, raw_csv_path, algos, envs, bounds)
    stoch_data = _build_test_norm_costs(raw_csv_path, algos, envs, bounds,
                                         col="test_episode_costs")
    det_data   = _build_test_norm_costs(raw_csv_path, algos, envs, bounds,
                                         col="det_test_episode_costs")

    panels = []
    if train_data:
        panels.append((train_data, "Training"))
    if stoch_data:
        panels.append((stoch_data, "Final policy (exploration)"))
    if det_data:
        panels.append((det_data, "Final policy (greedy)"))

    if not panels:
        print("No episode cost data found — skipping CDF figure.")
        return

    n_cols = len(panels)
    fig, axs = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.5), squeeze=False)
    fig.subplots_adjust(wspace=0.45)

    for col_idx, (data, title) in enumerate(panels):
        show_legend = (col_idx == n_cols - 1)
        _draw_cdf_panel(axs[0, col_idx], data, x_min, x_max,
                        title=title, show_legend=show_legend, n_bootstrap=n_bootstrap)
        if col_idx > 0:
            axs[0, col_idx].set_ylabel("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved CDF figure: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot aggregate CDF of D_norm.")
    p.add_argument("raw_csv", help="Raw CSV path.")
    p.add_argument("--out", default="figures/agg_cdf.pdf", help="Output file path.")
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--envs", nargs="+", default=None)
    p.add_argument("--bounds",  nargs="+", type=float, default=None)
    p.add_argument("--x_min",  type=float, default=-1.0)
    p.add_argument("--x_max", type=float, default=3.0)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    a = p.parse_args()
    plot_agg_cdf_figure(
        raw_csv_path=a.raw_csv,
        out_path=a.out,
        algos=a.algos,
        envs=a.envs,
        bounds=a.bounds,
        x_min=a.x_min,
        x_max=a.x_max,
        n_bootstrap=a.n_bootstrap,
    )

