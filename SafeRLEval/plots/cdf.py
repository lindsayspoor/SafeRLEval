"""Aggregate CDF plot of D_norm = (c - d) / d.

Main entry point:
    plot_agg_cdf_figure(raw_csv_path, algos, envs, bounds, out_path,
                        train_df=None, x_min=-1.0, x_max=3.0, n_bootstrap=2000)

X-axis is D_norm = (c - d) / d.  The constraint boundary is the vertical
dashed line at x = 0 (c = d).  Values < 0 are safe; > 0 are violating.

If train_df (a seed-curves DataFrame with columns [algo, bound, seed, metric, value, step])
is provided, a two-panel figure (Training | Test-time) is produced; otherwise
only the test panel is shown.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saferleval.common import BASELINES_COLORS, TRANSLATIONS, set_mpl_style

LINE_WIDTH = 1.8
FILL_ALPHA = 0.18

ALGO_ORDER = ["ppo", "ppo_lag", "focops", "p3o", "ppo_cost"]
ALGO_LABELS = {k: TRANSLATIONS.get(k, k) for k in ALGO_ORDER}

TRAIN_COST_KEYS = {"cost", "episodic/cost"}


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _color_for(algo: str, used: dict) -> tuple:
    if algo not in used:
        used[algo] = BASELINES_COLORS.get(algo, None)
    return used[algo]


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------

def _build_test_norm_costs(raw_csv_path: str,
                            algos: Optional[List[str]] = None,
                            envs:  Optional[List[str]] = None,
                            bounds: Optional[List[float]] = None) -> Dict[str, List[np.ndarray]]:
    """Load per-episode test costs normalised by bound from safety_spectrum_raw.csv.

    Returns {algo: [array_per_seed]} where each array contains D_norm = (c-d)/d
    for all deterministic test episodes of that seed.
    """
    df = pd.read_csv(raw_csv_path)
    if "det_test_episode_costs" not in df.columns:
        print("Warning: 'det_test_episode_costs' not found in raw CSV.")
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
        raw = row["det_test_episode_costs"]
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
    """Pool training window costs normalised by bound.

    Reads from seed-curves DataFrame (train_df) if available, otherwise falls
    back to the 'train_cost' column in the raw CSV.
    """
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
        algo  = row.get("algo")
        bound = row.get("bound")
        val   = row.get("train_cost")
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


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_cdf_ci(seed_arrays: List[np.ndarray], x_grid: np.ndarray,
                       n_bootstrap: int = 2000, alpha: float = 0.05):
    n_seeds  = len(seed_arrays)
    rng      = np.random.default_rng(0)
    boot_cdfs = np.empty((n_bootstrap, len(x_grid)))
    for b in range(n_bootstrap):
        idx    = rng.integers(0, n_seeds, size=n_seeds)
        pooled = np.sort(np.concatenate([seed_arrays[i] for i in idx]))
        boot_cdfs[b] = np.searchsorted(pooled, x_grid, side="right") / len(pooled)
    lower = np.percentile(boot_cdfs, 100 * alpha / 2, axis=0)
    upper = np.percentile(boot_cdfs, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _draw_cdf_panel(ax, data: Dict[str, List[np.ndarray]], used_colors: dict,
                    x_min: float, x_max: float, title: str, show_legend: bool,
                    n_bootstrap: int = 2000) -> None:
    x_grid = np.linspace(x_min, x_max, 500)
    sorted_algos = sorted(data.keys(),
                          key=lambda a: (ALGO_ORDER.index(a)
                                         if a in ALGO_ORDER else len(ALGO_ORDER)))
    for algo in sorted_algos:
        seed_arrs = data[algo]
        pooled    = np.sort(np.concatenate(seed_arrs))
        n         = len(pooled)
        color     = _color_for(algo, used_colors)
        label     = ALGO_LABELS.get(algo, algo)

        mean_cdf = np.searchsorted(pooled, x_grid, side="right") / n
        ax.plot(x_grid, mean_cdf, color=color, linewidth=LINE_WIDTH, label=label)

        if n_bootstrap > 0 and len(seed_arrs) >= 2:
            lo, hi = _bootstrap_cdf_ci(seed_arrs, x_grid, n_bootstrap)
            ax.fill_between(x_grid, lo, hi, color=color, alpha=FILL_ALPHA)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.4,
               label=r"$D_{\mathrm{norm}}=0$", zorder=10)
    ax.set_xlabel(r"$D_{\mathrm{norm}} = (\bar{c} - d)/d$")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(loc="upper left", frameon=False,
                  bbox_to_anchor=(1.01, 1.0), fontsize=10)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_agg_cdf_figure(raw_csv_path: str, out_path: str | Path,
                         algos: Optional[List[str]] = None,
                         envs:  Optional[List[str]] = None,
                         bounds: Optional[List[float]] = None,
                         train_df: Optional[pd.DataFrame] = None,
                         x_min: float = -1.0,
                         x_max: float = 3.0,
                         n_bootstrap: int = 2000) -> None:
    """Aggregate CDF of D_norm.

    Parameters
    ----------
    raw_csv_path : str
        Path to the safety_spectrum_raw CSV produced by safety_spectrum.py.
    out_path : str or Path
        Output PDF/PNG path.
    algos, envs, bounds : optional filter lists.
    train_df : optional seed-curves DataFrame for the training panel.
    x_min, x_max : x-axis limits.
    n_bootstrap : number of stratified bootstrap replicates for CI bands.
    """
    set_mpl_style()
    out_path = Path(out_path)

    test_data  = _build_test_norm_costs(raw_csv_path, algos, envs, bounds)
    if not test_data:
        print("No test episode cost data found — skipping CDF figure.")
        return

    train_data = _build_train_norm_costs(train_df, raw_csv_path, algos, envs, bounds)
    has_train  = bool(train_data)
    n_cols     = 2 if has_train else 1
    used_colors: dict = {}

    fig, axs = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.5), squeeze=False)
    fig.subplots_adjust(wspace=0.35)

    if has_train:
        _draw_cdf_panel(axs[0, 0], train_data, used_colors, x_min, x_max,
                        title="Training", show_legend=False, n_bootstrap=n_bootstrap)
        _draw_cdf_panel(axs[0, 1], test_data, used_colors, x_min, x_max,
                        title="Test-time (deterministic)", show_legend=True,
                        n_bootstrap=n_bootstrap)
        axs[0, 1].set_ylabel("")
    else:
        _draw_cdf_panel(axs[0, 0], test_data, used_colors, x_min, x_max,
                        title="Test-time (deterministic)", show_legend=True,
                        n_bootstrap=n_bootstrap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved CDF figure: {out_path}")
    plt.close(fig)
