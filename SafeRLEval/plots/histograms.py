"""Per-condition cost and reward histograms.

Main entry point:
    plot_condition_histograms(raw_csv_path, out_path,
                              algos=None, envs=None, bounds=None,
                              metric="test_cost", bins=20)

Produces one subplot per (env, bound) condition, showing the distribution of
the selected metric across seeds.  Useful for inspecting per-condition
variability and identifying outlier seeds.

Typical usage:
    # Per-condition test-time reward distribution
    plot_condition_histograms("data/aggregate/safety_spectrum_raw.csv",
                              "figures/hist_test_reward.pdf",
                              metric="test_reward")

    # Per-condition D_norm distribution at test time
    plot_condition_histograms("data/aggregate/safety_spectrum_raw.csv",
                              "figures/hist_dnorm_test.pdf",
                              metric="C_norm_test")   # computed on the fly
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saferleval.common import BASELINES_COLORS, TRANSLATIONS, set_mpl_style, nice_grid

DERIVED_METRICS = {"C_norm_train", "C_norm_test", "D_norm_train", "D_norm_test"}


def _compute_derived(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Add derived metric columns (C_norm, D_norm) to df in place."""
    if metric in ("C_norm_train", "D_norm_train"):
        if "train_cost" in df.columns and "bound" in df.columns:
            df = df.copy()
            df[metric] = (df["train_cost"].astype(float) - df["bound"].astype(float)) \
                         / df["bound"].astype(float)
    elif metric in ("C_norm_test", "D_norm_test"):
        if "test_cost" in df.columns and "bound" in df.columns:
            df = df.copy()
            df[metric] = (df["test_cost"].astype(float) - df["bound"].astype(float)) \
                         / df["bound"].astype(float)
    return df


def plot_condition_histograms(raw_csv_path: str,
                               out_path: str | Path,
                               algos:  Optional[List[str]] = None,
                               envs:   Optional[List[str]] = None,
                               bounds: Optional[List[float]] = None,
                               metric: str = "test_cost",
                               bins: int = 20,
                               fig_width: float = 4.0,
                               panel_h: float = 2.8) -> None:
    """Per-condition histogram of `metric` across seeds, one panel per (env, bound).

    Parameters
    ----------
    raw_csv_path : str
        Path to safety_spectrum_raw CSV.
    out_path : str or Path
        Output PDF/PNG.
    algos, envs, bounds : optional filter lists.
    metric : column name to plot (supports derived: C_norm_train/test, D_norm_train/test).
    bins : histogram bin count.
    """
    set_mpl_style()
    out_path = Path(out_path)

    df = pd.read_csv(raw_csv_path)
    if metric in DERIVED_METRICS:
        df = _compute_derived(df, metric)

    if algos:
        df = df[df["algo"].isin(algos)]
    if envs and "env" in df.columns:
        df = df[df["env"].isin(envs)]
    if bounds and "bound" in df.columns:
        df = df[df["bound"].isin(bounds)]

    if metric not in df.columns:
        print(f"Warning: metric '{metric}' not found in CSV. Available: {list(df.columns)}")
        return

    conditions = sorted(
        df[["env", "bound"]].drop_duplicates().itertuples(index=False),
        key=lambda r: (r.env, float(r.bound)),
    ) if "env" in df.columns and "bound" in df.columns else [None]

    n_cond = len(conditions)
    nrows, ncols = nice_grid(n_cond, max_cols=3)

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(fig_width * ncols, panel_h * nrows),
                            squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.40)

    algos_present = algos or sorted(df["algo"].unique().tolist()) if "algo" in df.columns else []

    for idx, cond in enumerate(conditions):
        ax = axs[idx // ncols, idx % ncols]

        if cond is not None:
            sub = df[(df["env"] == cond.env) & (df["bound"] == cond.bound)]
            title = f"{TRANSLATIONS.get(cond.env, cond.env)}, d={cond.bound:g}"
        else:
            sub = df
            title = metric

        all_vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if all_vals.empty:
            ax.set_visible(False)
            continue

        for algo in algos_present:
            if "algo" not in sub.columns:
                break
            algo_vals = pd.to_numeric(sub.loc[sub["algo"] == algo, metric],
                                      errors="coerce").dropna()
            if algo_vals.empty:
                continue
            color = BASELINES_COLORS.get(algo)
            label = TRANSLATIONS.get(algo, algo)
            ax.hist(algo_vals, bins=bins, alpha=0.6, color=color, label=label, density=True)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide empty panels
    for idx in range(n_cond, nrows * ncols):
        axs[idx // ncols, idx % ncols].set_visible(False)

    # Shared legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.04),
                   ncol=min(len(labels), 5), frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved histogram figure: {out_path}")
    plt.close(fig)
