"""Per-condition cost and reward histograms.
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
                               envs: Optional[List[str]] = None,
                               bounds: Optional[List[float]] = None,
                               metric: str = "test_cost",
                               bins: int = 20,
                               fig_width: float = 4.0,
                               panel_h: float = 2.8) -> None:

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

    for idx in range(n_cond, nrows * ncols):
        axs[idx // ncols, idx % ncols].set_visible(False)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.04),
                   ncol=min(len(labels), 5), frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved histogram figure: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot per-condition cost/reward histograms.")
    p.add_argument("raw_csv", help="Raw CSV path.")
    p.add_argument("--out",  default="figures/histograms.pdf", help="Output file path.")
    p.add_argument("--metric",  default="test_cost",
                   help="Metric to plot: test_cost, train_cost, D_norm_test, D_norm_train, etc.")
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--envs",  nargs="+", default=None)
    p.add_argument("--bounds",  nargs="+", type=float, default=None)
    p.add_argument("--bins",  type=int, default=20)
    a = p.parse_args()
    plot_condition_histograms(
        raw_csv_path=a.raw_csv,
        out_path=a.out,
        algos=a.algos,
        envs=a.envs,
        bounds=a.bounds,
        metric=a.metric,
        bins=a.bins,
    )
