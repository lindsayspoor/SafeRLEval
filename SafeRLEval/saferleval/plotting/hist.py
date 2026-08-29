"""Overlapping final-policy cost histograms: exploration vs greedy, per algorithm.

For a single env and one or more safety bounds, plots an (n_algos × n_bounds) grid.
Each cell overlays the per-episode cost distribution of:
  - the final policy with exploration noise  (test_episode_costs)
  - the final policy evaluated greedily      (det_test_episode_costs)
Each cell uses its own x-axis range, derived from that (algo, bound) pair's data.
Error bars show ±1σ across seeds.  The cost limit d is marked as a vertical dashed
line.  Column headers show the bound value; row labels show the algorithm name.

Usage:
    uv run python SafeRLEval/saferleval/plotting/hist.py \\
        experiments/data/aggregate/runs.csv \\
        --env safe_goal_point --bounds 15 25 50 \\
        --out figures/hist_goal.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# STYLE — mirror of plot_paper.py
# =============================================================================

FONT_FAMILY     = "times new roman"
FONT_SIZE       = 24
LABEL_SIZE      = 22
TITLE_SIZE      = 24
LEGEND_SIZE     = 22
TICK_SIZE       = 22
LINE_WIDTH      = 1.8

FIG_WIDTH       = 4.5    # width per column (inches)
PANEL_H         = 2.4    # height per subplot row (inches)

HIST_BINS       = 15
HIST_ERRBAR_CAP = 2
HIST_ERRBAR_LW  = 0.9

THRESHOLD_COLOR = "black"
THRESHOLD_STYLE = "--"
THRESHOLD_WIDTH = 2.2

ALPHA_EXPL      = 0.50   # exploration bars (behind, wide)
ALPHA_GREEDY    = 0.80   # greedy bars (in front, narrower)
LIGHTEN         = 0.48   # how much to lighten exploration colour toward white

ALGO_ORDER      = ["ppo", "ppo_lag", "p3o", "focops"]

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

# =============================================================================
# Style application
# =============================================================================

def _apply_rcparams() -> None:
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

# =============================================================================
# Colour helpers
# =============================================================================

def _color_for(algo: str) -> str:
    return ALGO_COLORS.get(algo, "#555555")


def _lighten(color, amount: float = LIGHTEN):
    """Blend a colour toward white."""
    try:
        r, g, b = mcolors.to_rgb(color)
    except Exception:
        return color
    return (r + (1 - r) * amount,
            g + (1 - g) * amount,
            b + (1 - b) * amount)


# =============================================================================
# Data helpers
# =============================================================================

def _parse_ep_costs(raw) -> Optional[np.ndarray]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    try:
        ep  = json.loads(raw) if isinstance(raw, str) else raw
        arr = np.array(ep, dtype=float)
        return arr if len(arr) > 0 else None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _collect(df_algo: pd.DataFrame, col: str) -> List[np.ndarray]:
    arrs = []
    if col not in df_algo.columns:
        return arrs
    for _, row in df_algo.iterrows():
        arr = _parse_ep_costs(row.get(col))
        if arr is not None:
            arrs.append(arr)
    return arrs


def _seed_densities(seed_arrays: List[np.ndarray],
                    bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean ± std density across seeds (each seed weighted equally)."""
    n_bins = len(bin_edges) - 1
    if not seed_arrays:
        return np.zeros(n_bins), np.zeros(n_bins)
    bw = bin_edges[1] - bin_edges[0]
    rows = []
    for arr in seed_arrays:
        counts, _ = np.histogram(arr, bins=bin_edges)
        n = len(arr)
        rows.append(counts / (n * bw) if n > 0 else np.zeros(n_bins))
    mat  = np.array(rows)
    mean = np.mean(mat, axis=0)
    std  = np.std(mat, axis=0, ddof=1) if len(rows) > 1 else np.zeros(n_bins)
    return mean, std


def _metrics(seed_arrays: List[np.ndarray], bound: float) -> Tuple[float, float]:
    """Return (V, D+_norm) averaged over seeds."""
    if not seed_arrays:
        return float("nan"), float("nan")
    V_vals, dp_vals = [], []
    for arr in seed_arrays:
        V_vals.append(float(np.mean(arr > bound)))
        viol = arr[arr > bound]
        if len(viol) > 0:
            dp_vals.append((float(np.mean(viol)) - bound) / bound)
    return float(np.mean(V_vals)), (float(np.mean(dp_vals)) if dp_vals else 0.0)

# =============================================================================
# Drawing
# =============================================================================

def _draw_policy_bars(ax, seed_arrays: List[np.ndarray], bin_edges: np.ndarray,
                       color, edge_color,
                       width_frac: float, alpha: float, zorder: int,
                       hatch: Optional[str] = None) -> None:
    """Draw density bars with ±1σ error bars."""
    bw      = bin_edges[1] - bin_edges[0]
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean, std = _seed_densities(seed_arrays, bin_edges)
    for c, dens, s in zip(centers, mean, std):
        if dens == 0 and s == 0:
            continue
        ax.bar(c, dens, width=bw * width_frac, align="center",
               color=color, alpha=alpha, hatch=hatch,
               edgecolor=edge_color, linewidth=0.25, zorder=zorder)
        if s > 0:
            ax.errorbar(c, dens, yerr=s, color="black",
                        capsize=HIST_ERRBAR_CAP, linewidth=HIST_ERRBAR_LW,
                        zorder=zorder + 1)


def _natural_ticks(x_lo: float, x_hi: float, n_ticks: int = 4) -> np.ndarray:
    """Pick ~n_ticks nicely rounded tick positions within [x_lo, x_hi]."""
    span = x_hi - x_lo
    if span <= 0:
        return np.array([x_lo])
    raw_step = span / n_ticks
    # Round step to 1 significant figure
    mag = 10 ** np.floor(np.log10(raw_step))
    nice = [1, 2, 2.5, 5, 10]
    step = mag * min(nice, key=lambda s: abs(s - raw_step / mag))
    lo = np.ceil(x_lo / step) * step
    ticks = np.arange(lo, x_hi + step * 0.01, step)
    return ticks[ticks <= x_hi + 1e-9]


def _draw_panel(ax,
                expl_arrs: List[np.ndarray],
                greedy_arrs: List[np.ndarray],
                algo: str,
                bound: float,
                show_xlabel: bool,
                show_ylabel: bool,
                show_col_title: bool,
                show_algo_title: bool) -> None:
    """Draw one (algo, bound) cell of the grid."""
    base_col   = _color_for(algo)
    expl_col   = _lighten(base_col)
    greedy_col = base_col

    algo_label = ALGO_LABELS.get(algo, algo)

    # Per-cell bin edges from pooled expl + greedy episode costs.
    all_costs = [c for arrs in (expl_arrs, greedy_arrs) for arr in arrs for c in arr]
    if not all_costs:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=LABEL_SIZE - 4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_algo_title:
            ax.text(0.5, 1.42, algo_label, transform=ax.transAxes,
                    ha="center", va="bottom", color=mcolors.to_hex(greedy_col),
                    fontsize=LABEL_SIZE, fontweight="bold", clip_on=False)
        if show_col_title:
            ax.text(0.5, 1.65, f"$d={bound:g}$", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=TITLE_SIZE - 2,
                    fontweight="bold", clip_on=False)
        return

    bin_edges = np.histogram_bin_edges(np.array(all_costs), bins=HIST_BINS)

    # Exploration: wide hatched bars, behind.
    _draw_policy_bars(ax, expl_arrs,   bin_edges,
                      expl_col,   greedy_col, 0.90, ALPHA_EXPL,   zorder=2, hatch="///")
    # Greedy: narrower solid bars, in front.
    _draw_policy_bars(ax, greedy_arrs, bin_edges,
                      greedy_col, "white",    0.55, ALPHA_GREEDY, zorder=3, hatch=None)

    # Cost limit
    ax.axvline(bound, color=THRESHOLD_COLOR, linestyle=THRESHOLD_STYLE,
               linewidth=THRESHOLD_WIDTH, zorder=10)

    # Metric annotations — floated above the axes, not inside the plot area.
    # Stack (bottom→top): expl | greedy | algo title | d-header
    V_e, dp_e = _metrics(expl_arrs,   bound)
    V_g, dp_g = _metrics(greedy_arrs, bound)
    dp_sym = r"$D_{\mathrm{norm}}^+$"
    ann_e = f"expl:    $V$={V_e:.0%},  {dp_sym}={dp_e:.3f}"
    ann_g = f"greedy: $V$={V_g:.0%},  {dp_sym}={dp_g:.3f}"
    fs = max(TITLE_SIZE - 8, 9)
    ax.text(0.03, 1.02, ann_e, transform=ax.transAxes, va="bottom",
            fontsize=fs, color=mcolors.to_hex(expl_col), clip_on=False)
    ax.text(0.03, 1.20, ann_g, transform=ax.transAxes, va="bottom",
            fontsize=fs, color=mcolors.to_hex(greedy_col), clip_on=False)

    # Algo title: centred above metric annotations, on middle column only.
    if show_algo_title:
        ax.text(0.5, 1.42, algo_label, transform=ax.transAxes,
                ha="center", va="bottom", color=mcolors.to_hex(greedy_col),
                fontsize=LABEL_SIZE, fontweight="bold", clip_on=False)

    # Column header: bound value (top row only).
    # Always at y=1.58 so all three d=... labels sit at the same height,
    # regardless of whether an algo title is present in this cell.
    if show_col_title:
        ax.text(0.5, 1.65, f"$d={bound:g}$", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=TITLE_SIZE - 6,
                fontweight="bold", clip_on=False)

    # Per-cell x range with natural ticks
    x_lo, x_hi = bin_edges[0], bin_edges[-1]
    ax.set_xlim(x_lo, x_hi)
    ticks = _natural_ticks(x_lo, x_hi, n_ticks=4)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))

    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Y-axis: label and ticks only on leftmost column
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=LABEL_SIZE, labelpad=4)
    else:
        ax.tick_params(labelleft=False)
        ax.set_ylabel("")

    # X-axis label only on bottom row; ticks always visible
    if show_xlabel:
        ax.set_xlabel("Cost", fontsize=LABEL_SIZE, labelpad=6)
    else:
        ax.set_xlabel("")

# =============================================================================
# Main
# =============================================================================

def plot_condition_hist(raw_csv: str,
                         env: str,
                         bounds: List[float],
                         algos: Optional[List[str]] = None,
                         out_path: str = "figures/hist.pdf",
                         x_max: Optional[float] = 250.0) -> None:

    _apply_rcparams()

    df = pd.read_csv(raw_csv)
    df = df[df["env"] == env]
    if df.empty:
        print(f"No data for env={env}.")
        return

    algo_list  = algos or [a for a in ALGO_ORDER if a in df["algo"].values]
    bound_list = sorted(bounds)

    if not algo_list:
        print("No matching algorithms found in the CSV.")
        return

    n_algos  = len(algo_list)
    n_bounds = len(bound_list)

    # Collect episode cost arrays per (algo, bound).
    seed_data: dict = {}
    for algo in algo_list:
        for bound in bound_list:
            df_ab = df[(df["algo"] == algo) & (df["bound"].astype(float) == bound)]
            expl_arrs   = _collect(df_ab, "test_episode_costs")
            greedy_arrs = _collect(df_ab, "det_test_episode_costs")
            if x_max is not None:
                expl_arrs   = [arr[arr <= x_max] for arr in expl_arrs]
                greedy_arrs = [arr[arr <= x_max] for arr in greedy_arrs]
            seed_data[(algo, bound)] = (expl_arrs, greedy_arrs)

    fig_w = FIG_WIDTH * n_bounds
    # Add 2.5 inches of headroom at the top for legend, suptitle, and above-axes annotations.
    fig_h = PANEL_H * n_algos + 2.5
    fig, axs = plt.subplots(n_algos, n_bounds,
                             figsize=(fig_w, fig_h),
                             squeeze=False)
    # Keep the subplot area the same absolute height as before; extra goes to the top.
    subplot_frac = (PANEL_H * n_algos) / fig_h
    top_frac     = 0.06 + subplot_frac
    fig.subplots_adjust(hspace=1.2,
                        left=0.18, right=0.97,
                        top=top_frac, bottom=0.06,
                        wspace=0.35)

    # Human-readable env name as figure suptitle
    from saferleval.common import TRANSLATIONS
    env_nice = TRANSLATIONS.get(env, env.replace("_", " ").title())
    fig.suptitle(env_nice, fontsize=TITLE_SIZE, y=1.03)

    for i, algo in enumerate(algo_list):
        for j, bound in enumerate(bound_list):
            expl_arrs, greedy_arrs = seed_data[(algo, bound)]
            ax = axs[i, j]
            _draw_panel(
                ax, expl_arrs, greedy_arrs, algo, bound,
                show_xlabel    = (i == n_algos - 1),
                show_ylabel    = (j == 0),
                show_col_title = (i == 0),
                show_algo_title= (j == n_bounds // 2),
            )

    # Shared legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#bbbbbb", alpha=ALPHA_EXPL, hatch="///",
              edgecolor="#555555", label="Exploration"),
        Patch(facecolor="#555555", alpha=ALPHA_GREEDY, label="Greedy"),
        Line2D([0], [0], color=THRESHOLD_COLOR, linestyle=THRESHOLD_STYLE,
               linewidth=THRESHOLD_WIDTH, label="$d$"),
    ]
    fig.legend(handles=legend_items, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=3,
               frameon=False, fontsize=LEGEND_SIZE - 4)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("raw_csv",
                   help="runs.csv produced by evaluate.py.")
    p.add_argument("--env",    required=True,
                   help="Environment name, e.g. safe_goal_point.")
    p.add_argument("--bounds", required=True, nargs="+", type=float,
                   help="One or more cost limits d, e.g. --bounds 15 25 50.")
    p.add_argument("--algos",  nargs="+", default=None,
                   help=f"Algorithms in order (default: {ALGO_ORDER}).")
    p.add_argument("--out",    default="figures/hist.pdf",
                   help="Output PDF path.")
    p.add_argument("--x_max",  type=float, default=250.0,
                   help="Clip episode costs above this value before binning.")
    a = p.parse_args()
    plot_condition_hist(
        raw_csv=a.raw_csv,
        env=a.env,
        bounds=a.bounds,
        algos=a.algos,
        out_path=a.out,
        x_max=a.x_max,
    )
