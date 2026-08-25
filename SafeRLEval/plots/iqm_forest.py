"""IQM + 95% bootstrap CI forest plot.

Main entry point:
    plot_iqm_figure(iqm_ci_csv, out_path)

Layout: 2 rows (Train | Test) × 4 cols (R̄, D_norm, V, D+_norm).

PPO is shown as an off-scale right-pointing triangle when its IQM is a clear
outlier; otherwise all algorithms are plotted as dots with CI error bars.
D_norm panels include a vertical dashed reference at x = 0 (c = d boundary).

The iqm_ci_csv is the CSV produced by safety_spectrum.save_data() when it
includes bootstrap CI data (columns: algo, <metric>_iqm, <metric>_iqm_ci_lower,
<metric>_iqm_ci_upper for each metric key).

Hardcoded defaults (from the paper's runs) are used when iqm_ci_csv is None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from saferleval.common import set_mpl_style

# ---------------------------------------------------------------------------
# Hardcoded defaults (paper values — override by passing iqm_ci_csv)
# ---------------------------------------------------------------------------

_IQM_CI_DATA: dict = {
    "PPO": {
        "r_bar_train":       (115.63, 114.87, 116.38),
        "r_bar_test":        ( 68.17,  67.08,  69.21),
        "C_norm_train":      (  4.77,   4.70,   4.85),
        "C_norm_test":       (  2.20,   2.11,   2.28),
        "V_train":           (  1.00,   1.00,   1.00),
        "V_test":            (  0.94,   0.92,   0.96),
        "D_plus_norm_train": (  4.79,   4.72,   4.87),
        "D_plus_norm_test":  (  2.29,   2.21,   2.36),
    },
    "PPO-Lag": {
        "r_bar_train":       ( 77.72,  73.31,  80.50),
        "r_bar_test":        ( 54.18,  51.92,  56.40),
        "C_norm_train":      (  0.00,  -0.01,   0.00),
        "C_norm_test":       ( -0.13,  -0.21,  -0.09),
        "V_train":           (  0.43,   0.42,   0.44),
        "V_test":            (  0.34,   0.31,   0.37),
        "D_plus_norm_train": (  0.25,   0.23,   0.28),
        "D_plus_norm_test":  (  0.50,   0.46,   0.56),
    },
    "P3O": {
        "r_bar_train":       ( 90.41,  89.32,  91.55),
        "r_bar_test":        ( 57.80,  56.04,  59.67),
        "C_norm_train":      (  0.55,   0.52,   0.57),
        "C_norm_test":       ( -0.03,  -0.05,  -0.01),
        "V_train":           (  0.70,   0.69,   0.72),
        "V_test":            (  0.42,   0.40,   0.43),
        "D_plus_norm_train": (  0.81,   0.78,   0.83),
        "D_plus_norm_test":  (  0.53,   0.49,   0.57),
    },
    "FOCOPS": {
        "r_bar_train":       (108.76, 107.77, 109.78),
        "r_bar_test":        ( 74.26,  72.63,  75.83),
        "C_norm_train":      (  0.06,   0.06,   0.06),
        "C_norm_test":       ( -0.19,  -0.26,  -0.13),
        "V_train":           (  0.41,   0.40,   0.41),
        "V_test":            (  0.29,   0.26,   0.32),
        "D_plus_norm_train": (  0.34,   0.33,   0.34),
        "D_plus_norm_test":  (  0.48,   0.43,   0.52),
    },
}

_ALGO_COLORS: dict = {
    "PPO":     "#7f7f7f",
    "PPO-Lag": "royalblue",
    "P3O":     "darkmagenta",
    "FOCOPS":  "orangered",
}

# (col_index, metric_key_stem, panel_title, add_zero_line)
_PANELS = [
    (0, "r_bar",       r"$\mathrm{IQM}(\bar{R})$",                 False),
    (1, "C_norm",      r"$\mathrm{IQM}(D_{\mathrm{norm}})$",       True),
    (2, "V",           r"$\mathrm{IQM}(V)$",                       False),
    (3, "D_plus_norm", r"$\mathrm{IQM}(D^+_{\mathrm{norm}})$",     False),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(iqm_ci_csv: Optional[str]) -> dict:
    if iqm_ci_csv is None:
        return _IQM_CI_DATA
    df   = pd.read_csv(iqm_ci_csv)
    data: dict = {}
    for _, row in df.iterrows():
        algo  = str(row["algo"])
        entry: dict = {}
        for _, stem, _, _ in _PANELS:
            for split in ("train", "test"):
                k   = f"{stem}_{split}"
                iqm = float(row.get(f"{k}_iqm", float("nan")))
                lo  = float(row.get(f"{k}_iqm_ci_lower", float("nan")))
                hi  = float(row.get(f"{k}_iqm_ci_upper", float("nan")))
                entry[k] = (iqm, lo, hi)
        data[algo] = entry
    return data


def _is_outlier(ppo_val: float, others: list, threshold: float = 0.8) -> bool:
    if not others or np.isnan(ppo_val):
        return False
    valid = [v for v in others if not np.isnan(v)]
    if not valid:
        return False
    lo, hi = min(valid), max(valid)
    if lo <= ppo_val <= hi:
        return False
    gap   = max(ppo_val - hi, lo - ppo_val)
    r     = max(hi - lo, 0.01)
    return gap > threshold * r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_iqm_figure(out_path: str | Path,
                    iqm_ci_csv: Optional[str] = None) -> None:
    """Forest plot: 2 rows (Train | Test) × 4 cols (R̄, D_norm, V, D+_norm).

    Parameters
    ----------
    out_path : str or Path
        Output file path (PDF recommended).
    iqm_ci_csv : str, optional
        CSV from safety_spectrum.save_data().  Uses paper defaults when None.
    """
    set_mpl_style()
    out_path = Path(out_path)

    data         = _load_data(iqm_ci_csv)
    algos_order  = ["FOCOPS", "P3O", "PPO-Lag", "PPO"]
    constrained  = ["FOCOPS", "P3O", "PPO-Lag"]
    y_pos        = {a: i for i, a in enumerate(algos_order)}
    n_algos      = len(algos_order)

    fig, axs = plt.subplots(2, 4, figsize=(15.0, 5.2))
    fig.subplots_adjust(wspace=0.50, hspace=0.60,
                        left=0.09, right=0.80, top=0.88, bottom=0.08)

    for col, stem, title, zero_line in _PANELS:
        for row, split in enumerate(("train", "test")):
            ax  = axs[row, col]
            key = f"{stem}_{split}"

            entries = {a: data[a][key] for a in algos_order
                       if a in data and key in data[a]}
            if not entries:
                continue

            ppo_iqm    = entries.get("PPO", (float("nan"),))[0]
            other_iqms = [entries[a][0] for a in constrained if a in entries]
            ppo_off    = _is_outlier(ppo_iqm, other_iqms)

            c_entries   = {a: v for a, v in entries.items() if a in constrained}
            ref_entries = c_entries if (ppo_off and c_entries) else entries
            lo_all = min(v[1] for v in ref_entries.values())
            hi_all = max(v[2] for v in ref_entries.values())
            pad    = 0.25 * max(hi_all - lo_all, 0.01)
            ax.set_xlim(lo_all - pad, hi_all + pad)

            trans = blended_transform_factory(ax.transAxes, ax.transData)

            for algo, (iqm, lo, hi) in entries.items():
                color = _ALGO_COLORS.get(algo, "black")
                y     = y_pos[algo]
                if algo == "PPO" and ppo_off:
                    ax.plot(1.0, y, marker=">", color=color, markersize=8,
                            clip_on=False, transform=trans, zorder=5,
                            linestyle="none")
                    ax.text(1.04, y, f"{iqm:.2f}\n[{lo:.2f},{hi:.2f}]",
                            color=color, fontsize=8, va="center", ha="left",
                            clip_on=False, transform=trans, linespacing=1.3)
                else:
                    ax.errorbar(iqm, y,
                                xerr=[[max(iqm - lo, 0.0)], [max(hi - iqm, 0.0)]],
                                fmt="o", color=color,
                                capsize=3, capthick=1.2, elinewidth=1.4, markersize=7)

            if zero_line and ax.get_xlim()[0] <= 0.0 <= ax.get_xlim()[1]:
                ax.axvline(0.0, linestyle="--", color="black",
                           linewidth=1.0, alpha=0.6, zorder=0)

            ax.set_yticks(range(n_algos))
            ax.set_yticklabels(algos_order if col == 0 else [""] * n_algos,
                               fontsize=10)
            ax.set_ylim(-0.6, n_algos - 0.4)
            ax.tick_params(axis="x", labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col == 0:
                ax.set_ylabel("Train" if row == 0 else "Test", fontsize=12)
            if row == 0:
                ax.set_title(title, fontsize=13, pad=5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved IQM forest figure: {out_path}")
    plt.close(fig)
