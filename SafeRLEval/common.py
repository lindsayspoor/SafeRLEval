"""Shared constants and plotting utilities for SafeRLEval.

Copied from CRAX/results/common.py and extended with SafeRLEval-specific entries.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRANSLATIONS = {
    # Metrics
    "reward": "Reward",
    "cost": "Cost",
    # Algorithms
    "ppo":      "PPO",
    "ppo_cost": "PPOCost",
    "ppo_lag":  "PPO-Lag",
    "ppo_pid":  "PPOPID",
    "ppo_saute":"PPOSaute",
    "p3o":      "P3O",
    "focops":   "FOCOPS",
    # Environments
    "safe_goal_point":   "Safe Goal Point",
    "safe_push_point":   "Safe Push Point",
    "safe_circle_point": "Safe Circle Point",
    "safe_button_point": "Safe Button Point",
    "safe_reacher":      "Safe Reacher",
}

_tab10 = cm.get_cmap("tab10").colors
BASELINES_COLORS: Dict[str, tuple] = {
    "ppo":      _tab10[7],   # grey
    "ppo_cost": _tab10[1],
    "ppo_lag":  _tab10[0],   # blue
    "ppo_pid":  _tab10[3],
    "ppo_saute":_tab10[4],
    "p3o":      _tab10[5],   # purple
    "focops":   _tab10[6],   # orange-red
}

DEFAULT_METRIC_COLS: Dict[str, str] = {
    "reward": "episodic/reward",
    "cost":   "episodic/cost",
}

REWARD_METRIC_MAP: Dict[str, str] = {}
DEFAULT_REWARD_METRIC = "episodic/reward"


def set_mpl_style() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "figure.dpi":      300,
        "font.size":       12.5,
        "axes.titlesize":  16,
        "axes.labelsize":  14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12.5,
        "lines.linewidth": 2.2,
        "axes.linewidth":  1.2,
    })


def nice_grid(n: int, max_cols: int = 3) -> Tuple[int, int]:
    if n <= 0:
        return 1, 1
    best = None
    for cols in range(1, max_cols + 1):
        rows = int(np.ceil(n / cols))
        score = (rows, abs(rows - cols), cols)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    return best[1], best[2]


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return data
    s = np.convolve(data, np.ones(window_size), "same")
    c = np.convolve(np.ones_like(data), np.ones(window_size), "same")
    return s / c


def align_and_stack(dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    if not dfs:
        return np.array([]), np.array([[]])
    T = int(min(len(d) for d in dfs))
    if T <= 0:
        return np.array([]), np.array([[]])
    trimmed = [d.sort_values("_step", kind="mergesort").iloc[:T] for d in dfs]
    steps = trimmed[0]["_step"].to_numpy(copy=True)
    vals  = np.stack([d["value"].to_numpy(copy=True) for d in trimmed], axis=0)
    return steps, vals


def get_series(df: pd.DataFrame, algo: str, metric: str,
               metric_cols: Optional[Dict[str, str]] = None,
               env_name: Optional[str] = None) -> Optional[pd.Series]:
    cols     = metric_cols or DEFAULT_METRIC_COLS
    cost_col = cols.get("cost", "episodic/cost")
    rew_col  = REWARD_METRIC_MAP.get(env_name or "", cols.get("reward", "episodic/reward"))

    if metric == "reward":
        if algo == "ppo_cost":
            if rew_col not in df.columns or cost_col not in df.columns:
                return None
            return df[rew_col].astype(np.float32) + df[cost_col].astype(np.float32)
        col = rew_col if rew_col in df.columns else cols.get("reward")
    elif metric == "cost":
        col = cost_col
    else:
        col = cols.get(metric, metric)

    if col is None or col not in df.columns:
        return None
    return df[col].astype(np.float32)
