#!/usr/bin/env python3
"""Summarize safety metrics across algorithms, fetched from wandb.

For each (algo, env, safety_bound) cell, reports:
  r̄, c̄  — mean reward and cost
  V       — violation rate
  C_norm  = (c̄ - b) / b
  D+_norm = (c̄⁺ - b) / b  (requires det_test/cost_mean_violating, see eval_deterministic.py)

Also prints mean and IQM of V, C_norm, D+_norm aggregated across all (env, bound) pairs.

Example:
    uv run python scripts/safety_spectrum.py \\
        --project crax \\
        --envs safe_goal_point safe_circle_point safe_push_point \\
        --algos ppo ppo_lag focops p3o ppo_cost \\
        --safety_bounds 10 25 50 \\
        --deterministic_test
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from saferleval.common import TRANSLATIONS

# Safety constraint metrics (used in heatmap and safety profile tables).
METRICS = {
    "violation_rate": ("episodic/cost_violation_rate", "test/cost_violation_rate"),
    "signed_dev":     ("episodic/cost_signed_dev_mean", "test/cost_signed_dev_mean"),
    "softness":       ("episodic/cost_signed_dev_std",  "test/cost_signed_dev_std"),
}
# Performance metrics: fetched and saved to the raw CSV, but kept out of the
# heatmap (which only shows safety constraint metrics).
PERF_METRICS = {
    "reward": ("episodic/reward",   "test/reward_mean"),
    "cost":   ("episodic/cost",     "test/cost_mean"),
}
# These use a full-history mean for the train split (not the last-window average
# used for VR/D/S), because reward and cost track learning progress across all
# of training, not just convergence behaviour in the last window.
FULL_HISTORY_TRAIN_METRICS = set(PERF_METRICS.keys())

BASELINE_COLUMNS = [f"{split}_{name}" for name in METRICS for split in ("train", "test")]
PERF_COLUMNS     = [f"{split}_{name}" for name in PERF_METRICS for split in ("train", "test")]
# Extra metrics: conditional cost mean (for D+_norm). train fetched as window avg,
# test always read from det_test/ regardless of --deterministic_test flag.
EXTRA_WINDOW_METRICS = {
    "cost_mean_violating": ("episodic/cost_mean_violating", "det_test/cost_mean_violating"),
}
EXTRA_COLUMNS = [f"{split}_cost_mean_violating" for split in ("train", "test")]
ALL_COLUMNS   = BASELINE_COLUMNS + PERF_COLUMNS + EXTRA_COLUMNS

# Lower is better for violation_rate/softness; signed_dev is best near 0 (handled separately in the heatmap).
LOWER_IS_BETTER = {f"{split}_{name}": (name != "signed_dev") for name in METRICS for split in ("train", "test")}
# signed_dev and softness are in raw cost units, which aren't comparable across tasks/bounds with
# different cost scales. --normalize divides these by the safety_bound in effect for that data
# point (mean and std both scale exactly linearly under division by a positive constant), turning
# them into "fraction of budget" before computing cross-task/cross-bound std. violation_rate is
# already dimensionless and is left untouched. reward/cost are excluded (different units entirely).
AMPLITUDE_COLUMNS = {f"{split}_{name}" for name in ("signed_dev", "softness") for split in ("train", "test")}


# Safety tiers: each tier requires ALL THREE profile dimensions to clear a threshold
# simultaneously (incidence, magnitude, stability), at test-time. (max violation_rate,
# max signed_dev/bound, max softness/bound), increasingly strict.
# Thresholds match Table 5 in spectrum_from_cache.py.
TIER_THRESHOLDS = [
    (0.50, 0.50, 1.00),   # Tier 1: weakly safe     (V≤0.50, D/d≤0.50, S/d≤1.00)
    (0.20, 0.10, 0.50),   # Tier 2: moderately safe  (V≤0.20, D/d≤0.10, S/d≤0.50)
    (0.00, 0.00, 0.20),   # Tier 3: strictly safe    (V=0,    D/d≤0,    S/d≤0.20)
]
TIER_NAMES = {0: "Unsafe", 1: "Weakly safe", 2: "Moderately safe", 3: "Strictly safe"}


def compute_tier(violation_rate: Optional[float], signed_dev_norm: Optional[float],
                  softness_norm: Optional[float]) -> Optional[int]:
    """Highest tier (0-3) whose (violation_rate, signed_dev/bound, softness/bound) all clear."""
    if violation_rate is None or signed_dev_norm is None or softness_norm is None:
        return None
    tier = 0
    for v_max, d_max, s_max in TIER_THRESHOLDS:
        if violation_rate <= v_max and signed_dev_norm <= d_max and softness_norm <= s_max:
            tier += 1
        else:
            break
    return tier


def compute_level(test_tier: Optional[int], train_tier: Optional[int]) -> Optional[int]:
    """Collapse the 2D (test_tier, train_tier) class into a single ordinal Level 0-6.

    Test-time tier is the gate (most theory only promises asymptotic/deployment safety);
    train-time tier is only a bonus on top, and only once test_tier has reached its ceiling
    (3) -- climbing the train axis before maxing test would otherwise reward an algorithm
    for being safe while learning despite being unsafe once deployed.
    """
    if test_tier is None:
        return None
    if test_tier < 3:
        return test_tier
    return 3 + (train_tier or 0)


def normalize_value(value: Optional[float], bound: Optional[float], already_normalized: bool) -> Optional[float]:
    """Normalize a raw amplitude metric by `bound`, unless it's already normalized
    (e.g. baseline_df was built with --normalize), to avoid dividing twice."""
    if value is None:
        return None
    if already_normalized or not bound:
        return value
    return value / bound


def compute_safety_levels(baseline: Dict[str, Dict[str, Optional[float]]], algos: List[str],
                           reference_bound: float, already_normalized: bool) -> pd.DataFrame:
    """Per-algorithm safety class: (test_tier, train_tier) and the collapsed Level 0-6."""
    rows = []
    for algo in algos:
        b = baseline.get(algo, {})
        test_tier = compute_tier(
            b.get("test_violation_rate"),
            normalize_value(b.get("test_signed_dev"), reference_bound, already_normalized),
            normalize_value(b.get("test_softness"), reference_bound, already_normalized),
        )
        train_tier = compute_tier(
            b.get("train_violation_rate"),
            normalize_value(b.get("train_signed_dev"), reference_bound, already_normalized),
            normalize_value(b.get("train_softness"), reference_bound, already_normalized),
        )
        level = compute_level(test_tier, train_tier)
        cls = "n/a" if test_tier is None else f"T{test_tier}-R{train_tier if train_tier is not None else '?'}"
        rows.append({
            "algo": TRANSLATIONS.get(algo, algo),
            "test_tier": test_tier,
            "test_tier_name": TIER_NAMES.get(test_tier, "n/a"),
            "train_tier": train_tier,
            "train_tier_name": TIER_NAMES.get(train_tier, "n/a"),
            "class": cls,
            "level": level,
        })
    return pd.DataFrame(rows).set_index("algo")


def normalize_rows(rows: List[dict], bound: float, normalize: bool) -> List[dict]:
    if not normalize or not bound:
        return rows
    out = []
    for r in rows:
        r2 = dict(r)
        for c in AMPLITUDE_COLUMNS:
            if r2.get(c) is not None:
                r2[c] = r2[c] / bound
        out.append(r2)
    return out


def build_filters(env: str, algo: str, level: int, bound: float, seeds: Optional[List[int]], state: str) -> dict:
    f = {"config.alg": algo, "config.env_name": env, "config.difficulty": level, "config.safety_bound": bound}
    if seeds:
        f["config.seed"] = {"$in": seeds}
    if state != "any":
        f["state"] = state
    return f


def fetch_train_window_avg(run, keys: List[str], samples: int, window_frac: float) -> Dict[str, float]:
    """Mean of each key over the last `window_frac` fraction of its logged history.

    Fetches each key separately (see scripts/plot_seed_curves.py for why: a
    single multi-key wandb history() call returns nothing if even one key was
    never logged for this run).
    """
    merged = None
    for key in keys:
        h = run.history(keys=["_step", key], samples=samples, x_axis="_step", pandas=True)
        if h is None or h.empty or key not in h.columns:
            continue
        h = h[["_step", key]].dropna()
        if h.empty:
            continue
        merged = h if merged is None else merged.merge(h, on="_step", how="outer")
    if merged is None or merged.empty:
        return {}
    merged = merged.sort_values("_step")
    cut = max(0, len(merged) - max(1, int(np.ceil(len(merged) * window_frac))))
    tail = merged.iloc[cut:]
    result = {}
    for key in keys:
        if key not in tail.columns:
            continue
        col = pd.to_numeric(tail[key], errors="coerce").dropna()
        if col.empty:
            continue
        result[key] = float(col.mean())
    return result


def fetch_train_full_avg(run, keys: List[str], samples: int) -> Dict[str, float]:
    """Mean of each key over the *entire* logged history (not just last window).

    Used for reward and cost, where the full-training average reflects learning
    efficiency rather than convergence behaviour.
    """
    out = {}
    for key in keys:
        h = run.history(keys=["_step", key], samples=samples, x_axis="_step", pandas=True)
        if h is None or h.empty or key not in h.columns:
            continue
        col = pd.to_numeric(h[key], errors="coerce").dropna()
        if col.empty:
            continue
        out[key] = float(col.mean())
    return out


def fetch_train_safety_from_cost_history(run, bound: float, samples: int) -> Dict[str, Optional[float]]:
    """Compute V_tr and cost_mean_violating_tr from the episodic/cost window history.

    Uses the same logic as plot_paper.py's histogram:
      V_tr               = fraction of training windows where mean episodic cost > bound
      cost_mean_violating = mean of those window costs (proxy for violating-episode mean)

    This is independent of episodic/cost_violation_rate (per-episode metric) and
    episodic/cost_mean_violating (not logged in older runs), so it is always available.
    """
    h = run.history(keys=["_step", "episodic/cost"], samples=samples,
                    x_axis="_step", pandas=True)
    if h is None or h.empty or "episodic/cost" not in h.columns:
        return {}
    w = pd.to_numeric(h["episodic/cost"], errors="coerce").dropna().to_numpy()
    if len(w) == 0:
        return {}
    viol = w[w > bound]
    return {
        "train_violation_rate":      float(np.mean(w > bound)),
        "train_cost_mean_violating": float(np.mean(viol)) if len(viol) > 0 else None,
    }


def fetch_test_summary(run, keys: List[str]) -> Dict[str, float]:
    out = {}
    for key in keys:
        val = run.summary.get(key)
        if val is not None:
            out[key] = float(val)
    return out


def collect_seed_rows(api: wandb.Api, project: str, env: str, algo: str, bound: float, level: int,
                       seeds: Optional[List[int]], state: str, samples: int, window_frac: float,
                       max_seeds: Optional[int] = None,
                       deterministic_test: bool = False) -> List[dict]:
    """One row per seed with train_<metric> and test_<metric> values for this (env, algo, bound)."""
    filters = build_filters(env, algo, level, bound, seeds, state)
    runs = api.runs(project, filters=filters, order="-created_at", per_page=200)

    # Safety constraint metrics: window-averaged train, summary test.
    window_train_keys = [k[0] for k in METRICS.values()]
    _orig_test_keys   = [k[1] for k in METRICS.values()]
    if deterministic_test:
        test_keys = [k.replace("test/", "det_test/") for k in _orig_test_keys]
    else:
        test_keys = _orig_test_keys
    _test_key_for = {name: tkey for name, tkey in zip(METRICS.keys(), test_keys)}

    # Performance metrics: full-history train avg, summary test.
    full_train_keys  = [k[0] for k in PERF_METRICS.values()]
    _orig_perf_test  = [k[1] for k in PERF_METRICS.values()]
    if deterministic_test:
        perf_test_keys = [k.replace("test/", "det_test/") for k in _orig_perf_test]
    else:
        perf_test_keys = _orig_perf_test
    _perf_test_key_for = {name: tkey for name, tkey in zip(PERF_METRICS.keys(), perf_test_keys)}

    # Extra window-averaged train metrics and always-det_test test metrics.
    extra_window_train_keys = [v[0] for v in EXTRA_WINDOW_METRICS.values()]
    extra_det_test_keys     = [v[1] for v in EXTRA_WINDOW_METRICS.values()]

    seen_seeds, rows = set(), []
    for run in runs:
        if max_seeds and len(seen_seeds) >= max_seeds:
            break
        seed = run.config.get("seed")
        if seed is None or seed in seen_seeds:
            continue
        seen_seeds.add(seed)

        train_window_vals = fetch_train_window_avg(run, window_train_keys, samples, window_frac)
        train_full_vals   = fetch_train_full_avg(run, full_train_keys, samples)
        test_vals         = fetch_test_summary(run, test_keys + perf_test_keys)
        extra_train_vals  = fetch_train_window_avg(run, extra_window_train_keys, samples, window_frac)
        extra_test_vals   = fetch_test_summary(run, extra_det_test_keys)
        # V_tr and cost_mean_violating_tr from window-mean cost history —
        # matches the hist plot and works even when episodic/cost_mean_violating
        # and episodic/cost_violation_rate are absent or use a different base.
        cost_safety_vals  = fetch_train_safety_from_cost_history(run, bound, samples)

        row = {"seed": seed}
        for name, (train_key, _) in METRICS.items():
            row[f"train_{name}"] = train_window_vals.get(train_key)
            row[f"test_{name}"]  = test_vals.get(_test_key_for[name])
        for name, (train_key, _) in PERF_METRICS.items():
            row[f"train_{name}"] = train_full_vals.get(train_key)
            row[f"test_{name}"]  = test_vals.get(_perf_test_key_for[name])
        for name, (train_key, test_key) in EXTRA_WINDOW_METRICS.items():
            row[f"train_{name}"] = extra_train_vals.get(train_key)
            row[f"test_{name}"]  = extra_test_vals.get(test_key)
        # Override train_violation_rate and train_cost_mean_violating with the
        # window-mean-cost-based versions so they match the hist plot exactly.
        row.update(cost_safety_vals)

        # Per-episode det-test costs (logged as a list by eval_deterministic.py).
        # Stored as a JSON string so pandas can serialise it to CSV; downstream
        # code (plot_paper.py, spectrum_from_cache.py) parses it back to a list.
        ep_costs_raw = run.summary.get("det_test/episode_costs")
        if ep_costs_raw is not None:
            row["det_test_episode_costs"] = (
                json.dumps(ep_costs_raw) if isinstance(ep_costs_raw, list)
                else str(ep_costs_raw)
            )
        else:
            row["det_test_episode_costs"] = None

        rows.append(row)
    return rows


def mean_over_seeds(rows: List[dict], field: str) -> Optional[float]:
    vals = [r[field] for r in rows if r.get(field) is not None]
    return float(np.mean(vals)) if vals else None


def std_over_seeds(rows: List[dict], field: str) -> Optional[float]:
    vals = [r[field] for r in rows if r.get(field) is not None]
    return float(np.std(vals)) if len(vals) > 1 else None


def compute_global_pooled(cells: List[dict]) -> Dict[str, Optional[float]]:
    """Fully pooled global safety metrics for one algorithm over all (task, limit) cells.

    Flat pooling over all (t, d) pairs — valid because the design is a fully balanced
    factorial grid (every task × every limit), so pooling tasks-then-limits or
    limits-then-tasks reduces to the same flat sum:

        Ṽ        = (1/N) Σ_{t,d} V_{t,d}
        D̃⁺       = Σ_{P+} D+_{t,d} / Σ_{P+} d      D+ = max(0, D),  P+ = {(t,d): D+>0}
        S̃        = Σ_{t,d} S_{t,d} / Σ_{t,d} d

    If P+ = ∅ (no cell has mean cost > limit) then D̃⁺ := 0 by convention.

    cells: list of {"bound": float, "means": {col: float}} — one per (env, bound) condition.
    Tries test_* columns first, falls back to train_*.
    """
    def _pick(means: dict, metric: str) -> Optional[float]:
        for prefix in ("test", "train"):
            v = means.get(f"{prefix}_{metric}")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
        return None

    v_vals: List[float] = []
    d_plus_num, d_plus_den = 0.0, 0.0
    s_num, s_den = 0.0, 0.0
    n_violating = 0

    for cell in cells:
        bound = cell.get("bound") or 0.0
        means = cell.get("means", {})
        v = _pick(means, "violation_rate")
        d = _pick(means, "signed_dev")
        s = _pick(means, "softness")

        if v is not None:
            v_vals.append(v)
        if d is not None:
            d_plus = max(0.0, d)
            if d_plus > 0 and bound:
                d_plus_num += d_plus
                d_plus_den += bound
                n_violating += 1
        if s is not None and bound:
            s_num += s
            s_den += bound

    return {
        "V_global":      float(np.mean(v_vals)) if v_vals else None,
        "D_plus_global": (d_plus_num / d_plus_den) if d_plus_den > 0 else 0.0,
        "S_global":      (s_num / s_den)            if s_den > 0       else None,
        "n_cells":       len(cells),
        "n_violating":   n_violating,
    }


def compute_iqm(values: List[Optional[float]]) -> Optional[float]:
    """Interquartile mean: average of the middle 50% of non-None, non-NaN values."""
    vals = sorted(v for v in values if v is not None and not np.isnan(v))
    if not vals:
        return None
    n = len(vals)
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    mid = vals[lo:hi] if lo < hi else vals
    return float(np.mean(mid)) if mid else None


def compute_cnorm(c_bar: Optional[float], bound: float) -> Optional[float]:
    """C_norm = (c̄ - b) / b.

    Positive means average cost is above the bound (unsafe), negative means below (safe).
    Normalising by b makes it scale-free so values are comparable across tasks and bounds.
    """
    if c_bar is None or abs(bound) < 1e-10:
        return None
    return (c_bar - bound) / bound


def compute_d_plus_norm(c_plus: Optional[float], V: Optional[float],
                         bound: float) -> Optional[float]:
    """D+_norm = (1/N⁺·b) · Σᵢ(cᵢ-b)⁺ = (c̄⁺ - b) / b.

    Mean overshoot above the bound among violating episodes only, normalised by b.
    Equivalently: the mean excess (c̄⁺ - b) divided by the bound.

    Returns 0 when V=0 (no violations). Returns None when c̄⁺ is unavailable (old run).
    """
    if V is None:
        return None
    if V <= 0.0:
        return 0.0
    if c_plus is None:
        return None  # det_test/cost_mean_violating not yet logged for this run
    return max(0.0, c_plus - bound) / bound


def compute_d_plus_norm_batch(c_bar: Optional[float], bound: float) -> Optional[float]:
    """Batch-level D+_norm = max(0, c̄ - b) / b.

    Training logs record window-averaged batch means, not individual episode costs.
    We treat each training window as the unit: if the batch mean c̄ exceeds b the
    window is 'violating' and its normalised overshoot is (c̄ - b) / b; otherwise 0.
    This is the positive part of C_norm and is a conservative lower bound on the
    episode-level D+_norm (batches where c̄ < b may still contain violating episodes).
    """
    if c_bar is None or abs(bound) < 1e-10:
        return None
    return max(0.0, c_bar - bound) / bound


def compute_d_plus_norm_from_episodes(
        episode_costs_per_seed: list, bound: float) -> Optional[float]:
    """Per-seed D+_norm averaged over seeds — matches the hist plot computation.

    For each seed: filter violating episodes (cost > bound), compute
    (mean_violating - b) / b, then average over seeds that had ≥1 violation.
    Seeds with no violations contribute 0 to the average implicitly (they are
    excluded from dp_per_seed but the denominator is all seeds, not just
    violating ones — wait, actually to match hist plot we average only over
    seeds that had violations). Returns 0.0 when no seed has violations.
    """
    if not episode_costs_per_seed or abs(bound) < 1e-10:
        return None
    dp_per_seed = []
    for sv in episode_costs_per_seed:
        arr = np.asarray(sv, dtype=float)
        viol = arr[arr > bound]
        if len(viol) > 0:
            dp_per_seed.append((float(np.mean(viol)) - bound) / bound)
    return float(np.mean(dp_per_seed)) if dp_per_seed else 0.0


def compute_d_plus_norm_std_from_episodes(episode_costs_per_seed, bound) -> Optional[float]:
    """Std over seeds of per-seed D+_norm (for det-test episode arrays)."""
    if not episode_costs_per_seed or abs(bound) < 1e-10:
        return None
    dp_per_seed = []
    for sv in episode_costs_per_seed:
        arr = np.asarray(sv, dtype=float)
        viol = arr[arr > bound]
        if len(viol) > 0:
            dp_per_seed.append((float(np.mean(viol)) - bound) / bound)
    if len(dp_per_seed) < 2:
        return None
    return float(np.std(dp_per_seed, ddof=1))


def compute_new_metrics(per_algo_cells: Dict[str, List[dict]],
                         algos: List[str]) -> Dict[str, List[dict]]:
    """Compute r̄, c̄, V, C_norm, D+_norm per (algo, env, bound) cell.

    D+_norm_train uses episodic/cost_mean_violating (mean cost of violating
    episodes per window, logged by the training logger) via compute_d_plus_norm.
    D+_norm_test uses per-episode costs (episode_costs_per_seed) when available
    for the exact formula; falls back to det_test/cost_mean_violating otherwise.
    """
    result: Dict[str, List[dict]] = {}
    for algo in algos:
        algo_rows = []
        for cell in per_algo_cells.get(algo, []):
            env, bound = cell["env"], cell["bound"]
            means = cell.get("means", {})
            stds  = cell.get("stds",  {})

            def g(split, name):
                return means.get(f"{split}_{name}")

            def gs(split, name):
                return stds.get(f"{split}_{name}")

            ep_per_seed = cell.get("episode_costs_per_seed")
            if ep_per_seed:
                d_plus_test     = compute_d_plus_norm_from_episodes(ep_per_seed, bound)
                d_plus_test_std = compute_d_plus_norm_std_from_episodes(ep_per_seed, bound)
            else:
                d_plus_test = compute_d_plus_norm(
                    g("test", "cost_mean_violating"), g("test", "violation_rate"), bound)
                cmv_std = gs("test", "cost_mean_violating")
                d_plus_test_std = cmv_std / bound if (cmv_std is not None and abs(bound) > 1e-10) else None

            cmv_tr_std = gs("train", "cost_mean_violating")
            d_plus_train_std = cmv_tr_std / bound if (cmv_tr_std is not None and abs(bound) > 1e-10) else None

            tr_cost_std = gs("train", "cost")
            te_cost_std = gs("test",  "cost")

            algo_rows.append({
                "env": env, "bound": bound,
                "r_bar_train":           g("train", "reward"),
                "r_bar_test":            g("test",  "reward"),
                "c_bar_train":           g("train", "cost"),
                "c_bar_test":            g("test",  "cost"),
                "V_train":               g("train", "violation_rate"),
                "V_test":                g("test",  "violation_rate"),
                "C_norm_train":          compute_cnorm(g("train", "cost"), bound),
                "C_norm_test":           compute_cnorm(g("test",  "cost"), bound),
                "D_plus_norm_train":     compute_d_plus_norm(
                    g("train", "cost_mean_violating"), g("train", "violation_rate"), bound),
                "D_plus_norm_test":      d_plus_test,
                # std over seeds for each metric
                "r_bar_train_std":       gs("train", "reward"),
                "r_bar_test_std":        gs("test",  "reward"),
                "c_bar_train_std":       gs("train", "cost"),
                "c_bar_test_std":        gs("test",  "cost"),
                "V_train_std":           gs("train", "violation_rate"),
                "V_test_std":            gs("test",  "violation_rate"),
                "C_norm_train_std":      (tr_cost_std / bound if (tr_cost_std is not None and abs(bound) > 1e-10) else None),
                "C_norm_test_std":       (te_cost_std / bound if (te_cost_std is not None and abs(bound) > 1e-10) else None),
                "D_plus_norm_train_std": d_plus_train_std,
                "D_plus_norm_test_std":  d_plus_test_std,
            })
        result[algo] = algo_rows
    return result


def compute_aggregate_new_metrics(new_metrics: Dict[str, List[dict]],
                                   algos: List[str]) -> Dict[str, dict]:
    """Mean and IQM of V, C_norm, D+_norm across all (env, bound) cells per algo."""
    result: Dict[str, dict] = {}
    keys = ("r_bar_train", "r_bar_test",
            "V_train", "V_test", "C_norm_train", "C_norm_test",
            "D_plus_norm_train", "D_plus_norm_test")
    for algo in algos:
        cells = new_metrics.get(algo, [])
        r: dict = {}
        for k in keys:
            vals = [c[k] for c in cells if c.get(k) is not None and not np.isnan(c[k])]
            r[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            r[f"{k}_iqm"]  = compute_iqm(vals)
        result[algo] = r
    return result


def print_per_cell_new_metrics(new_metrics: Dict[str, List[dict]], algos: List[str]) -> None:
    """Print per (env, bound) table: r̄, c̄, V, C_norm, D+_norm for each algo."""
    keys_seen: List = []
    keys_set: set = set()
    for algo in algos:
        for c in new_metrics.get(algo, []):
            k = (c["env"], c["bound"])
            if k not in keys_set:
                keys_seen.append(k)
                keys_set.add(k)
    keys_seen.sort(key=lambda x: (x[0], x[1]))

    lookup: Dict = {}
    for algo in algos:
        for c in new_metrics.get(algo, []):
            lookup[(algo, c["env"], c["bound"])] = c

    col_headers = [
        ("r̄_tr", "r_bar_train"), ("c̄_tr", "c_bar_train"), ("V_tr", "V_train"), ("C_tr", "C_norm_train"), ("D+_tr", "D_plus_norm_train"),
        ("r̄_te", "r_bar_test"),  ("c̄_te", "c_bar_test"),  ("V_te", "V_test"),  ("C_te", "C_norm_test"),  ("D+_te", "D_plus_norm_test"),
    ]
    col_w = 9
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)

    print("\n" + "=" * 100)
    print("PER-CONDITION NORMALIZED METRICS")
    print("  C_norm  = (c̄ - b) / b                         [tr=training window avg, te=det-test]")
    print("  D+_norm_tr = max(0, c̄_tr - b) / b             [batch-level: window mean as proxy]")
    print("  D+_norm_te = (c̄⁺ - b) / b                     [episode-level; n/a if cost_mean_violating not logged]")
    print("=" * 100)

    for env, bound in keys_seen:
        print(f"\n  env={env}, bound={bound:g}")
        print(f"  {'Algorithm':<{row_w}}", end="")
        for lbl, _ in col_headers:
            print(f"{lbl:>{col_w}}", end="")
        print()
        print("  " + "-" * (row_w + col_w * len(col_headers)))
        for algo in algos:
            c = lookup.get((algo, env, bound), {})
            label = TRANSLATIONS.get(algo, algo)
            print(f"  {label:<{row_w}}", end="")
            for _, key in col_headers:
                v = c.get(key)
                cell = f"{v:.4f}" if (v is not None and not np.isnan(v)) else "n/a"
                print(f"{cell:>{col_w}}", end="")
            print()
            # ± std row
            print(f"  {'±':<{row_w}}", end="")
            for _, key in col_headers:
                s = c.get(f"{key}_std")
                cell = f"{s:.4f}" if (s is not None and not np.isnan(s)) else "n/a"
                print(f"{cell:>{col_w}}", end="")
            print()
    print()


def _per_seed_derived(cache_rows: List[dict]) -> dict:
    """Compute derived per-seed metrics (C_norm, D_plus_norm, V, R_bar) from raw cache rows.

    Returns {(algo, env, bound): {metric_key: [per_seed_values]}}.
    Used as input to the stratified bootstrap for IQM confidence intervals.
    """
    from collections import defaultdict

    groups: dict = defaultdict(lambda: defaultdict(list))

    for row in cache_rows:
        algo  = row.get("algo")
        env   = row.get("env")
        bound = row.get("bound")
        if algo is None or env is None or bound is None:
            continue
        bound = float(bound)
        key   = (algo, env, bound)

        def _add(metric_key, raw_val):
            try:
                v = float(raw_val)
                if not np.isnan(v):
                    groups[key][metric_key].append(v)
            except (TypeError, ValueError):
                pass

        _add("r_bar_train", row.get("train_reward"))
        _add("r_bar_test",  row.get("test_reward"))
        _add("V_train",     row.get("train_violation_rate"))
        _add("V_test",      row.get("test_violation_rate"))

        for split, col in [("train", "train_cost"), ("test", "test_cost")]:
            try:
                v = float(row.get(col))
                if not np.isnan(v) and abs(bound) > 1e-10:
                    groups[key][f"C_norm_{split}"].append((v - bound) / bound)
            except (TypeError, ValueError):
                pass

        # D_plus_norm_train from window-mean cost_mean_violating
        cmv_tr = row.get("train_cost_mean_violating")
        vr_tr  = row.get("train_violation_rate")
        try:
            vr_f = float(vr_tr)
            if not np.isnan(vr_f):
                if vr_f <= 0:
                    groups[key]["D_plus_norm_train"].append(0.0)
                elif cmv_tr is not None and abs(bound) > 1e-10:
                    groups[key]["D_plus_norm_train"].append(
                        max(0.0, float(cmv_tr) - bound) / bound)
        except (TypeError, ValueError):
            pass

        # D_plus_norm_test from per-episode det-test costs.
        # Only append when the seed has ≥1 violation, matching compute_d_plus_norm_from_episodes
        # which averages over violating seeds only. This keeps point estimate and bootstrap
        # consistent (seeds with no violations are excluded from both).
        raw = row.get("det_test_episode_costs")
        if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
            try:
                ep   = json.loads(raw) if isinstance(raw, str) else raw
                arr  = np.array(ep, dtype=float)
                viol = arr[arr > bound]
                if len(viol) > 0:
                    groups[key]["D_plus_norm_test"].append(
                        (float(np.mean(viol)) - bound) / bound)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

    return dict(groups)


def compute_iqm_bootstrap_ci(cache_rows: List[dict], algos: List[str],
                               n_bootstrap: int = 2000,
                               alpha: float = 0.05) -> Dict[str, dict]:
    """Stratified bootstrap 95% CI for the IQM of each aggregate metric.

    For each replicate: resample seeds with replacement within each (env, bound)
    condition, recompute per-condition means, then recompute IQM over conditions.
    Returns {algo: {metric_iqm_ci_lower: float, metric_iqm_ci_upper: float}}.
    """
    per_seed    = _per_seed_derived(cache_rows)
    metric_keys = [
        "r_bar_train", "r_bar_test",
        "V_train",     "V_test",
        "C_norm_train","C_norm_test",
        "D_plus_norm_train", "D_plus_norm_test",
    ]
    rng    = np.random.default_rng(42)
    result: Dict[str, dict] = {}

    for algo in algos:
        conditions = [(env, bound)
                      for (a, env, bound) in per_seed.keys() if a == algo]
        if not conditions:
            result[algo] = {}
            continue

        boot_iqms: Dict[str, List[float]] = {k: [] for k in metric_keys}

        for _ in range(n_bootstrap):
            cond_means: Dict[str, List[float]] = {k: [] for k in metric_keys}
            for env, bound in conditions:
                group = per_seed.get((algo, env, bound), {})
                for k in metric_keys:
                    vals = group.get(k, [])
                    if not vals:
                        continue
                    idx  = rng.integers(0, len(vals), size=len(vals))
                    cond_means[k].append(float(np.mean([vals[i] for i in idx])))
            for k in metric_keys:
                iqm = compute_iqm(cond_means[k])
                if iqm is not None:
                    boot_iqms[k].append(iqm)

        algo_ci: dict = {}
        for k in metric_keys:
            if len(boot_iqms[k]) >= 2:
                algo_ci[f"{k}_iqm_ci_lower"] = float(
                    np.percentile(boot_iqms[k], 100 * alpha / 2))
                algo_ci[f"{k}_iqm_ci_upper"] = float(
                    np.percentile(boot_iqms[k], 100 * (1 - alpha / 2)))
        result[algo] = algo_ci

    return result


def print_iqm_bootstrap_table(agg: Dict[str, dict], ci: Dict[str, dict],
                                algos: List[str]) -> None:
    """Print IQM with 95% stratified bootstrap CI for each metric and algorithm."""
    metric_groups = [
        ("r̄_tr",    "r_bar_train"),
        ("r̄_te",    "r_bar_test"),
        ("V_tr",     "V_train"),
        ("V_te",     "V_test"),
        ("C_tr",     "C_norm_train"),
        ("C_te",     "C_norm_test"),
        ("D+_tr",    "D_plus_norm_train"),
        ("D+_te",    "D_plus_norm_test"),
    ]
    col_w = 24
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)

    print("\n" + "=" * (row_w + col_w * len(metric_groups)))
    print("IQM WITH 95% BOOTSTRAP CI  (stratified bootstrap, B=2000, resampling seeds within each condition)")
    print("  Format: IQM [CI_lower, CI_upper]")
    print("=" * (row_w + col_w * len(metric_groups)))
    print(f"{'Algorithm':<{row_w}}", end="")
    for name, _ in metric_groups:
        print(f"{name:^{col_w}}", end="")
    print()
    print("-" * (row_w + col_w * len(metric_groups)))

    for algo in algos:
        r = agg.get(algo, {})
        c = ci.get(algo, {})
        label = TRANSLATIONS.get(algo, algo)
        print(f"{label:<{row_w}}", end="")
        for _, base in metric_groups:
            iqm   = r.get(f"{base}_iqm")
            lower = c.get(f"{base}_iqm_ci_lower")
            upper = c.get(f"{base}_iqm_ci_upper")
            if iqm is not None and lower is not None and upper is not None:
                cell = f"{iqm:.3f} [{lower:.3f},{upper:.3f}]"
            elif iqm is not None:
                cell = f"{iqm:.3f}"
            else:
                cell = "n/a"
            print(f"{cell:^{col_w}}", end="")
        print()
    print()


def print_aggregate_new_metrics(agg: Dict[str, dict], algos: List[str]) -> None:
    """Print aggregate mean + IQM of V, C_norm, D+_norm across all tasks × bounds."""
    metric_groups = [
        ("r̄_train",      "r_bar_train_mean",        "r_bar_train_iqm"),
        ("r̄_test",       "r_bar_test_mean",          "r_bar_test_iqm"),
        ("V_train",       "V_train_mean",             "V_train_iqm"),
        ("V_test",        "V_test_mean",               "V_test_iqm"),
        ("C_norm_train",  "C_norm_train_mean",         "C_norm_train_iqm"),
        ("C_norm_test",   "C_norm_test_mean",          "C_norm_test_iqm"),
        ("D+_norm_train", "D_plus_norm_train_mean",    "D_plus_norm_train_iqm"),
        ("D+_norm_test",  "D_plus_norm_test_mean",     "D_plus_norm_test_iqm"),
    ]
    col_w = 10
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)

    print("\n" + "=" * 100)
    print("AGGREGATE METRICS  (mean + IQM across all tasks × bounds)")
    print("  IQM = mean of middle 50% of per-condition values   [tr=training window, te=det-test]")
    print("  D+_norm_tr = max(0, c̄_tr-b)/b  (batch-level approx)")
    print("  D+_norm_te = (c̄⁺-b)/b          (episode-level; n/a if cost_mean_violating not logged)")
    print("=" * 100)

    header_w = col_w * 2 + 1
    print(f"{'Algorithm':<{row_w}}", end="")
    for name, _, _ in metric_groups:
        print(f"{name:^{header_w}} ", end="")
    print()
    print(f"{'':>{row_w}}", end="")
    for _ in metric_groups:
        print(f"{'mean':>{col_w}} {'IQM':>{col_w}} ", end="")
    print()
    print("-" * (row_w + (header_w + 1) * len(metric_groups)))

    for algo in algos:
        r = agg.get(algo, {})
        label = TRANSLATIONS.get(algo, algo)
        print(f"{label:<{row_w}}", end="")
        for _, mean_key, iqm_key in metric_groups:
            m = r.get(mean_key)
            q = r.get(iqm_key)
            m_s = f"{m:.4f}" if m is not None else "n/a"
            q_s = f"{q:.4f}" if q is not None else "n/a"
            print(f"{m_s:>{col_w}} {q_s:>{col_w}} ", end="")
        print()
    print()


def build_summary(api: wandb.Api, args: argparse.Namespace):
    """Fetch wandb data for the full (env × bound) grid.

    Returns
    -------
    per_algo_cells  : algo -> list of {"env", "bound", "means", "stds"} for ALL (env, bound)
    cache_rows      : flat list of per-seed rows for CSV caching
    """
    per_algo_cells: Dict[str, List[dict]] = {algo: [] for algo in args.algos}
    cache_rows:     List[dict] = []

    for algo in args.algos:
        for env in args.envs:
            for bound in args.safety_bounds:
                rows = collect_seed_rows(
                    api, args.project, env, algo, bound, args.level,
                    args.seeds, args.state, args.samples, args.train_window_frac,
                    max_seeds=args.max_seeds, deterministic_test=args.deterministic_test,
                )
                if rows:
                    for r in rows:
                        cache_rows.append({"env": env, "algo": algo, "bound": bound, **r})
                    cell: dict = {
                        "env":   env,
                        "bound": bound,
                        "means": {c: mean_over_seeds(rows, c) for c in ALL_COLUMNS},
                        "stds":  {c: std_over_seeds(rows, c)  for c in ALL_COLUMNS},
                    }
                    # Build per-seed episode cost arrays for exact D+_norm_test formula.
                    ep_arrs = []
                    for r in rows:
                        raw = r.get("det_test_episode_costs")
                        if raw is None:
                            continue
                        try:
                            costs = json.loads(raw) if isinstance(raw, str) else raw
                            ep_arrs.append(np.array(costs, dtype=float))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                    if ep_arrs:
                        cell["episode_costs_per_seed"] = ep_arrs
                    per_algo_cells[algo].append(cell)
                else:
                    print(f"No runs for env={env} algo={algo} bound={bound}")

    return per_algo_cells, cache_rows


def to_frame(per_algo: Dict[str, Dict[str, Optional[float]]], algos: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({algo: per_algo.get(algo, {}) for algo in algos}).T
    df = df.reindex(columns=BASELINE_COLUMNS)
    df.index = [TRANSLATIONS.get(a, a) for a in df.index]
    return df


# Rename amplitude columns to show /d suffix when --normalize is active.
_NORMALIZED_COL_NAMES = {
    "train_signed_dev": "train_D/d",
    "test_signed_dev":  "test_D/d",
    "train_softness":   "train_S/d",
    "test_softness":    "test_S/d",
}


def _apply_col_names(df: pd.DataFrame, normalize: bool) -> pd.DataFrame:
    if not normalize:
        return df
    return df.rename(columns=_NORMALIZED_COL_NAMES)


def print_tables(baseline_df, cross_task_df, cross_bound_df, args):
    normalize = getattr(args, "normalize", False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
    print("\n" + "=" * 78)
    print(f"BASELINE SAFETY PROFILE  (env={args.reference_env}, safety_bound={args.reference_bound:g})")
    print("=" * 78)
    print(_apply_col_names(baseline_df, normalize).to_string())

    print("\n" + "=" * 78)
    print(f"CROSS-TASK GENERALIZATION  (std across envs={args.envs}, at safety_bound={args.reference_bound:g})")
    print("Lower = more consistent safety behavior across tasks.")
    print("=" * 78)
    print(_apply_col_names(cross_task_df, normalize).to_string())

    print("\n" + "=" * 78)
    print(f"CROSS-COST-LIMIT GENERALIZATION  (std across safety_bounds={args.safety_bounds}, at env={args.reference_env})")
    print("Lower = more consistent safety behavior across cost budgets.")
    print("=" * 78)
    print(_apply_col_names(cross_bound_df, normalize).to_string())
    print()


def print_global_pooled_table(global_pooled: Dict[str, dict], algos: List[str]) -> None:
    """Print the global pooled safety metrics table (all tasks × all limits)."""
    col_w = 12
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)
    print("\n" + "=" * 78)
    print("GLOBAL POOLED SAFETY METRICS  (all tasks × all limits)")
    print("  Ṽ   = (1/N) Σ V_{t,d}               flat mean violation rate")
    print("  D̃⁺  = Σ_{P+} D+_{t,d} / Σ_{P+} d    excess fraction (P+ = cells where mean cost > limit)")
    print("  S̃   = Σ S_{t,d} / Σ d                pooled cost-std relative to limit")
    print("  D̃⁺ = 0 when P+ = ∅ (never exceeds limit on average in any condition)")
    print("=" * 78)
    print(f"{'Algorithm':<{row_w}} {'Ṽ':>{col_w}} {'D̃⁺':>{col_w}} {'S̃':>{col_w}} {'cells (P+/N)':>{col_w + 4}}")
    print("-" * (row_w + col_w * 3 + col_w + 4 + 4))
    for algo in algos:
        r = global_pooled.get(algo, {})
        label = TRANSLATIONS.get(algo, algo)
        v = f"{r['V_global']:.4f}"      if r.get("V_global")      is not None else "n/a"
        d = f"{r['D_plus_global']:.4f}" if r.get("D_plus_global") is not None else "n/a"
        s = f"{r['S_global']:.4f}"      if r.get("S_global")      is not None else "n/a"
        cells = f"{r.get('n_violating','?')}/{r.get('n_cells','?')}"
        print(f"{label:<{row_w}} {v:>{col_w}} {d:>{col_w}} {s:>{col_w}} {cells:>{col_w + 4}}")
    print()


def print_safety_levels(levels_df: pd.DataFrame, args: argparse.Namespace) -> None:
    print("\n" + "=" * 78)
    print(f"SAFETY LEVEL / 2D CLASS  (env={args.reference_env}, safety_bound={args.reference_bound:g})")
    print("class = T<test_tier>-R<train_tier>; level collapses test_tier first, train_tier only")
    print("counts once test_tier=3 (3=strict-at-test ceiling, 4/5/6 add weak/moderate/strict train).")
    print("=" * 78)
    print(levels_df.to_string())
    print()


# Human-readable column labels with LaTeX subscripts for paper figures.
# T = train, R = rollout (test). Normalized variants append /d.
_COL_LABELS = {
    "train_violation_rate": r"$V_T$",
    "test_violation_rate":  r"$V_R$",
    "train_signed_dev":     r"$D_T$",
    "test_signed_dev":      r"$D_R$",
    "train_softness":       r"$S_T$",
    "test_softness":        r"$S_R$",
    "train_D/d":            r"$D_T/d$",
    "test_D/d":             r"$D_R/d$",
    "train_S/d":            r"$S_T/d$",
    "test_S/d":             r"$S_R/d$",
}

_SECTION_TITLES = {
    "baseline":    "Safety profile",
    "cross_task":  "Cross-task generalization",
    "cross_bound": "Cross-bound generalization",
}


def _heatmap_rcparams(args) -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":    "serif",
        "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":      getattr(args, "hm_font_size",   12),
        "axes.titlesize": getattr(args, "hm_title_size",  13),
        "axes.labelsize": getattr(args, "hm_label_size",  12),
        "xtick.labelsize":getattr(args, "hm_tick_size",   11),
        "ytick.labelsize":getattr(args, "hm_tick_size",   11),
        "pdf.fonttype":   42,
        "ps.fonttype":    42,
    })


def _draw_single_heatmap(ax, df: pd.DataFrame, title: str, args,
                          grey_rows: set = None):
    """Plain text table with no background colour.

    Best (lowest) value per column is bold, computed only over non-grey rows.
    Rows in grey_rows (unsafe baselines) are rendered in grey and excluded from
    the bold-best calculation. Returns None — no imshow, no colorbar needed.
    """
    grey_rows = grey_rows or set()
    col_labels = [_COL_LABELS.get(c, c) for c in df.columns]
    raw = df.to_numpy(dtype=float)
    n_rows, n_cols = raw.shape

    # Best (minimum) row per column, excluding grey baseline rows.
    best_row = np.full(n_cols, -1, dtype=int)
    for j in range(n_cols):
        finite = [
            (i, raw[i, j])
            for i in range(n_rows)
            if not np.isnan(raw[i, j]) and df.index[i] not in grey_rows
        ]
        if finite:
            best_row[j] = min(finite, key=lambda x: x[1])[0]

    # Column indices where a thick divider should appear on the right edge.
    # The column order is [train_VR, test_VR, train_D, test_D, train_S, test_S],
    # so dividers after index 1 (V_R | D_T) and after index 3 (D_R | S_T).
    _THICK_DIVIDERS_AFTER = {1, 3}

    # Draw cell borders.
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)   # row 0 at top
    ax.set_facecolor("white")
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="#cccccc", linewidth=0.7, zorder=0,
            ))

    # Thick vertical dividers between metric groups.
    for j in _THICK_DIVIDERS_AFTER:
        if j < n_cols - 1:
            ax.axvline(j + 0.5, color="#555555", linewidth=1.8, zorder=1)

    # Thick horizontal dividers between every algorithm row.
    for i in range(n_rows - 1):
        ax.axhline(i + 0.5, color="#555555", linewidth=1.8, zorder=1)

    annot_size = getattr(args, "hm_annot_size", 9)
    for i in range(n_rows):
        is_grey = df.index[i] in grey_rows
        for j in range(n_cols):
            val  = raw[i, j]
            text = "n/a" if np.isnan(val) else f"{val:.3f}"
            bold = (best_row[j] == i)
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=annot_size,
                    color="#aaaaaa" if is_grey else "black",
                    fontweight="bold" if bold else "normal")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(df.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    return None


def plot_heatmap(baseline_df, cross_task_df, cross_bound_df, args):
    _heatmap_rcparams(args)

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    algo_part  = "_".join(getattr(args, "algos", []) or [])
    ref_tag    = (f"_{args.reference_env}_bound{args.reference_bound:g}"
                  + (f"_{algo_part}" if algo_part else ""))

    n_algos  = len(baseline_df.index)
    fig_w    = getattr(args, "hm_fig_width",  5.0)
    fig_h    = getattr(args, "hm_fig_height", None) or max(2.0, 0.55 * n_algos + 1.8)

    sections = [
        ("baseline",    _SECTION_TITLES["baseline"],    baseline_df),
        ("cross_task",  _SECTION_TITLES["cross_task"],  cross_task_df),
        ("cross_bound", _SECTION_TITLES["cross_bound"], cross_bound_df),
    ]

    # Translate grey_algos names to the same labels used in the DataFrame index.
    grey_algos  = getattr(args, "grey_algos", ["ppo"])
    grey_labels = {TRANSLATIONS.get(a, a) for a in (grey_algos or [])}

    # --- Individual PDFs (one per section) ---
    for key, title, df in sections:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        _draw_single_heatmap(ax, df, title, args, grey_rows=grey_labels)
        fig.tight_layout()
        out_path = out_dir / f"{args.out_name}{ref_tag}_{key}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {out_path}")

    # --- Combined PDF: all three sections stacked vertically ---
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h * 3))
    for ax, (key, title, df) in zip(axes, sections):
        _draw_single_heatmap(ax, df, title, args, grey_rows=grey_labels)
    fig.tight_layout()
    out_path = out_dir / f"{args.out_name}{ref_tag}_combined.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run(args: argparse.Namespace) -> None:
    api = wandb.Api()
    per_algo_cells, cache_rows = build_summary(api, args)
    new_metrics = compute_new_metrics(per_algo_cells, args.algos)
    agg_metrics = compute_aggregate_new_metrics(new_metrics, args.algos)
    print_per_cell_new_metrics(new_metrics, args.algos)
    print_aggregate_new_metrics(agg_metrics, args.algos)
    print("Computing IQM bootstrap CIs (B=2000)...")
    iqm_ci = compute_iqm_bootstrap_ci(cache_rows, args.algos)
    print_iqm_bootstrap_table(agg_metrics, iqm_ci, args.algos)
    save_data(cache_rows, new_metrics, agg_metrics, args, iqm_ci=iqm_ci)


def save_data(cache_rows: List[dict],
              new_metrics: Dict[str, List[dict]], agg_metrics: Dict[str, dict],
              args: argparse.Namespace,
              iqm_ci: Optional[Dict[str, dict]] = None) -> None:
    """Save raw per-seed data and normalized metric tables to CSV."""
    data_dir = Path(args.output_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build a descriptive suffix from the env(s) and bound(s) used in this run.
    env_str   = "_".join(getattr(args, "envs", []) or [])
    bounds    = getattr(args, "safety_bounds", []) or []
    bound_str = "_".join(str(int(b) if b == int(b) else b) for b in bounds)
    suffix    = f"_{env_str}_{bound_str}" if (env_str or bound_str) else ""

    if cache_rows:
        raw_path = data_dir / f"{args.out_name}_raw{suffix}.csv"
        pd.DataFrame(cache_rows).to_csv(raw_path, index=False)
        print(f"Saved per-seed raw data: {raw_path}")

    if new_metrics:
        per_cell_rows = [{"algo": algo, **c}
                         for algo, cells in new_metrics.items() for c in cells]
        per_cell_path = data_dir / f"{args.out_name}_per_cell_normalized{suffix}.csv"
        pd.DataFrame(per_cell_rows).to_csv(per_cell_path, index=False)
        print(f"Saved per-condition normalized metrics: {per_cell_path}")

    if agg_metrics:
        agg_path = data_dir / f"{args.out_name}_aggregate_normalized{suffix}.csv"
        pd.DataFrame(agg_metrics).T.to_csv(agg_path)
        print(f"Saved aggregate normalized metrics: {agg_path}")

    if iqm_ci and agg_metrics:
        # Merge IQM point estimates with bootstrap CIs into one combined CSV.
        combined_rows = []
        for algo, ci_data in iqm_ci.items():
            row = {"algo": algo}
            if algo in agg_metrics:
                for k, v in agg_metrics[algo].items():
                    if k.endswith("_iqm"):
                        row[k] = v
            row.update(ci_data)
            combined_rows.append(row)
        ci_path = data_dir / f"{args.out_name}_iqm_ci{suffix}.csv"
        pd.DataFrame(combined_rows).to_csv(ci_path, index=False)
        print(f"Saved IQM bootstrap CIs: {ci_path}")


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize safety metrics across algorithms, fetched from wandb.")
    p.add_argument("--project",       type=str, required=True, help="WandB project name (e.g. 'crax').")
    p.add_argument("--envs",          type=str, nargs="+", required=True, help="Environments to evaluate.")
    p.add_argument("--algos",         type=str, nargs="+", required=True, help="Algorithms to summarize.")
    p.add_argument("--safety_bounds", type=float, nargs="+", required=True, help="Safety bounds to evaluate.")
    p.add_argument("--level",         type=int, default=1, help="Difficulty level (default: 1).")
    p.add_argument("--seeds",         type=int, nargs="+", default=None, help="Specific seeds to include (default: all found).")
    p.add_argument("--max_seeds",     type=int, default=None, help="Cap number of seeds per condition (default: all found).")
    p.add_argument("--state",         type=str, default="finished", choices=["finished", "running", "any"])
    p.add_argument("--samples",       type=int, default=10000, help="Max history rows to fetch per run from wandb.")
    p.add_argument("--train_window_frac", type=float, default=1.0,
                   help="Fraction of the end of training used for train-time metric averages (default: 0.2).")
    p.add_argument("--deterministic_test", action="store_true",
                   help="Read test metrics from det_test/* (eval_deterministic.py) instead of stochastic test/*.")
    p.add_argument("--output_data_dir", type=str, default="saved_data")
    p.add_argument("--out_name",        type=str, default="safety_spectrum")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    run(args)
