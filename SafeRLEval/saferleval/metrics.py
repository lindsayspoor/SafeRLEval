#!/usr/bin/env python3
"""Safety metrics for Safe RL evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from saferleval.common import TRANSLATIONS



# Wandb metric keys (used only in wandb mode)
_TRAIN_KEYS = {
    "violation_rate":"episodic/cost_violation_rate",
    "cost_mean_violating": "episodic/cost_mean_violating",
    "reward": "episodic/reward",
    "cost": "episodic/cost",
}

_TEST_KEYS = {
    "violation_rate":"test/cost_violation_rate",
    "cost_mean_violating": "test/cost_mean_violating",
    "reward":"test/reward_mean",
    "cost": "test/cost_mean",
}
_DET_TEST_KEYS = {
    "violation_rate":  "det_test/cost_violation_rate",
    "cost_mean_violating": "det_test/cost_mean_violating",
    "reward": "det_test/reward_mean",
    "cost":"det_test/cost_mean",
}




def compute_iqm(values: List[Optional[float]]) -> Optional[float]:
    vals = sorted(v for v in values if v is not None and not np.isnan(v))
    if not vals:
        return None
    n = len(vals)
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    mid = vals[lo:hi] if lo < hi else vals
    return float(np.mean(mid)) if mid else None


def compute_dnorm(c_bar: Optional[float], bound: float) -> Optional[float]:
    if c_bar is None or abs(bound) < 1e-10:
        return None
    return (c_bar - bound) / bound


def compute_d_plus_norm(c_plus: Optional[float], V: Optional[float],
                         bound: float) -> Optional[float]:
    if V is None:
        return None
    if V <= 0.0:
        return 0.0
    if c_plus is None:
        return None
    return max(0.0, c_plus - bound) / bound


def compute_d_plus_norm_from_episodes(
        episode_costs_per_seed: list, bound: float) -> Optional[float]:
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




def mean_over_seeds(rows: List[dict], field: str) -> Optional[float]:
    vals = [r[field] for r in rows if r.get(field) is not None]
    return float(np.mean(vals)) if vals else None


def std_over_seeds(rows: List[dict], field: str) -> Optional[float]:
    vals = [r[field] for r in rows if r.get(field) is not None]
    return float(np.std(vals)) if len(vals) > 1 else None



def _d_plus_for_split(cell: dict, split: str, bound: float,
                       ep_key: str, means: dict, stds: dict):
    ep = cell.get(ep_key)
    if ep:
        return (compute_d_plus_norm_from_episodes(ep, bound),
                compute_d_plus_norm_std_from_episodes(ep, bound))
    cmv = means.get(f"{split}_cost_mean_violating")
    vr  = means.get(f"{split}_violation_rate")
    cmv_std = stds.get(f"{split}_cost_mean_violating")
    d_plus = compute_d_plus_norm(cmv, vr, bound)
    d_plus_std = (cmv_std / bound if cmv_std is not None and abs(bound) > 1e-10 else None)
    return d_plus, d_plus_std


def compute_new_metrics(per_algo_cells: Dict[str, List[dict]],
                         algos: List[str]) -> Dict[str, List[dict]]:
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

            def _cost_std_norm(split):
                s = gs(split, "cost")
                return s / bound if s is not None and abs(bound) > 1e-10 else None

            d_plus_test, d_plus_test_std = _d_plus_for_split(
                cell, "test", bound, "test_episode_costs_per_seed", means, stds)
            d_plus_det, d_plus_det_std = _d_plus_for_split(
                cell, "det_test", bound, "episode_costs_per_seed", means, stds)
            d_plus_tr_std = gs("train", "cost_mean_violating")
            d_plus_tr_std = (d_plus_tr_std / bound
                             if d_plus_tr_std is not None and abs(bound) > 1e-10 else None)

            algo_rows.append({
                "env": env, "bound": bound,
                # Training
                "R_bar_train":               g("train", "reward"),
                "C_bar_train":               g("train", "cost"),
                "V_train":                   g("train", "violation_rate"),
                "D_norm_train":              compute_dnorm(g("train", "cost"), bound),
                "D_plus_norm_train":         compute_d_plus_norm(
                    g("train", "cost_mean_violating"), g("train", "violation_rate"), bound),
                "R_bar_train_std":           gs("train", "reward"),
                "C_bar_train_std":           gs("train", "cost"),
                "V_train_std":               gs("train", "violation_rate"),
                "D_norm_train_std":          _cost_std_norm("train"),
                "D_plus_norm_train_std":     d_plus_tr_std,
                # Stochastic test
                "R_bar_final_expl":                g("test", "reward"),
                "C_bar_final_expl":                g("test", "cost"),
                "V_final_expl":                    g("test", "violation_rate"),
                "D_norm_final_expl":               compute_dnorm(g("test", "cost"), bound),
                "D_plus_norm_final_expl":          d_plus_test,
                "R_bar_final_expl_std":            gs("test", "reward"),
                "C_bar_final_expl_std":            gs("test", "cost"),
                "V_final_expl_std":                gs("test", "violation_rate"),
                "D_norm_final_expl_std":           _cost_std_norm("test"),
                "D_plus_norm_final_expl_std":      d_plus_test_std,
                # Deterministic test
                "R_bar_final_greedy":            g("det_test", "reward"),
                "C_bar_final_greedy":            g("det_test", "cost"),
                "V_final_greedy":                g("det_test", "violation_rate"),
                "D_norm_final_greedy":           compute_dnorm(g("det_test", "cost"), bound),
                "D_plus_norm_final_greedy":      d_plus_det,
                "R_bar_final_greedy_std":        gs("det_test", "reward"),
                "C_bar_final_greedy_std":        gs("det_test", "cost"),
                "V_final_greedy_std":            gs("det_test", "violation_rate"),
                "D_norm_final_greedy_std":       _cost_std_norm("det_test"),
                "D_plus_norm_final_greedy_std":  d_plus_det_std,
            })
        result[algo] = algo_rows
    return result


def compute_aggregate_new_metrics(new_metrics: Dict[str, List[dict]],
                                   algos: List[str]) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    keys = (
        "R_bar_train",       "R_bar_final_expl",       "R_bar_final_greedy",
        "C_bar_train",       "C_bar_final_expl",       "C_bar_final_greedy",
        "V_train",           "V_final_expl",            "V_final_greedy",
        "D_norm_train",      "D_norm_final_expl",       "D_norm_final_greedy",
        "D_plus_norm_train", "D_plus_norm_final_expl",  "D_plus_norm_final_greedy",
    )
    for algo in algos:
        cells = new_metrics.get(algo, [])
        r: dict = {}
        for k in keys:
            vals = [c[k] for c in cells if c.get(k) is not None and not np.isnan(c[k])]
            r[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            r[f"{k}_iqm"]  = compute_iqm(vals)
        result[algo] = r
    return result



def _per_seed_derived(cache_rows: List[dict]) -> dict:
    from collections import defaultdict
    groups: dict = defaultdict(lambda: defaultdict(list))

    for row in cache_rows:
        algo = row.get("algo")
        env = row.get("env")
        bound = row.get("bound")
        if algo is None or env is None or bound is None:
            continue
        bound = float(bound)
        key = (algo, env, bound)

        def _add(metric_key, raw_val):
            try:
                v = float(raw_val)
                if not np.isnan(v):
                    groups[key][metric_key].append(v)
            except (TypeError, ValueError):
                pass

        _add("R_bar_train",       row.get("train_reward"))
        _add("R_bar_final_expl",  row.get("test_reward"))
        _add("R_bar_final_greedy",row.get("det_test_reward"))
        _add("C_bar_train",       row.get("train_cost"))
        _add("C_bar_final_expl",  row.get("test_cost"))
        _add("C_bar_final_greedy",row.get("det_test_cost"))
        _add("V_train",           row.get("train_violation_rate"))
        _add("V_final_expl",         row.get("test_violation_rate"))
        _add("V_final_greedy",     row.get("det_test_violation_rate"))

        for split, col in [("train",    "train_cost"),
                           ("final_expl",     "test_cost"),
                           ("final_greedy", "det_test_cost")]:
            try:
                v = float(row.get(col))
                if not np.isnan(v) and abs(bound) > 1e-10:
                    groups[key][f"D_norm_{split}"].append((v - bound) / bound)
            except (TypeError, ValueError):
                pass

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

        for ep_col, metric_key, vr_col, cmv_col in [
            ("det_test_episode_costs", "D_plus_norm_final_greedy",
             "det_test_violation_rate", "det_test_cost_mean_violating"),
            ("test_episode_costs",     "D_plus_norm_final_expl",
             "test_violation_rate",    "test_cost_mean_violating"),
        ]:
            raw = row.get(ep_col)
            added = False
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                try:
                    ep   = json.loads(raw) if isinstance(raw, str) else raw
                    arr  = np.array(ep, dtype=float)
                    viol = arr[arr > bound]
                    if len(viol) > 0:
                        groups[key][metric_key].append(
                            (float(np.mean(viol)) - bound) / bound)
                    else:
                        groups[key][metric_key].append(0.0)
                    added = True
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
            if not added:

                vr_val  = row.get(vr_col)
                cmv_val = row.get(cmv_col)
                try:
                    vr_f = float(vr_val)
                    if not np.isnan(vr_f):
                        if vr_f <= 0:
                            groups[key][metric_key].append(0.0)
                        elif cmv_val is not None and abs(bound) > 1e-10:
                            groups[key][metric_key].append(
                                max(0.0, float(cmv_val) - bound) / bound)
                except (TypeError, ValueError):
                    pass

    return dict(groups)


def compute_iqm_bootstrap_ci(cache_rows: List[dict], algos: List[str],
                               n_bootstrap: int = 2000,
                               alpha: float = 0.05) -> Dict[str, dict]:

    per_seed    = _per_seed_derived(cache_rows)
    metric_keys = [
        "R_bar_train",       "R_bar_final_expl",       "R_bar_final_greedy",
        "C_bar_train",       "C_bar_final_expl",       "C_bar_final_greedy",
        "V_train",           "V_final_expl",            "V_final_greedy",
        "D_norm_train",      "D_norm_final_expl",       "D_norm_final_greedy",
        "D_plus_norm_train", "D_plus_norm_final_expl",  "D_plus_norm_final_greedy",
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


_SPLIT_DISPLAY = {"train": "training", "final_expl": "final policy (exploration)", "final_greedy": "final policy (greedy)"}


def _split_of_key(key: str) -> Optional[str]:
    for s in ("final_greedy", "final_expl", "train"):  
        if key.endswith(f"_{s}"):
            return s
    return None


def print_per_cell_new_metrics(new_metrics: Dict[str, List[dict]], algos: List[str],
                                splits: Optional[List[str]] = None) -> None:

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

    all_col_headers = [
        ("bar{R}_tr",  "R_bar_train"),    ("bar{C}_tr",  "C_bar_train"),
        ("V_tr",   "V_train"),         ("Dnorm_tr",   "D_norm_train"),    ("Dnorm+_tr",  "D_plus_norm_train"),
        ("bar{R}_final_expl",  "R_bar_final_expl"),     ("bar{C}_final_expl",  "C_bar_final_expl"),
        ("V_final_expl",   "V_final_expl"),          ("Dnorm_final_expl",   "D_norm_final_expl"),     ("Dnorm+_final_expl",  "D_plus_norm_final_expl"),
        ("bar{R}_final_greedy",  "R_bar_final_greedy"), ("bar{C}_final_greedy",  "C_bar_final_greedy"),
        ("V_final_greedy",   "V_final_greedy"),      ("Dnorm_final_greedy",   "D_norm_final_greedy"), ("Dnorm+_final_greedy",  "D_plus_norm_final_greedy"),
    ]
    col_headers = ([(lbl, key) for lbl, key in all_col_headers if _split_of_key(key) in splits]
                   if splits else all_col_headers)

    split_labels = "  ".join(_SPLIT_DISPLAY.get(s, s) for s in (splits or ["train", "final_expl", "final_greedy"]))
    col_w = 9
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)
    total_w = max(100, row_w + col_w * len(col_headers) + 4)

    print("\n" + "=" * total_w)
    print(f"PER-CONDITION METRICS  [{split_labels}]  (± = std over seeds)")

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
            print(f"  {'±':<{row_w}}", end="")
            for _, key in col_headers:
                s = c.get(f"{key}_std")
                cell = f"{s:.4f}" if (s is not None and not np.isnan(s)) else "n/a"
                print(f"{cell:>{col_w}}", end="")
            print()
    print()


def print_aggregate_new_metrics(agg: Dict[str, dict], algos: List[str],
                                 splits: Optional[List[str]] = None) -> None:

    all_metric_groups = [
        ("R̄_tr",    "R_bar_train_mean",              "R_bar_train_iqm"),
        ("R̄_fe",    "R_bar_final_expl_mean",         "R_bar_final_expl_iqm"),
        ("R̄_fg",    "R_bar_final_greedy_mean",       "R_bar_final_greedy_iqm"),
        ("C̄_tr",    "C_bar_train_mean",              "C_bar_train_iqm"),
        ("C̄_fe",    "C_bar_final_expl_mean",         "C_bar_final_expl_iqm"),
        ("C̄_fg",    "C_bar_final_greedy_mean",       "C_bar_final_greedy_iqm"),
        ("V_tr",     "V_train_mean",                  "V_train_iqm"),
        ("V_fe",     "V_final_expl_mean",             "V_final_expl_iqm"),
        ("V_fg",     "V_final_greedy_mean",           "V_final_greedy_iqm"),
        ("Dn_tr",    "D_norm_train_mean",             "D_norm_train_iqm"),
        ("Dn_fe",    "D_norm_final_expl_mean",        "D_norm_final_expl_iqm"),
        ("Dn_fg",    "D_norm_final_greedy_mean",      "D_norm_final_greedy_iqm"),
        ("Dn+_tr",   "D_plus_norm_train_mean",        "D_plus_norm_train_iqm"),
        ("Dn+_fe",   "D_plus_norm_final_expl_mean",   "D_plus_norm_final_expl_iqm"),
        ("Dn+_fg",   "D_plus_norm_final_greedy_mean", "D_plus_norm_final_greedy_iqm"),
    ]
    metric_groups = (
        [(n, mk, ik) for n, mk, ik in all_metric_groups
         if _split_of_key(mk.removesuffix("_mean").removesuffix("_iqm")) in splits]
        if splits else all_metric_groups
    )

    split_labels = "  ".join(_SPLIT_DISPLAY.get(s, s) for s in (splits or ["train", "final_expl", "final_greedy"]))
    col_w = 10
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)
    header_w = col_w * 2 + 1
    total_w = max(100, row_w + (header_w + 1) * len(metric_groups))

    print("\n" + "=" * total_w)
    print(f"AGGREGATE METRICS  (mean + IQM across all tasks × bounds)  [{split_labels}]")

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


def print_iqm_bootstrap_table(agg: Dict[str, dict], ci: Dict[str, dict],
                               algos: List[str],
                               splits: Optional[List[str]] = None) -> None:

    all_metric_groups = [
        ("R̄_tr",   "R_bar_train"),
        ("R̄_fe",   "R_bar_final_expl"),
        ("R̄_fg",   "R_bar_final_greedy"),
        ("C̄_tr",   "C_bar_train"),
        ("C̄_fe",   "C_bar_final_expl"),
        ("C̄_fg",   "C_bar_final_greedy"),
        ("V_tr",    "V_train"),
        ("V_fe",    "V_final_expl"),
        ("V_fg",    "V_final_greedy"),
        ("Dn_tr",   "D_norm_train"),
        ("Dn_fe",   "D_norm_final_expl"),
        ("Dn_fg",   "D_norm_final_greedy"),
        ("Dn+_tr",  "D_plus_norm_train"),
        ("Dn+_fe",  "D_plus_norm_final_expl"),
        ("Dn+_fg",  "D_plus_norm_final_greedy"),
    ]
    metric_groups = (
        [(n, base) for n, base in all_metric_groups if _split_of_key(base) in splits]
        if splits else all_metric_groups
    )

    split_labels = "  ".join(_SPLIT_DISPLAY.get(s, s) for s in (splits or ["train", "final_expl", "final_greedy"]))
    col_w = 24
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)
    total_w = max(80, row_w + col_w * len(metric_groups))

    print("\n" + "=" * total_w)
    print(f"IQM WITH 95% BOOTSTRAP CI  (B=2000, stratified by condition)  [{split_labels}]")
    print(f"{'Algorithm':<{row_w}}", end="")
    for name, _ in metric_groups:
        print(f"{name:^{col_w}}", end="")
    print()
    print("-" * total_w)

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



def save_data(cache_rows: List[dict],
              new_metrics: Dict[str, List[dict]], agg_metrics: Dict[str, dict],
              args,
              iqm_ci: Optional[Dict[str, dict]] = None) -> None:
    data_dir = Path(args.output_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if cache_rows:
        raw_path = data_dir / "runs.csv"
        pd.DataFrame(cache_rows).to_csv(raw_path, index=False)
        print(f"Saved per-seed run data: {raw_path}")

    if new_metrics:
        per_cell_rows = [{"algo": algo, **c}
                         for algo, cells in new_metrics.items() for c in cells]
        per_cell_path = data_dir / "per_condition.csv"
        pd.DataFrame(per_cell_rows).to_csv(per_cell_path, index=False)
        print(f"Saved per-condition metrics: {per_cell_path}")

    if agg_metrics:
        agg_path = data_dir / "aggregate.csv"
        pd.DataFrame(agg_metrics).T.to_csv(agg_path)
        print(f"Saved aggregate metrics:{agg_path}")

    if iqm_ci and agg_metrics:
        combined_rows = []
        for algo, ci_data in iqm_ci.items():
            row = {"algo": algo}
            if algo in agg_metrics:
                for k, v in agg_metrics[algo].items():
                    if k.endswith("_iqm"):
                        row[k] = v
            row.update(ci_data)
            combined_rows.append(row)
        ci_path = data_dir / "iqm_ci.csv"
        pd.DataFrame(combined_rows).to_csv(ci_path, index=False)
        print(f"Saved IQM bootstrap CIs:{ci_path}")




def _fetch_train_window_avg(run, keys: List[str], samples: int,
                             window_frac: float) -> Dict[str, float]:
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
    cut  = max(0, len(merged) - max(1, int(np.ceil(len(merged) * window_frac))))
    tail = merged.iloc[cut:]
    result = {}
    for key in keys:
        if key not in tail.columns:
            continue
        col = pd.to_numeric(tail[key], errors="coerce").dropna()
        if not col.empty:
            result[key] = float(col.mean())
    return result


def _fetch_train_full_avg(run, keys: List[str], samples: int) -> Dict[str, float]:
    out = {}
    for key in keys:
        h = run.history(keys=["_step", key], samples=samples, x_axis="_step", pandas=True)
        if h is None or h.empty or key not in h.columns:
            continue
        col = pd.to_numeric(h[key], errors="coerce").dropna()
        if not col.empty:
            out[key] = float(col.mean())
    return out


def _fetch_test_summary(run, keys: List[str]) -> Dict[str, float]:
    out = {}
    for key in keys:
        val = run.summary.get(key)
        if val is not None:
            out[key] = float(val)
    return out


_ALL_CELL_COLS = [
    "train_reward",    "train_cost",    "train_violation_rate",    "train_cost_mean_violating",
    "test_reward",     "test_cost",     "test_violation_rate",     "test_cost_mean_violating",
    "det_test_reward", "det_test_cost", "det_test_violation_rate", "det_test_cost_mean_violating",
]


def _collect_seed_rows(api, project: str, env: str, algo: str, bound: float,
                        level: int, seeds, state: str, samples: int,
                        window_frac: float, max_seeds=None,
                        deterministic_test: bool = False) -> List[dict]:
    f = {"config.alg": algo, "config.env_name": env, "config.difficulty": level,
         "config.safety_bound": bound}
    if seeds:
        f["config.seed"] = {"$in": seeds}
    if state != "any":
        f["state"] = state
    runs = api.runs(project, filters=f, order="-created_at", per_page=200)

    window_keys = [_TRAIN_KEYS["violation_rate"], _TRAIN_KEYS["cost_mean_violating"]]
    full_keys   = [_TRAIN_KEYS["reward"], _TRAIN_KEYS["cost"]]
    # Always fetch both stochastic (test/) and deterministic (det_test/) summaries.
    stoch_keys = list(_TEST_KEYS.values())
    det_keys   = list(_DET_TEST_KEYS.values())

    seen_seeds, rows = set(), []
    for run in runs:
        if max_seeds and len(seen_seeds) >= max_seeds:
            break
        seed = run.config.get("seed")
        if seed is None or seed in seen_seeds:
            continue
        seen_seeds.add(seed)

        train_window = _fetch_train_window_avg(run, window_keys, samples, window_frac)
        train_full   = _fetch_train_full_avg(run, full_keys, samples)
        stoch_vals   = _fetch_test_summary(run, stoch_keys)
        det_vals     = _fetch_test_summary(run, det_keys)

        row: dict = {"seed": seed}
        for metric, key in [("violation_rate",      _TRAIN_KEYS["violation_rate"]),
                             ("cost_mean_violating", _TRAIN_KEYS["cost_mean_violating"])]:
            row[f"train_{metric}"] = train_window.get(key)
        for metric, key in [("reward", _TRAIN_KEYS["reward"]),
                             ("cost",   _TRAIN_KEYS["cost"])]:
            row[f"train_{metric}"] = train_full.get(key)
        for metric, key in _TEST_KEYS.items():
            row[f"test_{metric}"] = stoch_vals.get(key)
        for metric, key in _DET_TEST_KEYS.items():
            row[f"det_test_{metric}"] = det_vals.get(key)

        for wandb_key, csv_col in [("det_test/episode_costs", "det_test_episode_costs"),
                                    ("test/episode_costs",     "test_episode_costs")]:
            raw = run.summary.get(wandb_key)
            row[csv_col] = (
                json.dumps(raw) if isinstance(raw, list)
                else str(raw)   if raw is not None
                else None
            )
        rows.append(row)
    return rows


def build_summary(api, args) -> tuple:

    per_algo_cells: Dict[str, List[dict]] = {algo: [] for algo in args.algos}
    cache_rows: List[dict] = []

    for algo in args.algos:
        for env in args.envs:
            for bound in args.safety_bounds:
                rows = _collect_seed_rows(
                    api, args.project, env, algo, bound, args.level,
                    args.seeds, args.state, args.samples, args.train_window_frac,
                    max_seeds=args.max_seeds,
                    deterministic_test=args.deterministic_test,
                )
                if not rows:
                    print(f"No runs for env={env} algo={algo} bound={bound}")
                    continue
                for r in rows:
                    cache_rows.append({"env": env, "algo": algo, "bound": bound, **r})
                cell: dict = {
                    "env": env,
                    "bound":bound,
                    "means":{c: mean_over_seeds(rows, c) for c in _ALL_CELL_COLS},
                    "stds": {c: std_over_seeds(rows, c) for c in _ALL_CELL_COLS},
                }
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

    return per_algo_cells, cache_rows
