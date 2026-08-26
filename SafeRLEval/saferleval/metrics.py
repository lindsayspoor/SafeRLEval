#!/usr/bin/env python3
"""Safety metrics for Safe RL evaluation.

Metrics:
  V: violation rate
  D_norm: mean normalized cost deviation
  D+_norm: normalized violation magnitude = (bar{C} − d) / d  (violating episodes only)
  IQM:interquartile mean over per-condition values
  Bootstrap CI: stratified bootstrap 95% CIs for IQM
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



def compute_new_metrics(per_algo_cells: Dict[str, List[dict]],
                         algos: List[str]) -> Dict[str, List[dict]]:
    """Compute bar{R}, bar{C}, V, D_norm, D+_norm per (algo, task, bound)"""
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
                d_plus_test  = compute_d_plus_norm_from_episodes(ep_per_seed, bound)
                d_plus_test_std = compute_d_plus_norm_std_from_episodes(ep_per_seed, bound)
            else:
                d_plus_test = compute_d_plus_norm(
                    g("test", "cost_mean_violating"), g("test", "violation_rate"), bound)
                cmv_std = gs("test", "cost_mean_violating")
                d_plus_test_std = (cmv_std / bound
                                   if cmv_std is not None and abs(bound) > 1e-10 else None)

            cmv_tr_std  = gs("train", "cost_mean_violating")
            tr_cost_std = gs("train", "cost")
            te_cost_std = gs("test",  "cost")

            algo_rows.append({
                "env": env, "bound": bound,
                "r_bar_train": g("train", "reward"),
                "r_bar_test": g("test",  "reward"),
                "c_bar_train": g("train", "cost"),
                "c_bar_test": g("test",  "cost"),
                "V_train": g("train", "violation_rate"),
                "V_test": g("test",  "violation_rate"),
                "D_norm_train": compute_dnorm(g("train", "cost"), bound),
                "D_norm_test":  compute_dnorm(g("test",  "cost"), bound),
                "D_plus_norm_train": compute_d_plus_norm(
                    g("train", "cost_mean_violating"), g("train", "violation_rate"), bound),
                "D_plus_norm_test": d_plus_test,
                "r_bar_train_std": gs("train", "reward"),
                "r_bar_test_std": gs("test",  "reward"),
                "c_bar_train_std": gs("train", "cost"),
                "c_bar_test_std":  gs("test",  "cost"),
                "V_train_std": gs("train", "violation_rate"),
                "V_test_std": gs("test",  "violation_rate"),
                "D_norm_train_std": (tr_cost_std / bound
                                          if tr_cost_std is not None and abs(bound) > 1e-10 else None),
                "D_norm_test_std": (te_cost_std / bound
                                          if te_cost_std is not None and abs(bound) > 1e-10 else None),
                "D_plus_norm_train_std": (cmv_tr_std / bound
                                          if cmv_tr_std is not None and abs(bound) > 1e-10 else None),
                "D_plus_norm_test_std":  d_plus_test_std,
            })
        result[algo] = algo_rows
    return result


def compute_aggregate_new_metrics(new_metrics: Dict[str, List[dict]],
                                   algos: List[str]) -> Dict[str, dict]:
    """Mean and IQM of V, D_norm, D+_norm across all (task, bound) cells per algorithm"""
    result: Dict[str, dict] = {}
    keys = ("r_bar_train", "r_bar_test",
            "V_train", "V_test", "D_norm_train", "D_norm_test",
            "D_plus_norm_train", "D_plus_norm_test")
    for algo in algos:
        cells = new_metrics.get(algo, [])
        r: dict = {}
        for k in keys:
            vals = [c[k] for c in cells if c.get(k) is not None and not np.isnan(c[k])]
            r[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            r[f"{k}_iqm"] = compute_iqm(vals)
        result[algo] = r
    return result



def _per_seed_derived(cache_rows: List[dict]) -> dict:
    """Per-seed derived metrics grouped by (algo, task, bound) for stratified bootstrap"""
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

        _add("r_bar_train", row.get("train_reward"))
        _add("r_bar_test", row.get("test_reward"))
        _add("V_train", row.get("train_violation_rate"))
        _add("V_test", row.get("test_violation_rate"))

        for split, col in [("train", "train_cost"), ("test", "test_cost")]:
            try:
                v = float(row.get(col))
                if not np.isnan(v) and abs(bound) > 1e-10:
                    groups[key][f"D_norm_{split}"].append((v - bound) / bound)
            except (TypeError, ValueError):
                pass

        cmv_tr = row.get("train_cost_mean_violating")
        vr_tr= row.get("train_violation_rate")
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

        raw = row.get("det_test_episode_costs")
        if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
            try:
                ep = json.loads(raw) if isinstance(raw, str) else raw
                arr= np.array(ep, dtype=float)
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
    """Stratified bootstrap 95% CI for the IQM of each aggregate metric"""
    per_seed    = _per_seed_derived(cache_rows)
    metric_keys = [
        "r_bar_train", "r_bar_test",
        "V_train", "V_test",
        "D_norm_train","D_norm_test",
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


def print_per_cell_new_metrics(new_metrics: Dict[str, List[dict]], algos: List[str]) -> None:
    """Print per (task, bound) table: bar{R}, bar{C}, V, D_norm, D+_norm for each algorithm"""
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
        ("bar{R}_train", "r_bar_train"),("bar{C}_train", "c_bar_train"),
        ("V_train", "V_train"), ("Dnorm_train",   "D_norm_train"), ("D+norm_train", "D_plus_norm_train"),
        ("bar{R}_test",  "r_bar_test"), ("bar{C}_test",  "c_bar_test"),
        ("V_test", "V_test"), ("Dnorm_test",   "D_norm_test"), ("D+norm_test", "D_plus_norm_test"),
    ]
    col_w = 9
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)

    print("\n" + "=" * 100)
    print("PER-CONDITION METRICS")

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


def print_aggregate_new_metrics(agg: Dict[str, dict], algos: List[str]) -> None:
    """Print aggregate mean and IQM of V, D_norm, D+_norm across all tasks x bounds"""
    metric_groups = [
        ("bar{R}_train","r_bar_train_mean","r_bar_train_iqm"),
        ("bar{R}_test", "r_bar_test_mean", "r_bar_test_iqm"),
        ("V_train",  "V_train_mean",  "V_train_iqm"),
        ("V_test", "V_test_mean",  "V_test_iqm"),
        ("D_norm_train", "D_norm_train_mean", "D_norm_train_iqm"),
        ("D_norm_test","D_norm_test_mean",  "D_norm_test_iqm"),
        ("D+_norm_train", "D_plus_norm_train_mean","D_plus_norm_train_iqm"),
        ("D+_norm_test", "D_plus_norm_test_mean", "D_plus_norm_test_iqm"),
    ]
    col_w = 10
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)

    print("\n" + "=" * 100)
    print("AGGREGATE METRICS (mean + IQM across all tasks and bounds)")

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


def print_iqm_bootstrap_table(agg: Dict[str, dict], ci: Dict[str, dict],
                                algos: List[str]) -> None:
    metric_groups = [
        ("bar{R}_train", "r_bar_train"),
        ("bar{R}_test", "r_bar_test"),
        ("V_train", "V_train"),
        ("V_test", "V_test"),
        ("D_train","D_norm_train"),
        ("D_test", "D_norm_test"),
        ("D+_train", "D_plus_norm_train"),
        ("D+_test", "D_plus_norm_test"),
    ]
    col_w = 24
    row_w = max(20, max((len(TRANSLATIONS.get(a, a)) for a in algos), default=10) + 2)

    print("\n" + "=" * (row_w + col_w * len(metric_groups)))
    print("IQM WITH 95% BOOTSTRAP CI")
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
            iqm = r.get(f"{base}_iqm")
            lower= c.get(f"{base}_iqm_ci_lower")
            upper= c.get(f"{base}_iqm_ci_upper")
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
    "train_reward", "train_cost", "train_violation_rate", "train_cost_mean_violating",
    "test_reward", "test_cost", "test_violation_rate", "test_cost_mean_violating",
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
    full_keys= [_TRAIN_KEYS["reward"], _TRAIN_KEYS["cost"]]
    test_map = _DET_TEST_KEYS if deterministic_test else _TEST_KEYS

    seen_seeds, rows = set(), []
    for run in runs:
        if max_seeds and len(seen_seeds) >= max_seeds:
            break
        seed = run.config.get("seed")
        if seed is None or seed in seen_seeds:
            continue
        seen_seeds.add(seed)

        train_window = _fetch_train_window_avg(run, window_keys, samples, window_frac)
        train_full = _fetch_train_full_avg(run, full_keys, samples)
        test_vals = _fetch_test_summary(run, list(test_map.values()))

        row: dict = {"seed": seed}
        for metric, key in [("violation_rate", _TRAIN_KEYS["violation_rate"]),
                             ("cost_mean_violating", _TRAIN_KEYS["cost_mean_violating"])]:
            row[f"train_{metric}"] = train_window.get(key)
        for metric, key in [("reward", _TRAIN_KEYS["reward"]),
                             ("cost",   _TRAIN_KEYS["cost"])]:
            row[f"train_{metric}"] = train_full.get(key)
        for metric, key in test_map.items():
            row[f"test_{metric}"] = test_vals.get(key)

        ep_costs_raw = run.summary.get("det_test/episode_costs")
        row["det_test_episode_costs"] = (
            json.dumps(ep_costs_raw) if isinstance(ep_costs_raw, list)
            else str(ep_costs_raw) if ep_costs_raw is not None
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
