# SafeRLEval

Reproducibility code for the paper **"Evaluating Safe Reinforcement Learning: A Safety Spectrum"**.

---

## Repository layout

```
SafeRLEval/
├── SafeRLEval/              ← installable evaluation library
│   ├── safety_spectrum.py   ← fetch metrics from wandb (IQM, V, D_norm, D+_norm)
│   ├── spectrum_from_cache.py ← recompute metrics from local CSV (no wandb needed)
│   ├── common.py            ← shared constants, colours, plotting helpers
│   └── plots/
│       ├── cdf.py           ← aggregate CDF of D_norm
│       ├── iqm_forest.py    ← IQM + 95% CI forest plot
│       └── histograms.py    ← per-condition cost/reward histograms
│
└── experiments/
    ├── crax/                ← modified CRAX safe-RL library (editable install)
    ├── train/
    │   ├── train.py         ← training + deterministic evaluation in one script
    │   └── configs/         ← one YAML per environment (paper hyperparameters)
    ├── data/
    │   ├── raw/             ← per-run CSVs written by train.py
    │   └── aggregate/       ← aggregate CSVs produced by SafeRLEval/safety_spectrum.py
    └── plots/
        └── plot_training_curves.py  ← training curve plotter (reads wandb or local CSVs)
```

---

## Installation

Requires Python 3.10.  We use [uv](https://github.com/astral-sh/uv) for environment management.

```bash
git clone https://github.com/lindsayspoor/SafeRLEval.git
cd SafeRLEval
uv sync          # creates .venv and installs all dependencies
```

This installs:
- `experiments/crax/` as the editable `brax` package (the modified CRAX library).
- `SafeRLEval/` as the editable `saferleval` package.

---

## Reproducing experiments

### 1. Train

Each run trains one seed of one algorithm on one environment with one cost limit.

```bash
uv run python experiments/train/train.py \
    --env_name safe_goal_point \
    --alg ppo_lag \
    --safety_bound 25 \
    --seeds 0 1 2 3 4 \
    --num_timesteps 5e8 \
    --env_kwargs '{"episode_length": 2000}' \
    --lagrangian_coef_rate 3.0
```

Results are saved to **wandb** (if `--use_wandb` is set, the default) **and** to local CSVs in `experiments/data/raw/`:
- `{run_name}_history.csv` — per-step training metrics (for training curve plots).
- `{run_name}_summary.csv` — one-row summary compatible with `spectrum_from_cache.py`.

To disable wandb and use local storage only:

```bash
uv run python experiments/train/train.py --no_wandb ...
```

Paper sweep: 4 environments × 4 algorithms × 3 cost limits × 30 seeds.  
See `experiments/train/configs/` for the exact hyperparameters used per environment.

### 2. Evaluate (aggregate metrics)

**From wandb** (requires wandb access):

```bash
uv run python SafeRLEval/safety_spectrum.py \
    --project <wandb-project> \
    --envs safe_goal_point safe_circle_point safe_push_point safe_button_point \
    --algos ppo ppo_lag focops p3o \
    --safety_bounds 15 25 50 \
    --deterministic_test \
    --output_data_dir experiments/data/aggregate
```

**From local CSVs** (no wandb needed):

```bash
# First combine per-run summaries into one raw CSV:
cat experiments/data/raw/*_summary.csv > experiments/data/aggregate/safety_spectrum_raw.csv

# Then recompute metrics:
uv run python SafeRLEval/spectrum_from_cache.py \
    experiments/data/aggregate/safety_spectrum_raw.csv \
    --algos ppo ppo_lag focops p3o
```

### 3. Plot

**Training curves** (from wandb):

```bash
uv run python experiments/plots/plot_training_curves.py \
    --project <wandb-project> \
    --envs safe_goal_point \
    --algos ppo_lag \
    --metrics reward cost
```

**IQM forest plot**:

```python
from saferleval.plots import plot_iqm_figure
plot_iqm_figure("figures/iqm_forest.pdf",
                iqm_ci_csv="experiments/data/aggregate/safety_spectrum_iqm_ci.csv")
```

**Aggregate CDF**:

```python
from saferleval.plots import plot_agg_cdf_figure
plot_agg_cdf_figure(
    raw_csv_path="experiments/data/aggregate/safety_spectrum_raw.csv",
    out_path="figures/agg_cdf.pdf",
    algos=["ppo", "ppo_lag", "focops", "p3o"],
    x_min=-1.0, x_max=3.0,
)
```

**Per-condition histograms**:

```python
from saferleval.plots import plot_condition_histograms
plot_condition_histograms(
    raw_csv_path="experiments/data/aggregate/safety_spectrum_raw.csv",
    out_path="figures/hist_test_cost.pdf",
    metric="test_cost",
)
```

---

## Metrics

| Symbol | Definition |
|--------|-----------|
| $\bar{R}$ | Mean episode reward |
| $\bar{C}$ | Mean episode cost |
| $V$ | Violation rate: fraction of episodes where $c > d$ |
| $D_{\text{norm}} = (\bar{c}-d)/d$ | Normalised cost deviation; $< 0$ = safe |
| $D^+_{\text{norm}} = (\bar{c}^+ - d)/d$ | Mean overshoot among violating episodes only |
| IQM | Interquartile mean (middle 50%) over conditions |

---

## Citation

```bibtex
@article{spoor2025saferleval,
  title  = {Evaluating Safe Reinforcement Learning: A Safety Spectrum},
  author = {Spoor, Lindsay J.},
  year   = {2025},
}
```
