# SafeRLEval

Reproducibility code for the paper "Evaluation Metrics for Safe Reinforcement Learning" (under review).

Safe reinforcement learning algorithms are commonly evaluated by reporting the cost on average, alongside the average reward. We argue that average cost alone is insufficient to capture safety and reliability of an algorithm: it does not indicate whether a policy ever violates the safety constraint, how severely it does so when it does, and whether behaviour differs between training and test time.

This repository accompanies the paper and provides:

- **New evaluation metrics**: the violation rate $V$, the mean normalised cost deviation $D_{\text{norm}}$, and the normalised violation magnitude $D^+_{\text{norm}}$ among violating episodes, each capturing a distinct aspect of safety that average cost misses.
- **A safety tier system**: (Tiers 0–4) that categorises algorithms based on thresholds of $D_{\text{norm}}$, $V$, and $D^+_{\text{norm}}$.
- **Aggregate CDF visualisation** to show the full distribution.
- **Per-condition histograms** of episodic cost distributions to reveal task- and bound-specific behavior hidden by aggregate metrics.

The `SafeRLEval/` Python package can be installed standalone (`pip install -e SafeRLEval/`) and used independently of the `experiments/` folder used for the purpose of reproducing our results.

---


## Installation

Requires Python 3.10.  We use [uv](https://github.com/astral-sh/uv) for environment management.

```bash
git clone https://github.com/lindsayspoor/SafeRLEval.git
cd SafeRLEval
uv sync          # creates .venv and installs all dependencies
```

This installs:
- `experiments/crax/` as the editable `brax` package (the modified CRAX library, original library from https://github.com/TTomilin/CRAX/tree/main).
- `SafeRLEval/` as the editable `saferleval` package.

---

## Quick test

Run the full pipeline in a few minutes using the quicktest config:

```bash
# make sure older runs in this folder don't mix in with your evaluation
rm -rf experiments/data/raw/ experiments/data/aggregate/

# 1. Train
uv run python experiments/train/train.py \
    --config experiments/train/configs/quicktest.yaml \
    --alg ppo_lag --safety_bound 25 --seeds 0 1 --no_wandb

# 2. Compute metrics
uv run python SafeRLEval/evaluate.py experiments/data/raw/

# 3. Run stochastic evaluation (per-episode costs for final policy with exploration noise)
uv run python experiments/eval_stoch.py \
    --runs_csv experiments/data/aggregate/runs.csv \
    --models_dir models/models \
    --envs safe_goal_point --algos ppo_lag --bounds 25

# 4. Plot training curves
uv run python experiments/plots/plot_training_curves.py \
    --data_dir experiments/data/raw/ \
    --envs safe_goal_point --algos ppo_lag --metrics reward cost

# 5. Plot aggregate CDF
uv run python SafeRLEval/saferleval/plotting/cdf.py \
    experiments/data/aggregate/runs.csv \
    --out figures/quicktest_cdf.pdf

# 6. Plot per-condition histograms
uv run python SafeRLEval/saferleval/plotting/hist.py \
    experiments/data/aggregate/runs.csv \
    --env safe_goal_point --bounds 25 --algos ppo_lag \
    --out figures/quicktest_hist.pdf
```


---

## Usage

### 1. Train

Each run trains one seed of one algorithm on one environment with one safety bound.

Per-environment configs with the paper hyperparameters are in `experiments/train/configs/`.
Pass one with `--config` and override individual values on the command line as needed:

```bash
uv run python experiments/train/train.py \
    --config experiments/train/configs/safe_goal_point.yaml \
    --alg ppo_lag \
    --safety_bound 25 \
    --seeds 0 1 2 3 4
```

Or specify all hyperparameters explicitly:

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

Results are saved to wandb(default) and to local CSVs in `experiments/data/raw/`.
To disable wandb and use local storage only:

```bash
uv run python experiments/train/train.py \
    --config experiments/train/configs/safe_goal_point.yaml \
    --alg ppo_lag --safety_bound 25 --seeds 0 1 2 3 4 \
    --no_wandb
```

### 2. Evaluate (aggregate metrics)

Metrics are computed with `SafeRLEval/evaluate.py`.

#### Option A: local CSVs (no wandb required)

After training, pass the directory of per-run summary files written by `train.py`:

```bash
uv run python SafeRLEval/evaluate.py experiments/data/raw/
```

All `*_summary.csv` files in that directory are combined automatically.
You can optionally filter by algorithm, environment, or safety bound:

```bash
uv run python SafeRLEval/evaluate.py experiments/data/raw/ \
    --algos ppo ppo_lag focops p3o \
    --envs safe_goal_point safe_circle_point \
    --safety_bounds 15 25 50
```

#### Option B: wandb

If you logged runs to wandb, pass `--project` instead of a source path.
`--envs`, `--algos`, and `--safety_bounds` are required in this mode:

```bash
uv run python SafeRLEval/evaluate.py \
    --project <wandb-project> \
    --envs safe_goal_point safe_circle_point safe_push_point safe_button_point \
    --algos ppo ppo_lag focops p3o \
    --safety_bounds 15 25 50 \
    --deterministic_test
```

Both modes print per-condition and aggregate metric tables to the terminal output and write
output CSVs to `experiments/data/aggregate/` (override with `--output_data_dir`).

### 3. Stochastic evaluation

`evaluate.py` populates the greedy (deterministic) test metrics. To also populate
per-episode costs for the final policy **with exploration noise**, run `eval_stoch.py`:

```bash
uv run python experiments/eval_stoch.py \
    --runs_csv experiments/data/aggregate/runs.csv \
    --models_dir models/models \
    --envs safe_goal_point safe_circle_point safe_push_point safe_button_point \
    --algos ppo ppo_lag focops p3o \
    --bounds 15 25 50
```

This writes `test_episode_costs` back into `runs.csv` and is required before plotting
the exploration-noise panels of the CDF and per-condition histograms.

### 4. Plot

**Training curves (local)**:

```bash
uv run python experiments/plots/plot_training_curves.py \
    --data_dir experiments/data/raw/ \
    --envs safe_goal_point \
    --algos ppo ppo_lag focops p3o \
    --metrics reward cost
```

**Training curves (wandb)**:

```bash
uv run python experiments/plots/plot_training_curves.py \
    --project <wandb-project> \
    --envs safe_goal_point \
    --algos ppo ppo_lag focops p3o \
    --metrics reward cost
```

**Aggregate CDF**:

```bash
uv run python SafeRLEval/saferleval/plotting/cdf.py \
    experiments/data/aggregate/runs.csv \
    --out figures/agg_cdf.pdf \
    --algos ppo ppo_lag focops p3o \
    --x_min -1.0 --x_max 3.0
```

**Individual task and safety bounds histograms**:

```bash
uv run python SafeRLEval/saferleval/plotting/hist.py \
    experiments/data/aggregate/runs.csv \
    --env safe_goal_point --bounds 15 25 50 \
    --algos ppo ppo_lag focops p3o \
    --out figures/hist_goal.pdf
```

---


## Citation

The paper and repo are currently under review.
