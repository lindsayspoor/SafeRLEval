"""Training entry point for SafeRLEval experiments.

Trains a safe-RL agent and, immediately after training, runs deterministic
evaluation of the final policy."""

import csv
import functools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import yaml


CRAX_DIR = Path(__file__).parent.parent / "crax"
sys.path.insert(0, str(CRAX_DIR))

import wandb
from brax import envs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from configs.training_config import build_base_parser
from run_utils import (
    collect_rollout_metrics,
    collect_deterministic_eval,
    record_episode_video,
    setup_gpu_environment,
    get_algorithm_train_fn,
    filter_kwargs_for_fn,
    custom_progress_fn,
    make_vision_network_factory,
)

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"



def run_det_episodes(policy_fn, env_name: str, safety_bound: float,
                     num_episodes: int, episode_length: int,
                     level: int, env_kwargs: dict, seed: int) -> dict:
    """Roll out deterministic episodes."""

    env = envs.get_environment(env_name, level=level, **env_kwargs)

    def run_one(ep_key):
        state = env.reset(ep_key)

        def step_fn(carry, _):
            state, rng, cum_cost, cum_reward, done = carry
            rng, act_rng = jax.random.split(rng)
            action, _ = policy_fn(state.obs, act_rng)
            next_state = env.step(state, action)
            cost = next_state.metrics.get("cost", jax.numpy.zeros(()))
            reward = next_state.metrics.get("reward",jax.numpy.zeros(()))
            cum_cost = cum_cost + jax.numpy.where(done, 0.0, cost)
            cum_reward = cum_reward + jax.numpy.where(done, 0.0, reward)
            new_done = done | (next_state.done > 0)
            return (next_state, rng, cum_cost, cum_reward, new_done), None

        init = (state, ep_key,
                jax.numpy.zeros(()), jax.numpy.zeros(()),
                jax.numpy.bool_(False))
        (_, _, cum_cost, cum_reward, _), _ = jax.lax.scan(
            step_fn, init, None, length=episode_length)
        return cum_cost, cum_reward

    ep_keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    cum_costs, cum_rewards = jax.jit(jax.vmap(run_one))(ep_keys)

    costs = np.asarray(cum_costs)
    rewards = np.asarray(cum_rewards)
    dev = costs - safety_bound
    viol = costs[costs > safety_bound]
    return {
        "det_test/cost_violation_rate": float(np.mean(costs > safety_bound)),
        "det_test/cost_mean": float(np.mean(costs)),
        "det_test/cost_mean_violating": float(np.mean(viol)) if len(viol) > 0 else 0.0,
        "det_test/cost_signed_dev_mean":float(np.mean(dev)),
        "det_test/cost_signed_dev_std": float(np.std(dev)),
        "det_test/cost_p90": float(np.percentile(costs, 90)),
        "det_test/cost_max": float(np.max(costs)),
        "det_test/reward_mean": float(np.mean(rewards)),
        "det_test/n_episodes":float(costs.size),
        "det_test/episode_costs": costs.tolist(),
    }



class _LocalWriter:
    """Buffer per-step training metrics, write history + summary in CSV format."""

    _STEP_KEYS = [
        "eval/episode_reward",
        "eval/episode_cost",
        "episodic/reward",
        "episodic/cost",
        "episodic/cost_violation_rate",
        "episodic/cost_mean_violating",
        "training/lambda_lagr",
    ]

    def __init__(self, run_name: str, meta: dict, data_dir: Path):
        self.run_name = run_name
        self.meta = meta        # env, alg, bound, seed, level
        self.data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self._steps: list = []      # [{step, metric, value}]

    def log_step(self, num_steps: int, metrics: dict) -> None:
        for key in self._STEP_KEYS:
            val = metrics.get(key)
            if val is not None:
                try:
                    self._steps.append({
                        "step": int(num_steps),
                        "metric": key,
                        "value": float(val),
                    })
                except (TypeError, ValueError):
                    pass

    def save_history(self) -> Path:
        path = self.data_dir / f"{self.run_name}_history.csv"
        if not self._steps:
            return path
        fieldnames = ["step", "metric", "value"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._steps)
        print(f"Training history saved: {path}")
        return path

    def save_summary(self, train_final: dict, det_summary: dict,
                     stoch_ep_summary: dict = None) -> Path:
        """Write a one-row CSV.

        Columns:
          train_*      — training-window averages
          test_*       — stochastic final-policy evaluation
          det_test_*   — deterministic final-policy evaluation
        """
        path = self.data_dir / f"{self.run_name}_summary.csv"

        def _mean(key):
            vals = [r["value"] for r in self._steps if r["metric"] == key]
            return float(np.mean(vals)) if vals else None

        s = stoch_ep_summary or {}
        row = {
            "run_name": self.run_name,
            "env":      self.meta.get("env"),
            "algo":     self.meta.get("algo"),
            "bound":    self.meta.get("bound"),
            "seed":     self.meta.get("seed"),
            "level":    self.meta.get("level"),

            # Training
            "train_reward":              _mean("episodic/reward"),
            "train_cost":                _mean("episodic/cost"),
            "train_violation_rate":      _mean("episodic/cost_violation_rate"),
            "train_cost_mean_violating": _mean("episodic/cost_mean_violating"),

            # Stochastic final-policy test
            "test_reward":               s.get("det_test/reward_mean"),
            "test_cost":                 s.get("det_test/cost_mean"),
            "test_violation_rate":       s.get("det_test/cost_violation_rate"),
            "test_cost_mean_violating":  s.get("det_test/cost_mean_violating"),
            "test_episode_costs":        json.dumps(s.get("det_test/episode_costs", [])),

            # Deterministic final-policy test
            "det_test_reward":               det_summary.get("det_test/reward_mean"),
            "det_test_cost":                 det_summary.get("det_test/cost_mean"),
            "det_test_violation_rate":       det_summary.get("det_test/cost_violation_rate"),
            "det_test_cost_mean_violating":  det_summary.get("det_test/cost_mean_violating"),
            "det_test_episode_costs":        json.dumps(det_summary.get("det_test/episode_costs", [])),
        }
        
        for k, v in train_final.items():
            if k not in row and v is not None:
                row[k] = v

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"Summary saved: {path}")
        return path




def main():

    parser = build_base_parser(description="Train safe RL agents")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file. "
                             "CLI args override values from the file.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging (local CSV only)")
    parser.add_argument("--data_dir", type=str, default=str(_DEFAULT_DATA_DIR),
                        help="Directory for local CSV outputs")
    parser.add_argument("--det_episodes", type=int, default=100,
                        help="Number of deterministic evaluation episodes after training.")


    pre, _ = parser.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**{k: v for k, v in yaml_cfg.items() if not k.startswith("#")})

    config = parser.parse_args()

    if config.no_wandb:
        config.use_wandb = False

    env_name = config.env_name
    alg_name = config.alg
    difficulty= config.difficulty
    use_wandb= config.use_wandb
    data_dir= Path(config.data_dir)

    setup_gpu_environment()

    for seed in config.seeds:
        print(f"\n{'=' * 60}")
        print(f"env={env_name}, alg={alg_name}, bound={config.safety_bound}, seed={seed}")
        print(f"{'=' * 60}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safety_bound_str = f"{config.safety_bound:g}"
        run_name = (f"{env_name}_Level_{difficulty}_{alg_name}"
                           f"_bound{safety_bound_str}_seed{seed}_{timestamp}")

        meta = {
            "env": env_name,
            "algo":alg_name,
            "bound": config.safety_bound,
            "seed":seed,
            "level": difficulty,
        }
        local_writer = _LocalWriter(run_name, meta, data_dir)


        vision_kwargs = None
        if config.vision:
            vision_kwargs = dict(
                camera=config.vision_camera,
                height=config.vision_height,
                width=config.vision_width,
                obs_mode=config.vision_obs_mode,
                frame_stack=config.vision_frame_stack,
            )

        env_kwargs = config.env_kwargs or {}
        if env_name == "safe_velocity":
            env_kwargs["agent"] = config.agent
        env = envs.get_environment(
            env_name, level=difficulty,
            vision=config.vision, vision_kwargs=vision_kwargs, **env_kwargs,
        )
        eval_env = envs.get_environment(
            env_name, level=difficulty,
            vision=config.vision, vision_kwargs=vision_kwargs, **env_kwargs,
        )
        episode_length = env_kwargs.get("episode_length") or getattr(env, "episode_length", None)


        cli_cfg      = vars(config)
        runtime_cfg  = {"seed": seed, "timestamp": timestamp,
                        "episode_length": episode_length}
        cfg = {**cli_cfg, **runtime_cfg}

        if use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=run_name, id=run_name, config=cfg,
                group=config.wandb_group or env_name,
                job_type=alg_name,
                tags=config.wandb_tags,
            )

        if config.store_model:
            ckpt_root = Path(config.model_dir) / run_name
            os.makedirs(ckpt_root, exist_ok=True)
            cfg["save_checkpoint_path"] = ckpt_root

        _wandb_progress = functools.partial(
            custom_progress_fn, use_wandb=use_wandb, verbose=not config.quiet)

        def progress_fn(num_steps, metrics):
            _wandb_progress(num_steps, metrics)
            local_writer.log_step(num_steps, metrics)

        train_fn_base = get_algorithm_train_fn(alg_name)
        train_kwargs  = filter_kwargs_for_fn(train_fn_base, cfg)

        if config.vision:
            state_obs_key = "state" if config.vision_obs_mode == "pixels+state" else ""
            train_kwargs["network_factory"] = make_vision_network_factory(
                alg_name,
                policy_obs_key=state_obs_key,
                value_obs_key=state_obs_key,
            )
            train_kwargs["augment_pixels"] = True

        train_fn = functools.partial(train_fn_base, **train_kwargs)

        print(f"Training {alg_name} on {env_name} …")
        make_inference_fn, params, final_metrics, eval_env = train_fn(
            environment=env,
            eval_env=eval_env,
            progress_fn=progress_fn,
        )
        print("Training finished.")

        # Log final metrics to wandb summary
        final_log: dict = {}
        if final_metrics:
            for k, v in final_metrics.items():
                if v is not None:
                    if np.ndim(v) > 0:
                        v = np.asarray(v).mean()
                    try:
                        final_log[k] = float(v)
                    except (TypeError, ValueError):
                        final_log[k] = v
        if use_wandb and wandb.run is not None and final_log:
            wandb.run.summary.update(final_log)


        # Stochastic rollout evaluation
        if not config.skip_rollout:
            print("\nStochastic rollout evaluation...")
            rollout_metrics = collect_rollout_metrics(
                env_name=env_name,
                make_inference_fn=make_inference_fn,
                params=params,
                num_steps=config.rollout_steps,
                seed=seed,
                save_trajectory=True,
                save_plots=True,
                level=config.difficulty,
                env_kwargs=config.env_kwargs,
                safety_bound=config.safety_bound,
                num_episodes=config.rollout_episodes,
            )
            safety_summary = rollout_metrics.get("safety_summary")
            if use_wandb and wandb.run is not None and safety_summary:
                wandb.run.summary.update(safety_summary)

        # Deterministic evaluation
        print("\nDeterministic evaluation...")
        det_ep_length = episode_length or getattr(eval_env, "episode_length", 2000) or 2000
        det_env_kwargs = dict(env_kwargs)
        if "episode_length" not in det_env_kwargs:
            det_env_kwargs["episode_length"] = det_ep_length

        # Try collect_deterministic_eval from run_utils first
        try:
            det_summary = collect_deterministic_eval(
                env_name=env_name,
                make_inference_fn=make_inference_fn,
                params=params,
                num_episodes=config.det_episodes,
                episode_length=det_ep_length,
                seed=seed,
                level=config.difficulty,
                env_kwargs=config.env_kwargs,
                safety_bound=config.safety_bound,
            )
        except Exception as exc:
            print(f"collect_deterministic_eval raised {exc!r}; falling back to run_det_episodes.")
            policy_fn = make_inference_fn(params, deterministic=True)
            det_summary = run_det_episodes(
                policy_fn=policy_fn,
                env_name=env_name,
                safety_bound=config.safety_bound,
                num_episodes=config.det_episodes,
                episode_length=det_ep_length,
                level=config.difficulty,
                env_kwargs=det_env_kwargs,
                seed=seed,
            )

        # Ensure per-episode cost list is always present (needed for CDF plots).
        if not det_summary.get("det_test/episode_costs"):
            policy_fn = make_inference_fn(params, deterministic=True)
            ep_summary = run_det_episodes(
                policy_fn=policy_fn,
                env_name=env_name,
                safety_bound=config.safety_bound,
                num_episodes=config.det_episodes,
                episode_length=det_ep_length,
                level=config.difficulty,
                env_kwargs=det_env_kwargs,
                seed=seed,
            )
            det_summary["det_test/episode_costs"] = ep_summary["det_test/episode_costs"]

        print("det_test/reward_mean:"
              f"{det_summary.get('det_test/reward_mean', 'n/a'):.4f}")
        print("det_test/cost_mean:"
              f"{det_summary.get('det_test/cost_mean', 'n/a'):.4f}")
        print("det_test/cost_violation_rate:"
              f"{det_summary.get('det_test/cost_violation_rate', 'n/a'):.4f}")

        if use_wandb and wandb.run is not None:
            wandb.run.summary.update(det_summary)

        # Stochastic per-episode evaluation (for stochastic CDF and D+_norm).
        print("\nStochastic per-episode evaluation...")
        stoch_policy_fn = make_inference_fn(params, deterministic=False)
        stoch_ep_summary = run_det_episodes(
            policy_fn=stoch_policy_fn,
            env_name=env_name,
            safety_bound=config.safety_bound,
            num_episodes=config.det_episodes,
            episode_length=det_ep_length,
            level=config.difficulty,
            env_kwargs=det_env_kwargs,
            seed=seed,
        )
        # Log stochastic metrics to wandb under test/ prefix
        if use_wandb and wandb.run is not None:
            wandb.run.summary.update({
                "test/reward_mean":          stoch_ep_summary.get("det_test/reward_mean"),
                "test/cost_mean":            stoch_ep_summary.get("det_test/cost_mean"),
                "test/cost_violation_rate":  stoch_ep_summary.get("det_test/cost_violation_rate"),
                "test/cost_mean_violating":  stoch_ep_summary.get("det_test/cost_mean_violating"),
                "test/episode_costs":        stoch_ep_summary.get("det_test/episode_costs"),
            })

        # Local CSV save (always)
        local_writer.save_history()
        local_writer.save_summary(final_log, det_summary, stoch_ep_summary)


        # Video
        if not config.skip_video:
            video_length = config.video_length or episode_length or \
                getattr(eval_env, "episode_length", None) or \
                getattr(eval_env, "default_episode_length", None)
            video_env = envs.get_environment(env_name, level=difficulty, **env_kwargs)
            record_episode_video(
                env=video_env,
                make_inference_fn=make_inference_fn,
                params=params,
                steps=video_length,
                cameras=config.cameras,
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                frame_stride=config.video_frame_stride,
                out_name=run_name,
                log_to_wandb=use_wandb,
                seed=seed,
                num_episodes=config.num_video_episodes,
            )

        if use_wandb and wandb.run is not None:
            wandb.finish()

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
