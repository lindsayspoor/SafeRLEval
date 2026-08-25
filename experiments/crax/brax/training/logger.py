# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logger for training metrics."""

import collections
import logging
import time

import numpy as np
from jax import numpy as jnp


class MetricsLogger:
    """Logs training metrics at each training step."""

    def __init__(self, buffer_size, steps_between_logging, progress_fn, safety_bound=None):
        self._metrics_buffer = collections.defaultdict(lambda: collections.deque(maxlen=buffer_size))
        self._buffer_size = buffer_size
        self._steps_between_logging = steps_between_logging
        self._num_steps = 0
        self._last_log_steps = 0
        self._log_count = 0
        self._progress_fn = progress_fn
        self._episodic_metrics_updated = set()
        self._sps_last_time = None
        self._sps_last_steps = 0
        # Raw (unaveraged) per-episode cost values since the last log, used to
        # compute distributional safety stats (violation rate, signed deviation
        # from the safety bound, percentiles) rather than just the window mean.
        self._safety_bound = safety_bound
        self._cost_samples = []

    def _record_episode_costs(self, name, metric, dones):
        if self._safety_bound is not None and name == "cost":
            self._cost_samples.extend(np.asarray(metric[dones.astype(bool)]).flatten().tolist())

    def update_env_metrics(self, metrics, dones, env_steps):
        self._num_steps = int((np.uint64(env_steps.hi) << 32) + np.uint64(env_steps.lo))

        # Log metrics of whole episodes that finished at this step
        if jnp.sum(dones) > 0:
            for name, metric in metrics.items():
                self._record_episode_costs(name, metric, dones)
                done_metrics = np.mean(metric[dones.astype(bool)].flatten()).item()
                metric_key = f"episodic/{name}"
                self._metrics_buffer[metric_key].append(done_metrics)
                self._episodic_metrics_updated.add(metric_key)

        self.maybe_log_metrics()

    def update_step_metrics(self, episode_metrics, dones, train_metrics, env_steps):
        """Update both episodic and training metrics in one call, then log once."""
        self._num_steps = int((np.uint64(env_steps.hi) << 32) + np.uint64(env_steps.lo))

        if jnp.sum(dones) > 0:
            for name, metric in episode_metrics.items():
                self._record_episode_costs(name, metric, dones)
                done_metrics = np.mean(metric[dones.astype(bool)].flatten()).item()
                metric_key = f"episodic/{name}"
                self._metrics_buffer[metric_key].append(done_metrics)
                self._episodic_metrics_updated.add(metric_key)

        for name, metric in train_metrics.items():
            arr = np.asarray(metric)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                continue
            value = arr.reshape(-1).mean().item()
            self._metrics_buffer[f"training/{name}"].append(value)

        self.maybe_log_metrics()

    def update_train_metrics(self, metrics, env_steps):
        """Update training metrics buffer and log if needed.

        Args:
          metrics: Dictionary of training metrics (losses, lambda values, etc.)
          env_steps: Current environment step count
        """
        self._num_steps = int((np.uint64(env_steps.hi) << 32) + np.uint64(env_steps.lo))

        # Add metrics to buffer
        for name, metric in metrics.items():
            arr = np.asarray(metric)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                continue
            value = arr.reshape(-1).mean().item()
            self._metrics_buffer[f"training/{name}"].append(value)
        self.maybe_log_metrics()

    def maybe_log_metrics(self, pad=35):
        """Log metrics to console."""
        # Log if enough steps have passed
        if self._steps_between_logging is None:
            return
        if self._num_steps - self._last_log_steps < self._steps_between_logging:
            return
        now = time.time()
        self._last_log_steps = self._num_steps
        self._log_count += 1
        log_string = (
            f"\n{'Steps':>{pad}} Env: {self._num_steps} Log: {self._log_count}\n"
        )
        mean_metrics = {}
        for metric_name in self._metrics_buffer:
            if len(self._metrics_buffer[metric_name]) > 0:
                if not metric_name.startswith("episodic/") or metric_name in self._episodic_metrics_updated:
                    avg = np.mean(self._metrics_buffer[metric_name])
                    mean_metrics[metric_name] = avg
                    log_string += (f"{f'Training {metric_name}:':>{pad}} {avg:.4f}\n")

        # Distributional safety stats over raw per-episode costs since the last
        # log (not just the windowed mean): violation rate, signed deviation
        # from the safety bound (positive = overshoot/unsafe, negative =
        # undershoot/conservative), spread, and tail value.
        if self._safety_bound is not None and len(self._cost_samples) > 0:
            costs = np.asarray(self._cost_samples, dtype=np.float64)
            deviation = costs - self._safety_bound
            safety_stats = {
                "episodic/cost_violation_rate": float(np.mean(costs > self._safety_bound)),
                "episodic/cost_signed_dev_mean": float(np.mean(deviation)),
                "episodic/cost_signed_dev_std": float(np.std(deviation)),
                "episodic/cost_p90": float(np.percentile(costs, 90)),
                "episodic/cost_max": float(np.max(costs)),
                "episodic/cost_n_episodes": float(costs.size),
                "episodic/cost_mean_violating": (
                    float(np.mean(costs[costs > self._safety_bound]))
                    if np.any(costs > self._safety_bound) else float("nan")
                ),
            }
            mean_metrics.update(safety_stats)
            for k, v in safety_stats.items():
                log_string += (f"{f'Training {k}:':>{pad}} {v:.4f}\n")

        # Compute SPS from wall-clock time between consecutive log calls.
        # Skip the first interval (no prior timestamp) to avoid including JIT compile time.
        if self._sps_last_time is not None:
            elapsed = now - self._sps_last_time
            steps_delta = self._num_steps - self._sps_last_steps
            if elapsed > 0:
                sps = steps_delta / elapsed
                mean_metrics['training/sps'] = sps
                log_string += (f"{'Training training/sps:':>{pad}} {sps:.0f}\n")
        self._sps_last_time = now
        self._sps_last_steps = self._num_steps

        # Clear the updated episodic metrics set after logging
        self._episodic_metrics_updated.clear()

        logging.info(log_string)
        self._progress_fn(int(self._num_steps), mean_metrics)

        # Clear buffers after logging to avoid re-averaging the same data
        for key in list(self._metrics_buffer.keys()):
            self._metrics_buffer[key].clear()
        self._cost_samples.clear()
