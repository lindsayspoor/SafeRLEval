"""SafeRLEval — evaluation metrics and visualisations for Safe RL.

Core modules
------------
safety_spectrum     : fetch metrics from wandb, compute IQM / V / D_norm / D+_norm
spectrum_from_cache : recompute metrics from local CSV (no wandb needed)

Plotting
--------
plots.cdf           : aggregate CDF of D_norm
plots.iqm_forest    : IQM + 95% bootstrap CI forest plot
plots.histograms    : per-condition cost/reward histograms
"""
from saferleval.safety_spectrum import (
    compute_iqm,
    compute_cnorm,
    compute_d_plus_norm,
    compute_new_metrics,
    compute_aggregate_new_metrics,
    compute_iqm_bootstrap_ci,
)

__version__ = "0.1.0"
__all__ = [
    "compute_iqm",
    "compute_cnorm",
    "compute_d_plus_norm",
    "compute_new_metrics",
    "compute_aggregate_new_metrics",
    "compute_iqm_bootstrap_ci",
]
