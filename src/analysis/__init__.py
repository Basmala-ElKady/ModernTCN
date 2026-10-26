"""Analysis routines: kernel ablation, ERF autograd, failure analysis, and RevIN ablation."""
from src.analysis.ablation import aggregate_kernel_results, count_parameters, run_kernel_ablation
from src.analysis.failure_analysis import compare_window_errors, compute_per_window_mse, run_failure_analysis
from src.analysis.receptive_field import compute_effective_receptive_field, plot_receptive_field, run_erf_analysis
from src.analysis.revin_ablation import aggregate_results, run_ablation

__all__ = [
    "aggregate_kernel_results",
    "count_parameters",
    "run_kernel_ablation",
    "compare_window_errors",
    "compute_per_window_mse",
    "run_failure_analysis",
    "compute_effective_receptive_field",
    "plot_receptive_field",
    "run_erf_analysis",
    "aggregate_results",
    "run_ablation",
]
