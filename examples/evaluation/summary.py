import torch
import numpy as np
import arviz as az

from BayesianDLL.Evaluation import summary
from BayesianDLL.Samplers._result import SamplingResult


print("=" * 80)
print("TESTING BAYESIANDLL MCMC SUMMARY FUNCTION")
print("=" * 80)

torch.manual_seed(42)
np.random.seed(42)

n_chains = 4
n_samples = 1000

# 1. Create realistic simulated posterior traces with double precision:
intercept_draws = torch.randn(n_chains, n_samples, dtype=torch.float64) * 0.2 + 1.5
slope_draws = torch.randn(n_chains, n_samples, dtype=torch.float64) * 0.1 + 2.5
weights_draws = torch.randn(n_chains, n_samples, 2, dtype=torch.float64) * 0.3 + torch.tensor([0.5, -0.5], dtype=torch.float64)
sigma_draws = torch.abs(torch.randn(n_chains, n_samples, dtype=torch.float64) * 0.2 + 0.8)
r2_draws = 0.85 + 0.02 * torch.randn(n_chains, n_samples, dtype=torch.float64)

trace = {
    "intercept": intercept_draws,
    "slope": slope_draws,
    "weights": weights_draws,
    "sigma": sigma_draws,
}
deterministic_trace = {
    "R2": r2_draws
}

sampling_result = SamplingResult(
    trace=trace,
    divergences=[0, 0, 0, 0],
    acceptance_probabilities=[[0.8], [0.8], [0.8], [0.8]],
    step_sizes=[[0.05], [0.05], [0.05], [0.05]],
    deterministic_trace=deterministic_trace
)

print("\n--- BayesianDLL sampling_result.summary(hdi_prob=0.94) ---")
df_summary = sampling_result.summary(hdi_prob=0.94, round_to=3, include_deterministic=True)
print(df_summary)

print("\n--- ArviZ az.summary() on the exact same draws ---")
az_dict = {
    "intercept": intercept_draws.numpy(),
    "slope": slope_draws.numpy(),
    "weights": weights_draws.numpy(),
    "sigma": sigma_draws.numpy(),
    "R2": r2_draws.numpy()
}
idata = az.convert_to_dataset(az_dict)
az_df = az.summary(idata, ci_prob=0.94, ci_kind="eti")
print(az_df)

print("\n" + "=" * 80)
print("EXACT NUMERICAL COMPARISON CHECK (Unrounded raw values):")
print("=" * 80)
df_summary_raw = sampling_result.summary(hdi_prob=0.94, round_to=None, include_deterministic=True)
az_df_raw = az.summary(idata, ci_prob=0.94, ci_kind="eti", round_to="none")

col_pairs = [
    ("mean", "mean"),
    ("sd", "sd"),
    ("eti_3%", "eti94_lb"),
    ("eti_97%", "eti94_ub"),
    ("ess_bulk", "ess_bulk"),
    ("ess_tail", "ess_tail"),
    ("r_hat", "r_hat"),
    ("mcse_mean", "mcse_mean"),
    ("mcse_sd", "mcse_sd"),
]
for col_bdll, col_az in col_pairs:
    if col_bdll in df_summary_raw.columns and col_az in az_df_raw.columns:
        diff = np.max(np.abs(df_summary_raw[col_bdll].values.astype(float) - az_df_raw[col_az].values.astype(float)))
        print(f"  Max absolute difference on '{col_bdll}': {diff:.10f}")
print("=" * 80)
