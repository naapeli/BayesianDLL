import torch
import numpy as np
import arviz as az

from BayesianDLL.Evaluation import gelman_rubin, effective_sample_size


print("=" * 75)
print("MCMC CONVERGENCE DIAGNOSTICS: GELMAN-RUBIN (R-hat) & EFFECTIVE SAMPLE SIZE (ESS)")
print("=" * 75)

torch.manual_seed(42)
np.random.seed(42)

# ===========================================================================
# Test Case 1: Independent Converged Draws (4 chains, 1000 draws ~ N(0, 1))
# ===========================================================================
print("\n" + "=" * 75)
print("--- Test Case 1: Independent Converged Draws (4 chains x 1000 samples) ---")
print("=" * 75)
n_chains, n_samples = 4, 1000
converged_chains = torch.randn(n_chains, n_samples, dtype=torch.float64)
sample_dict = {"theta": converged_chains}

# 1. Gelman-Rubin R-hat
print("\n[R-hat Convergence Diagnostic]")
for method in ["classical", "split", "rank"]:
    rhat = gelman_rubin(sample_dict, method=method)["theta"].item()
    print(f"  BayesianDLL ({method:9s}): R-hat = {rhat:.4f}")

az_data = az.convert_to_dataset(converged_chains.numpy())
rhat_rank = float(az.rhat(az_data, method="rank")["x"].values)
rhat_split = float(az.rhat(az_data, method="split")["x"].values)
rhat_classical = float(az.rhat(az_data, method="identity")["x"].values)
print(f"  ArviZ       (classical): R-hat = {rhat_classical:.4f}")
print(f"  ArviZ       (split)    : R-hat = {rhat_split:.4f}")
print(f"  ArviZ       (rank)     : R-hat = {rhat_rank:.4f}")

# 2. Effective Sample Size (ESS)
print("\n[Effective Sample Size (ESS)] (Theoretical ~ 4000 for independent draws)")
ess_mean = effective_sample_size(sample_dict, method="mean")["theta"].item()
ess_bulk = effective_sample_size(sample_dict, method="bulk")["theta"].item()
print(f"  BayesianDLL (mean ESS) : {ess_mean:.1f}")
print(f"  BayesianDLL (bulk ESS) : {ess_bulk:.1f}")

az_ess_mean = float(az.ess(az_data, method="mean")["x"].values)
az_ess_bulk = float(az.ess(az_data, method="bulk")["x"].values)
print(f"  ArviZ       (mean ESS) : {az_ess_mean:.1f}")
print(f"  ArviZ       (bulk ESS) : {az_ess_bulk:.1f}")

# ===========================================================================
# Test Case 2: Autocorrelated AR(1) Process (phi = 0.7)
# ===========================================================================
print("\n" + "=" * 75)
print("--- Test Case 2: Autocorrelated AR(1) Markov Chains (phi = 0.7) ---")
print("=" * 75)
# Generate AR(1) chains: x_t = 0.7 * x_{t-1} + eps_t
phi = 0.7
ar_chains = np.zeros((n_chains, n_samples))
for c in range(n_chains):
    x = 0.0
    for t in range(n_samples):
        x = phi * x + np.random.normal(0, np.sqrt(1 - phi ** 2))
        ar_chains[c, t] = x

ar_tensor = torch.tensor(ar_chains, dtype=torch.float64)
sample_dict_ar = {"theta": ar_tensor}

# Theoretical ESS for AR(1): N_total * (1 - phi) / (1 + phi)
theoretical_ess = (n_chains * n_samples) * (1 - phi) / (1 + phi)

print("\n[R-hat Convergence Diagnostic] (Should be ~ 1.00 since chains have same target)")
for method in ["classical", "split", "rank"]:
    rhat = gelman_rubin(sample_dict_ar, method=method)["theta"].item()
    print(f"  BayesianDLL ({method:9s}): R-hat = {rhat:.4f}")

az_data_ar = az.convert_to_dataset(ar_chains)
rhat_rank = float(az.rhat(az_data_ar, method="rank")["x"].values)
rhat_split = float(az.rhat(az_data_ar, method="split")["x"].values)
rhat_classical = float(az.rhat(az_data_ar, method="identity")["x"].values)
print(f"  ArviZ       (classical): R-hat = {rhat_classical:.4f}")
print(f"  ArviZ       (split)    : R-hat = {rhat_split:.4f}")
print(f"  ArviZ       (rank)     : R-hat = {rhat_rank:.4f}")

print(f"\n[Effective Sample Size (ESS)] (Theoretical ~ {theoretical_ess:.1f})")
ess_ar_mean = effective_sample_size(sample_dict_ar, method="mean")["theta"].item()
ess_ar_bulk = effective_sample_size(sample_dict_ar, method="bulk")["theta"].item()
print(f"  BayesianDLL (mean ESS) : {ess_ar_mean:.1f}")
print(f"  BayesianDLL (bulk ESS) : {ess_ar_bulk:.1f}")

az_ar_mean = float(az.ess(az_data_ar, method="mean")["x"].values)
az_ar_bulk = float(az.ess(az_data_ar, method="bulk")["x"].values)
print(f"  ArviZ       (mean ESS) : {az_ar_mean:.1f}")
print(f"  ArviZ       (bulk ESS) : {az_ar_bulk:.1f}")

# ===========================================================================
# Test Case 3: Non-Converged / Drifting Chains (Distinct chain means)
# ===========================================================================
print("\n" + "=" * 75)
print("--- Test Case 3: Non-Converged Chains (Means separated: 0, 1, 2, 3) ---")
print("=" * 75)
drifting_chains = torch.randn(n_chains, n_samples, dtype=torch.float64) + torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64).unsqueeze(1)
sample_dict_drift = {"theta": drifting_chains}

print("\n[R-hat Convergence Diagnostic] (Should be > 1.2 indicating poor convergence)")
for method in ["classical", "split", "rank"]:
    rhat = gelman_rubin(sample_dict_drift, method=method)["theta"].item()
    print(f"  BayesianDLL ({method:9s}): R-hat = {rhat:.4f}")

az_data_drift = az.convert_to_dataset(drifting_chains.numpy())
rhat_rank = float(az.rhat(az_data_drift, method="rank")["x"].values)
rhat_split = float(az.rhat(az_data_drift, method="split")["x"].values)
rhat_classical = float(az.rhat(az_data_drift, method="identity")["x"].values)
print(f"  ArviZ       (classical): R-hat = {rhat_classical:.4f}")
print(f"  ArviZ       (split)    : R-hat = {rhat_split:.4f}")
print(f"  ArviZ       (rank)     : R-hat = {rhat_rank:.4f}")

print("\n[Effective Sample Size (ESS)]")
ess_drift_mean = effective_sample_size(sample_dict_drift, method="mean")["theta"].item()
ess_drift_bulk = effective_sample_size(sample_dict_drift, method="bulk")["theta"].item()
print(f"  BayesianDLL (mean ESS) : {ess_drift_mean:.1f}")
print(f"  BayesianDLL (bulk ESS) : {ess_drift_bulk:.1f}")

az_drift_mean = float(az.ess(az_data_drift, method="mean")["x"].values)
az_drift_bulk = float(az.ess(az_data_drift, method="bulk")["x"].values)
print(f"  ArviZ       (mean ESS) : {az_drift_mean:.1f}")
print(f"  ArviZ       (bulk ESS) : {az_drift_bulk:.1f}")

# ===========================================================================
# Test Case 4: Heavy-Tailed Chains (Student-t with df=2, infinite variance)
# ===========================================================================
print("\n" + "=" * 75)
print("--- Test Case 4: Heavy-Tailed Chains (Student-t, df=2) ---")
print("=" * 75)
from scipy.stats import t as student_t
t_samples = torch.tensor(student_t.rvs(df=2, size=(n_chains, n_samples)), dtype=torch.float64)
sample_dict_t = {"theta": t_samples}

print("\n[R-hat Convergence Diagnostic] (Rank-normalized R-hat handles infinite variance)")
for method in ["classical", "split", "rank"]:
    rhat = gelman_rubin(sample_dict_t, method=method)["theta"].item()
    print(f"  BayesianDLL ({method:9s}): R-hat = {rhat:.4f}")

az_data_t = az.convert_to_dataset(t_samples.numpy())
rhat_rank = float(az.rhat(az_data_t, method="rank")["x"].values)
rhat_split = float(az.rhat(az_data_t, method="split")["x"].values)
rhat_classical = float(az.rhat(az_data_t, method="identity")["x"].values)
print(f"  ArviZ       (classical): R-hat = {rhat_classical:.4f}")
print(f"  ArviZ       (split)    : R-hat = {rhat_split:.4f}")
print(f"  ArviZ       (rank)     : R-hat = {rhat_rank:.4f}")

print("\n[Effective Sample Size (ESS)]")
ess_t_mean = effective_sample_size(sample_dict_t, method="mean")["theta"].item()
ess_t_bulk = effective_sample_size(sample_dict_t, method="bulk")["theta"].item()
print(f"  BayesianDLL (mean ESS) : {ess_t_mean:.1f}")
print(f"  BayesianDLL (bulk ESS) : {ess_t_bulk:.1f}")

az_t_mean = float(az.ess(az_data_t, method="mean")["x"].values)
az_t_bulk = float(az.ess(az_data_t, method="bulk")["x"].values)
print(f"  ArviZ       (mean ESS) : {az_t_mean:.1f}")
print(f"  ArviZ       (bulk ESS) : {az_t_bulk:.1f}")

print("\n" + "=" * 75)
print("ALL TESTS COMPLETED SUCCESSFULLY.")
print("=" * 75)

