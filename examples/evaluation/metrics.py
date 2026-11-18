"""
Convergence Diagnostics
=======================

Two numbers summarise whether MCMC chains can be trusted:

* **R-hat** compares the variance between chains with the variance within
  them. Chains that sample the same distribution give values close to 1; the
  usual threshold is 1.01. ``gelman_rubin`` offers the classical statistic,
  the split version (each chain cut in half, which also catches drift within a
  chain) and the rank-normalised split version (Vehtari et al., 2021), which
  also works for heavy tails.
* **Effective sample size (ESS)** is the number of independent draws that
  would give the same Monte Carlo error. ``effective_sample_size`` computes
  the bulk ESS (rank-normalised, for the centre of the distribution), the tail
  ESS (for the 5% and 95% quantiles, or another ``prob``) and the classic ESS
  of the mean.

Four synthetic sets of four chains with 1000 draws each test the diagnostics:
independent draws, autocorrelated AR(1) chains with a known ESS, chains stuck
at different levels, and heavy-tailed Student-t draws. Every value is
compared with ArviZ, and the figure shows the chains with their
diagnostics.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

from BayesianDLL.Evaluation import effective_sample_size, gelman_rubin


torch.manual_seed(42)
rng = np.random.default_rng(42)
n_chains, n_draws = 4, 1000

phi = 0.7  # AR(1) coefficient
ar = np.zeros((n_chains, n_draws))
for t in range(1, n_draws):
    ar[:, t] = phi * ar[:, t - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(n_chains)

cases = {
    "independent N(0, 1)": torch.randn(n_chains, n_draws, dtype=torch.float64),
    f"AR(1), phi = {phi}": torch.tensor(ar),
    "chains at different levels": torch.randn(n_chains, n_draws, dtype=torch.float64)
    + torch.arange(n_chains, dtype=torch.float64).unsqueeze(1),
    "Student-t, 2 dof": torch.tensor(student_t.rvs(df=2, size=(n_chains, n_draws), random_state=rng)),
}
expected_ess = {f"AR(1), phi = {phi}": n_chains * n_draws * (1 - phi) / (1 + phi), "independent N(0, 1)": n_chains * n_draws}

arviz_methods = {"classical": "identity", "split": "split", "rank": "rank"}
diagnostics = {}
for name, chains in cases.items():
    draws = {"theta": chains}
    dataset = az.convert_to_dataset(chains.numpy())
    print(f"\n{name}")
    print(f"  {'diagnostic':18s} {'BayesianDLL':>12s} {'ArviZ':>10s}")
    for method, arviz_method in arviz_methods.items():
        ours = gelman_rubin(draws, method=method)["theta"].item()
        theirs = float(az.rhat(dataset, method=arviz_method)["x"].values)
        print(f"  R-hat ({method:9s}) {ours:12.4f} {theirs:10.4f}")
    for method in ("bulk", "tail", "mean"):
        ours = effective_sample_size(draws, method=method)["theta"].item()
        theirs = float(az.ess(dataset, method=method)["x"].values)
        print(f"  ESS ({method:4s})        {ours:12.1f} {theirs:10.1f}")
    if name in expected_ess:
        print(f"  theoretical ESS of the mean: {expected_ess[name]:.0f}")
    diagnostics[name] = (gelman_rubin(draws)["theta"].item(), effective_sample_size(draws)["theta"].item())

fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained", sharex=True)
for ax, (name, chains) in zip(axes.flat, cases.items()):
    for chain in range(n_chains):
        ax.plot(chains[chain, :300], linewidth=0.8, alpha=0.8, label=f"chain {chain + 1}")
    r_hat, ess = diagnostics[name]
    ax.set_title(f"{name}\nrank R-hat {r_hat:.3f}, bulk ESS {ess:.0f}", fontsize="medium")
    ax.set_ylabel("value")
    if name.startswith("Student"):
        ax.set_ylim(-15, 15)
for ax in axes[1]:
    ax.set_xlabel("iteration (first 300)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=n_chains, frameon=False)
plt.show()
