"""
Sequential Monte Carlo
======================

Sequential Monte Carlo (SMC) starts with particles drawn from the prior and
moves them through the tempered posteriors ``p(theta) p(y | theta)^beta`` as
the inverse temperature ``beta`` rises from 0 to 1. Each stage picks the next
``beta`` so that the importance weights keep a target effective sample size,
resamples the particles and moves them with a few MCMC steps.

Two properties make SMC useful beside NUTS:

* the particles start spread over the whole prior, so they populate every
  mode of a multimodal posterior, while one MCMC chain usually stays in the
  mode where it starts;
* the weights give an estimate of the marginal likelihood ``p(y)``, from which
  two models can be compared with a Bayes factor.

The first model observes only ``|theta|``, so ``theta`` has two symmetric
posterior modes; ``|theta|`` is the deterministic block ``Elementwise``. The
second part compares a Normal and a Student-t model for data with outliers.
Both answers are checked against numerical integration on a grid. The figure
shows the posterior of ``theta`` from SMC and from a single NUTS chain over the
exact density, the tempering schedule, both fitted models over the data, and
the log evidences.
"""

import math

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate, sample, sample_smc
from BayesianDLL.Deterministic import Elementwise
from BayesianDLL.Distributions import Gamma, Normal, StudentT


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


# %%
# A bimodal posterior
# -------------------
y = 2.5 + torch.randn(20)
with Model() as bimodal:
    theta = RandomParameter("theta", Normal(0.0, 3.0))
    magnitude = Elementwise("magnitude", torch.abs, lambda t: (torch.sign(t),), theta)
    with plate("observations", len(y)):
        ObservedParameter("y", Normal(magnitude, 1.0), y)

smc = sample_smc(bimodal, n_particles=2000, kernel="metropolis", seed=0, progress_bar=False)
# One chain only: several NUTS chains would land in different modes and
# disagree, which is the failure this part illustrates.
nuts = sample(1000, 300, n_chains=1, model=bimodal, seed=0, check_convergence=False, progress_bar=False)

theta_grid = torch.linspace(-4.5, 4.5, 2001)
log_posterior = (torch.distributions.Normal(0.0, 3.0).log_prob(theta_grid)
                 + torch.distributions.Normal(theta_grid.abs().unsqueeze(1), 1.0).log_prob(y).sum(1))
exact_theta = torch.softmax(log_posterior, 0) / (theta_grid[1] - theta_grid[0])
print(f"share of draws with theta > 0: SMC {(smc['theta'] > 0).double().mean():.2f}, "
      f"NUTS {(nuts['theta'] > 0).double().mean():.2f}, exact 0.50")
print(f"{len(smc.beta_schedule) - 1} tempering stages")


# %%
# Bayes factor from the log evidence
# ----------------------------------
data = torch.cat([torch.randn(40), torch.tensor([6.0, -5.5, 7.0])])
likelihoods = {
    "Normal": (lambda mu, sigma: Normal(mu, sigma), lambda mu, sigma: torch.distributions.Normal(mu, sigma)),
    "Student-t (3 dof)": (lambda mu, sigma: StudentT(3.0, mu, sigma),
                          lambda mu, sigma: torch.distributions.StudentT(3.0, mu, sigma)),
}
mu_grid = torch.linspace(-3.0, 3.0, 300)
sigma_grid = torch.linspace(0.02, 8.0, 400)
M, S = torch.meshgrid(mu_grid, sigma_grid, indexing="ij")
cell = (mu_grid[1] - mu_grid[0]) * (sigma_grid[1] - sigma_grid[0])

fits, exact_evidence = {}, {}
for k, (name, (model_likelihood, torch_likelihood)) in enumerate(likelihoods.items()):
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        sigma = RandomParameter("sigma", Gamma(2.0, 1.0))
        with plate("data", len(data)):
            ObservedParameter("data", model_likelihood(mu, sigma), data)
    fits[name] = sample_smc(model, n_particles=2000, seed=1 + k, progress_bar=False)
    log_joint = (torch.distributions.Normal(0.0, 5.0).log_prob(M) + torch.distributions.Gamma(2.0, 1.0).log_prob(S)
                 + torch_likelihood(M.unsqueeze(-1), S.unsqueeze(-1)).log_prob(data).sum(-1))
    exact_evidence[name] = (torch.logsumexp(log_joint.reshape(-1), 0) + torch.log(cell)).item()
    print(f"log p(y | {name}): SMC {fits[name].log_marginal_likelihood:.2f}, grid {exact_evidence[name]:.2f}")

log_bayes_factor = fits["Student-t (3 dof)"].log_marginal_likelihood - fits["Normal"].log_marginal_likelihood
print(f"Bayes factor Student-t : Normal = {math.exp(log_bayes_factor):.3g}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

ax = axes[0, 0]
bins = torch.linspace(-4.5, 4.5, 80).numpy()
ax.hist(smc["theta"].reshape(-1).numpy(), bins=bins, density=True, alpha=0.5, label="SMC particles")
ax.hist(nuts["theta"].reshape(-1).numpy(), bins=bins, density=True, histtype="step", linewidth=1.5,
        label="one NUTS chain")
ax.plot(theta_grid, exact_theta, color="black", label="exact")
ax.set_xlabel("theta")
ax.set_ylabel("density")
ax.set_title("Bimodal posterior")
ax.legend(frameon=False, fontsize="small", loc="upper center")

ax = axes[0, 1]
ax.plot(smc.beta_schedule, marker="o")
ax.set_xlabel("stage")
ax.set_ylabel("inverse temperature beta")
ax.set_title("Adaptive tempering schedule")

ax = axes[1, 0]
grid = torch.linspace(-8, 9, 400)
ax.hist(data.numpy(), bins=30, density=True, color="lightgray", label="data")
for name, (_, torch_likelihood) in likelihoods.items():
    mu_draws, sigma_draws = fits[name]["mu"].reshape(-1, 1), fits[name]["sigma"].reshape(-1, 1)
    density = torch_likelihood(mu_draws, sigma_draws).log_prob(grid).exp().mean(0)
    ax.plot(grid, density, label=name)
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_xlabel("y")
ax.set_ylabel("density (log scale)")
ax.set_title("Posterior predictive densities")
ax.legend(frameon=False, fontsize="small")

ax = axes[1, 1]
names = list(likelihoods)
positions = torch.arange(len(names), dtype=torch.float64)
ax.plot(positions, [fits[n].log_marginal_likelihood for n in names], "o", markersize=10, label="SMC estimate")
ax.plot(positions, [exact_evidence[n] for n in names], "kx", markersize=10, label="grid integration")
ax.set_xticks(positions, names)
ax.set_xlim(-0.5, len(names) - 0.5)
ax.set_ylabel("log p(y)")
ax.set_title(f"Log evidence (log Bayes factor {log_bayes_factor:.1f})")
ax.legend(frameon=False, fontsize="small")
plt.show()
