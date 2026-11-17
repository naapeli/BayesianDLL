"""
Joint Posterior of a Mean and a Scale With NUTS
===============================================

One hundred measurements are modelled as ``Normal(mean, sigma)`` with
weakly informative priors ``mean ~ Normal(0, 10)`` and
``sigma ~ HalfNormal(5)``. NUTS updates both parameters together; ``sigma``
is sampled as ``log(sigma)``, on which the posterior is nearly Gaussian and
unbounded.

Four chains are run, and R-hat and the effective sample size check that they
agree and how many independent draws they are worth. The exact joint
posterior is computed on a grid for reference. The figures show the model
graph, the joint draws over the exact posterior contours with the true
values, the marginal posteriors, and the fitted density over the data.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Evaluation import Graphics, effective_sample_size, gelman_rubin


torch.manual_seed(0)

n_obs = 100
mean_true, sigma_true = 5.0, 3 ** 0.5
y = mean_true + sigma_true * torch.randn(n_obs, dtype=torch.float64)

with Model() as model:
    mean = RandomParameter("mean", Normal(0.0, 10.0))
    sigma = RandomParameter("sigma", HalfNormal(5.0))
    with plate("observations", n_obs):
        ObservedParameter("y", Normal(mean, sigma), y)

Graphics.plot_model(model)
result = model.sample(1000, 500, n_chains=4, seed=0, delta=0.85, progress_bar=False)

for name in ("mean", "sigma"):
    draws = {name: result[name]}
    print(f"{name:5s}: posterior mean {result[name].mean():.3f}, R-hat {gelman_rubin(draws)[name].item():.4f}, "
          f"bulk ESS {effective_sample_size(draws)[name].item():.0f}, true {[mean_true, sigma_true][name == 'sigma']:.3f}")
print(f"divergent transitions: {sum(result.divergences)}")

# %%
# Exact posterior on a grid
# -------------------------
mean_grid = torch.linspace(4.0, 6.0, 200, dtype=torch.float64)
sigma_grid = torch.linspace(1.4, 2.8, 200, dtype=torch.float64)
M, S = torch.meshgrid(mean_grid, sigma_grid, indexing="ij")
log_post = (torch.distributions.Normal(0.0, 10.0).log_prob(M) + torch.distributions.HalfNormal(5.0).log_prob(S)
            + torch.distributions.Normal(M.unsqueeze(-1), S.unsqueeze(-1)).log_prob(y).sum(-1))
post = torch.exp(log_post - log_post.max())
post_mean_marginal = post.sum(1) / (post.sum() * (mean_grid[1] - mean_grid[0]))
post_sigma_marginal = post.sum(0) / (post.sum() * (sigma_grid[1] - sigma_grid[0]))

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), layout="constrained")

ax = axes[0, 0]
ax.scatter(result["mean"].reshape(-1)[::4], result["sigma"].reshape(-1)[::4], s=4, alpha=0.4, label="NUTS draws")
ax.contour(M, S, post, levels=[0.011, 0.135, 0.607], colors="black", linewidths=1, zorder=3)
ax.plot(mean_true, sigma_true, "r*", markersize=12, label="true values")
ax.set_xlabel("mean")
ax.set_ylabel("sigma")
ax.set_title("Joint posterior (contours: exact)")
ax.legend(frameon=False, fontsize="small", loc="lower left")

for ax, name, grid, marginal, truth in ((axes[0, 1], "mean", mean_grid, post_mean_marginal, mean_true),
                                        (axes[1, 0], "sigma", sigma_grid, post_sigma_marginal, sigma_true)):
    ax.hist(result[name].reshape(-1).numpy(), bins=50, density=True, alpha=0.5, label="NUTS draws")
    ax.plot(grid, marginal, color="black", label="exact")
    ax.axvline(truth, color="C3", linestyle="--", label="true value")
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    ax.set_title(f"Marginal posterior of {name}")
    ax.legend(frameon=False, fontsize="small")

ax = axes[1, 1]
grid = torch.linspace(y.min() - 1, y.max() + 1, 300, dtype=torch.float64)
densities = torch.distributions.Normal(result["mean"].reshape(-1, 1), result["sigma"].reshape(-1, 1)).log_prob(grid).exp()
ax.hist(y.numpy(), bins=20, density=True, color="lightgray", label="data")
ax.fill_between(grid, densities.quantile(0.05, 0), densities.quantile(0.95, 0), color="C0", alpha=0.3, linewidth=0,
                label="90% band")
ax.plot(grid, densities.mean(0), color="C0", label="fitted density")
ax.plot(grid, torch.distributions.Normal(mean_true, sigma_true).log_prob(grid).exp(), "--", color="C3",
        label="true density")
ax.set_xlabel("y")
ax.set_ylabel("density")
ax.set_title("Fitted density over the data")
ax.legend(frameon=False, fontsize="small")
plt.show()
