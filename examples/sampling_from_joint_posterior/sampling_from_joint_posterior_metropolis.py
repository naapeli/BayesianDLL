"""
Joint Posterior of a Mean and a Scale With Metropolis
=====================================================

The same model as in the NUTS example, ``y ~ Normal(mean, sigma)`` with
``mean ~ Normal(0, 10)`` and ``sigma ~ HalfNormal(5)``, is sampled with
random-walk Metropolis. Both parameters request ``sampler="metropolis"``, so
they form one block with a joint Gaussian proposal on the unconstrained scale
(``sigma`` moves as ``log(sigma)``). The proposal width adapts during warmup
towards an acceptance rate of 0.234.

Metropolis uses no gradient, so it is simple and cheap per step, but its
successive draws are strongly correlated: it needs many more iterations than
NUTS for the same effective sample size. Here it runs 5000 iterations per
chain against 1000 for NUTS. The figure overlays both sets of draws on the
exact posterior contours, compares the marginals and the autocorrelations,
and prints the effective sample sizes.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Evaluation import effective_sample_size, gelman_rubin


torch.manual_seed(0)

n_obs = 100
mean_true, sigma_true = 5.0, 3 ** 0.5
y = mean_true + sigma_true * torch.randn(n_obs, dtype=torch.float64)


def normal_model(sampler):
    with Model() as model:
        mean = RandomParameter("mean", Normal(0.0, 10.0), sampler=sampler)
        sigma = RandomParameter("sigma", HalfNormal(5.0), sampler=sampler)
        with plate("observations", n_obs):
            ObservedParameter("y", Normal(mean, sigma), y)
    return model


runs = {
    "Metropolis": normal_model("metropolis").sample(5000, 1000, n_chains=4, seed=0, chain_method="vectorized",
                                                    progress_bar=False),
    "NUTS": normal_model("nuts").sample(1000, 500, n_chains=4, seed=0, delta=0.85, chain_method="vectorized",
                                        progress_bar=False),
}
for label, result in runs.items():
    draws = {name: result[name] for name in ("mean", "sigma")}
    r_hat = max(value.item() for value in gelman_rubin(draws).values())
    ess = {name: value.item() for name, value in effective_sample_size(draws).items()}
    total = result["mean"].numel()
    print(f"{label:10s}: {total} draws, max R-hat {r_hat:.4f}, bulk ESS mean {ess['mean']:.0f}, sigma {ess['sigma']:.0f} "
          f"({min(ess.values()) / total:.1%} of the draws)")
print("Metropolis acceptance rate:", f"{runs['Metropolis'].sample_stats['accepted'].double().mean():.2f}")

# Exact posterior on a grid.
mean_grid = torch.linspace(4.0, 6.0, 200, dtype=torch.float64)
sigma_grid = torch.linspace(1.4, 2.8, 200, dtype=torch.float64)
M, S = torch.meshgrid(mean_grid, sigma_grid, indexing="ij")
log_post = (torch.distributions.Normal(0.0, 10.0).log_prob(M) + torch.distributions.HalfNormal(5.0).log_prob(S)
            + torch.distributions.Normal(M.unsqueeze(-1), S.unsqueeze(-1)).log_prob(y).sum(-1))
post = torch.exp(log_post - log_post.max())
marginals = {"mean": (mean_grid, post.sum(1) / (post.sum() * (mean_grid[1] - mean_grid[0]))),
             "sigma": (sigma_grid, post.sum(0) / (post.sum() * (sigma_grid[1] - sigma_grid[0])))}


def autocorrelation(chain, max_lag=60):
    chain = chain - chain.mean()
    variance = (chain * chain).mean()
    return torch.stack([(chain[: len(chain) - k] * chain[k:]).mean() / variance for k in range(max_lag + 1)])


colors = {"Metropolis": "C0", "NUTS": "C1"}
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), layout="constrained")

ax = axes[0, 0]
for label, result in runs.items():
    ax.scatter(result["mean"].reshape(-1)[:1000], result["sigma"].reshape(-1)[:1000], s=4, alpha=0.4,
               color=colors[label])
ax.contour(M, S, post, levels=[0.011, 0.135, 0.607], colors="black", linewidths=1, zorder=3)
ax.plot(mean_true, sigma_true, "r*", markersize=12)
ax.set_xlabel("mean")
ax.set_ylabel("sigma")
ax.set_title("1000 draws each on the exact contours (star: truth)")

for ax, name in zip((axes[0, 1], axes[1, 0]), ("mean", "sigma")):
    for label, result in runs.items():
        ax.hist(result[name].reshape(-1).numpy(), bins=50, density=True, histtype="step", linewidth=1.5,
                color=colors[label])
    grid, marginal = marginals[name]
    ax.plot(grid, marginal, color="black")
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    ax.set_title(f"Marginal posterior of {name} (black: exact)")

ax = axes[1, 1]
for label, result in runs.items():
    ax.plot(autocorrelation(result["mean"][0].reshape(-1)), color=colors[label], marker=".", label=label)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("lag")
ax.set_ylabel("autocorrelation of mean")
ax.set_title("Autocorrelation within one chain")
ax.legend(frameon=False, fontsize="small")
handles = [plt.Line2D([], [], color=colors[label], label=label) for label in runs]
fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
plt.show()
