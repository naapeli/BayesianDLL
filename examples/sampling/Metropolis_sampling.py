"""
Metropolis Sampling
===================

Random-walk Metropolis proposes a move and accepts it with probability
``min(1, p(new) / p(old))``. It needs only the log density, no gradient, so it
also works for discrete parameters. Here it draws from five fixed
distributions, three discrete and two continuous, so the result can be
checked against the exact probabilities.

* For a discrete parameter every coordinate proposes a step of ``+1`` or
  ``-1``; there is nothing to tune.
* For a continuous parameter the proposal is Gaussian on the unconstrained
  scale (log for positive, logit for bounded variables). Its width adapts
  during warmup towards an acceptance rate of 0.234.

Each distribution is sampled with four chains in one process
(``chain_method="vectorized"``). The figure compares the draws with the exact
probability mass or density, and the last panel shows how a random walk moves
through an Exponential distribution: successive draws are strongly
correlated.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, RandomParameter, sample
from BayesianDLL.Distributions import Bernoulli, Beta, Binomial, Exponential, Poisson
from BayesianDLL.Evaluation import effective_sample_size, gelman_rubin


torch.manual_seed(0)


def metropolis_draws(distribution, seed):
    with Model() as model:
        RandomParameter("x", distribution, sampler="metropolis")
    result = sample(4000, 1000, n_chains=4, model=model, seed=seed, chain_method="vectorized", progress_bar=False)
    draws = {"x": result["x"].double()}
    print(f"{type(distribution).__name__:12s}: R-hat {gelman_rubin(draws)['x'].max():.3f}, "
          f"bulk ESS {effective_sample_size(draws)['x'].min():6.0f} of {draws['x'].numel()} draws, "
          f"acceptance {result.sample_stats['accepted'].double().mean():.2f}")
    return result["x"]


fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")

discrete = [
    ("Bernoulli(0.1)", Bernoulli(0.1), torch.arange(0, 2)),
    ("Binomial(10, 0.2)", Binomial(10, 0.2), torch.arange(0, 11)),
    ("Poisson(3)", Poisson(3.0), torch.arange(0, 13)),
]
for k, (ax, (title, distribution, support)) in enumerate(zip(axes[0], discrete)):
    draws = metropolis_draws(distribution, seed=k).reshape(-1).long()
    frequency = torch.bincount(draws, minlength=int(support.max()) + 1)[: len(support)].double() / draws.numel()
    exact = distribution.pdf(support.double().unsqueeze(-1)).reshape(-1)
    ax.bar(support, frequency, alpha=0.6, label="Metropolis draws")
    ax.plot(support, exact, "ko", markersize=4, label="exact pmf")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("probability")

continuous = [
    ("Exponential(rate 0.3)", Exponential(0.3), torch.linspace(1e-3, 20, 300)),
    ("Beta(2, 5)", Beta(2.0, 5.0), torch.linspace(1e-3, 1 - 1e-3, 300)),
]
traces = {}
for k, (ax, (title, distribution, grid)) in enumerate(zip(axes[1, :2], continuous)):
    draws = metropolis_draws(distribution, seed=10 + k)
    traces[title] = draws
    ax.hist(draws.reshape(-1).numpy(), bins=50, range=(0, float(grid.max())), density=True, alpha=0.6,
            label="Metropolis draws")
    ax.plot(grid, distribution.pdf(grid.double().unsqueeze(-1)).reshape(-1), color="black", label="exact density")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("density")

ax = axes[1, 2]
ax.plot(traces["Exponential(rate 0.3)"][0, :300].reshape(-1), color="C0")
ax.set_title("Random walk: first 300 draws of one chain")
ax.set_xlabel("iteration")
ax.set_ylabel("value (Exponential)")

handles, labels = axes[0, 0].get_legend_handles_labels()
handles2, labels2 = axes[1, 0].get_legend_handles_labels()
fig.legend(handles + handles2[1:], labels + labels2[1:], loc="outside lower center", ncols=3, frameon=False)
fig.suptitle("Metropolis draws against the exact distributions")
plt.show()
