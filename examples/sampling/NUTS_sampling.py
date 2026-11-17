"""
NUTS Sampling
=============

The No-U-Turn Sampler (NUTS) simulates Hamiltonian dynamics with the gradient
of the log density and stops a trajectory when it starts to turn back. It
adapts its step size (towards the acceptance statistic ``delta``) and a
diagonal mass matrix during warmup, so it needs no hand tuning.

NUTS moves on an unconstrained scale: positive variables are sampled as their
logarithm, variables on an interval through a logit, and simplex-valued
variables through a stick-breaking map. The draws are mapped back, so they
always respect the support, and the log density includes the Jacobian of the
map.

Nine distributions with bounded, heavy-tailed, skewed, bimodal and simplex
supports are sampled with four vectorized chains each. The figure overlays
the draws on the exact densities; for the Dirichlet distribution the draws
are shown on the probability simplex over its exact density.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from BayesianDLL import Model, RandomParameter, sample
from BayesianDLL.Distributions import Beta, Dirichlet, Exponential, HalfCauchy, InvGamma, Mixture, Normal, Uniform
from BayesianDLL.Evaluation import effective_sample_size, gelman_rubin


def nuts_draws(name, distribution, seed):
    with Model() as model:
        RandomParameter("x", distribution)
    result = sample(2000, 500, n_chains=4, model=model, seed=seed, delta=0.9, chain_method="vectorized",
                    progress_bar=False)
    draws = {"x": result["x"].double()}
    print(f"{name:24s}: R-hat {gelman_rubin(draws)['x'].max():.3f}, bulk ESS {effective_sample_size(draws)['x'].min():5.0f}, "
          f"divergences {sum(result.divergences)}")
    return result["x"].double()


univariate = [
    ("Normal(5, sd 1.7)", Normal(5.0, 3 ** 0.5), (-1.0, 11.0), "linear"),
    ("Beta(2, 5)", Beta(2.0, 5.0), (0.0, 1.0), "linear"),
    ("Beta(0.5, 0.5)", Beta(0.5, 0.5), (0.0, 1.0), "linear"),
    ("Exponential(rate 0.3)", Exponential(0.3), (0.0, 20.0), "linear"),
    ("Uniform(2, 5)", Uniform(2.0, 5.0), (1.5, 5.5), "linear"),
    ("InvGamma(3, 6)", InvGamma(3.0, 6.0), (0.5, 30.0), "log"),
    ("HalfCauchy(2)", HalfCauchy(2.0), (0.05, 500.0), "log"),
    ("Mixture of two Normals", Mixture([Normal(-1.0, 0.5), Normal(2.0, 1.0)], [0.3, 0.7]), (-3.5, 5.5), "linear"),
]

fig, axes = plt.subplots(3, 3, figsize=(13, 11), layout="constrained")
for k, (ax, (name, distribution, (low, high), scale)) in enumerate(zip(axes.flat, univariate)):
    draws = nuts_draws(name, distribution, seed=k).reshape(-1)
    if scale == "log":
        edges = np.logspace(np.log10(low), np.log10(high), 50)
        grid = torch.logspace(np.log10(low), np.log10(high), 400, dtype=torch.float64)
        ax.hist(draws.numpy(), bins=edges, density=True, alpha=0.6, label="NUTS draws")
        ax.plot(grid, distribution.pdf(grid.unsqueeze(-1)).reshape(-1), color="black", label="exact density")
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        grid = torch.linspace(low, high, 400, dtype=torch.float64)
        inside = grid.clone()
        if isinstance(distribution, Beta):
            inside = grid.clamp(1e-3, 1 - 1e-3)
        density = distribution.pdf(inside.unsqueeze(-1)).reshape(-1)
        if isinstance(distribution, Uniform):
            density = torch.where((grid > 2.0) & (grid < 5.0), density, torch.zeros_like(density))
        ax.hist(draws.numpy(), bins=50, range=(low, high), density=True, alpha=0.6, label="NUTS draws")
        ax.plot(inside, density, color="black", label="exact density")
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("density")

# %%
# A Dirichlet distribution on the 3-simplex
# -----------------------------------------
alpha = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)
dirichlet = Dirichlet(alpha)
draws = nuts_draws("Dirichlet(2, 4, 8)", dirichlet, seed=20).reshape(-1, 3)
corners = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 3 ** 0.5 / 2]], dtype=torch.float64)

resolution = 120
grid = torch.tensor([[i, j, resolution - i - j] for i in range(1, resolution) for j in range(1, resolution - i)],
                    dtype=torch.float64) / resolution
plane = grid @ corners
density = dirichlet.pdf(grid).reshape(-1)
ax = axes.flat[-1]
ax.tripcolor(plane[:, 0], plane[:, 1], density, shading="gouraud", cmap="Blues")
points = draws[:800] @ corners
ax.scatter(points[:, 0], points[:, 1], s=2, color="C1", alpha=0.6)
ax.plot(*torch.cat([corners, corners[:1]]).T, color="black")
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Dirichlet(2, 4, 8): draws on the exact density")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
fig.suptitle("NUTS draws against the exact densities")
plt.show()
