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

Eight univariate distributions with bounded, heavy-tailed, skewed and bimodal
supports are sampled with four vectorized chains each, and the first figure
overlays the draws on the exact densities. The second figure shows a
Dirichlet distribution: its draws on the probability simplex over the exact
density, and the three marginals against their exact Beta densities.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from BayesianDLL import Model, RandomParameter, sample
from BayesianDLL.Distributions import Beta, Dirichlet, Exponential, HalfCauchy, InvGamma, Mixture, Normal, Uniform
from BayesianDLL.Evaluation import effective_sample_size, gelman_rubin


def nuts_draws(name, distribution, seed, n_samples=800):
    with Model() as model:
        RandomParameter("x", distribution)
    result = sample(n_samples, 300, n_chains=4, model=model, seed=seed, delta=0.9, chain_method="vectorized",
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

fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), layout="constrained")
for k, (ax, (name, distribution, (low, high), scale)) in enumerate(zip(axes.flat, univariate)):
    # The chains must hop between the two modes of the mixture: give it more draws.
    draws = nuts_draws(name, distribution, seed=k, n_samples=2500 if isinstance(distribution, Mixture) else 800)
    draws = draws.reshape(-1)
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

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
fig.suptitle("NUTS draws against the exact densities")

# %%
# A Dirichlet distribution on the 3-simplex
# -----------------------------------------
# A draw from a Dirichlet distribution is a probability vector: three positive
# numbers that sum to one. Every such vector is a point in a triangle, the
# 3-simplex, whose corners are the vectors (1, 0, 0), (0, 1, 0) and (0, 0, 1).
# NUTS samples an unconstrained two-dimensional vector that the stick-breaking
# transform maps into the triangle.
alpha = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)
dirichlet = Dirichlet(alpha)
draws = nuts_draws("Dirichlet(2, 4, 8)", dirichlet, seed=20).reshape(-1, 3)
corners = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 3 ** 0.5 / 2]], dtype=torch.float64)

resolution = 120
grid = torch.tensor([[i, j, resolution - i - j] for i in range(1, resolution) for j in range(1, resolution - i)],
                    dtype=torch.float64) / resolution
plane = grid @ corners
density = dirichlet.pdf(grid).reshape(-1)

fig, (ax_simplex, ax_marginals) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
mesh = ax_simplex.tripcolor(plane[:, 0], plane[:, 1], density, shading="gouraud", cmap="Blues")
points = draws[:800] @ corners
ax_simplex.scatter(points[:, 0], points[:, 1], s=2, color="C1", alpha=0.6, label="NUTS draws")
ax_simplex.plot(*torch.cat([corners, corners[:1]]).T, color="black")
for corner, label in zip(corners, ("$x_1 = 1$", "$x_2 = 1$", "$x_3 = 1$")):
    ax_simplex.annotate(label, corner, textcoords="offset points", xytext=(0, -14 if corner[1] == 0 else 6),
                        ha="center", fontsize="small")
ax_simplex.set_aspect("equal")
ax_simplex.axis("off")
ax_simplex.set_title("Draws on the exact Dirichlet(2, 4, 8) density")
ax_simplex.legend(loc="upper left", fontsize="small", markerscale=4)
fig.colorbar(mesh, ax=ax_simplex, shrink=0.7, label="density")

# The marginal of each component is Beta(alpha_k, sum(alpha) - alpha_k).
x = torch.linspace(1e-3, 1 - 1e-3, 400, dtype=torch.float64)
for k in range(3):
    ax_marginals.hist(draws[:, k].numpy(), bins=50, range=(0, 1), density=True, alpha=0.45, color=f"C{k}",
                      label=f"$x_{k + 1}$ draws")
    exact = Beta(alpha[k], alpha.sum() - alpha[k]).pdf(x.unsqueeze(-1)).reshape(-1)
    ax_marginals.plot(x, exact, color=f"C{k}")
ax_marginals.plot([], [], color="black", label="exact Beta marginals")
ax_marginals.set(xlabel="component value", ylabel="density", title="Marginals of the three components")
ax_marginals.legend(fontsize="small")
plt.show()
