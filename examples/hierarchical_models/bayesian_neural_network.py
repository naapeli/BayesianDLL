"""
Bayesian Neural Network
=======================

A neural network with one hidden layer of ``H`` units fitted to noisy 1-D
regression data:

.. math::

    h = \\tanh(x W_1 + b_1), \\quad f(x) = h w_2 + b_2, \\quad
    y \\sim N(f(x), \\sigma).

All weights get standard normal priors and the posterior is sampled with
NUTS. The network is assembled from ``MatMul``, ``Add`` and ``Tanh`` blocks,
whose hand-written vector-Jacobian products give the gradients for NUTS. The
posterior predictive mean and interval are computed from the posterior draws
of the weights; the uncertainty grows away from the training data.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Add, MatMul, Tanh
from BayesianDLL.Distributions import HalfNormal, Normal


torch.manual_seed(5)

n_observations, n_hidden = 40, 6
x = torch.cat([torch.linspace(-3.0, -0.8, n_observations // 2), torch.linspace(0.8, 3.0, n_observations // 2)])
x = x.to(torch.float64)
y = torch.sin(1.3 * x) + 0.15 * torch.randn(n_observations, dtype=torch.float64)

with Model() as model:
    inputs = Data("inputs", x.unsqueeze(-1))  # (N, 1)
    W1 = RandomParameter("W1", Normal(0.0, 1.0), shape=(1, n_hidden))
    b1 = RandomParameter("b1", Normal(0.0, 1.0), shape=(n_hidden,))
    w2 = RandomParameter("w2", Normal(0.0, 1.0), shape=(n_hidden,))
    b2 = RandomParameter("b2", Normal(0.0, 1.0))
    sigma = RandomParameter("sigma", HalfNormal(0.5))

    with plate("observations", n_observations):
        hidden = Tanh("hidden", Add("pre_activation", MatMul("x_W1", inputs, W1), b1))  # (N, H)
        f = Add("f", MatMul("h_w2", hidden, w2), b2)  # (N,)
        ObservedParameter("y", Normal(f, sigma), y)

trace = model.sample(n_samples=300, warmup_length=300, n_chains=2, progress_bar=False, seed=0, delta=0.8)
print("divergences:", sum(trace.divergences))
print(trace.summary().loc[["sigma[0]", "b2[0]"]])

# %%
# Posterior predictive on a grid
# ------------------------------
# Every posterior draw of the weights defines one network; evaluating all of
# them on a grid gives the predictive distribution of ``f`` and of new ``y``.

grid = torch.linspace(-4.5, 4.5, 200, dtype=torch.float64)
W1_draws = trace["W1"].flatten(0, 1)  # (S, 1, H)
b1_draws = trace["b1"].flatten(0, 1)  # (S, H)
w2_draws = trace["w2"].flatten(0, 1)  # (S, H)
b2_draws = trace["b2"].flatten(0, 1)  # (S, 1)
sigma_draws = trace["sigma"].flatten(0, 1)  # (S, 1)

hidden_grid = torch.tanh(grid.reshape(1, -1, 1) @ W1_draws + b1_draws.unsqueeze(1))  # (S, G, H)
f_grid = (hidden_grid @ w2_draws.unsqueeze(-1)).squeeze(-1) + b2_draws  # (S, G)
y_grid = f_grid + sigma_draws * torch.randn_like(f_grid)

quantiles = torch.tensor([0.05, 0.95], dtype=grid.dtype)
f_lower, f_upper = f_grid.quantile(quantiles, dim=0)
y_lower, y_upper = y_grid.quantile(quantiles, dim=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(grid, y_lower, y_upper, color="C0", alpha=0.15, label="90% predictive interval of y")
ax.fill_between(grid, f_lower, f_upper, color="C0", alpha=0.35, label="90% interval of f")
ax.plot(grid, f_grid.mean(0), color="C0", label="posterior mean of f")
ax.plot(grid, torch.sin(1.3 * grid), color="black", linestyle="--", label="true function")
ax.scatter(x, y, color="black", s=12, label="observations")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Bayesian neural network with {n_hidden} hidden units")
ax.legend(fontsize=8)
plt.show()
