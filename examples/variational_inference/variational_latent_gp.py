"""
Variational Inference for a Latent GP
=====================================

A latent Gaussian process ``f ~ GP(0, k_RBF)`` is observed with Normal
noise, ``y_i ~ N(f(x_i), 0.29)``. ``LatentGP`` represents the function in
**whitened** (non-centred) form: ``f = L z`` with ``z ~ N(0, I)`` and ``L`` the
Cholesky factor of the kernel matrix. The latent ``z`` has a standard normal
prior whatever the kernel, which makes it far easier to approximate than
``f`` itself, whose coordinates are strongly correlated.

To focus on the function, the kernel hyperparameters are fixed. With a
Gaussian likelihood the posterior of ``f`` is then Gaussian and known in
closed form, which gives an exact reference. Two guides over ``z`` are
compared:

* a **hand-written mean-field guide** (a ``Guide``): an independent Normal
  for every coordinate of ``z`` with log-scale standard deviations;
  ``guide.sample`` maps its draws through the model's ``LatentGP`` node;
* ``Guide(model, default_family="full_rank")``, a full-rank Normal over ``z``, which contains
  the exact posterior.

The figures show the ELBO of both guides, the posterior standard deviation
of ``f`` at the training inputs, and the posterior of the function on a grid
under both guides and the exact posterior. The full-rank guide reproduces the
exact posterior. The mean-field guide does not: the posterior of ``z`` is
correlated, and independent noise in ``z`` maps through the triangular
``L`` to the wrong pattern of uncertainty in ``f``. It is too confident at the
left end of the data and far too uncertain at the right end, and its ELBO
is about 8 nats lower.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Guide, Model, ObservedParameter, RandomParameter, VariationalParameter, plate
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import RBF, LatentGP, gp_predictive
from BayesianDLL.Variational import BBVI


torch.manual_seed(0)

n_observations = 18
lengthscale, variance, noise_std = 0.37, 1.0, 0.29
x = torch.linspace(0.0, 1.0, n_observations, dtype=torch.float64)
y = torch.sin(2 * torch.pi * x) + 0.08 * torch.randn_like(x)

with Model() as model:
    inputs = Data("inputs", x)
    f = LatentGP("f", inputs, RBF(lengthscale, variance))  # creates the latent "f_white" ~ N(0, I)
    with plate("observations", x):
        ObservedParameter("y", Normal(f, noise_std), y)

with Guide(model) as guide:
    z_loc = VariationalParameter("z_loc", torch.zeros(n_observations, dtype=torch.float64))
    z_scale = VariationalParameter("z_scale", torch.ones(n_observations, dtype=torch.float64), log_scale=True)
    RandomParameter("f_white", Normal(z_loc, z_scale), shape=(n_observations,))

# %%
# Fitting both guides
# -------------------

history_mean_field = BBVI(model, guide, n_samples=4, epochs=2000, lr=0.02, lrd=0.999, callback_frequency=10, verbose=False)
mean_field = guide.sample(1000)

full_rank_guide = Guide(model, default_family="full_rank")
history_full_rank = BBVI(model, full_rank_guide, n_samples=4, epochs=2000, lr=0.02, lrd=0.999, callback_frequency=10, verbose=False)
full_rank = full_rank_guide.sample(1000)

for label, history in (("mean-field", history_mean_field), ("full-rank", history_full_rank)):
    print(f"{label:>10} guide: final ELBO estimate {sum(history[-20:]) / 20:.2f}")

# %%
# Exact posterior of f
# --------------------
# For fixed hyperparameters, ``f(x*) | y`` is Normal with mean
# ``K*^T (K + s^2 I)^-1 y`` and covariance ``K** - K*^T (K + s^2 I)^-1 K*``.

kernel = RBF(lengthscale, variance)
x_new = torch.linspace(-0.2, 1.2, 120, dtype=torch.float64)
K = kernel(x, x) + noise_std ** 2 * torch.eye(n_observations, dtype=torch.float64)
factor = torch.linalg.cholesky(K)


def exact_posterior(points):
    cross = kernel(x, points)
    mean = cross.T @ torch.cholesky_solve(y.unsqueeze(-1), factor).squeeze(-1)
    covariance = kernel(points, points) - cross.T @ torch.cholesky_solve(cross, factor)
    return mean, covariance.diagonal().clamp_min(0).sqrt()


exact_mean, exact_sd = exact_posterior(x_new)
_, exact_sd_train = exact_posterior(x)

# %%
# Comparison
# ----------
# ``gp_predictive`` extends each posterior draw of ``f`` at the training
# inputs to the grid by sampling from the GP conditional.

traces = {"mean-field guide": mean_field, "full-rank guide": full_rank}
colors = {"mean-field guide": "C1", "full-rank guide": "C0"}

fig = plt.figure(figsize=(13, 8.5), layout="constrained")
upper, lower_ax = fig.subfigures(2, 1, height_ratios=[1, 1.15])
history_ax, sd_ax = upper.subplots(1, 2)
steps = range(10, 10 * len(history_mean_field) + 1, 10)
history_ax.plot(steps, history_mean_field, color=colors["mean-field guide"], alpha=0.8, label="mean-field guide")
history_ax.plot(steps, history_full_rank, color=colors["full-rank guide"], alpha=0.8, label="full-rank guide")
history_ax.set_ylim(history_full_rank[-1] - 40, history_full_rank[-1] + 10)
history_ax.set_xlabel("epoch")
history_ax.set_ylabel("ELBO estimate")
history_ax.set_title("Optimisation")
history_ax.legend(fontsize="small")

sd_ax.plot(x, exact_sd_train, color="black", lw=3, alpha=0.4, label="exact")
for label, trace in traces.items():
    sd_ax.plot(x, trace["f"].reshape(-1, n_observations).std(0), "o--", color=colors[label], ms=4, label=label)
sd_ax.set_ylim(0, None)
sd_ax.set_xlabel("x")
sd_ax.set_ylabel("posterior sd of f(x)")
sd_ax.set_title("Uncertainty at the training inputs")
sd_ax.legend(fontsize="small")

ax = lower_ax.subplots()
ax.fill_between(x_new, exact_mean - 1.96 * exact_sd, exact_mean + 1.96 * exact_sd, color="grey", alpha=0.25, label="exact 95% band")
ax.plot(x_new, exact_mean, color="black", label="exact mean")
for label, trace in traces.items():
    draws = gp_predictive(model.deterministic_params["f"], trace, x_new).reshape(-1, x_new.numel())
    lower, upper_band = draws.quantile(torch.tensor([0.025, 0.975], dtype=torch.float64), dim=0)
    ax.plot(x_new, lower, color=colors[label], linestyle="--", lw=1.2, label=f"{label} 95% band")
    ax.plot(x_new, upper_band, color=colors[label], linestyle="--", lw=1.2)
ax.scatter(x, y, color="C3", s=20, zorder=3, label="data")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Posterior of the latent function")
ax.legend(fontsize="small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.show()
