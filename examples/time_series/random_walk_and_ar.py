r"""
Random Walk and Autoregressive Priors over a Series
===================================================

:class:`~BayesianDLL.TimeSeries.GaussianRandomWalk` and
:class:`~BayesianDLL.TimeSeries.AutoRegressive` are distributions over a whole
series, so a latent series ``x`` is a single parameter,
``RandomParameter("x", GaussianRandomWalk(sigma, T=T))``, whose joint density
and gradient are exact. Their parameters (here the step scale) are
learned together with the series.

* A *random walk*, :math:`x_t \sim \mathcal{N}(x_{t-1}, \sigma^2)`, is a prior for a
  smooth latent trend. Here it is the log-rate of Poisson counts, so the
  observations are not Gaussian and the series cannot be filtered exactly.
* An *AR(2) process*, :math:`x_t - \mu = \phi_1 (x_{t-1} - \mu) + \phi_2 (x_{t-2} - \mu) + \sigma \varepsilon_t`,
  is used as the exact likelihood of an observed series. Its first two values
  come from the stationary distribution (``init="stationary"``), so the
  density is proper and could equally serve as a prior. The coefficients are
  parameterised through partial autocorrelations by
  :class:`~BayesianDLL.TimeSeries.StationaryAR`, which keeps every draw
  stationary.

The latent random-walk rate is recovered with a 90% band; for the AR(2) series
the posterior of the coefficients is shown inside the stationary triangle.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import HalfNormal, Normal, Poisson, Uniform
from BayesianDLL.Evaluation import summary
from BayesianDLL.TimeSeries import AutoRegressive, GaussianRandomWalk, StationaryAR

torch.manual_seed(3)
dtype = torch.float64

# %%
# Poisson counts with a random-walk log-rate
# ------------------------------------------
# The true log-rate is a smooth curve and the counts are Poisson around its
# exponential. The random walk only encodes that neighbouring values are close;
# the data decide the shape.
T = 60
t = torch.arange(T, dtype=dtype)
log_rate_true = 3.5 + 0.8 * torch.sin(t / 18.0) + 0.3 * torch.sin(t / 5.0)
counts = torch.poisson(torch.exp(log_rate_true))

with Model() as walk_model:
    sigma = RandomParameter("sigma", HalfNormal(0.5))
    x = RandomParameter("x", GaussianRandomWalk(sigma, init_sigma=3.0, T=T))
    rate = Exp("rate", x)
    ObservedParameter("counts", Poisson(rate), counts)

walk = walk_model.sample(350, 350, n_chains=4, seed=1, chain_method="vectorized", progress_bar=False)
log_rate = walk["x"].reshape(-1, T)
inside = ((log_rate_true > log_rate.quantile(0.05, 0)) & (log_rate_true < log_rate.quantile(0.95, 0))).double().mean()
print(summary({"sigma": walk["sigma"]}).to_string())
print(f"posterior mean of the step scale sigma: {walk['sigma'].mean():.3f}")
print(f"root-mean-square error of the log-rate: {((log_rate.mean(0) - log_rate_true) ** 2).mean().sqrt():.3f}; "
      f"true log-rate inside the 90% band at {100 * inside:.0f}% of the times")

# %%
# An AR(2) series with a stationary start
# ---------------------------------------
# A quasi-periodic AR(2) series is observed directly, so the series itself is
# the value of the ``AutoRegressive`` distribution and the likelihood is the
# exact joint density of all ``T`` values, including the stationary
# distribution of the first two. The coefficients, the innovation scale and the
# mean are inferred.
T_ar = 150
phi_true = torch.tensor([1.2, -0.6], dtype=dtype)
sigma_true, mu_true = 0.6, 1.0
y = AutoRegressive(phi_true, sigma_true, mu_true, T=T_ar).sample(1)[0]

with Model() as ar_model:
    pacf = RandomParameter("pacf", Uniform(-1.0, 1.0), shape=2)
    phi = StationaryAR("phi", pacf)
    sigma = RandomParameter("sigma", HalfNormal(1.0))
    mu = RandomParameter("mu", Normal(0.0, 3.0))
    ObservedParameter("y", AutoRegressive(phi, sigma, mu), y)

ar = ar_model.sample(300, 300, n_chains=4, seed=2, chain_method="vectorized", progress_bar=False)
print(summary({"phi": ar["phi"], "sigma": ar["sigma"], "mu": ar["mu"]}).to_string())
print(f"true values: phi = {phi_true.tolist()}, sigma = {sigma_true}, mu = {mu_true}")

# %%
# Recovered series and parameters
# -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")

ax = axes[0]
ax.scatter(t, counts, s=10, color="gray", label="counts")
rate_draws = torch.exp(log_rate)
ax.fill_between(t, rate_draws.quantile(0.05, 0), rate_draws.quantile(0.95, 0), color="C0", alpha=0.3, linewidth=0)
ax.plot(t, rate_draws.mean(0), color="C0", label="posterior mean rate (90% band)")
ax.plot(t, torch.exp(log_rate_true), "k--", label="true rate")
ax.set_xlabel("t")
ax.set_title("Random-walk log-rate under Poisson counts")
ax.legend(frameon=False, fontsize="small")

ax = axes[1]
ax.plot(y, color="C1", linewidth=1)
ax.axhline(mu_true, color="black", linestyle="--", linewidth=1, label=r"true mean $\mu$")
ax.set_xlabel("t")
ax.set_title("Simulated AR(2) series")
ax.legend(frameon=False, fontsize="small")

ax = axes[2]
phi_draws = ar["phi"].reshape(-1, 2)
ax.fill([-2, 2, 0], [-1, -1, 1], color="gray", alpha=0.15, linewidth=0, label="stationary region")
ax.scatter(phi_draws[::4, 0], phi_draws[::4, 1], s=4, alpha=0.4, color="C1", label="posterior draws")
ax.scatter(*phi_true, color="black", marker="x", s=60, label="true value")
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 1)
ax.set_xlabel(r"$\phi_1$")
ax.set_ylabel(r"$\phi_2$")
ax.set_title("Posterior of the AR coefficients")
ax.legend(frameon=False, fontsize="small", loc="lower left")
plt.show()
