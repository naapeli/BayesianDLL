r"""
Censored Regression with a Detection Limit
==========================================

A concentration depends linearly on a covariate, but the instrument cannot
report values below a detection limit :math:`a`: such readings are recorded as
:math:`a` and only tell us that the true value is at most :math:`a`. With a
Gaussian model :math:`y_i \sim \mathrm{Normal}(\alpha + \beta x_i, \sigma)` the
likelihood of reading :math:`i` is

.. math::

    p(y_i) = \begin{cases}
        \Phi\!\left(\dfrac{a - \mu_i}{\sigma}\right) & y_i = a, \\
        \mathrm{Normal}(y_i \mid \mu_i, \sigma) & y_i > a,
    \end{cases}

where the first line is the probability of falling below the limit.
``Censored(Normal(mu, sigma), low=a)`` is exactly this likelihood, with
gradients obtained from the distribution function of the base. We compare it
with the naive model that treats the recorded limit as a real measurement.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import Censored, HalfNormal, Normal

torch.manual_seed(5)

n_obs = 300
alpha_true, beta_true, sigma_true = -1.0, 1.5, 1.2
limit = 0.0
x = 3.0 * torch.rand(n_obs, dtype=torch.float64)
latent = alpha_true + beta_true * x + sigma_true * torch.randn(n_obs, dtype=torch.float64)
y = latent.clamp(min=limit)  # readings below the limit are recorded as the limit
censored = y == limit
x_mean = x.mean()
truth_alpha = alpha_true + beta_true * x_mean
print(f"censored readings: {int(censored.sum())} of {n_obs} ({censored.double().mean():.0%})")

# %%
# Two models
# ----------
# Both models share the priors and the linear predictor. They differ only in
# the likelihood: ``Censored`` knows that the values recorded at the limit are
# bounds, ``Normal`` takes them at face value. The covariate is centred, so
# ``alpha`` is the mean of the latent value at the average :math:`x` and is
# almost uncorrelated with ``beta``.


def build(censoring):
    with Model() as model:
        inputs = Data("inputs", x - x_mean)
        alpha = RandomParameter("alpha", Normal(0.0, 5.0))
        beta = RandomParameter("beta", Normal(0.0, 5.0))
        sigma = RandomParameter("sigma", HalfNormal(3.0))
        mu = Linear("mu", inputs, coefficients=beta, intercept=alpha)
        likelihood = Censored(Normal(mu, sigma), low=limit) if censoring else Normal(mu, sigma)
        with plate("observations", inputs):
            ObservedParameter("y", likelihood, y)
    return model


fits = {}
for label, censoring in (("censored", True), ("naive", False)):
    start = time.perf_counter()
    fits[label] = build(censoring).sample(300, 250, n_chains=4, seed=2, progress_bar=False, chain_method="vectorized")
    elapsed = time.perf_counter() - start
    summary = fits[label].summary()
    print(f"\n{label} model")
    print(summary.to_string())
    print(f"sampling: {elapsed:.1f} s, divergences: {sum(fits[label].divergences)}, max R-hat: {summary['r_hat'].max():.3f}")
print(f"\ntrue values: alpha = {truth_alpha:.3f} (latent mean at the average x), beta = {beta_true}, sigma = {sigma_true}")

# %%
# Posterior of the parameters
# ---------------------------
# The naive fit is confidently wrong: the recorded limits pull the line up at
# small :math:`x`, so the slope is too small, the intercept (here at the mean of
# :math:`x`) too large and the noise too small, with the truth about three to five
# posterior standard deviations away. The censored model is within two posterior standard
# deviations of all three.

truth = {"alpha": truth_alpha, "beta": beta_true, "sigma": sigma_true}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
for ax, name in zip(axes, truth):
    for label, color in (("censored", "C0"), ("naive", "C1")):
        ax.hist(fits[label][name].reshape(-1), bins=40, density=True, color=color, alpha=0.6, label=label)
    ax.axvline(truth[name], color="black", linestyle="--", label="truth")
    ax.set(xlabel=name, yticks=[])
axes[0].legend(frameon=False, fontsize="small")
plt.show()

# %%
# Fitted lines
# ------------
# The naive line flattens where the data pile up at the limit, while the
# censored line follows the latent trend through the censored region.

x_grid = torch.linspace(0.0, 3.0, 100, dtype=torch.float64)
fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
ax.scatter(x[~censored], y[~censored], s=10, color="black", label="measured")
ax.scatter(x[censored], y[censored], s=10, color="C3", label="recorded at the limit")
for label, color in (("censored", "C0"), ("naive", "C1")):
    lines = fits[label]["alpha"].reshape(-1, 1) + fits[label]["beta"].reshape(-1, 1) * (x_grid - x_mean)
    ax.fill_between(x_grid, lines.quantile(0.03, 0), lines.quantile(0.97, 0), color=color, alpha=0.3, linewidth=0)
    ax.plot(x_grid, lines.mean(0), color=color, label=f"{label} fit")
ax.plot(x_grid, alpha_true + beta_true * x_grid, color="black", linestyle="--", label="true line")
ax.axhline(limit, color="0.5", linewidth=0.8)
ax.set(xlabel="x", ylabel="y")
ax.legend(frameon=False, fontsize="small")
plt.show()
