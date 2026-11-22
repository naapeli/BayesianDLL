r"""
Trend plus Seasonality with an Exact GP
=======================================

A seasonal signal on top of a linear trend is observed 40 times with
Gaussian noise. We model it as ``y = f(x) + ε`` with a periodic
Gaussian-process prior on ``f`` and a learned linear mean function:

.. math::

    m(x) = \text{slope} \cdot x + \text{intercept}, \quad k(x, x') = k_{\text{periodic}}(x, x').

The periodic kernel captures repeating oscillations while ``LinearMean``
models the linear trend. Because the mean continues outside the data,
predictions extrapolate the trend beyond the observations.

Because the likelihood is Gaussian, ``f`` can be integrated out
analytically: ``ExactGP`` evaluates ``y ~ Normal(mean, K + σ² I)`` directly,
so NUTS only has to explore the hyperparameters and mean coefficients.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import ExactGP, LinearMean, Periodic, exact_gp_predictive


torch.manual_seed(0)


def true_function(x):
    return 0.6 * (x - 2.0) + torch.sin(2 * torch.pi * x) + 0.4 * torch.cos(4 * torch.pi * x)


x = torch.sort(torch.rand(40, dtype=torch.float64) * 4.0).values
y = true_function(x) + 0.3 * torch.randn_like(x)

# %%
# The model
# ---------
# The prior on the log-period is centred on one time unit with a spread of
# roughly a factor 1.3, i.e. we know the season length only approximately.
# The periodic and noise variances get log-normal priors around 1 and 0.1, and
# the trend is captured by a ``LinearMean(slope, intercept)`` with weak priors.

with Model() as model:
    inputs = Data("inputs", x)
    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 0.7))
    log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
    log_period = RandomParameter("log_period", Normal(0.0, 0.25))
    log_noise = RandomParameter("log_noise", Normal(-2.3, 1.0))
    slope = RandomParameter("slope", Normal(0.6, 1.0))
    intercept = RandomParameter("intercept", Normal(0.0, 1.0))

    seasonal = Periodic(
        lengthscale=Exp("lengthscale", log_lengthscale),
        variance=Exp("variance", log_variance),
        period=Exp("period", log_period),
    )
    f = ExactGP(
        "f", inputs, seasonal, noise_variance=Exp("noise_variance", log_noise),
        mean=LinearMean(slope, intercept),
    )
    ObservedParameter("y", f, y)

# %%
# Sampling
# --------
# A periodic posterior has aliases: a function that repeats every time unit
# also repeats every two, so a period near 2 (with a shorter lengthscale) is a
# second, much less probable mode. A chain that starts there stays there, and
# the chains then disagree (R-hat far above 1). We therefore move the model to
# its MAP first and start the four chains close to it.

model.find_MAP(lr=0.05, epochs=300, verbose=False)
start = time.perf_counter()
trace = model.sample(500, 300, n_chains=4, init_scale=0.1, seed=0, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary(include_deterministic=True)
print(summary)
print(f"sampling: {elapsed:.1f} s, divergences: {sum(trace.divergences)}, max R-hat: {summary['r_hat'].max():.3f}")

# %%
# Posterior predictive fit
# ------------------------
# ``exact_gp_predictive`` conditions the GP on the observations for every
# posterior draw of the hyperparameters, so the band includes both the
# uncertainty about ``f`` and about the kernel. The right panels show what the
# data taught us about the period and the noise level.

grid = torch.linspace(-0.2, 5.5, 240, dtype=torch.float64)
draws = exact_gp_predictive(f, trace.thin(10), y, grid, n_samples=5).reshape(-1, grid.numel())
lower, median, upper = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)

fig = plt.figure(figsize=(12, 5), layout="constrained")
grid_spec = fig.add_gridspec(2, 3)
ax = fig.add_subplot(grid_spec[:, :2])
ax.fill_between(grid, lower, upper, color="tab:blue", alpha=0.25, label="90% band of f")
ax.plot(grid, median, color="tab:blue", label="median of f")
ax.plot(grid, true_function(grid), "k--", label="true f")
ax.scatter(x, y, color="black", s=15, zorder=3, label="observations")
ax.axvspan(4.0, 5.5, color="0.9", label="extrapolation")
ax.set(xlabel="x", ylabel="y", title="Exact GP with periodic kernel & LinearMean: posterior predictive")
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize="small")

ax_period = fig.add_subplot(grid_spec[0, 2])
ax_period.hist(trace["period"].reshape(-1), bins=30, color="tab:orange", density=True)
ax_period.axvline(1.0, color="k", linestyle="--", label="truth")
ax_period.set(xlabel="period", ylabel="density", title="Posterior of the period")
ax_period.legend(loc="best", fontsize="small")

ax_noise = fig.add_subplot(grid_spec[1, 2])
ax_noise.hist(trace["noise_variance"].reshape(-1).sqrt(), bins=30, color="tab:green", density=True)
ax_noise.axvline(0.3, color="k", linestyle="--", label="truth")
ax_noise.set(xlabel="noise standard deviation", ylabel="density", title="Posterior of the noise")
ax_noise.legend(loc="best", fontsize="small")
plt.show()
