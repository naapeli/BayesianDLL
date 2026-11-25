r"""
Locating a Change in Behaviour with a Change-Point Kernel
=========================================================

A sensor signal is smooth for the first part of the record and then starts
to oscillate quickly. A single stationary kernel cannot describe both
regimes: it either over-smooths the oscillation or fits noise in the smooth
part. The change-point kernel switches between two kernels at a location
:math:`c`:

.. math::

    k(x, x') = \bar{s}(x)\bar{s}(x') k_1(x, x') + s(x)s(x') k_2(x, x'),
    \quad s(x) = \sigma\big(a (x - c)\big), \quad \bar{s} = 1 - s,

with the logistic sigmoid :math:`\sigma`. The kernel :math:`k_1` describes
the function before :math:`c` and :math:`k_2` after it, and the steepness
:math:`a` sets how abrupt the switch is. Because the location is a
parameter, it has a posterior, which NUTS samples together with the
hyperparameters of both regimes.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import RBF, ChangePoint, ExactGP


torch.manual_seed(0)

true_change = 6.0


def true_function(x):
    return torch.sin(0.5 * x) + 0.8 * torch.sin(5.0 * x) * torch.sigmoid(5.0 * (x - true_change))


n = 80
x = torch.sort(torch.rand(n, dtype=torch.float64) * 10.0).values
y = true_function(x) + 0.15 * torch.randn_like(x)

# %%
# The model
# ---------
# Both regimes use an RBF kernel with their own lengthscale and variance, and
# the switch has a fixed steepness of 5: the weight of the second kernel rises
# from 10% to 90% within about 0.9 time units around the location. The prior
# on the location is wide, ``Normal(5, 2)``. The lengthscale priors are broad
# (lognormal with medians 1.6 before and 1 after the change), so the data
# decides which regime is the smooth one. The variance priors are
# centred on the unit signal scale and fairly tight (standard deviation 0.5
# on the log scale): looser priors left the smooth regime weakly identified
# (lengthscale and variance trade off) and slowed NUTS down.

with Model() as model:
    inputs = Data("x", x)
    log_lengthscale_before = RandomParameter("log_lengthscale_before", Normal(0.5, 1.0))
    log_variance_before = RandomParameter("log_variance_before", Normal(0.0, 0.5))
    log_lengthscale_after = RandomParameter("log_lengthscale_after", Normal(0.0, 1.0))
    log_variance_after = RandomParameter("log_variance_after", Normal(0.0, 0.5))
    location = RandomParameter("location", Normal(5.0, 2.0))
    log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0))

    kernel = ChangePoint(
        RBF(Exp("lengthscale_before", log_lengthscale_before), Exp("variance_before", log_variance_before)),
        RBF(Exp("lengthscale_after", log_lengthscale_after), Exp("variance_after", log_variance_after)),
        location,
        steepness=5.0,
    )
    f = ExactGP("f", inputs, kernel, noise_variance=Exp("noise", log_noise))
    ObservedParameter("y", f, y)

# %%
# Sampling
# --------
# Type-II MAP provides a starting point near a mode, and four chains of
# NUTS then sample the joint posterior.

model.find_MAP(lr=0.05, epochs=200, verbose=False)
start = time.perf_counter()
trace = model.sample(200, 150, n_chains=4, init_scale=0.1, seed=0, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary(include_deterministic=True)
print(summary)
print(f"sampling: {elapsed:.1f} s, divergences: {sum(trace.divergences)}, max R-hat: {summary['r_hat'].max():.3f}")

change = trace["location"].reshape(-1)
low, high = torch.quantile(change, torch.tensor([0.05, 0.95], dtype=change.dtype))
print(f"change point: median {change.median():.2f}, 90% interval [{low:.2f}, {high:.2f}] (true {true_change})")

# %%
# Posterior predictive fit
# ------------------------
# The right panels show the posterior of the change location and of the
# lengthscale of each regime: the regime before the change is far smoother
# than the one after it.

grid = torch.linspace(0.0, 10.0, 400, dtype=torch.float64)
inputs.set_value(grid)
draws = model.posterior_predicative(trace.thin(10), samples_per_step=5, progress_bar=False)["f"].reshape(-1, grid.numel())
lower, median, upper = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)

fig = plt.figure(figsize=(12, 5), layout="constrained")
grid_spec = fig.add_gridspec(2, 3)
ax = fig.add_subplot(grid_spec[:, :2])
ax.fill_between(grid, lower, upper, color="tab:blue", alpha=0.25, label="90% band of f")
ax.plot(grid, median, color="tab:blue", label="median of f")
ax.plot(grid, true_function(grid), "k--", linewidth=1, label="true f")
ax.scatter(x, y, color="black", s=12, zorder=3, label="observations")
ax.axvline(true_change, color="tab:red", linestyle=":", label="true change")
ax.set(xlabel="x", ylabel="y", title="Change-point GP: posterior predictive")
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize="small")

ax_location = fig.add_subplot(grid_spec[0, 2])
ax_location.hist(change, bins=40, color="tab:orange", density=True)
ax_location.axvline(true_change, color="k", linestyle="--", label="truth")
ax_location.set(xlabel="change location", ylabel="density", title="Posterior of the change point")
ax_location.legend(loc="best", fontsize="small")

ax_length = fig.add_subplot(grid_spec[1, 2])
ax_length.hist(trace["lengthscale_before"].reshape(-1), bins=30, color="tab:green", alpha=0.7, density=True, label="before")
ax_length.hist(trace["lengthscale_after"].reshape(-1), bins=30, color="tab:purple", alpha=0.7, density=True, label="after")
ax_length.set(xlabel="lengthscale", ylabel="density", title="Lengthscale of each regime")
ax_length.legend(loc="best", fontsize="small")
plt.show()
