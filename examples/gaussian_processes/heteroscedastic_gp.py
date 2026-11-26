r"""
Known Measurement Errors with Per-Observation Noise
===================================================

When every reading comes with its own standard error, the noise is not one
number. ``ExactGP`` accepts a ``noise_variance`` with one value per observation
and evaluates

.. math::

    \mathbf{y} \sim \mathcal{N}\big(m(X),\ K + \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)\big),

so precise readings pull the function towards them and imprecise ones are
discounted. Here a quantity is measured 60 times, and the instrument is
unreliable between ``x = 4`` and ``x = 7``: the reported standard errors
are large there. The model that uses these known errors is compared with
one that assumes a single noise level shared by all readings.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import ExactGP, Matern52


torch.manual_seed(2)


def true_function(x):
    return torch.sin(1.2 * x) + 0.4 * torch.cos(2.7 * x)


def standard_error(x):
    return 0.08 + 0.8 * torch.exp(-0.5 * ((x - 5.5) / 0.8) ** 2)


n = 60
x = torch.sort(torch.rand(n, dtype=torch.float64) * 10.0).values
sd = standard_error(x)
y = true_function(x) + sd * torch.randn_like(x)

# %%
# Two models
# ----------
# The kernel hyperparameters get the same priors in both models. The
# heteroscedastic model passes the reported variances as a ``Data`` node, so
# they are fixed; the shared model learns a single noise variance.

def build(heteroscedastic):
    with Model() as model:
        inputs = Data("x", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
        if heteroscedastic:
            noise_variance = Data("noise_variance", sd.square())
        else:
            log_noise = RandomParameter("log_noise", Normal(-2.0, 1.5))
            noise_variance = Exp("noise_variance", log_noise)
        kernel = Matern52(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
        f = ExactGP("f", inputs, kernel, noise_variance=noise_variance)
        ObservedParameter("y", f, y)
    return model


models = {"known": build(True), "shared": build(False)}

# %%
# Type-II MAP
# -----------
# Only the kernel hyperparameters (and for the shared model the noise level)
# are optimised.

results = {}
for name, model in models.items():
    start = time.perf_counter()
    results[name] = model.find_MAP(lr=0.05, epochs=200, verbose=False)
    elapsed = time.perf_counter() - start
    message = (f"{name}: MAP in {elapsed:.1f} s, lengthscale {model.deterministic_params['lengthscale'].constrained_value.item():.2f}, "
               f"variance {model.deterministic_params['variance'].constrained_value.item():.2f}")
    if name == "shared":
        message += f", noise sd {model.deterministic_params['noise_variance'].constrained_value.item() ** 0.5:.3f}"
    print(message)
print(f"reported standard errors range from {sd.min():.2f} to {sd.max():.2f}")

# %%
# Prediction at new inputs
# ------------------------
# The function is predicted on a dense grid. With per-observation noise, the noise
# variance is a ``Data`` node of length ``n``, so it needs a value at the new inputs
# too: ``Data.set_value`` updates it together with the inputs. The function
# itself does not depend on it, but the shapes must agree. The grid's noise
# is the error an instrument would report there.

grid = torch.linspace(0.0, 10.0, 300, dtype=torch.float64)
bands = {}
for name, model in models.items():
    model.data["x"].set_value(grid)
    if name == "known":
        model.data["noise_variance"].set_value(standard_error(grid).square())
    draws = model.predict(results[name].mode(), samples_per_step=300)["f"].reshape(-1, grid.numel())
    bands[name] = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)

truth = true_function(grid)
noisy = (grid > 4.0) & (grid < 7.0)
for name, (lower, median, upper) in bands.items():
    covered = ((truth >= lower) & (truth <= upper)).double()
    rmse = (median - truth).square().mean().sqrt()
    print(f"{name}: RMSE {rmse:.3f}, 90% band covers the truth at {covered.mean():.0%} of the grid "
          f"({covered[noisy].mean():.0%} in the noisy window), mean band width {(upper - lower).mean():.2f} "
          f"({(upper - lower)[noisy].mean():.2f} in the noisy window)")

# %%
# Comparison
# ----------
# The shared model has to compromise on one noise level. In the unreliable
# window it takes the scattered readings at face value, so its band is
# narrower than the known-error band there and often misses the true function,
# while the known-error model widens the band where the readings are poor.
# The printed coverage and band widths quantify this.

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, layout="constrained")
for ax, (name, title, color) in zip(axes, (("known", "Known per-observation errors", "tab:blue"), ("shared", "One shared noise level", "tab:red"))):
    lower, median, upper = bands[name]
    ax.fill_between(grid, lower, upper, color=color, alpha=0.25, label="90% band of f")
    ax.plot(grid, median, color=color, label="median of f")
    ax.plot(grid, truth, "k--", linewidth=1, label="true f")
    ax.errorbar(x, y, yerr=1.645 * sd, fmt="o", color="black", markersize=3, elinewidth=0.8, capsize=0, zorder=3, label="observations (90% error bars)")
    ax.set(xlabel="x", title=title)
axes[0].set_ylabel("y")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize="small")
plt.show()
