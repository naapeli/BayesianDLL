"""
Minibatch Variational Inference
===============================

With many observations every full-data ELBO gradient is expensive. Passing
``batch_size=B`` to ``BBVI`` turns it into stochastic variational inference:
each step evaluates the likelihood on ``B`` randomly chosen observations and
multiplies it by ``N / B``, which keeps the ELBO and its gradient unbiased.
The subsampled plate must take its size from a ``Data`` node
(``plate("observations", x_data)``) so that the covariate and the observed
values are sliced together.

The model is a linear regression with a known noise scale, ``N = 20 000``
observations and a standardised covariate,

.. math::

    a, b \\sim N(0, 5), \\quad y_i \\sim N(a + b x_i, 1).

Its posterior is Gaussian and known exactly, and with a standardised
covariate the two coefficients are nearly independent, so the mean-field
``AutoNormal`` guide can match it. The same guide is fitted twice, with
minibatches of 1000 and with the full data, and both are compared with the
exact posterior. The figures show the ELBO traces against the amount of data
processed,
the marginal posteriors of both coefficients (the exact posterior is very
narrow at this sample size), and the fitted line over a subset of the data.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import AutoNormal, Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import Normal
from BayesianDLL.Variational import BBVI


torch.manual_seed(0)

n_observations, batch_size, noise = 20_000, 1000, 1.0
x = torch.randn(n_observations, dtype=torch.float64)
y = 0.5 + 1.5 * x + noise * torch.randn(n_observations, dtype=torch.float64)

with Model() as model:
    intercept = RandomParameter("intercept", Normal(0.0, 5.0))
    slope = RandomParameter("slope", Normal(0.0, 5.0))
    x_data = Data("x", x)
    with plate("observations", x_data):
        line = Linear("line", x_data, coefficients=slope, intercept=intercept)
        ObservedParameter("y", Normal(line, noise), y)

# Exact Gaussian posterior of (intercept, slope): known noise and Normal priors.
design = torch.stack([torch.ones_like(x), x], dim=1)
precision = design.T @ design / noise ** 2 + torch.eye(2, dtype=x.dtype) / 5.0 ** 2
exact_cov = torch.linalg.inv(precision)
exact_mean = exact_cov @ (design.T @ y / noise ** 2)

# %%
# Minibatch and full-batch fits
# -----------------------------
# Both runs take the same number of steps; the full-batch run touches 20
# times more data per step. (For a model this small the per-step overhead
# dominates the run time, so the plot below measures the cost by the number
# of likelihood terms evaluated rather than by wall-clock time.)

runs = {}
for label, size in (("minibatch (B = 1000)", batch_size), ("full batch", None)):
    torch.manual_seed(1)
    guide = AutoNormal(model)
    start = time.perf_counter()
    history = BBVI(model, guide, n_samples=4, epochs=2500, lr=0.02, lrd=0.998, batch_size=size,
                   callback_frequency=10, verbose=False)
    seconds = time.perf_counter() - start
    runs[label] = (guide, history, seconds)
    means = [guide.locs[name].value.item() for name in ("intercept", "slope")]
    sds = [guide.scales[name].value.item() for name in ("intercept", "slope")]
    print(f"{label:>20}: {seconds:5.1f} s, mean {[round(v, 4) for v in means]}, sd {[round(v, 4) for v in sds]}")
print(f"{'exact':>20}:          mean {[round(v, 4) for v in exact_mean.tolist()]}, "
      f"sd {[round(v, 4) for v in exact_cov.diagonal().sqrt().tolist()]}")

# %%
# Comparison with the exact posterior
# -----------------------------------
# The minibatch run reaches the optimum after evaluating about 20 times fewer
# likelihood terms. Its ELBO estimates stay noisy, and its final means wander
# by a fraction of a posterior standard deviation, while the full-batch guide
# lies on top of the exact posterior. Both widths match the exact ones.

colors = {"minibatch (B = 1000)": "C0", "full batch": "C1"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
ax = axes[0, 0]
window = 20
for label, (guide, history, seconds) in runs.items():
    smoothed = torch.tensor(history).unfold(0, window, 1).mean(-1)  # moving average of 20 records
    per_record = 10 * (batch_size if label.startswith("minibatch") else n_observations) * 4  # 4 ELBO draws
    processed = per_record * torch.arange(1, len(history) + 1, dtype=torch.float64)
    ax.plot(processed[window - 1:], smoothed, color=colors[label], label=label)
ax.set_xscale("log")
final = runs["full batch"][1][-1]
ax.set_ylim(final - 3000, final + 500)
ax.set_xlabel("likelihood terms evaluated")
ax.set_ylabel("ELBO estimate (moving average)")
ax.set_title("Optimisation")
ax.legend(fontsize="small")

for ax, index, name in ((axes[0, 1], 0, "intercept"), (axes[1, 0], 1, "slope")):
    sd = exact_cov[index, index].sqrt()
    grid = torch.linspace(exact_mean[index] - 5 * sd, exact_mean[index] + 5 * sd, 300, dtype=torch.float64)
    ax.plot(grid, Normal(exact_mean[index], sd).pdf(grid).flatten(), color="black", lw=3, alpha=0.4, label="exact")
    for label, (guide, history, seconds) in runs.items():
        density = Normal(guide.locs[name].value, guide.scales[name].value).pdf(grid).flatten()
        ax.plot(grid, density, color=colors[label], linestyle="--", label=label)
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    ax.set_title(f"Posterior of the {name}")
    ax.legend(fontsize="small")

ax = axes[1, 1]
shown = torch.randperm(n_observations)[:400]
ax.scatter(x[shown], y[shown], s=6, color="grey", alpha=0.5, label="400 of the data points")
grid = torch.linspace(-3.5, 3.5, 100, dtype=torch.float64)
guide = runs["minibatch (B = 1000)"][0]
draws = guide.sample(2000)
lines = draws["intercept"].reshape(-1, 1) + draws["slope"].reshape(-1, 1) * grid
ax.plot(grid, lines.mean(0), color="C0", label="minibatch fit, mean line")
ax.fill_between(grid, lines.mean(0) - 1.96 * noise, lines.mean(0) + 1.96 * noise,
                color="C0", alpha=0.15, label="95% interval of new y")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Fitted line")
ax.legend(fontsize="small", loc="upper left")
plt.show()
