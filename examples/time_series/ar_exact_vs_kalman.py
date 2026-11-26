r"""
AR(p): Exact Density versus Kalman Filter
=========================================

The same autoregressive model can be fitted in two ways:

* :class:`~BayesianDLL.TimeSeries.AR` evaluates the exact joint density of the series directly, in
  :math:`O(T p^2)` operations, using the stationary distribution of the first :math:`p` values.
* :class:`~BayesianDLL.TimeSeries.SARIMAX` with only ``ar`` writes the same process as a state-space
  model and evaluates the likelihood with the Kalman filter, which is a general tool (it also handles
  moving-average terms, differencing, seasonality and regressors) but costs more per step.

Both are the exact likelihood of the same model, so the posteriors must agree up to Monte Carlo error.
This example fits an AR(2) process both ways with NUTS (the coefficients through
:class:`~BayesianDLL.Deterministic.StationaryAR`, which keeps every draw stationary), compares the
posteriors and compares the run times. Use ``AR`` when the model is a pure autoregression and
``SARIMAX`` when you need its extra features.
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import StationaryAR
from BayesianDLL.Distributions import HalfNormal, Normal, Uniform
from BayesianDLL.Evaluation import summary
from BayesianDLL.TimeSeries import AR, SARIMAX

torch.manual_seed(3)
dtype = torch.float64

# %%
# Simulated AR(2) series
# ----------------------
n_steps = 80
phi_true = torch.tensor([0.6, -0.3], dtype=dtype)
sigma_true, mu_true = 0.8, 1.5
y = AR(phi_true, sigma_true, mu_true, n_steps=n_steps).sample(1)[0]


# %%
# The same model with two likelihoods
# -----------------------------------
# Only the distribution of ``y`` differs: ``AR(phi, sigma, mu)`` against ``SARIMAX(phi, sigma=sigma, mu=mu)``.
def build(likelihood):
    with Model() as model:
        pacf = RandomParameter("pacf", Uniform(-1.0, 1.0), shape=2)
        phi = StationaryAR("phi", pacf)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        ObservedParameter("y", likelihood(phi, sigma, mu), y)
    return model


likelihoods = {
    "AR (exact density)": lambda phi, sigma, mu: AR(phi, sigma, mu),
    "SARIMAX (Kalman filter)": lambda phi, sigma, mu: SARIMAX(phi, sigma=sigma, mu=mu),
}
results, seconds = {}, {}
for name, likelihood in likelihoods.items():
    model = build(likelihood)
    start = time.perf_counter()
    results[name] = model.sample(200, 200, n_chains=4, seed=1, chain_method="vectorized", progress_bar=False)
    seconds[name] = time.perf_counter() - start
    print(f"{name}: divergences = {sum(results[name].divergences)}, {seconds[name]:.1f} s")

# %%
# Posteriors agree, run times do not
# ----------------------------------
names = list(likelihoods)
rows = []
for name in names:
    table = summary({"phi": results[name]["phi"], "sigma": results[name]["sigma"], "mu": results[name]["mu"]})
    table.insert(0, "model", name)
    rows.append(table)
print(pd.concat(rows).to_string())
print(f"run time of SARIMAX relative to AR: {seconds[names[1]] / seconds[names[0]]:.1f}x")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
truths = [phi_true[0], phi_true[1], mu_true]
labels = [r"$\phi_1$", r"$\phi_2$", r"$\mu$"]
for i, (ax, truth, label) in enumerate(zip(axes, truths, labels)):
    for name, color in zip(names, ("C0", "C1")):
        r = results[name]
        draws = (r["phi"][..., i] if i < 2 else r["mu"]).reshape(-1)
        ax.hist(draws, bins=40, density=True, alpha=0.5, color=color, label=name)
    ax.axvline(float(truth), color="black", linestyle="--", linewidth=1)
    ax.set_title(label)
axes[0].legend(frameon=False, fontsize="small")
plt.show()
