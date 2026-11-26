"""
ARMA(1, 1) with posterior forecasts
===================================

A series is simulated from an ARMA(1, 1) process (a :class:`~BayesianDLL.TimeSeries.SARIMAX`
with only ``ar`` and ``ma``), the coefficients are
inferred with NUTS and the series is forecast from the posterior.

The AR and MA coefficients are given through partial autocorrelations in
:math:`(-1, 1)` (:class:`~BayesianDLL.Deterministic.StationaryAR` and
:class:`~BayesianDLL.Deterministic.InvertibleMA`), which keeps every draw
stationary and invertible. The likelihood is the exact Gaussian likelihood of
the series, evaluated by the Kalman filter with the states integrated out.
``forecast`` simulates the continuation exactly given the observed history,
once for every posterior draw.
"""

import warnings

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, sample
from BayesianDLL.Distributions import HalfNormal, Uniform
from BayesianDLL.Deterministic import InvertibleMA, StationaryAR
from BayesianDLL.TimeSeries import SARIMAX

torch.manual_seed(0)
truth = dict(ar=0.7, ma=0.4, sigma=1.0)
y = SARIMAX(torch.tensor([truth["ar"]]), torch.tensor([truth["ma"]]), truth["sigma"], n_steps=50).sample(1)[0]

with Model() as model:
    r = RandomParameter("r", Uniform(-1, 1), shape=1)
    q = RandomParameter("q", Uniform(-1, 1), shape=1)
    sigma = RandomParameter("sigma", HalfNormal(2.0))
    phi = StationaryAR("phi", r)
    theta = InvertibleMA("theta", q)
    ObservedParameter("y", SARIMAX(phi, theta, sigma), y)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = sample(80, 80, n_chains=4, model=model, progress_bar=False, seed=1, chain_method="vectorized")

draws = {"phi": result["r"].reshape(-1, 1), "theta": -result["q"].reshape(-1, 1)}  # for a single coefficient the map is phi = r, theta = -q
posterior = SARIMAX(draws["phi"], draws["theta"], result["sigma"].reshape(-1))
forecast = posterior.forecast(y, 10, n_samples=1)[0]  # (draws, 10)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(range(50), y, color="black", label="series")
future = range(50, 60)
lo, med, hi = torch.quantile(forecast, torch.tensor([0.05, 0.5, 0.95], dtype=forecast.dtype), dim=0)
ax.plot(future, med, color="C0", label="forecast median")
ax.fill_between(future, lo, hi, color="C0", alpha=0.3, label="90% interval")
ax.legend()
ax.set_xlabel("t")
plt.tight_layout()
plt.show()
