r"""
Seasonal ARIMA with a Regressor
===============================

:class:`~BayesianDLL.TimeSeries.SARIMAX` models a series :math:`y_t` as a regression plus a seasonal
ARMA error. After :math:`d` ordinary and :math:`D` seasonal differences,

.. math::

    w_t = \delta + x_t^\top \beta + u_t, \qquad
    \phi(L)\,\Phi(L^s)\, u_t = \theta(L)\,\Theta(L^s)\, \sigma \varepsilon_t,

with seasonal period :math:`s`; :math:`\delta` is ``mu`` without differencing and ``drift`` with it. The
likelihood is the exact Gaussian likelihood from the Kalman filter. There is only one class: MA, ARMA, ARIMA and
SARIMA models are the same :class:`~BayesianDLL.TimeSeries.SARIMAX` with the unused arguments left out.

A quarterly series (:math:`s = 4`) is generated from a regression on a known
regressor plus a seasonal AR(1) error. Four models are fitted to the first 80
quarters and compared on the next 12: an MA(2), an ARIMA(1, 1, 0), a
SARIMA(1, 0, 0)(1, 0, 0)\ :sub:`4` and the SARIMAX with the regressor. The
AR and MA coefficients go through partial autocorrelations
(:class:`~BayesianDLL.Deterministic.StationaryAR`,
:class:`~BayesianDLL.Deterministic.InvertibleMA`), so every draw is stationary
and invertible. The likelihood is one joint density of the whole series,
with no pointwise terms for PSIS-LOO to leave out, so the models are compared
by their forecasts of the held-out quarters. All four are fitted quickly by MAP estimation, and the SARIMAX is then
sampled with NUTS for a forecast that includes parameter uncertainty.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import InvertibleMA, StationaryAR
from BayesianDLL.Distributions import HalfNormal, Normal, Uniform
from BayesianDLL.Evaluation import summary
from BayesianDLL.TimeSeries import SARIMAX

torch.manual_seed(11)
dtype = torch.float64

# %%
# Simulated quarterly series
# --------------------------
n_train, horizon, period = 80, 12, 4
T = n_train + horizon
truth = dict(mu=2.0, beta=1.2, phi=0.4, Phi=0.6, sigma=0.5)
tt = torch.arange(T, dtype=dtype)
x = (1.5 * torch.sin(2 * torch.pi * tt / 40) + torch.randn(T, dtype=dtype)).unsqueeze(-1)
error = SARIMAX(torch.tensor([truth["phi"]], dtype=dtype), sigma=truth["sigma"],
                seasonal_ar=torch.tensor([truth["Phi"]], dtype=dtype), period=period, n_steps=T).sample(1)[0]
y_all = truth["mu"] + truth["beta"] * x[:, 0] + error
y, x_train, x_future = y_all[:n_train], x[:n_train], x[n_train:]
y_future = y_all[n_train:]

# %%
# Four models by MAP
# ------------------
# Each builder returns the model and a function that turns the parameters into
# the distribution used for forecasting. Only the SARIMAX uses the regressor.
# The four models are first fitted by maximum a posteriori (MAP) estimation,
# which is quick, and forecast with the parameters held at the mode; the
# forecast still simulates the future innovations exactly given the history.
def coefficient(name, block, size):
    return block(name, RandomParameter(name + "_pacf", Uniform(-1.0, 1.0), shape=size))


def ma2():
    with Model() as model:
        theta = coefficient("theta", InvertibleMA, 2)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        ObservedParameter("y", SARIMAX(ma=theta, sigma=sigma, mu=mu), y)
    return model, lambda d: SARIMAX(ma=d["theta"], sigma=d["sigma"], mu=d["mu"])


def arima110():
    with Model() as model:
        phi = coefficient("phi", StationaryAR, 1)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        drift = RandomParameter("drift", Normal(0.0, 1.0))  # constant of the differenced series
        ObservedParameter("y", SARIMAX(phi, sigma=sigma, difference=1, drift=drift), y)
    return model, lambda d: SARIMAX(d["phi"], sigma=d["sigma"], difference=1, drift=d["drift"])


def sarima():
    with Model() as model:
        phi = coefficient("phi", StationaryAR, 1)
        Phi = coefficient("Phi", StationaryAR, 1)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        ObservedParameter("y", SARIMAX(phi, sigma=sigma, mu=mu, seasonal_ar=Phi, period=period), y)
    return model, lambda d: SARIMAX(d["phi"], sigma=d["sigma"], mu=d["mu"], seasonal_ar=d["Phi"], period=period)


def sarimax():
    with Model() as model:
        phi = coefficient("phi", StationaryAR, 1)
        Phi = coefficient("Phi", StationaryAR, 1)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        beta = RandomParameter("beta", Normal(0.0, 3.0), shape=1)
        ObservedParameter("y", SARIMAX(phi, sigma=sigma, mu=mu, seasonal_ar=Phi, beta=beta, exog=x_train, period=period), y)
    return model, lambda d: SARIMAX(d["phi"], sigma=d["sigma"], mu=d["mu"], seasonal_ar=d["Phi"], beta=d["beta"], exog=x_train, period=period)


builders = {"MA(2)": ma2, "ARIMA(1,1,0)": arima110, "SARIMA": sarima, "SARIMAX": sarimax}
forecasts, models = {}, {}
for name, build in builders.items():
    model, forecaster = build()
    models[name] = (model, forecaster)
    mode = model.find_MAP(lr=0.1, epochs=300, verbose=False).mode()
    kwargs = {"future_exog": x_future} if name == "SARIMAX" else {}
    forecasts[name] = forecaster(mode).forecast(y, horizon, n_samples=500, **kwargs).reshape(-1, horizon)  # (500, horizon)
    print(f"{name:<13} MAP: " + ", ".join(f"{key} = {[round(v, 2) for v in value.flatten().tolist()]}" for key, value in mode.items() if not key.endswith("pacf")))

# %%
# Full posterior of the SARIMAX
# -----------------------------
# The SARIMAX is then sampled with NUTS, and the forecast is simulated once
# per posterior draw, so the band also carries the parameter uncertainty.
model, forecaster = models["SARIMAX"]
result = model.sample(100, 100, n_chains=4, seed=1, chain_method="vectorized", progress_bar=False)
print(summary({"phi": result["phi"], "Phi": result["Phi"], "sigma": result["sigma"], "mu": result["mu"], "beta": result["beta"]}).to_string())
print(f"true values: {truth}")
draws = {"phi": result["phi"].reshape(-1, 1), "Phi": result["Phi"].reshape(-1, 1), "sigma": result["sigma"].reshape(-1),
         "mu": result["mu"].reshape(-1), "beta": result["beta"].reshape(-1, 1)}
forecasts["SARIMAX (NUTS)"] = forecaster(draws).forecast(y, horizon, n_samples=1, future_exog=x_future)[0]

# %%
# Forecast comparison
# -------------------
# The RMSE is that of the forecast median over the 12 held-out quarters, the
# coverage the fraction of them inside the 90% band. With only 12 correlated
# values the coverage is a rough guide. The regressor varies from quarter to
# quarter, so the models without it cannot anticipate its effect, while the
# SARIMAX forecasts with its known future values.
print(f"\n{'model':<16}{'RMSE':>8}{'90% coverage':>14}")
quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=dtype)
for name, f in forecasts.items():
    lo, med, hi = f.quantile(quantiles, dim=0)
    rmse = ((med - y_future) ** 2).mean().sqrt()
    coverage = ((y_future > lo) & (y_future < hi)).double().mean()
    print(f"{name:<16}{rmse:>8.3f}{coverage:>14.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
future_t = torch.arange(n_train, T)
for ax, name, color in zip(axes, ("SARIMA", "SARIMAX (NUTS)"), ("C1", "C0")):
    lo, med, hi = forecasts[name].quantile(quantiles, dim=0)
    ax.plot(range(n_train - 30, n_train), y[-30:], color="black")
    ax.plot(future_t, y_future, color="black", linestyle="--", label="held-out series")
    ax.fill_between(future_t, lo, hi, color=color, alpha=0.3, linewidth=0)
    ax.plot(future_t, med, color=color, label="forecast median (90% band)")
    ax.set_xlabel("quarter")
    ax.set_title(f"{name} forecast")
    ax.legend(frameon=False, fontsize="small", loc="upper left")
plt.show()
