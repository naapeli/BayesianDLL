"""
Sparse GPs: VFE, DTC and FITC
=============================

An exact GP costs ``O(n³)`` per likelihood evaluation, which becomes the
bottleneck for thousands of observations. Inducing-point methods summarise
the function by its values ``u = f(Z)`` at ``m << n`` *inducing inputs*
``Z`` and only factorise ``m x m`` matrices, at ``O(n m²)`` cost. They differ
in how they treat what ``u`` cannot explain:

* **DTC** assumes ``f`` is fully determined by ``u``: the prior covariance
  becomes ``Q_nn = K_nm K_mm⁻¹ K_mn``. Its predictive variance collapses
  away from the inducing inputs.
* **FITC** adds back the exact prior variance on the diagonal,
  ``Q_nn + diag(K_nn - Q_nn)``, i.e. heteroscedastic extra noise.
* **VFE** (Titsias) uses DTC's Gaussian term minus the penalty
  ``tr(K_nn - Q_nn) / (2σ²)``. It is a lower bound on the exact log
  marginal likelihood, so maximising it cannot overfit the approximation.

Here 400 noisy observations of a 1-D function are fitted with ``m = 10``
fixed, evenly spaced inducing inputs. Each method's hyperparameters are set
to a type-II MAP estimate (``find_MAP`` maximises the method's marginal
likelihood times weak log-normal priors), and the four fits are compared
with the exact GP. NUTS then samples the VFE hyperparameter posterior, and
a last section adds a learned ``LinearMean`` for data with a trend.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import ExactGP, LinearMean, Matern52, SparseGP, exact_gp_predictive, sparse_gp_predictive


torch.manual_seed(0)


def true_function(x):
    return torch.sin(x) + 0.3 * torch.cos(3.0 * x)


n, m = 400, 10
x = torch.sort(torch.rand(n, dtype=torch.float64) * 10.0).values
y = true_function(x) + 0.2 * torch.randn(n, dtype=torch.float64)
inducing = torch.linspace(0.5, 9.5, m, dtype=torch.float64)
grid = torch.linspace(-0.5, 10.5, 200, dtype=torch.float64)


def build(method):
    """One model per method; ``method="exact"`` uses the exact marginal likelihood."""
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0))
        kernel = Matern52(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
        noise = Exp("noise", log_noise)
        if method == "exact":
            f = ExactGP("f", inputs, kernel, noise_variance=noise)
        else:
            f = SparseGP("f", inputs, kernel, inducing_points=inducing, noise_variance=noise, method=method)
        ObservedParameter("y", f, y)
    return model, f


def point_trace(model):
    """A one-draw 'trace' holding the current hyperparameter values."""
    return {name: model.deterministic_params[name].constrained_value.reshape(1, 1, -1)
            for name in ("lengthscale", "variance", "noise")}


def band(f, trace, method):
    predict = exact_gp_predictive if method == "exact" else sparse_gp_predictive
    draws = predict(f, trace, y, grid, n_samples=300).reshape(-1, grid.numel())
    return torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)


# %%
# Type-II MAP estimate for each method
# ------------------------------------

fits = {}
for method in ("exact", "vfe", "dtc", "fitc"):
    model, f = build(method)
    start = time.perf_counter()
    model.find_MAP(lr=0.05, epochs=250, verbose=False)
    elapsed = time.perf_counter() - start
    fits[method] = (model, f, point_trace(model))
    values = {k: v.item() for k, v in fits[method][2].items()}
    print(f"{method:>5}: log p(y) = {f.log_pdf(y).item():8.2f}  lengthscale {values['lengthscale']:.2f}  "
          f"noise sd {values['noise'] ** 0.5:.3f}  ({elapsed:.1f} s)")

# %%
# Comparing the fits
# ------------------
# With only ten inducing inputs, DTC's band pinches between and beyond the
# inducing inputs, FITC absorbs the misfit as extra noise, and VFE stays
# closest to the exact GP, typically choosing a slightly longer length-scale
# that the inducing inputs can represent.

titles = {"exact": "Exact GP", "vfe": "VFE", "dtc": "DTC", "fitc": "FITC"}
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, layout="constrained")
for ax, method in zip(axes.flat, titles):
    _, f, trace = fits[method]
    lower, median, upper = band(f, trace, method)
    ax.scatter(x, y, s=3, color="0.65", label="observations")
    ax.fill_between(grid, lower, upper, color="tab:blue", alpha=0.3, label="90% band of f")
    ax.plot(grid, median, color="tab:blue", label="median of f")
    ax.plot(grid, true_function(grid), "k--", linewidth=1, label="true f")
    if method != "exact":
        ax.plot(inducing, torch.full_like(inducing, -2.1), "^", color="tab:red", label="inducing inputs")
    ax.set_title(f"{titles[method]}  (log p(y) = {f.log_pdf(y).item():.1f})")
for ax in axes[1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize="small")
plt.show()

# %%
# Full Bayesian VFE
# -----------------
# The VFE bound is cheap (``O(n m²)``), so NUTS can sample the three
# log-hyperparameters directly. With only ten inducing inputs the
# approximation cannot follow every wiggle, and VFE attributes the remaining
# misfit to noise: the noise posterior sits somewhat above the true value.
# More (or learned, see the next example) inducing inputs close this gap.

model, f, _ = fits["vfe"]
start = time.perf_counter()
trace = model.sample(300, 300, n_chains=2, max_depth=5, seed=2, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary(include_deterministic=True)
print(summary)
print(f"sampling: {elapsed:.1f} s, divergences: {sum(trace.divergences)}, max R-hat: {summary['r_hat'].max():.3f}")

draws = sparse_gp_predictive(f, trace.thin(10), y, grid, n_samples=10).reshape(-1, grid.numel())
lower, median, upper = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)
fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained", gridspec_kw={"width_ratios": [2.2, 1]})
ax.scatter(x, y, s=3, color="0.65", label="observations")
ax.fill_between(grid, lower, upper, color="tab:green", alpha=0.3, label="90% band of f")
ax.plot(grid, median, color="tab:green", label="median of f")
ax.plot(grid, true_function(grid), "k--", linewidth=1, label="true f")
ax.plot(inducing, torch.full_like(inducing, -2.1), "^", color="tab:red", label="inducing inputs")
ax.set(xlabel="x", ylabel="y", title="VFE posterior predictive (NUTS)")
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize="small")
ax_hist.hist(trace["noise"].reshape(-1).sqrt(), bins=30, color="tab:green", density=True)
ax_hist.axvline(0.2, color="k", linestyle="--", label="true value")
ax_hist.set(xlabel="noise standard deviation", ylabel="density", title="Noise posterior")
ax_hist.legend(fontsize="small")
plt.show()

# %%
# A learned linear mean
# ---------------------
# Every GP takes a mean function ``m(x)`` (``mean=``, default
# ``ZeroMean()``): the GP then models ``f(x) - m(x)``. Here the data get an
# upward trend, ``y + 0.35 x``, and the model is fitted twice with VFE: with
# the default zero mean and with ``LinearMean(slope, intercept)``, whose
# slope and intercept are ordinary ``RandomParameter`` nodes with weak priors,
# estimated by ``find_MAP`` together with the kernel hyperparameters (NUTS
# would sample them in the same way, and the predictive reads them from the
# trace draw by draw).
#
# Away from the data a stationary GP reverts to its mean. With a zero mean
# the prediction beyond the data (shaded, ``x > 10``) falls back to zero; with the learned
# linear mean it continues the trend, and the kernel only describes the
# wiggles around it.
#
# **Linear mean vs. linear kernel.** The ExactGP example models a trend with
# a ``Linear`` *kernel*, ``k(x, x') = v x x'``. That is the covariance of
# ``f(x) = β x`` with a *random* slope ``β ~ Normal(0, v)`` integrated out of
# the likelihood: the slope's uncertainty is part of the GP, and the
# predictive band widens linearly with ``|x|``. A ``LinearMean`` is a fixed
# trend whose slope is a model parameter: it is learned (a point estimate
# here, or a posterior with NUTS) rather than integrated out, the kernel
# stays stationary (so it also works with ``ToeplitzGP``), and the band far
# from the data is that of the stationary kernel alone.

y_trend = y + 0.35 * x
grid_far = torch.linspace(-0.5, 15.0, 250, dtype=torch.float64)


def build_trend(learned_mean):
    with Model() as model:
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0))
        kernel = Matern52(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
        mean = None
        if learned_mean:
            slope = RandomParameter("slope", Normal(0.0, 1.0))
            intercept = RandomParameter("intercept", Normal(0.0, 2.0))
            mean = LinearMean(slope, intercept)
        f = SparseGP("f", Data("inputs", x), kernel, inducing_points=inducing,
                     noise_variance=Exp("noise", log_noise), mean=mean)
        ObservedParameter("y", f, y_trend)
    return model, f


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout="constrained")
for ax, learned_mean in zip(axes, (False, True)):
    model, f = build_trend(learned_mean)
    model.find_MAP(lr=0.05, epochs=250, verbose=False)
    trace = point_trace(model)
    title = "zero mean"
    if learned_mean:
        for name in ("slope", "intercept"):
            trace[name] = model.params[name].constrained_value.reshape(1, 1, -1)
        title = f"LinearMean (slope {trace['slope'].item():.2f}, intercept {trace['intercept'].item():.2f})"
    draws = sparse_gp_predictive(f, trace, y_trend, grid_far, n_samples=300).reshape(-1, grid_far.numel())
    lower, median, upper = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)
    ax.scatter(x, y_trend, s=3, color="0.65", label="observations")
    ax.fill_between(grid_far, lower, upper, color="tab:purple", alpha=0.3, label="90% band of f")
    ax.plot(grid_far, median, color="tab:purple", label="median of f")
    ax.plot(grid_far, true_function(grid_far) + 0.35 * grid_far, "k--", linewidth=1, label="true f")
    ax.axvspan(10.0, 15.0, color="0.9", zorder=0)
    ax.set(xlabel="x", title=f"VFE, {title}")
axes[0].set_ylabel("y")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize="small")
plt.show()
