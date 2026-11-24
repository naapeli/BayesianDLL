"""
Polynomial Regression
=====================

A smooth nonlinear curve is fitted with a degree-5 polynomial. The data come
from a degree-9 polynomial, so the fit is an approximation on purpose.

Two choices keep the posterior well conditioned. The polynomial is written in
the Chebyshev basis on ``[-1, 1]`` instead of in powers of ``x``: the basis
functions are close to orthogonal, so their coefficients are nearly
uncorrelated a posteriori (the monomials ``x, x^3, x^5`` are almost
collinear). The response is standardised, so a ``Normal(0, 1)`` prior on each
coefficient and a ``HalfNormal(1)`` prior on the noise are weakly informative
whatever the units of ``y``. The deterministic blocks ``Affine`` and
``Multiply`` map the mean and the noise back to the original scale.

The figures compare the prior and the posterior predictive distributions with
the data, and show the fitted curve with credible bands next to the true
function. A degree-5 polynomial cannot follow the degree-9 curve exactly, so
the estimated noise level is larger than the true one: it absorbs the misfit.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Affine, Linear, Multiply
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Evaluation import Graphics, summary


torch.manual_seed(7)

true_coefficients = [0.0, 3.488378906, 0.0, -0.855187500, 0.0, 0.107675000, 0.0, -0.005857143, 0.0, 0.000111111]
x_min, x_max = -5.0, 5.0
n_obs = 100
sigma_true = 0.3


def true_function(x):
    return sum(c * x ** k for k, c in enumerate(true_coefficients))


x = torch.linspace(x_min, x_max, n_obs, dtype=torch.float64)
y = true_function(x) + sigma_true * torch.randn(n_obs, dtype=torch.float64)
y_mean, y_sd = y.mean(), y.std()

degree = 5


def chebyshev_features(x):
    """Chebyshev polynomials T_0, ..., T_degree of x rescaled to [-1, 1]."""
    u = (2 * (x - x_min) / (x_max - x_min) - 1).clamp(-1, 1)
    return torch.stack([torch.cos(k * torch.acos(u)) for k in range(degree + 1)], dim=1)


# %%
# The model
# ---------
with Model() as model:
    features = Data("features", chebyshev_features(x))
    coefficients = RandomParameter("coefficients", Normal(0.0, 1.0), shape=degree + 1)
    sigma_std = RandomParameter("sigma_std", HalfNormal(1.0))

    curve_std = Linear("curve_std", features, coefficients=coefficients)
    mu = Affine("mu", curve_std, loc=y_mean, scale=y_sd)
    sigma = Multiply("sigma", sigma_std, y_sd)

    with plate("observations", n_obs):
        ObservedParameter("y", Normal(mu, sigma), y)

prior_predictive = model.prior_predicative(30, samples_per_step=10, progress_bar=False)
result = model.sample(1000, 500, n_chains=4, seed=2, delta=0.85, progress_bar=False)
posterior_predictive = model.posterior_predicative(result, n_samples=30, samples_per_step=10, progress_bar=False)

print(summary({name: result[name] for name in ("coefficients", "sigma")}).to_string())
print(f"true noise standard deviation: {sigma_true} (the estimate also absorbs the misfit of degree {degree})")

# %%
# Prior and posterior predictive distributions
# --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
Graphics.plot_predicative_distribution(prior_predictive, y, kind="pdf", ax=axes[0])
axes[0].set_title("Prior predictive density vs data")
Graphics.plot_predicative_distribution(posterior_predictive, y, kind="pdf", ax=axes[1])
axes[1].set_title("Posterior predictive density vs data")
for ax in axes:
    ax.set_xlabel("y")
    ax.set_ylabel("density")
    ax.legend(loc="upper center", fontsize="small", frameon=False)

# %%
# Fitted curve
# ------------
x_grid = torch.linspace(x_min, x_max, 300, dtype=torch.float64)
draws = result["coefficients"].reshape(-1, degree + 1)
curves = y_mean + y_sd * draws @ chebyshev_features(x_grid).T
noise = result["sigma"].reshape(-1, 1)
new_obs = curves + noise * torch.randn_like(curves)

fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
ax.fill_between(x_grid, new_obs.quantile(0.025, 0), new_obs.quantile(0.975, 0), color="C0", alpha=0.15,
                linewidth=0, label="95% interval for a new observation")
ax.fill_between(x_grid, curves.quantile(0.025, 0), curves.quantile(0.975, 0), color="C0", alpha=0.35,
                linewidth=0, label="95% interval for the curve")
ax.plot(x_grid, curves.mean(0), color="black", label="posterior mean curve")
ax.plot(x_grid, true_function(x_grid), color="C3", linestyle="--", label="true function (degree 9)")
ax.scatter(x, y, s=14, color="C0", alpha=0.7, label="observations")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Degree-{degree} Chebyshev fit")
ax.legend(frameon=False, fontsize="small", loc="upper left")
plt.show()
