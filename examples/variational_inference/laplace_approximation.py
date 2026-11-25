"""
Laplace Approximation
=====================

``laplace_approximation`` finds the posterior mode in the unconstrained space,
computes the Hessian ``H`` of the log joint density there (finite differences
of the model's gradient), and approximates the posterior by the Gaussian
``N(mode, (-H)^-1)``. It also returns the Laplace estimate of the log
marginal likelihood. The method is cheap and accurate when the posterior is
close to Gaussian in the unconstrained coordinates.

``hessian="fd"`` (default) is the exact Hessian. ``hessian="ggn"`` uses the
positive semi-definite generalised Gauss-Newton matrix instead, which is the
better choice for neural networks (see the Bayesian neural network example).
For the canonical logit link of the logistic regression below both agree.

Two models are approximated, each with an exact or sampled reference:

1. **Beta-Bernoulli.** A success probability ``p ~ Beta(2, 2)`` with 30 coin
   flips. The posterior ``Beta(2 + k, 2 + n - k)`` and the marginal
   likelihood are known in closed form. The Gaussian lives on ``logit(p)``,
   so the approximation of ``p`` itself is a logit-normal, which follows the
   skewed Beta closely.
2. **Logistic regression.** ``y_i ~ Bernoulli(logits = a + b x_i)`` with a
   standardised covariate, compared with NUTS. ``Bernoulli(logits=...)``
   evaluates the likelihood in a numerically stable way, and ``Linear``
   builds the logits.

The figures show the approximate and exact posterior of ``p``; the joint and
marginal posteriors of the regression coefficients (Laplace ellipse over
NUTS draws); and the fitted probability curve with its 90% band under both
methods.
"""

import math

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, laplace_approximation, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import Bernoulli, Beta, Normal


torch.manual_seed(0)

# %%
# Beta-Bernoulli: comparison with the exact posterior
# ---------------------------------------------------

a, b = 2.0, 2.0
flips = (torch.rand(30, dtype=torch.float64) < 0.7).double()
k, n = int(flips.sum()), len(flips)

with Model() as coin_model:
    p = RandomParameter("p", Beta(a, b))
    with plate("flips", n):
        ObservedParameter("y", Bernoulli(p), flips)

coin_laplace = laplace_approximation(coin_model, map_epochs=300, lr=0.05)
exact_log_evidence = (
    math.lgamma(a + k) + math.lgamma(b + n - k) - math.lgamma(a + b + n)
    - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
)
print(f"{k} successes in {n} flips; p at the mode of logit(p): {coin_laplace.map_estimate()['p'].item():.3f}")
print(f"log marginal likelihood: Laplace {coin_laplace.log_marginal_likelihood.item():.4f}, exact {exact_log_evidence:.4f}")
coin_draws = coin_laplace.sample(20000)["p"].flatten()

# %%
# Logistic regression: comparison with NUTS
# -----------------------------------------

n_observations = 80
x_raw = 10.0 + 3.0 * torch.randn(n_observations, dtype=torch.float64)
x_mean, x_std = x_raw.mean(), x_raw.std()
x = (x_raw - x_mean) / x_std  # standardised: the coefficients are on a unit scale
true_logits = 0.4 + 1.6 * x
labels = (torch.rand(n_observations, dtype=torch.float64) < torch.sigmoid(true_logits)).double()

with Model() as logistic_model:
    intercept = RandomParameter("intercept", Normal(0.0, 2.5))
    slope = RandomParameter("slope", Normal(0.0, 2.5))
    x_data = Data("x", x)
    with plate("observations", x_data):
        logits = Linear("logits", x_data, coefficients=slope, intercept=intercept)
        ObservedParameter("y", Bernoulli(logits=logits), labels)

logistic_laplace = laplace_approximation(logistic_model, map_epochs=500, lr=0.05)
logistic_ggn = laplace_approximation(logistic_model, map_epochs=500, lr=0.05, hessian="ggn")
print("Gauss-Newton Laplace sd:", logistic_ggn.cov.diagonal().sqrt().tolist())
laplace_trace = logistic_laplace.sample(4000)
nuts_trace = logistic_model.sample(n_samples=1000, warmup_length=500, n_chains=4, progress_bar=False, seed=1, delta=0.9,
                                  chain_method="vectorized")
summary = nuts_trace.summary()
print(summary)
print(f"NUTS: {sum(nuts_trace.divergences)} divergences, largest R-hat {summary['r_hat'].max():.3f}")
print("Laplace mean:", logistic_laplace.mean.tolist(), " sd:", logistic_laplace.cov.diagonal().sqrt().tolist())

# %%
# Figures
# -------
# Left: the logit-normal Laplace approximation of ``p`` against the exact
# Beta posterior. Middle: NUTS draws of the regression coefficients with the
# 50% and 95% contours of the Laplace Gaussian. Right: the posterior
# probability curve ``sigmoid(a + b x)`` with 90% bands from both methods,
# over the 0/1 data.

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
grid_p = torch.linspace(0.3, 0.98, 400, dtype=torch.float64)
axes[0].hist(coin_draws, bins=80, density=True, alpha=0.4, color="C0", label="Laplace (logit-normal) draws")
axes[0].plot(grid_p, Beta(a + k, b + n - k).pdf(grid_p), color="black", label="exact Beta posterior")
axes[0].set_xlabel("p")
axes[0].set_ylabel("density")
axes[0].set_title("Beta-Bernoulli")
axes[0].legend(fontsize="small", loc="upper left")

angles = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
circle = torch.stack([angles.cos(), angles.sin()])
axes[1].scatter(nuts_trace["intercept"].flatten(), nuts_trace["slope"].flatten(), s=3, alpha=0.2, color="black", label="NUTS draws")
for radius, style in ((1.177, "--"), (2.448, "-")):  # 50% and 95% regions of a 2-D Gaussian
    contour = logistic_laplace.mean.unsqueeze(-1) + radius * logistic_laplace.scale_tril @ circle
    axes[1].plot(contour[0], contour[1], color="C3", linestyle=style, lw=2,
                 label=f"Laplace {'50' if radius < 2 else '95'}% contour")
axes[1].set_xlabel("intercept")
axes[1].set_ylabel("slope")
axes[1].set_title("Logistic regression coefficients")
axes[1].legend(fontsize="small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))

grid_raw = torch.linspace(x_raw.min() - 1, x_raw.max() + 1, 150, dtype=torch.float64)
grid = (grid_raw - x_mean) / x_std
axes[2].scatter(x_raw, labels, s=12, color="grey", alpha=0.6, label="data")
for label, trace, color in (("NUTS", nuts_trace, "black"), ("Laplace", laplace_trace, "C3")):
    curves = torch.sigmoid(trace["intercept"].reshape(-1, 1) + trace["slope"].reshape(-1, 1) * grid)
    lower, upper = curves.quantile(torch.tensor([0.05, 0.95], dtype=torch.float64), dim=0)
    axes[2].plot(grid_raw, curves.mean(0), color=color, label=f"{label} mean")
    axes[2].fill_between(grid_raw, lower, upper, color=color, alpha=0.15, label=f"{label} 90% band")
axes[2].plot(grid_raw, torch.sigmoid(0.4 + 1.6 * grid), color="C2", linestyle=":", label="true curve")
axes[2].set_xlabel("x")
axes[2].set_ylabel("P(y = 1)")
axes[2].set_title("Fitted probability curve")
axes[2].legend(fontsize="small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
plt.show()
