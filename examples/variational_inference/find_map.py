"""
Maximum a Posteriori Estimation
===============================

``find_MAP`` maximises the log posterior density of a model with Adam and
leaves every latent at its optimum. It returns a point estimate, not a
distribution, which makes it a quick first fit or a starting point for
sampling.

The model is a straight-line regression with a positive noise scale,

.. math::

    a \\sim N(0, 5), \\quad b \\sim N(0, 5), \\quad
    \\sigma \\sim \\text{HalfNormal}(1), \\quad
    y_i \\sim N(a + b x_i, \\sigma).

The optimiser works in the unconstrained space, so ``sigma`` is optimised
through ``log(sigma)`` and can never become negative. The covariate is
centred, which decouples the intercept from the slope and lets Adam converge
in a few hundred steps.

With 500 observations and wide priors the MAP estimate should agree with
the ordinary least-squares fit, which is computed in closed form as a
reference. The figures show the log posterior during the optimisation and
the MAP line with a ``+-1.96 sigma`` band for single observations, next to
the least-squares line.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, find_MAP, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import HalfNormal, Normal


torch.manual_seed(7)

n_observations = 500
true_intercept, true_slope, true_sigma = 1.0, 2.5, 0.7
x_raw = torch.rand(n_observations, dtype=torch.float64)
y = true_intercept + true_slope * x_raw + true_sigma * torch.randn(n_observations, dtype=torch.float64)
x_centre = x_raw.mean()
x = x_raw - x_centre  # with a centred covariate the intercept is the mean of y at x = x_centre

with Model() as model:
    intercept = RandomParameter("intercept", Normal(0.0, 5.0))
    slope = RandomParameter("slope", Normal(0.0, 5.0))
    sigma = RandomParameter("sigma", HalfNormal(1.0))
    x_data = Data("x", x)
    with plate("observations", x_data):
        line = Linear("line", x_data, coefficients=slope, intercept=intercept)
        ObservedParameter("y", Normal(line, sigma), y)

history = find_MAP(model, lr=0.05, epochs=300, callback_frequency=10, verbose=False)
estimate = {name: model.params[name].constrained_value.item() for name in ("intercept", "slope", "sigma")}

# Closed-form least-squares reference.
design = torch.stack([torch.ones_like(x), x], dim=1)
coefficients = torch.linalg.lstsq(design, y.unsqueeze(-1)).solution.squeeze(-1)
residual_sd = (y - design @ coefficients).std(correction=2)
reference = {"intercept": coefficients[0].item(), "slope": coefficients[1].item(), "sigma": residual_sd.item()}

print(f"{'':>10} {'MAP':>8} {'least sq.':>10} {'true':>8}")
truth = {"intercept": true_intercept + true_slope * x_centre.item(), "slope": true_slope, "sigma": true_sigma}
for name in estimate:
    print(f"{name:>10} {estimate[name]:8.3f} {reference[name]:10.3f} {truth[name]:8.3f}")

# %%
# Optimisation trace and fitted line
# ----------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
axes[0].plot(range(10, 10 * len(history) + 1, 10), history)
axes[0].set_xlabel("Adam step")
axes[0].set_ylabel("log posterior density")
axes[0].set_ylim(history[-1] - 200, history[-1] + 20)
axes[0].set_title("Convergence of find_MAP")

grid = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)
map_line = estimate["intercept"] + estimate["slope"] * (grid - x_centre)
reference_line = reference["intercept"] + reference["slope"] * (grid - x_centre)
axes[1].scatter(x_raw, y, s=8, alpha=0.4, color="grey", label="data")
axes[1].plot(grid, map_line, color="C0", lw=2, label="MAP line")
axes[1].fill_between(grid, map_line - 1.96 * estimate["sigma"], map_line + 1.96 * estimate["sigma"],
                     color="C0", alpha=0.15, label=r"MAP line $\pm 1.96\,\sigma$")
axes[1].plot(grid, reference_line, color="black", linestyle="--", label="least-squares line")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("MAP fit")
axes[1].legend(fontsize="small", ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
plt.show()
