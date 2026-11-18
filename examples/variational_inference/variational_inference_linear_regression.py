"""
Hand-Written Guide for a Linear Regression
==========================================

A straight-line regression with unknown noise,

.. math::

    a \\sim N(0, 5), \\quad b \\sim N(0, 5), \\quad
    \\sigma \\sim \\text{HalfNormal}(1), \\quad y_i \\sim N(a + b x_i, \\sigma),

is approximated with a mean-field guide written by hand in a
``Guide`` block:

* ``a`` and ``b`` get Normal factors with a free mean and a standard
  deviation that is a ``VariationalParameter`` with ``log_scale=True``, so
  the optimiser updates ``log(sd)`` and the scale stays positive;
* ``sigma`` must stay positive, so the guide puts a Normal on
  ``log_sigma`` and defines ``sigma = Exp(log_sigma)``. A deterministic
  guide node with the name of a model latent tells ``BBVI`` how the guide
  produces that latent. The result is a log-normal factor for ``sigma``.

The covariate is centred, which makes the intercept and slope nearly
independent a posteriori, so the factorised guide loses little.

The hand-written guide is compared with the automatic ``AutoNormal`` guide,
which builds the same mean-field family, and with NUTS. The figures show the
ELBO of both guides, the three marginal posteriors under all three methods,
and the fitted line with its 90% band and a 95% predictive band.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import AutoNormal, Data, Guide, Model, ObservedParameter, RandomParameter, VariationalParameter, plate
from BayesianDLL.Deterministic import Exp, Linear
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Variational import BBVI


torch.manual_seed(7)

n_observations = 100
true_intercept, true_slope, true_sigma = 1.0, 2.5, 0.7
x_raw = torch.rand(n_observations, dtype=torch.float64)
y = true_intercept + true_slope * x_raw + true_sigma * torch.randn(n_observations, dtype=torch.float64)
x_centre = x_raw.mean()
x = x_raw - x_centre

with Model() as model:
    intercept = RandomParameter("intercept", Normal(0.0, 5.0))
    slope = RandomParameter("slope", Normal(0.0, 5.0))
    sigma = RandomParameter("sigma", HalfNormal(1.0))
    x_data = Data("x", x)
    with plate("observations", x_data):
        line = Linear("line", x_data, coefficients=slope, intercept=intercept)
        ObservedParameter("y", Normal(line, sigma), y)


def normal_factor(name, mean=0.0, sd=1.0):
    """A Normal guide factor with a free mean and a log-scale standard deviation."""
    loc = VariationalParameter(f"{name}_loc", torch.tensor([mean], dtype=torch.float64))
    scale = VariationalParameter(f"{name}_scale", torch.tensor([sd], dtype=torch.float64), log_scale=True)
    return RandomParameter(name, Normal(loc, scale))


with Guide() as guide:
    normal_factor("intercept")
    normal_factor("slope")
    Exp("sigma", normal_factor("log_sigma", mean=-1.0, sd=0.5))

# %%
# Fitting the guides and the reference
# ------------------------------------
# ``guide.sample`` draws every guide latent directly from its distribution
# (no MCMC) and maps the draws onto the model, so the result holds the model
# latents ``intercept``, ``slope`` and ``sigma``, like the other traces.

history = BBVI(model, guide, n_samples=8, epochs=800, lr=0.02, lrd=0.998, callback_frequency=5, verbose=False)
handwritten = guide.sample(4000)

auto_guide = AutoNormal(model)
auto_history = BBVI(model, auto_guide, n_samples=8, epochs=800, lr=0.02, lrd=0.998, callback_frequency=5, verbose=False)
automatic = auto_guide.sample(4000)

nuts = model.sample(n_samples=1000, warmup_length=500, n_chains=4, progress_bar=False, seed=0, delta=0.9,
                    chain_method="vectorized")
summary = nuts.summary()
print(summary)
print(f"NUTS: {sum(nuts.divergences)} divergences, largest R-hat {summary['r_hat'].max():.3f}")
print(f"final ELBO: hand-written {sum(history[-30:]) / 30:.2f}, AutoNormal {sum(auto_history[-30:]) / 30:.2f}")

traces = {"hand-written guide": handwritten, "AutoNormal": automatic, "NUTS": nuts}
colors = {"hand-written guide": "C0", "AutoNormal": "C1", "NUTS": "black"}
for name in ("intercept", "slope", "sigma"):
    print(f"{name:>10}: " + ", ".join(
        f"{label} {trace[name].mean().item():.3f} +- {trace[name].std().item():.3f}" for label, trace in traces.items()))

# %%
# Marginal posteriors and fitted line
# -----------------------------------
# The two guides belong to the same family and reach the same optimum; both
# agree with NUTS on all three marginals. The intercept refers to the centre
# of the covariate, ``x = mean(x)``.

fig = plt.figure(figsize=(13, 8), layout="constrained")
panels = fig.subplots(2, 3)
ax = panels[0, 0]
steps = range(5, 5 * len(history) + 1, 5)
ax.plot(steps, history, color=colors["hand-written guide"], alpha=0.8, label="hand-written guide")
ax.plot(steps, auto_history, color=colors["AutoNormal"], alpha=0.8, label="AutoNormal")
ax.set_ylim(history[-1] - 60, history[-1] + 15)
ax.set_xlabel("epoch")
ax.set_ylabel("ELBO estimate")
ax.set_title("Optimisation")
ax.legend(fontsize="small")

for ax, name in zip((panels[0, 1], panels[0, 2], panels[1, 0]), ("intercept", "slope", "sigma")):
    for label, trace in traces.items():
        ax.hist(trace[name].flatten(), bins=50, density=True, histtype="step", lw=1.8, color=colors[label], label=label)
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    ax.set_title(f"Posterior of {name}")
handles, labels = panels[0, 1].get_legend_handles_labels()
panels[0, 1].legend(handles, labels, fontsize="small")

ax = fig.add_subplot(2, 3, (5, 6))
panels[1, 1].remove()
panels[1, 2].remove()
grid = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)
lines = handwritten["intercept"].reshape(-1, 1) + handwritten["slope"].reshape(-1, 1) * (grid - x_centre)
new_y = lines + handwritten["sigma"].reshape(-1, 1) * torch.randn_like(lines)
quantiles = torch.tensor([0.05, 0.95, 0.025, 0.975], dtype=torch.float64)
line_lower, line_upper, _, _ = lines.quantile(quantiles, dim=0)
_, _, y_lower, y_upper = new_y.quantile(quantiles, dim=0)
ax.scatter(x_raw, y, s=12, color="grey", label="data")
ax.fill_between(grid, y_lower, y_upper, color="C0", alpha=0.12, label="95% interval of new y")
ax.fill_between(grid, line_lower, line_upper, color="C0", alpha=0.4, label="90% band of the line")
ax.plot(grid, lines.mean(0), color="C0", label="posterior mean line")
ax.plot(grid, true_intercept + true_slope * grid, color="black", linestyle="--", label="true line")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Fit of the hand-written guide")
ax.legend(fontsize="small", ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.show()
