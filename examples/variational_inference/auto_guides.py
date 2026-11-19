"""
Mean-Field vs Full-Rank Automatic Guides
========================================

A straight-line regression ``y ~ N(a + b x, sigma)`` is fitted with
black-box variational inference (``BBVI``) using two guides that are built
automatically from the model:

* ``Guide(model, default_family="mean_field")``: an independent Normal for
  every latent in the unconstrained space (``sigma`` enters through its
  logarithm), i.e. a mean-field guide;
* ``Guide(model, default_family="full_rank")``: one Normal with a full
  covariance matrix over all unconstrained latents.

The covariate lies between 2 and 4, far from zero, so the intercept and the
slope are strongly negatively correlated in the posterior: a steeper line
needs a lower intercept to pass through the data. This is left in on purpose
to show what each guide can represent. (In a real analysis, centering ``x``
would remove the correlation and make the mean-field guide adequate.)

A NUTS run with a dense mass matrix serves as the reference posterior. The
figures show the ELBO histories, the joint posterior of intercept and slope
(NUTS draws with the 95% ellipses of both guides), the marginal posteriors,
and the regression line with its 90% band under each approximation.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Guide, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Variational import BBVI


torch.manual_seed(1)

n_observations = 40
x = torch.linspace(2.0, 4.0, n_observations, dtype=torch.float64)
y = 1.0 + 0.8 * x + 0.4 * torch.randn(n_observations, dtype=torch.float64)

with Model() as model:
    intercept = RandomParameter("intercept", Normal(0.0, 5.0))
    slope = RandomParameter("slope", Normal(0.0, 5.0))
    sigma = RandomParameter("sigma", HalfNormal(1.0))
    x_data = Data("x", x)
    with plate("observations", x_data):
        line = Linear("line", x_data, coefficients=slope, intercept=intercept)
        ObservedParameter("y", Normal(line, sigma), y)

# %%
# Fitting both guides and the reference
# -------------------------------------
# Each guide starts from the model's current values and is optimised with
# four Monte Carlo draws per ELBO estimate. ``guide.sample`` draws
# independent samples directly from the fitted Normal and maps them back to
# the constrained space.

guides = {"mean-field": Guide(model, default_family="mean_field"), "full-rank": Guide(model, default_family="full_rank")}
histories = {}
posteriors = {}
for label, guide in guides.items():
    histories[label] = BBVI(model, guide, n_samples=4, epochs=1500, lr=0.03, lrd=0.999, verbose=False)
    posteriors[label] = guide.sample(2000)
    print(f"{label:>22}: final ELBO estimate {sum(histories[label][-100:]) / 100:.2f}")

reference = model.sample(n_samples=1000, warmup_length=500, n_chains=4, progress_bar=False, seed=0, metric="dense", delta=0.9,
                         chain_method="vectorized")
summary = reference.summary()
print(summary)
print(f"NUTS reference: {sum(reference.divergences)} divergences, largest R-hat {summary['r_hat'].max():.3f}")
posteriors["NUTS"] = reference


def pair(trace):
    return torch.stack([trace["intercept"].flatten(), trace["slope"].flatten()], dim=-1)


for label, trace in posteriors.items():
    draws = pair(trace)
    print(f"{label:>22}: corr(intercept, slope) = {torch.corrcoef(draws.T)[0, 1].item():+.3f}, "
          f"sd(slope) = {draws[:, 1].std().item():.3f}")

# %%
# Comparing the approximations
# ----------------------------
# The full-rank ellipse follows the tilted cloud of NUTS draws. The
# mean-field ellipse is axis-aligned and much too narrow: an independent
# guide that has to stay inside a thin diagonal ridge can only do so by
# shrinking both marginals. Its regression band is consequently too narrow,
# most visibly at the ends of the data range, while the full-rank band
# matches the NUTS band.

colors = {"mean-field": "C1", "full-rank": "C2", "NUTS": "black"}
angles = torch.linspace(0, 2 * torch.pi, 200, dtype=torch.float64)
circle = torch.stack([angles.cos(), angles.sin()]) * 2.448  # radius of the 95% region in 2-D

fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
ax = axes[0, 0]
for label, history in histories.items():
    ax.plot(history, color=colors[label], label=label)
final = min(sum(h[-100:]) / 100 for h in histories.values())
ax.set_ylim(final - 40, max(max(h) for h in histories.values()) + 5)
ax.set_xlabel("epoch")
ax.set_ylabel("ELBO estimate")
ax.set_title("Optimisation")
ax.legend()

ax = axes[0, 1]
nuts_pair = pair(reference)
ax.scatter(nuts_pair[:, 0], nuts_pair[:, 1], s=3, alpha=0.2, color="black", label="NUTS draws")
for label in guides:
    draws = pair(posteriors[label])
    ellipse = draws.mean(0, keepdim=True).T + torch.linalg.cholesky(torch.cov(draws.T)) @ circle
    ax.plot(ellipse[0], ellipse[1], color=colors[label], lw=2, label=f"{label} (95%)")
ax.set_xlabel("intercept")
ax.set_ylabel("slope")
ax.set_title("Joint posterior of intercept and slope")
ax.legend(fontsize="small")

ax = axes[1, 0]
for label, trace in posteriors.items():
    ax.hist(trace["slope"].flatten(), bins=60, density=True, histtype="step", lw=1.8, color=colors[label], label=label)
ax.set_xlabel("slope")
ax.set_ylabel("density")
ax.set_title("Marginal posterior of the slope")
ax.legend()

ax = axes[1, 1]
grid = torch.linspace(1.8, 4.2, 100, dtype=torch.float64)
ax.scatter(x, y, s=12, color="grey", label="data")
for label, trace in posteriors.items():
    lines = trace["intercept"].reshape(-1, 1) + trace["slope"].reshape(-1, 1) * grid
    lower, upper = lines.quantile(torch.tensor([0.05, 0.95], dtype=torch.float64), dim=0)
    ax.plot(grid, lines.mean(0), color=colors[label], lw=1)
    ax.fill_between(grid, lower, upper, color=colors[label], alpha=0.2, label=f"{label}, 90% band")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Fitted regression line")
ax.legend(fontsize="small", ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
plt.show()
