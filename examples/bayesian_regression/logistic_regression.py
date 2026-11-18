"""
Logistic Regression
===================

Binary outcomes ``y_i`` are modelled as ``Bernoulli(logits = a + b x_i)``.
The linear predictor goes straight into ``Bernoulli(logits=...)``, which is
numerically safer than computing probabilities first.

The covariate is standardised, ``z = (x - mean) / sd``, before it enters the
model. On this scale a ``Normal(0, 2.5)`` prior is weakly informative for both
coefficients, and the intercept and slope are close to uncorrelated, which
lets NUTS take large steps. The coefficients on the original scale of ``x``
are recovered with the deterministic blocks ``Multiply`` and ``Affine``.

The figures show the model graph and the fitted probability curve with a 95%
credible band, the true curve and the binned observed frequencies.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Affine, Linear, Multiply
from BayesianDLL.Distributions import Bernoulli, Normal
from BayesianDLL.Evaluation import Graphics, summary


torch.manual_seed(7)

n_obs = 500
a_true, b_true = -0.4, 1.6
x = 3.0 * torch.rand(n_obs, dtype=torch.float64) + 0.5 * torch.randn(n_obs, dtype=torch.float64) - 1.0
y = torch.bernoulli(torch.sigmoid(a_true + b_true * x))
x_mean, x_sd = x.mean(), x.std()

# %%
# The model on the standardised scale
# -----------------------------------
with Model() as model:
    z = Data("z", (x - x_mean) / x_sd)
    alpha = RandomParameter("alpha", Normal(0.0, 2.5))
    beta = RandomParameter("beta", Normal(0.0, 2.5))
    logits = Linear("logits", z, coefficients=beta, intercept=alpha)

    # Coefficients per unit of the original covariate x.
    b = Multiply("b", beta, 1.0 / x_sd)
    Affine("a", b, loc=alpha, scale=-x_mean)

    with plate("observations", z):
        ObservedParameter("y", Bernoulli(logits=logits), y)

Graphics.plot_model(model)

result = model.sample(1000, 500, n_chains=4, seed=3, delta=0.85, progress_bar=False)
print(summary({name: result[name] for name in ("alpha", "beta", "a", "b")}).to_string())
print(f"true values: a = {a_true}, b = {b_true}")

# %%
# Fitted probability curve
# ------------------------
x_grid = torch.linspace(x.min(), x.max(), 200, dtype=torch.float64)
curves = torch.sigmoid(result["a"].reshape(-1, 1) + result["b"].reshape(-1, 1) * x_grid)

edges = torch.linspace(x.min(), x.max(), 13, dtype=torch.float64)
bin_index = torch.bucketize(x, edges[1:-1])
centres = 0.5 * (edges[1:] + edges[:-1])
frequency = torch.stack([y[bin_index == k].mean() for k in range(len(centres))])

fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
ax.scatter(x, y + 0.03 * torch.randn(n_obs), s=8, alpha=0.3, color="C0", label="observations (jittered)")
ax.scatter(centres, frequency, color="C1", edgecolors="black", zorder=3, label="observed frequency per bin")
ax.fill_between(x_grid, curves.quantile(0.025, 0), curves.quantile(0.975, 0), color="C0", alpha=0.3,
                linewidth=0, label="95% credible band")
ax.plot(x_grid, curves.mean(0), color="black", label="posterior mean")
ax.plot(x_grid, torch.sigmoid(a_true + b_true * x_grid), color="C3", linestyle="--", label="true curve")
ax.set_xlabel("x")
ax.set_ylabel("P(y = 1)")
ax.set_title("Logistic regression: fitted probability")
ax.legend(frameon=False, fontsize="small", loc="center left")
plt.show()
