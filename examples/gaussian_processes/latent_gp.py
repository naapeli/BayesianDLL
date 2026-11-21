"""
GP Classification with a Latent Function
========================================

Forty binary outcomes are recorded at roughly evenly spaced inputs; the chance
of a success rises and falls smoothly with ``x``. We model the log-odds as an
unknown smooth function ``f(x)`` with a Gaussian-process prior and observe

``y_i ~ Bernoulli(sigmoid(f(x_i)))``.

The Bernoulli likelihood is not Gaussian, so ``f`` cannot be integrated out
as in ``ExactGP``: the function values are sampled together with the
hyperparameters. ``LatentGP`` does this in *whitened* (non-centred) form,
``f = L z`` with ``L L^T = K`` and ``z ~ Normal(0, I)``. The sampler then
moves in the well-conditioned ``z`` space, whose prior does not change with
the length-scale, which avoids the funnel-shaped posterior of the centred
parameterisation. Hyperparameters live on the log scale and are mapped back
with ``Exp``.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Bernoulli, Normal
from BayesianDLL.GP import ConstantMean, LatentGP, RBF, gp_predictive


torch.manual_seed(3)


def true_log_odds(x):
    return 3.0 * torch.sin(1.3 * x) - 0.5


x = torch.linspace(-3.0, 3.0, 40, dtype=torch.float64) + 0.05 * torch.randn(40, dtype=torch.float64)
y = torch.bernoulli(torch.sigmoid(true_log_odds(x)))

# %%
# The model
# ---------
# The latent vector ``f_white`` (40 standard-normal values) is created by
# ``LatentGP`` automatically. A length-scale prior centred on 1 keeps the
# function from fitting each label individually, which binary data alone
# cannot rule out.

with Model() as model:
    inputs = Data("inputs", x)
    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 0.5))
    log_variance = RandomParameter("log_variance", Normal(1.0, 0.7))
    intercept = RandomParameter("intercept", Normal(0.0, 1.0))
    kernel = RBF(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
    f = LatentGP("f", inputs, kernel, mean=ConstantMean(intercept))
    ObservedParameter("y", Bernoulli(logits=f), y)

# %%
# Sampling
# --------

start = time.perf_counter()
trace = model.sample(300, 300, n_chains=2, max_depth=5, delta=0.85, seed=0, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary()
print(summary.loc[["log_lengthscale[0]", "log_variance[0]"]])
print(f"sampling: {elapsed:.1f} s, divergences: {sum(trace.divergences)}, max R-hat: {summary['r_hat'].max():.3f}")

# %%
# Latent function and probabilities
# ---------------------------------
# ``gp_predictive`` extends every posterior draw of ``f`` to a fine grid by
# conditioning on the sampled training values. Pushing the draws through
# the sigmoid gives the posterior of the success probability; the band is
# wide where data are sparse and at the edges of the input range.

grid = torch.linspace(-3.5, 3.5, 150, dtype=torch.float64)
latent = gp_predictive(f, trace.thin(4), grid, n_samples=4).reshape(-1, grid.numel())
probability = torch.sigmoid(latent)
levels = torch.tensor([0.05, 0.5, 0.95], dtype=latent.dtype)
latent_q = torch.quantile(latent, levels, dim=0)
probability_q = torch.quantile(probability, levels, dim=0)

fig, (ax_latent, ax_probability) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
ax_latent.fill_between(grid, latent_q[0], latent_q[2], color="tab:blue", alpha=0.25, label="90% band")
ax_latent.plot(grid, latent_q[1], color="tab:blue", label="median")
ax_latent.plot(grid, true_log_odds(grid), "k--", label="truth")
ax_latent.set(xlabel="x", ylabel="f(x) (log-odds)", title="Latent function")
ax_latent.legend(loc="best", fontsize="small")

ax_probability.fill_between(grid, probability_q[0], probability_q[2], color="tab:orange", alpha=0.25, label="90% band")
ax_probability.plot(grid, probability_q[1], color="tab:orange", label="median")
ax_probability.plot(grid, torch.sigmoid(true_log_odds(grid)), "k--", label="truth")
jitter = 0.03 * torch.randn_like(y)
ax_probability.scatter(x, y + jitter, s=12, color="black", alpha=0.6, label="labels")
ax_probability.set(xlabel="x", ylabel="P(y = 1)", title="Success probability", ylim=(-0.1, 1.1))
ax_probability.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
plt.show()
