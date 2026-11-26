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

A second fit uses the Hilbert-space approximation of the same GP
(``LatentGP(..., n_basis=m, boundary_factor=c)``): ``f`` becomes a sum of ``m``
fixed sine basis functions with sampled weights, so the sampler no longer
works with a 40 x 40 Cholesky factor. With only 40 inputs the approximation
cannot be much faster, so the comparison is about accuracy and mixing.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Bernoulli, Normal
from BayesianDLL.GP import ConstantMean, LatentGP, RBF


torch.manual_seed(3)


def true_log_odds(x):
    return 3.0 * torch.sin(1.3 * x) - 0.5


x = torch.linspace(-3.0, 3.0, 40, dtype=torch.float64) + 0.05 * torch.randn(40, dtype=torch.float64)
y = torch.bernoulli(torch.sigmoid(true_log_odds(x)))
# Held-out points for comparing the two fits.
x_test = torch.rand(300, dtype=torch.float64) * 6.0 - 3.0
y_test = torch.bernoulli(torch.sigmoid(true_log_odds(x_test)))

# %%
# The models
# ----------
# The latent vector (``f_white`` for the dense GP, ``f_basis_white`` for the
# Hilbert-space GP) is created by ``LatentGP`` automatically. A length-scale
# prior centred on 1 keeps the function from fitting each label individually,
# which binary data alone cannot rule out. The Hilbert-space version uses
# ``m = 20`` basis functions on the data range extended by a factor
# ``c = 1.5``, which is ample for length-scales near 1. It keeps the default
# non-centred form: the 40 binary labels are only weakly informative, and the
# centred form (weights sampled directly) was much worse on this data
# (R-hat 1.24 and 30 divergences in a trial run of the same length).


def build(**options):
    with Model() as built:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 0.5))
        log_variance = RandomParameter("log_variance", Normal(1.0, 0.7))
        intercept = RandomParameter("intercept", Normal(0.0, 1.0))
        kernel = RBF(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
        f = LatentGP("f", inputs, kernel, mean=ConstantMean(intercept), **options)
        ObservedParameter("y", Bernoulli(logits=f), y)
    return built, inputs


dense_model, dense_inputs = build()
hilbert_model, hilbert_inputs = build(n_basis=20, boundary_factor=1.5)

# %%
# Sampling
# --------

traces, summaries = {}, {}
for label, current in (("dense", dense_model), ("Hilbert space", hilbert_model)):
    start = time.perf_counter()
    traces[label] = current.sample(250, 250, n_chains=2, max_depth=5, delta=0.85, seed=0, progress_bar=False)
    elapsed = time.perf_counter() - start
    summaries[label] = traces[label].summary()
    print(f"--- {label}")
    print(summaries[label].loc[["log_lengthscale[0]", "log_variance[0]"], ["mean", "sd", "ess_bulk", "r_hat"]])
    print(f"sampling: {elapsed:.1f} s, divergences: {sum(traces[label].divergences)}, "
          f"max R-hat: {summaries[label]['r_hat'].max():.3f}, min bulk ESS: {summaries[label]['ess_bulk'].min():.0f}")

# %%
# Latent function and probabilities
# ---------------------------------
# ``model.posterior_predicative`` extends every posterior draw of ``f`` to new
# inputs: the dense GP by conditioning on the sampled training values, the
# Hilbert-space GP by evaluating its basis. Pushing the draws through the
# sigmoid gives the posterior of the success probability; the band is wide
# where data are sparse and at the edges of the input range. The held-out
# points score both fits by the posterior mean probability: its root mean
# squared error against the true probability and the log loss of the held-out labels.
# The two fits are practically the same: both predict 81% of the 300
# held-out labels correctly, with a log loss of 0.459 and 0.458 (0.410 for
# the true probabilities, the floor given the label noise) and an RMSE of
# 0.161 for the probability. Sampling is 2-3 times faster with the Hilbert-space
# GP here (16 s against 40 s), and its length-scale is better mixed (bulk ESS
# 296 against 133), as the 20 weights replace a 40 x 40 factorisation.
# The saving grows with the number of inputs, since the cost is linear in it.

grid = torch.linspace(-3.5, 3.5, 150, dtype=torch.float64)
levels = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
bands = {}
for label, current, inputs in (("dense", dense_model, dense_inputs), ("Hilbert space", hilbert_model, hilbert_inputs)):
    thinned = traces[label].thin(4)
    inputs.set_value(grid)
    latent = current.posterior_predicative(thinned, samples_per_step=4, progress_bar=False)["f"].reshape(-1, grid.numel())
    bands[label] = (torch.quantile(latent, levels, dim=0), torch.quantile(torch.sigmoid(latent), levels, dim=0))
    inputs.set_value(x_test)
    latent_test = current.posterior_predicative(thinned, samples_per_step=4, progress_bar=False)["f"].reshape(-1, x_test.numel())
    probability = torch.sigmoid(latent_test).mean(0)
    rmse = (probability - torch.sigmoid(true_log_odds(x_test))).square().mean().sqrt()
    log_loss = -(y_test * probability.log() + (1 - y_test) * (1 - probability).log()).mean()
    accuracy = ((probability > 0.5) == (y_test > 0.5)).double().mean()
    print(f"{label:>13}: held-out log loss {log_loss:.3f}, accuracy {accuracy:.3f}, RMSE of P(y = 1) {rmse:.3f}")
    inputs.set_value(x)
true_loss = -(y_test * torch.sigmoid(true_log_odds(x_test)).log() + (1 - y_test) * (1 - torch.sigmoid(true_log_odds(x_test))).log()).mean()
print(f"{'truth':>13}: held-out log loss {true_loss:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout="constrained")
for row, (label, (latent_q, probability_q)) in enumerate(bands.items()):
    ax_latent, ax_probability = axes[row]
    ax_latent.fill_between(grid, latent_q[0], latent_q[2], color="tab:blue", alpha=0.25, label="90% band")
    ax_latent.plot(grid, latent_q[1], color="tab:blue", label="median")
    ax_latent.plot(grid, true_log_odds(grid), "k--", label="truth")
    ax_latent.set(ylabel="f(x) (log-odds)", title=f"Latent function, {label}")
    ax_probability.fill_between(grid, probability_q[0], probability_q[2], color="tab:orange", alpha=0.25, label="90% band")
    ax_probability.plot(grid, probability_q[1], color="tab:orange", label="median")
    ax_probability.plot(grid, torch.sigmoid(true_log_odds(grid)), "k--", label="truth")
    ax_probability.scatter(x, y + 0.03 * torch.randn_like(y), s=12, color="black", alpha=0.6, label="labels")
    ax_probability.set(ylabel="P(y = 1)", title=f"Success probability, {label}", ylim=(-0.1, 1.1))
axes[0, 0].legend(loc="best", fontsize="small")
axes[0, 1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
for ax in axes[1]:
    ax.set_xlabel("x")
plt.show()
