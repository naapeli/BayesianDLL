r"""
Counts with Excess Zeros: the Hurdle Model
==========================================

The number of visits a person makes to a clinic in a year has many zeros:
most people do not go at all, and those who do go a few times. A Poisson
model cannot produce this, because one rate sets both the zero fraction and
the typical positive count. A hurdle model separates the two. A zero occurs
with probability :math:`\pi`, and a positive count follows a Poisson
distribution truncated to :math:`\{1, 2, \dots\}`:

.. math::

    P(0) = \pi, \qquad
    P(k) = (1 - \pi)\,\frac{\mathrm{Poisson}(k \mid \lambda)}{1 - e^{-\lambda}}
    \quad (k \ge 1).

``Hurdle(Poisson(rate), zero_prob)`` is this distribution. Unlike
``ZeroInflated``, :math:`\pi` is the *total* zero probability, so it is
modelled independently of the rate. We fit it and a plain Poisson model to
the same data and compare their posterior predictive distributions.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import Beta, Gamma, Hurdle, Poisson

torch.manual_seed(11)

n_obs = 400
pi_true, rate_true = 0.5, 3.0
y = Hurdle(Poisson(rate_true), pi_true).sample(n_obs)
print(f"observed zero fraction: {(y == 0).double().mean():.3f}, mean count: {y.mean():.2f}, mean positive count: {y[y > 0].mean():.2f}")

# %%
# The two models
# --------------
# Both use the same ``Gamma`` prior on the rate and the same observations.
# The hurdle model adds the zero probability with a uniform ``Beta(1, 1)``
# prior.

with Model() as hurdle:
    pi = RandomParameter("pi", Beta(1.0, 1.0))
    rate = RandomParameter("rate", Gamma(2.0, 0.5))
    with plate("observations", y):
        ObservedParameter("y", Hurdle(Poisson(rate), pi), y)

with Model() as plain:
    rate = RandomParameter("rate", Gamma(2.0, 0.5))
    with plate("observations", y):
        ObservedParameter("y", Poisson(rate), y)

fits = {}
for label, model in (("hurdle", hurdle), ("plain Poisson", plain)):
    start = time.perf_counter()
    fits[label] = model.sample(400, 300, n_chains=4, seed=1, progress_bar=False, chain_method="vectorized")
    elapsed = time.perf_counter() - start
    summary = fits[label].summary()
    print(f"\n{label} model")
    print(summary.to_string())
    print(f"sampling: {elapsed:.1f} s, divergences: {sum(fits[label].divergences)}, max R-hat: {summary['r_hat'].max():.3f}")
print(f"\ntrue values: pi = {pi_true}, rate = {rate_true}")

# %%
# Posterior predictive check
# --------------------------
# For every posterior draw we simulate a data set of the same size and compare
# it with the observed one. The plain Poisson model has to compromise: its
# rate is pulled to the overall mean, which is too low for the positive counts
# and leaves far too few zeros. The hurdle model reproduces the zero
# fraction and the whole histogram.

n_draws = 400
replicates = {}
for label in fits:
    draws = fits[label]
    rate_draws = draws["rate"].reshape(-1)[::len(draws["rate"].reshape(-1)) // n_draws][:n_draws]
    if label == "hurdle":
        pi_draws = draws["pi"].reshape(-1)[::len(draws["pi"].reshape(-1)) // n_draws][:n_draws]
        replicates[label] = Hurdle(Poisson(rate_draws), pi_draws).sample(n_obs)  # (n_obs, draws)
    else:
        replicates[label] = Poisson(rate_draws).sample(n_obs)
for label, rep in replicates.items():
    zero = (rep == 0).double().mean(0)
    print(f"{label}: predicted zero fraction {zero.mean():.3f} "
          f"(94% interval {zero.quantile(0.03):.3f} to {zero.quantile(0.97):.3f}), observed {(y == 0).double().mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
max_count = 10
ks = torch.arange(max_count + 1)
observed = torch.stack([(y == k).double().mean() for k in ks])
for color, (label, rep) in zip(("C0", "C1"), replicates.items()):
    freq = torch.stack([(rep == k).double().mean(0) for k in ks])  # (counts, draws)
    lo, med, hi = freq.quantile(torch.tensor([0.03, 0.5, 0.97], dtype=freq.dtype), dim=1)
    axes[0].errorbar(ks + (0.12 if color == "C1" else -0.12), med, yerr=torch.stack([med - lo, hi - med]),
                     fmt="o", color=color, capsize=3, label=f"{label}: predicted")
    axes[1].hist((rep == 0).double().mean(0), bins=30, color=color, alpha=0.6, label=label)
axes[0].bar(ks, observed, color="0.8", edgecolor="black", zorder=0, label="observed")
axes[0].set(xlabel="count", ylabel="relative frequency", title="Count histogram")
axes[0].legend(frameon=False, fontsize="small")
axes[1].axvline((y == 0).double().mean(), color="black", linestyle="--", label="observed")
axes[1].set(xlabel="fraction of zeros", yticks=[], title="Predicted zero fraction")
axes[1].legend(frameon=False, fontsize="small")
plt.show()
