r"""
Overdispersed Counts: Beta-Binomial and Dirichlet-Multinomial
=============================================================

A Binomial model assumes every trial succeeds with the same probability, and a
Multinomial model that every trial falls in the same category probabilities.
When the probabilities vary between groups, the counts spread out more than
these models allow, and fitting them anyway gives intervals that are too
narrow. Letting the probability itself be random removes the problem:

.. math::

    p \sim \mathrm{Beta}(\alpha, \beta), \quad y \mid p \sim \mathrm{Binomial}(n, p)
    \;\Longrightarrow\; y \sim \mathrm{BetaBinomial}(n, \alpha, \beta),

and likewise ``DirichletMultinomial(n, concentration)`` for a probability
vector with a Dirichlet prior, integrated out analytically. The variance of a
beta-binomial count is the binomial variance inflated by
:math:`(\alpha + \beta + n) / (\alpha + \beta + 1)`: the smaller the
concentration :math:`\alpha + \beta`, the stronger the overdispersion.

Two small problems follow. Each is fitted with the overdispersed distribution
and with its plain counterpart, and the models are compared by a posterior
predictive check of the spread of the counts.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Add, Divide
from BayesianDLL.Distributions import Beta, BetaBinomial, Binomial, Dirichlet, DirichletMultinomial, Gamma, Multinomial

torch.manual_seed(2)


def fit(model, label, n=250, warmup=200):
    start = time.perf_counter()
    result = model.sample(n, warmup, n_chains=4, seed=1, progress_bar=False, chain_method="vectorized")
    elapsed = time.perf_counter() - start
    summary = result.summary(include_deterministic=True)
    print(f"\n{label}")
    print(summary.to_string())
    print(f"sampling: {elapsed:.1f} s, divergences: {sum(result.divergences)}, max R-hat: {summary['r_hat'].max():.3f}")
    return result


def thin(draws, n=400):
    flat = draws.reshape(-1, *draws.shape[2:])
    return flat[:: max(1, len(flat) // n)][:n]


# %%
# Seed germination: beta-binomial
# -------------------------------
# In 60 plots, 20 seeds each are sown and the germinated ones are counted. The
# germination probability differs between plots, here
# :math:`p \sim \mathrm{Beta}(4, 6)` (mean 0.4, concentration 10), so the
# variance of the counts is 2.7 times that of a Binomial(20, 0.4). The
# beta-binomial model gets weak ``Gamma`` priors on :math:`\alpha` and
# :math:`\beta`, and the mean :math:`\alpha / (\alpha + \beta)` and the
# concentration :math:`\alpha + \beta` are derived from them.

n_plots, n_seeds = 60, 20
y_bb = BetaBinomial(n_seeds, 4.0, 6.0).sample(n_plots)
p_hat = y_bb.mean() / n_seeds
print(f"germinated per plot: mean {y_bb.mean():.2f}, sd {y_bb.std():.2f}; "
      f"binomial sd at the same mean: {(n_seeds * p_hat * (1 - p_hat)) ** 0.5:.2f}")

with Model() as beta_binomial:
    alpha = RandomParameter("alpha", Gamma(2.0, 0.2))
    beta = RandomParameter("beta", Gamma(2.0, 0.2))
    total = Add("concentration", alpha, beta)
    Divide("mean", alpha, total)
    with plate("plots", y_bb):
        ObservedParameter("y", BetaBinomial(n_seeds, alpha, beta), y_bb)

with Model() as binomial:
    p = RandomParameter("p", Beta(1.0, 1.0))
    with plate("plots", y_bb):
        ObservedParameter("y", Binomial(n_seeds, p), y_bb)

fit_bb = fit(beta_binomial, "beta-binomial model")
fit_bin = fit(binomial, "binomial model")
print("true values: mean = 0.4, concentration = 10")

# %%
# Predictive check of the spread
# ------------------------------
# For each posterior draw we simulate 60 plots and compute the standard
# deviation of the counts. The binomial model cannot reach the observed spread,
# whatever its parameter, while the beta-binomial model puts it in the centre
# of its predictions.

a, b = thin(fit_bb["alpha"]).reshape(-1), thin(fit_bb["beta"]).reshape(-1)
rep_bb = BetaBinomial(n_seeds, a, b).sample(n_plots)  # (plots, draws)
rep_bin = Binomial(n_seeds, thin(fit_bin["p"]).reshape(-1)).sample(n_plots)
for label, rep in (("beta-binomial", rep_bb), ("binomial", rep_bin)):
    sd = rep.std(0)
    print(f"{label}: predicted sd of the counts {sd.mean():.2f} (94% interval {sd.quantile(0.03):.2f} to {sd.quantile(0.97):.2f}), observed {y_bb.std():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
ks = torch.arange(n_seeds + 1)
axes[0].bar(ks, torch.stack([(y_bb == k).double().mean() for k in ks]), color="0.8", edgecolor="black", label="observed")
for color, label, rep, shift in (("C0", "beta-binomial", rep_bb, -0.15), ("C1", "binomial", rep_bin, 0.15)):
    freq = torch.stack([(rep == k).double().mean(0) for k in ks])
    lo, med, hi = freq.quantile(torch.tensor([0.03, 0.5, 0.97], dtype=freq.dtype), dim=1)
    axes[0].errorbar(ks + shift, med, yerr=torch.stack([med - lo, hi - med]), fmt="o", color=color, capsize=2, markersize=3, label=f"{label}: predicted")
    axes[1].hist(rep.std(0), bins=30, color=color, alpha=0.6, label=label)
axes[0].set(xlabel="germinated seeds per plot", ylabel="relative frequency", title="Counts")
axes[0].legend(frameon=False, fontsize="small")
axes[1].axvline(y_bb.std(), color="black", linestyle="--", label="observed")
axes[1].set(xlabel="standard deviation of the counts", yticks=[], title="Predicted spread")
axes[1].legend(frameon=False, fontsize="small")
plt.show()

# %%
# Category proportions: Dirichlet-multinomial
# -------------------------------------------
# Now 80 samples are classified into four categories with 50 observations
# each, and the category probabilities differ between samples, drawn from a
# Dirichlet distribution with concentration :math:`(8, 4, 2, 1.5)`. The
# Multinomial model with a flat ``Dirichlet`` prior on the shared probabilities
# is compared with ``DirichletMultinomial`` and a ``Gamma`` prior on each
# concentration.

n_samples, n_trials = 80, 50
conc_true = torch.tensor([8.0, 4.0, 2.0, 1.5], dtype=torch.float64)
y_dm = DirichletMultinomial(n_trials, conc_true).sample(n_samples)
print(f"\nsd of the first proportion across samples: {(y_dm[:, 0] / n_trials).std():.3f}")

with Model() as dirichlet_multinomial:
    conc = RandomParameter("conc", Gamma(2.0, 0.2), shape=4)
    with plate("samples", y_dm):
        ObservedParameter("y", DirichletMultinomial(n_trials, conc), y_dm)

with Model() as multinomial:
    probs = RandomParameter("probs", Dirichlet(torch.ones(4, dtype=torch.float64)))
    with plate("samples", y_dm):
        ObservedParameter("y", Multinomial(n_trials, probs=probs), y_dm)

fit_dm = fit(dirichlet_multinomial, "Dirichlet-multinomial model")
fit_mn = fit(multinomial, "multinomial model")
print(f"true concentration: {conc_true.tolist()}")

# %%
# Predictive check for the proportions
# ------------------------------------
# The check is the standard deviation, across the 80 samples, of each
# category's proportion. As above, the plain model predicts a much smaller
# spread than observed for every category, and the overdispersed model covers it.

rep_dm = DirichletMultinomial(n_trials, thin(fit_dm["conc"])).sample(n_samples)  # (samples, draws, 4)
rep_mn = Multinomial(n_trials, probs=thin(fit_mn["probs"])).sample(n_samples)
observed_sd = (y_dm / n_trials).std(0)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True, layout="constrained")
for j, ax in enumerate(axes):
    for color, label, rep in (("C0", "Dirichlet-multinomial", rep_dm), ("C1", "multinomial", rep_mn)):
        sd = (rep[..., j] / n_trials).std(0)
        ax.hist(sd, bins=30, color=color, alpha=0.6, label=label)
        print(f"category {j}, {label}: predicted sd {sd.mean():.3f} (94% interval {sd.quantile(0.03):.3f} to {sd.quantile(0.97):.3f}), observed {observed_sd[j]:.3f}")
    ax.axvline(observed_sd[j], color="black", linestyle="--", label="observed")
    ax.set(xlabel="sd of the proportion", title=f"category {j}", yticks=[])
axes[0].legend(frameon=False, fontsize="small")
plt.show()
