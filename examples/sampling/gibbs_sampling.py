"""
Gibbs Sampling of a Change Point
================================

Yearly event counts ``y_t`` follow a Poisson distribution whose rate drops
from ``early_rate`` to ``late_rate`` at an unknown year ``tau``. The change
point is a discrete parameter with a finite support (a ``DiscreteUniform``
prior), so the sampler picks the exact Gibbs sampler (``FiniteGibbs``) for it
automatically, while NUTS updates the two rates.

A Gibbs update evaluates the target at every value of the support and draws
the new value from the normalised full conditional ``p(tau | rates, y)``.
Unlike Metropolis it needs no proposal and no tuning, and nothing is ever
rejected. The same model is also sampled with a random-walk Metropolis update
for ``tau``: it proposes neighbouring years, rejects many proposals and moves
slowly across the posterior.

With conjugate ``Gamma`` priors on the rates, ``p(tau | y)`` is known in closed
form, which the figures use as the reference. They show the data with the
inferred rates, the posterior of ``tau`` from both samplers against the exact
one, the full conditional that a Gibbs update draws from, and the traces and
autocorrelations of ``tau``. The ``rate`` node is built with the deterministic
block ``Elementwise``.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Elementwise
from BayesianDLL.Distributions import DiscreteUniform, Gamma, Poisson
from BayesianDLL.Evaluation import effective_sample_size, summary


torch.manual_seed(4)

n_years = 40
tau_true, early_true, late_true = 24, 3.5, 1.2
years = torch.arange(n_years, dtype=torch.float64)
counts = torch.poisson(torch.where(years < tau_true, early_true, late_true) * torch.ones(n_years, dtype=torch.float64))

prior_shape, prior_rate = 2.0, 1.0  # Gamma(2, 1) prior on both rates


def piecewise(early, late, tau):
    return torch.where(years < tau, early, late)


def piecewise_partials(early, late, tau):
    before = (years < tau).to(early.dtype)
    return before, 1 - before, None  # tau is discrete: no gradient


def change_point_model(tau_sampler):
    with Model() as model:
        tau = RandomParameter(
            "tau",
            DiscreteUniform(1, n_years - 1),
            sampler=tau_sampler,
            initial_value=torch.tensor([n_years // 2], dtype=torch.float64),
        )
        early = RandomParameter("early_rate", Gamma(prior_shape, prior_rate), delta=0.9)
        late = RandomParameter("late_rate", Gamma(prior_shape, prior_rate), delta=0.9)
        rate = Elementwise("rate", piecewise, piecewise_partials, early, late, tau)
        with plate("years", n_years):
            ObservedParameter("counts", Poisson(rate), counts)
    return model


# %%
# Exact posterior of the change point
# -----------------------------------
# Integrating out each Gamma-distributed rate leaves a Gamma-Poisson marginal
# likelihood for each of the two segments.
def log_segment_evidence(segment):
    total, length = segment.sum(), len(segment)
    return (torch.lgamma(prior_shape + total) - torch.lgamma(torch.tensor(prior_shape))
            + prior_shape * torch.log(torch.tensor(prior_rate))
            - (prior_shape + total) * torch.log(prior_rate + torch.tensor(float(length))))


support = torch.arange(1, n_years)
log_evidence = torch.stack([log_segment_evidence(counts[:t]) + log_segment_evidence(counts[t:]) for t in support])
exact = torch.softmax(log_evidence, dim=0)

# %%
# Gibbs and Metropolis updates of tau
# -----------------------------------
# Four Gibbs chains converge quickly. The Metropolis variant is run as one
# longer chain: it mixes so slowly that several short chains would disagree
# (their R-hat would be far above 1.01), which is exactly the point here.
gibbs = change_point_model("auto").sample(500, 300, n_chains=4, seed=0, progress_bar=False,
                                          chain_method="vectorized")
metropolis = change_point_model("metropolis").sample(2000, 500, n_chains=1, seed=0, progress_bar=False,
                                                    check_convergence=False, chain_method="vectorized")

print("samplers per block (Gibbs run):", list(zip(gibbs.sample_stats.params, gibbs.sample_stats.samplers)))
print(summary(gibbs).to_string())
for name, result in (("Gibbs", gibbs), ("Metropolis", metropolis)):
    ess = effective_sample_size({"tau": result["tau"].double()})["tau"].item()
    print(f"{name:10s}: bulk ESS of tau = {ess:6.0f}, mean tau = {result['tau'].double().mean():.2f}")
tau_block = [i for i, params in enumerate(metropolis.sample_stats.params) if params == ["tau"]][0]
print(f"Metropolis acceptance rate for tau: {metropolis.sample_stats[tau_block]['accepted'].double().mean():.2f}")


def autocorrelation(chain, max_lag=40):
    chain = chain - chain.mean()
    variance = (chain * chain).mean()
    return torch.stack([(chain[: len(chain) - k] * chain[k:]).mean() / variance for k in range(max_lag + 1)])


# At fixed rates, a Gibbs update draws ``tau`` from its full conditional
# exactly. The lower-left panel shows it with the rates at their posterior
# means.
early_mean, late_mean = gibbs["early_rate"].mean(), gibbs["late_rate"].mean()
log_conditional = torch.stack([
    torch.distributions.Poisson(piecewise(early_mean, late_mean, t)).log_prob(counts).sum() for t in support
])
window = support[(support > tau_true - 10) & (support < tau_true + 10)]

fig = plt.figure(figsize=(14, 8), layout="constrained")
grid = fig.add_gridspec(2, 3)

ax = fig.add_subplot(grid[0, :2])
ax.bar(years, counts, color="lightgray", label="yearly counts")
rate_draws = gibbs["rate"].reshape(-1, n_years)
ax.fill_between(years, rate_draws.quantile(0.05, 0), rate_draws.quantile(0.95, 0), color="C0", alpha=0.3,
                step="mid", label="90% band of the rate")
ax.step(years, rate_draws.mean(0), where="mid", color="C0", label="posterior mean rate")
ax.step(years, piecewise(torch.tensor(early_true), torch.tensor(late_true), tau_true), where="mid",
        color="C3", linestyle="--", label="true rate")
ax.set_ylim(0, 1.35 * counts.max())
ax.set_xlabel("year")
ax.set_ylabel("count")
ax.set_title("Data and inferred rate")
ax.legend(frameon=False, ncols=4, loc="upper center", fontsize="small")

ax = fig.add_subplot(grid[0, 2])
for offset, (name, result) in zip((-0.2, 0.2), (("Gibbs", gibbs), ("Metropolis", metropolis))):
    draws = result["tau"].reshape(-1).long()
    frequency = torch.bincount(draws, minlength=n_years).double() / len(draws)
    ax.bar(window + offset, frequency[window], width=0.4, alpha=0.8, label=name)
ax.plot(window, exact[window - 1], "ko", markersize=4, label="exact")
ax.set_xlabel("change point tau")
ax.set_ylabel("posterior probability")
ax.set_title("Posterior of the change point")
ax.legend(frameon=False, fontsize="small")

ax = fig.add_subplot(grid[1, 0])
ax.bar(window, torch.softmax(log_conditional, 0)[window - 1], color="C0", alpha=0.8, label="p(tau | rates, y)")
ax.plot(window, exact[window - 1], "ko", markersize=4, label="p(tau | y), exact")
ax.set_xlabel("change point tau")
ax.set_ylabel("probability")
ax.set_title("Full conditional of one Gibbs update")
ax.legend(frameon=False, fontsize="small")

ax = fig.add_subplot(grid[1, 1])
ax.plot(gibbs["tau"][0].reshape(-1), drawstyle="steps-mid", label="Gibbs")
ax.plot(metropolis["tau"][0, :500].reshape(-1), drawstyle="steps-mid", alpha=0.8, label="Metropolis")
ax.set_xlabel("iteration after warmup (first chain)")
ax.set_ylabel("tau")
ax.set_title("Trace of the change point")
ax.legend(frameon=False, fontsize="small")

ax = fig.add_subplot(grid[1, 2])
for name, result in (("Gibbs", gibbs), ("Metropolis", metropolis)):
    ax.plot(autocorrelation(result["tau"][0].reshape(-1).double()), marker=".", label=name)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("lag")
ax.set_ylabel("autocorrelation")
ax.set_title("Autocorrelation of tau")
ax.legend(frameon=False, fontsize="small")
plt.show()
