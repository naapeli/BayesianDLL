"""
Structured Variational Groups
=============================

In a hierarchical model the spread of the group effects depends on the
population scale: when ``tau`` is small, every ``theta_j`` must be close to
``mu``. With weakly informative data this dependence dominates the
posterior, which then has the shape of a funnel in ``(log tau, theta_j)``.
A mean-field guide cannot represent it, because it treats ``tau`` and
``theta`` as independent.

A **structured group** is a ``Group`` used as a context manager without a
``family``. Inside it, guide latents may depend on each other, and ``q``
factorises along their graph. Here the guide is

.. math::

    q(\\mu)\\, q(\\tau)\\, q(\\theta \\mid \\mu, \\tau), \\qquad
    \\theta_j = \\mu + \\tau (a_j + b_j \\varepsilon_j),

written with deterministic blocks whose arguments are guide latents and
``VariationalParameter`` objects. ``tau`` gets a ``LogNormal`` entry, which
describes ``tau`` itself. Draws are ancestral and exact (first ``mu`` and
``tau``, then ``theta`` given them), and the ELBO gradient is
reparameterised through the whole chain.

The model is the eight schools model in its centered form,

.. math::

    \\mu \\sim N(0, 5), \\quad \\tau \\sim \\text{HalfCauchy}(5), \\quad
    \\theta_j \\sim N(\\mu, \\tau), \\quad y_j \\sim N(\\theta_j, \\sigma_j).

A mean-field guide and the structured group are fitted to it. NUTS on the
equivalent non-centered model, which has the same posterior and samples
without divergences, is the reference.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Group, Guide, Model, ObservedParameter, RandomParameter, VariationalParameter
from BayesianDLL.Deterministic import Add, Affine, Multiply
from BayesianDLL.Distributions import HalfCauchy, LogNormal, LogTransform, Normal
from BayesianDLL.Variational import BBVI

torch.manual_seed(0)

y = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0], dtype=torch.float64)
sigma = torch.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0], dtype=torch.float64)
n_schools = len(y)

with Model() as model:
    mu = RandomParameter("mu", Normal(0.0, 5.0))
    tau = RandomParameter("tau", HalfCauchy(5.0))
    theta = RandomParameter("theta", Normal(mu, tau), shape=(n_schools,))
    ObservedParameter("y", Normal(theta, sigma), y)

with Model() as non_centered:
    mu_nc = RandomParameter("mu", Normal(0.0, 5.0))
    tau_nc = RandomParameter("tau", HalfCauchy(5.0))
    z = RandomParameter("z", Normal(0.0, 1.0), shape=(n_schools,))
    theta_nc = Affine("theta", z, loc=mu_nc, scale=tau_nc)
    ObservedParameter("y", Normal(theta_nc, sigma), y)


# %%
# The structured guide
# --------------------
# ``Group(["mu", "tau", "theta"])`` lists the model latents that the block
# defines; the deterministic nodes ``theta_shift``, ``theta_loc`` and
# ``theta_scale`` belong to the group too. The group is independent of
# everything outside it. Here it covers the whole model.


def vp(name, value, shape=(1,), transform=None):
    return VariationalParameter(name, torch.full(shape, value, dtype=torch.float64), transform=transform)


with Guide(model) as structured:
    with Group(["mu", "tau", "theta"]):
        q_mu = RandomParameter("mu", Normal(vp("mu_loc", 0.0), vp("mu_scale", 2.0, transform=LogTransform())))
        q_tau = RandomParameter("tau", LogNormal(vp("tau_loc", 1.0), vp("tau_scale", 0.5, transform=LogTransform())))
        shift = Multiply("theta_shift", q_tau, vp("a", 0.0, (n_schools,)))
        loc = Add("theta_loc", q_mu, shift)
        scale = Multiply("theta_scale", q_tau, vp("b", 1.0, (n_schools,), transform=LogTransform()))
        RandomParameter("theta", Normal(loc, scale), shape=(n_schools,))

mean_field = Guide(model, default_family="mean_field", init_scale=1.0)

histories, traces, seconds = {}, {}, {}
for label, guide in (("mean-field", mean_field), ("structured", structured)):
    torch.manual_seed(1)
    start = time.perf_counter()
    histories[label] = BBVI(model, guide, n_samples=2, epochs=4500, lr=0.02, lrd=0.999, callback_frequency=5, verbose=False)
    seconds[label] = time.perf_counter() - start
    traces[label] = guide.sample(2000)
    print(f"{label:>10}: {seconds[label]:4.1f} s, final ELBO {sum(histories[label][-100:]) / 100:.2f}")

traces["NUTS"] = non_centered.sample(n_samples=80, warmup_length=80, n_chains=2, progress_bar=False, seed=0, delta=0.9,
                                     chain_method="vectorized")
print(f"NUTS (non-centered): {sum(traces['NUTS'].divergences)} divergences")
for label, trace in traces.items():
    log_tau = torch.log(trace["tau"]).flatten()
    theta_1 = trace["theta"][..., 0].flatten()
    print(f"{label:>10}: mean tau {trace['tau'].mean().item():5.2f}, sd log tau {log_tau.std().item():.2f}, "
          f"corr(log tau, |theta_1 - mu|) {torch.corrcoef(torch.stack([log_tau, (theta_1 - trace['mu'].flatten()).abs()]))[0, 1].item():+.2f}")

# %%
# The funnel
# ----------
# Each panel shows ``theta_1 - mu``, the first school's deviation from the
# population mean, against ``log tau``. NUTS draws fill a funnel: the
# deviations narrow as ``tau`` shrinks. The structured guide reproduces this
# shape, because its ``theta`` scales with the sampled ``tau``. The
# mean-field draws form an axis-aligned blob: deviations of the same size at
# every ``tau``. To keep away from small ``tau``, where such deviations would
# be improbable, the mean-field guide also narrows the marginal of
# ``log tau`` to a fraction of its posterior width. The structured guide
# reaches a higher ELBO and a ``log tau`` marginal close to the NUTS one.
# Its remaining error is in the shape of each factor: ``q(tau)`` is
# log-normal, and ``theta_1`` given ``tau`` is symmetric about ``mu``, while
# the posterior of the first school leans towards its large observed effect.

colors = {"mean-field": "C1", "structured": "C0", "NUTS": "black"}
fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
for ax, label in zip(axes[:3], ("NUTS", "mean-field", "structured")):
    trace = traces[label]
    deviation = (trace["theta"][..., 0] - trace["mu"][..., 0]).flatten()
    ax.scatter(torch.log(trace["tau"]).flatten(), deviation, s=3, alpha=0.25, color=colors[label])
    ax.set_xlim(-4, 4)
    ax.set_ylim(-30, 40)
    ax.set_xlabel("log tau")
    ax.set_ylabel("theta_1 - mu")
    ax.set_title(label)
ax = axes[3]
for label, trace in traces.items():
    ax.hist(torch.log(trace["tau"]).flatten(), bins=60, range=(-4, 4), density=True, histtype="step", lw=1.6,
            color=colors[label], label=label)
ax.set_xlabel("log tau")
ax.set_ylabel("density")
ax.set_title("Marginal of log tau")
ax.legend(fontsize="small")
plt.show()

# %%
# Moments of a structured group
# -----------------------------
# Inside a structured group the latents depend on each other through
# arbitrary blocks, so ``guide.covariance(...)`` estimates its moments by
# Monte Carlo in both spaces, from exact ancestral draws, and lists them in
# ``estimated``.

mean = structured.mean("constrained", n_samples=2000)
result = structured.covariance("constrained", n_samples=2000)
print(f"estimated: {result.estimated}, n = {result.n_samples}")
print(f"E[tau] = {mean['tau'].item():.2f} +- {result.standard_error['tau'].item():.2f}")
