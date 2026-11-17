"""
Eight Schools: Centered vs Non-Centered
=======================================

Eight schools ran the same coaching programme, and each reports an
estimated effect ``y_j`` on test scores with a known standard error
``sigma_j`` (Rubin, 1981). The school effects ``theta_j`` are modelled as
draws from a common population:

.. math::

    \\mu \\sim N(0, 5), \\quad \\tau \\sim \\text{HalfCauchy}(5), \\quad
    \\theta_j \\sim N(\\mu, \\tau), \\quad y_j \\sim N(\\theta_j, \\sigma_j).

The standard errors (9 to 18) are large compared with the spread of the
``y_j``, so the data say little about each ``theta_j`` and little about
``tau``. The posterior then has the shape of a funnel: for large ``tau`` the
effects may spread widely, for small ``tau`` they are squeezed together
around ``mu``. The step size that suits the wide mouth is too large for the
narrow neck, so NUTS on the **centered** model (``theta`` sampled directly)
reports divergent transitions and fails to reach small ``tau``. Its estimates
are biased, which no amount of extra sampling fixes.

The **non-centered** model samples standardised effects ``z_j ~ N(0, 1)``
and computes ``theta = mu + tau * z`` with the ``Affine`` block. A priori,
``z`` is independent of ``tau``, and the weak data barely change that, so the
sampler sees a nearly round posterior without a funnel. Both models describe
the same distribution of ``theta``; only the coordinates differ.

The comparison is the point of this example, so both models are run with
identical settings. The figures show the draws of ``(theta_1, log tau)``
with the divergent transitions marked, the posterior of ``tau`` under both
models, and the school effects of both against the reported ``y_j``.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Affine
from BayesianDLL.Distributions import HalfCauchy, Normal


y = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0], dtype=torch.float64)
sigma = torch.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0], dtype=torch.float64)
n_schools = len(y)


def centered_model():
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        tau = RandomParameter("tau", HalfCauchy(5.0))
        theta = RandomParameter("theta", Normal(mu, tau), shape=(n_schools,))
        ObservedParameter("y", Normal(theta, sigma), y)
    return model


def non_centered_model():
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        tau = RandomParameter("tau", HalfCauchy(5.0))
        z = RandomParameter("z", Normal(0.0, 1.0), shape=(n_schools,))
        theta = Affine("theta", z, loc=mu, scale=tau)
        ObservedParameter("y", Normal(theta, sigma), y)
    return model


settings = dict(n_samples=500, warmup_length=500, n_chains=4, progress_bar=False, seed=2, delta=0.9)
results = {
    "centered": centered_model().sample(**settings),
    "non-centered": non_centered_model().sample(**settings),
}

n_draws = settings["n_samples"] * settings["n_chains"]
for name, result in results.items():
    summary = result.summary(include_deterministic=True)
    print(f"{name:>12}: {sum(result.divergences)} divergent transitions in {n_draws} draws, "
          f"largest R-hat {summary['r_hat'].max():.3f}, smallest bulk ESS {summary['ess_bulk'].min():.0f}, "
          f"smallest tau {result['tau'].min().item():.4f}")
    print(summary.loc[["mu[0]", "tau[0]"]])

# %%
# The funnel and the posterior of tau
# -----------------------------------
# The centered chains stop short of the neck of the funnel (hardly any draws
# below ``log tau = -0.5``), and the divergent transitions they report (red)
# sit at the lower edge of the explored region: the sampler signals that it
# could not follow the posterior further. How many divergences appear varies
# from run to run; any number of them is a warning. The non-centered draws
# reach far down the neck without a single divergence. As a result the centered model
# underrepresents small ``tau``, as the histogram of ``log tau`` shows.

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
colors = {"centered": "C1", "non-centered": "C0"}
for ax, (name, result) in zip(axes[:2], results.items()):
    log_tau = torch.log(result["tau"]).flatten()
    theta0 = result["theta"][..., 0].flatten()
    diverging = result.sample_stats["diverging"].flatten().bool()
    ax.scatter(theta0[~diverging], log_tau[~diverging], s=4, alpha=0.3, color=colors[name], label="draws")
    ax.scatter(theta0[diverging], log_tau[diverging], s=16, color="red", label=f"divergent ({int(diverging.sum())})")
    ax.set_title(f"{name} parameterisation")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\log \tau$")
    ax.set_ylim(-8, 4)
    ax.legend(fontsize="small", loc="lower right")

for name, result in results.items():
    axes[2].hist(torch.log(result["tau"]).flatten(), bins=60, range=(-8, 4), density=True, histtype="step", lw=2,
                 color=colors[name], label=name)
axes[2].set_xlabel(r"$\log \tau$")
axes[2].set_ylabel("density")
axes[2].set_title(r"Posterior of $\log \tau$")
axes[2].legend(fontsize="small", loc="upper left")
plt.show()

# %%
# School effects
# --------------
# Partial pooling pulls every school's effect from its own estimate ``y_j``
# (black crosses) towards the overall mean, more strongly for the schools
# with large standard errors. The two parameterisations agree on the means
# but differ slightly in the intervals, because the centered chains miss the
# neck of the funnel, where all effects are close to ``mu``.

positions = torch.arange(n_schools, dtype=torch.float64)
fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
ax.errorbar(positions - 0.25, y, yerr=sigma, fmt="x", color="black", capsize=3, label=r"reported $y_j \pm \sigma_j$")
for offset, (name, result) in zip((0.0, 0.25), results.items()):
    theta = result["theta"].flatten(0, 1)
    mean = theta.mean(0)
    lower, upper = theta.quantile(torch.tensor([0.05, 0.95], dtype=torch.float64), dim=0)
    ax.errorbar(positions + offset, mean, yerr=torch.stack([mean - lower, upper - mean]), fmt="o", color=colors[name],
                capsize=3, label=f"{name}: mean and 90% interval")
ax.axhline(results["non-centered"]["mu"].mean(), color="grey", linestyle=":", label=r"posterior mean of $\mu$")
ax.set_xticks(positions, [f"school {j + 1}" for j in range(n_schools)])
ax.set_ylabel("effect")
ax.set_title("School effects: reported estimates and partially pooled posteriors")
ax.legend(fontsize="small", ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
plt.show()
