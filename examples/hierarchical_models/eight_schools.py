"""
Eight Schools: Centered vs Non-Centered
=======================================

The classic eight schools model (Rubin, 1981) estimates the effect of a
coaching programme in eight schools from the estimated effects ``y_j`` and
their standard errors ``sigma_j``:

.. math::

    \\mu \\sim N(0, 5), \\quad \\tau \\sim \\text{HalfCauchy}(5), \\quad
    \\theta_j \\sim N(\\mu, \\tau), \\quad y_j \\sim N(\\theta_j, \\sigma_j).

Written like this (the *centered* parameterisation) the posterior has a
funnel: when ``tau`` is small every ``theta_j`` is forced close to ``mu``. NUTS
cannot adapt one step size to both the wide mouth and the narrow neck of the
funnel and reports divergent transitions, and the draws of small ``tau`` are
biased.

The *non-centered* parameterisation samples standardised effects
``z_j ~ N(0, 1)`` and builds ``theta = mu + tau * z`` with the ``Affine`` block.
The sampler then explores a posterior without the funnel, which removes the
divergences. ``Index`` broadcasts the school effects to the observations.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Affine, Index
from BayesianDLL.Distributions import HalfCauchy, Normal


y = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0], dtype=torch.float64)
sigma = torch.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0], dtype=torch.float64)
n_schools = len(y)


def centered_model():
    with Model() as model:
        school = Data("school", torch.arange(n_schools))
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        tau = RandomParameter("tau", HalfCauchy(5.0))
        theta = RandomParameter("theta", Normal(mu, tau), shape=(n_schools,))
        ObservedParameter("y", Normal(Index("theta_obs", theta, school), sigma), y)
    return model


def non_centered_model():
    with Model() as model:
        school = Data("school", torch.arange(n_schools))
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        tau = RandomParameter("tau", HalfCauchy(5.0))
        z = RandomParameter("z", Normal(0.0, 1.0), shape=(n_schools,))
        theta = Affine("theta", z, loc=mu, scale=tau)
        ObservedParameter("y", Normal(Index("theta_obs", theta, school), sigma), y)
    return model


settings = dict(n_samples=500, warmup_length=500, n_chains=2, progress_bar=False, seed=2, delta=0.8)
results = {
    "centered": centered_model().sample(**settings),
    "non-centered": non_centered_model().sample(**settings),
}

for name, result in results.items():
    print(f"{name:>12}: {sum(result.divergences)} divergent transitions "
          f"(out of {settings['n_samples'] * settings['n_chains']} draws), "
          f"smallest tau draw {result['tau'].min().item():.4f}")
    print(result.summary(include_deterministic=True))

# %%
# The divergences of the centered model concentrate in the neck of the funnel
# (small ``tau``): the centered chains never reach small values of ``tau``,
# while the non-centered model explores the whole funnel without divergences.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")
for ax, (name, result) in zip(axes, results.items()):
    log_tau = torch.log(result["tau"]).flatten()
    theta0 = result["theta"][..., 0].flatten()
    diverging = result.sample_stats["diverging"].flatten().bool()
    ax.scatter(theta0[~diverging], log_tau[~diverging], s=6, alpha=0.4, label="draws")
    ax.scatter(theta0[diverging], log_tau[diverging], s=14, color="red", label=f"divergent ({int(diverging.sum())})")
    ax.set_title(f"{name} parameterisation")
    ax.set_xlabel(r"$\theta_1$")
    ax.legend()
axes[0].set_ylabel(r"$\log \tau$")
plt.show()
