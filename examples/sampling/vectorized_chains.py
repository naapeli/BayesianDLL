"""
Vectorized Chains
=================

``sample`` can run its chains in two ways:

* ``chain_method="parallel"`` (the default on a CPU) starts one worker
  process per chain; every process evaluates the model for its own chain;
* ``chain_method="vectorized"`` runs all chains in one process. Every latent
  parameter gets a leading chain dimension, and one evaluation of the log
  density and its gradient serves all chains at once. NUTS builds the
  trajectories of all chains in lock step and masks chains that have
  finished. Step sizes, mass matrices and statistics stay per chain. This is
  also the mode that runs on a GPU (``device="cuda"``).

Both return the same kind of result. Here 16 chains sample the eight schools
model in its non-centred form, ``theta = mu + tau * eta`` with
``eta ~ Normal(0, 1)``, written with the deterministic blocks ``Affine`` and
``Index``. The centred form has a funnel between ``tau`` and ``theta`` that
causes divergent transitions; the non-centred form removes it. The figure
compares the wall times and overlays the posteriors of ``mu`` and ``tau`` from
the two modes.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate, sample
from BayesianDLL.Deterministic import Affine, Index
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Evaluation import gelman_rubin, summary

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Estimated treatment effects and their standard errors in eight schools.
effects = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
std_errors = torch.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def eight_schools():
    with Model() as model:
        school = Data("school", torch.arange(8))
        mu = RandomParameter("mu", Normal(0.0, 10.0))
        tau = RandomParameter("tau", HalfNormal(10.0))
        with plate("schools", 8):
            eta = RandomParameter("eta", Normal(0.0, 1.0), shape=())
        theta = Affine("theta", eta, loc=mu, scale=tau)
        ObservedParameter("effects", Normal(Index("theta_school", theta, school), std_errors), effects)
    return model


n_chains, n_samples, warmup = 16, 250, 250
settings = {"n_chains": n_chains, "seed": 1, "delta": 0.9, "progress_bar": False}

# %%
# All chains in one process
# -------------------------
start = time.perf_counter()
vectorized = sample(n_samples, warmup, model=eight_schools(), chain_method="vectorized", device=device, **settings)
times = {f"vectorized ({device})": time.perf_counter() - start}

# %%
# One process per chain
# ---------------------
# Only available on the CPU; the worker processes share the machine's cores.
start = time.perf_counter()
parallel = sample(n_samples, warmup, model=eight_schools(), chain_method="parallel", **settings)
times["parallel (CPU)"] = time.perf_counter() - start

print(summary({name: vectorized[name] for name in ("mu", "tau")}).to_string())
for label, result in zip(times, (vectorized, parallel)):
    r_hat = max(value.max().item() for value in gelman_rubin(result.trace).values())
    print(f"{label:18s}: {times[label]:5.1f} s, {sum(result.divergences)} divergences, max R-hat {r_hat:.3f}, "
          f"mu {result['mu'].mean():.2f}, tau {result['tau'].mean():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
axes[0].bar(list(times), list(times.values()), color=["C0", "C1"])
axes[0].set_ylabel("wall time (s)")
axes[0].set_title(f"{n_chains} chains, {n_samples + warmup} iterations each")
for ax, name in zip(axes[1:], ("mu", "tau")):
    for label, result, color in zip(times, (vectorized, parallel), ("C0", "C1")):
        ax.hist(result[name].reshape(-1).cpu().numpy(), bins=60, density=True, histtype="step", linewidth=1.5,
                color=color, label=label)
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    ax.set_title(f"Posterior of {name}")
axes[1].legend(frameon=False, fontsize="small")
plt.show()
