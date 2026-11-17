"""
Vectorized chains
=================

Runs 64 NUTS chains of the non-centered eight schools model in one process
with ``chain_method="vectorized"`` and compares the wall time with the
default ``chain_method="parallel"``, which runs one worker process per chain.

In vectorized mode the model is *batched*: every latent parameter gets a
leading chain dimension, and a single evaluation of the log density and its
gradient covers all chains. The NUTS trajectories of all chains are built in
lock step; chains whose trajectory has already turned around are masked.
Step sizes, metrics and all statistics remain per chain, and the result is
the same :class:`~BayesianDLL.SamplingResult` as in parallel mode.

The vectorized mode is also the way to sample on a GPU: with a CUDA (or, with
the ROCm build of PyTorch, an AMD) GPU, ``device="cuda"`` moves the model to
the GPU (``Model.to``) and the chains run there.
"""

import time
import warnings

import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate, sample
from BayesianDLL.Deterministic import Affine, Index
from BayesianDLL.Distributions import HalfCauchy, Normal
from BayesianDLL.Evaluation import summary

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# The non-centered eight schools model
# ------------------------------------
y = torch.tensor([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = torch.tensor([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def build_model():
    with Model() as model:
        school = Data("school", torch.arange(8))
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        tau = RandomParameter("tau", HalfCauchy(5.0))
        with plate("schools", 8):
            eta = RandomParameter("eta", Normal(0.0, 1.0), shape=())
        theta = Affine("theta", eta, loc=mu, scale=tau)
        ObservedParameter("y", Normal(Index("theta_obs", theta, school), sigma), y)
    return model


# %%
# 64 vectorized chains
# --------------------
n_chains, n_samples, warmup = 64, 100, 100
start = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vectorized = sample(n_samples, warmup, n_chains=n_chains, model=build_model(), chain_method="vectorized",
                        device=device, delta=0.8, seed=1, progress_bar=False)
vectorized_time = time.perf_counter() - start
print(f"vectorized on {device}: {vectorized_time:.1f} s for {n_chains} chains")
print(summary(vectorized))

# %%
# The same chains with one process per chain
# ------------------------------------------
# Only possible on the CPU. The worker processes share the machine's cores.
start = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    parallel = sample(n_samples, warmup, n_chains=n_chains, model=build_model(), chain_method="parallel",
                      delta=0.8, seed=1, progress_bar=False)
parallel_time = time.perf_counter() - start
print(f"parallel: {parallel_time:.1f} s for {n_chains} chains")
print(f"speedup of the vectorized chains: {parallel_time / vectorized_time:.1f}x")

# %%
# Both modes sample the same posterior
# ------------------------------------
for name in ("mu", "tau"):
    print(f"{name}: vectorized mean {vectorized[name].mean():.2f}, parallel mean {parallel[name].mean():.2f}")
print("divergences (vectorized, parallel):", sum(vectorized.divergences), sum(parallel.divergences))
