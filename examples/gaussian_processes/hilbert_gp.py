"""
Hilbert-Space Approximate GPs
=============================

A latent GP on ``n`` inputs factorises an ``n x n`` covariance matrix at every
evaluation, ``O(n³)``. For a stationary kernel on low-dimensional inputs, the
Hilbert-space approximation (Solin & Särkkä, 2020; Riutort-Mayol et al.,
2023) needs no matrix factorisation at all. On a box ``[c - L, c + L]`` the
covariance operator is expanded in the eigenfunctions of the Laplacian,

``f(x) ≈ Σ_j φ_j(x) sqrt(S(sqrt(λ_j))) z_j``,  ``z_j ~ Normal(0, 1)``,

where ``φ_j(x) = L^{-1/2} sin(π j (x - c + L) / (2L))`` are fixed sine
waves with eigenvalues ``λ_j = (π j / (2L))²`` and ``S`` is the kernel's
spectral density. The hyperparameters only rescale the weights of the basis
functions, so an evaluation costs ``O(n m)`` for ``m`` basis functions,
and predicting at new inputs simply evaluates the basis there.

``LatentGP(..., n_basis=m, boundary_factor=c_b)`` builds this approximation
with ``L = c_b S`` for the half-range ``S`` of the inputs. Here it models the
log rate of 2000 Poisson counts.
"""

import math
import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal, Poisson
from BayesianDLL.GP import ConstantMean, LatentGP, Matern52


torch.manual_seed(1)


def true_log_rate(t):
    return 1.0 + 0.8 * torch.sin(0.9 * t) + 0.4 * torch.cos(2.3 * t)


n = 2000
t = torch.sort(torch.rand(n, dtype=torch.float64) * 20.0).values
counts = torch.poisson(torch.exp(true_log_rate(t)))

# %%
# How many basis functions?
# -------------------------
# The approximation is good when the basis covers the kernel's spectrum
# (``m`` large enough for the lengthscale) and the boundary is far enough
# from the data (``c_b`` around 1.5 or more). Riutort-Mayol et al. suggest
# ``m ≳ 1.75 c_b S / l``: for the half-range ``S = 10``, ``c_b = 1.5`` and a
# lengthscale ``l = 1`` that is about 26. The approximate covariance
# ``Σ_j S(sqrt(λ_j)) φ_j(x) φ_j(x')`` is compared with the Matérn 5/2 kernel
# at ``x = 10`` (the centre of the data) and ``x = 1`` (near its edge).

kernel = Matern52(1.0, 1.0)
c, L = 10.0, 1.5 * 10.0
distances = torch.linspace(0.0, 4.0, 200, dtype=torch.float64)


def approximate_covariance(x0, m):
    j = torch.arange(1, m + 1, dtype=torch.float64)
    sqrt_eigenvalues = math.pi * j / (2 * L)

    def basis(x):
        return torch.sin(sqrt_eigenvalues * (x[:, None] - c + L)) / math.sqrt(L)

    density = kernel.spectral_density(sqrt_eigenvalues)
    return (basis(torch.tensor([x0], dtype=torch.float64)) * density) @ basis(x0 + distances).T


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout="constrained")
for ax, x0 in zip(axes, (10.0, 1.0)):
    ax.plot(distances, kernel(distances, torch.zeros(1, dtype=torch.float64))[:, 0], "k", linewidth=2.5, label="Matérn 5/2")
    for m, color in zip((8, 16, 32), ("tab:red", "tab:orange", "tab:blue")):
        ax.plot(distances, approximate_covariance(x0, m)[0], "--", color=color, label=f"m = {m}")
    ax.set(xlabel="distance x' - x", title=f"k(x, x') at x = {x0:g}")
axes[0].set_ylabel("covariance")
axes[0].legend(fontsize="small")
plt.show()

# %%
# The model
# ---------
# The latent parameters are the 30 basis weights, whatever the number of
# observations, and the rate is ``exp(f)``. By default ``LatentGP`` samples
# standard-normal weights ``z`` and scales them by ``sqrt(S)`` (the
# non-centred form). With 2000 counts the data pin the function down, so the
# weights ``sqrt(S) z`` are nearly fixed and every change of the lengthscale
# or variance has to rescale all ``z`` at once, a curved posterior in which
# the hyperparameters mix slowly. ``centred=True`` samples the weights
# themselves (``f_weights``, with prior standard deviations
# ``f_basis_sd = sqrt(S)``): their posterior then hardly depends on the
# hyperparameters, which here gives several times the effective sample size
# for the same number of draws. The non-centred form is the better choice
# when the data are weakly informative.

with Model() as model:
    inputs = Data("t", t)
    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 0.5))
    log_variance = RandomParameter("log_variance", Normal(-1.0, 0.7))
    intercept = RandomParameter("intercept", Normal(0.0, 2.0))
    kernel = Matern52(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
    f = LatentGP("f", inputs, kernel, mean=ConstantMean(intercept), n_basis=30, boundary_factor=1.5, centred=True)
    ObservedParameter("counts", Poisson(Exp("rate", f)), counts)

# %%
# Cost of one evaluation
# ----------------------
# One log density and gradient evaluation, compared with the dense latent GP
# on the same 2000 inputs (one ``2000 x 2000`` Cholesky factorisation and its
# derivative).

with Model() as dense_model:
    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 0.5))
    log_variance = RandomParameter("log_variance", Normal(-1.0, 0.7))
    dense_kernel = Matern52(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
    dense_f = LatentGP("f", Data("t", t), dense_kernel)
    ObservedParameter("counts", Poisson(Exp("rate", dense_f)), counts)

for label, current in (("Hilbert space, m = 30", model), ("dense", dense_model)):
    current.joint_grad_log_prob()
    start = time.perf_counter()
    current.model_log_prob()
    current.joint_grad_log_prob()
    print(f"{label:>22}: log density + gradient in {1000 * (time.perf_counter() - start):.1f} ms")

# %%
# Sampling
# --------

start = time.perf_counter()
trace = model.sample(400, 400, n_chains=4, chain_method="vectorized", max_depth=6, init_scale=0.1, seed=0, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary()
print(summary.loc[["log_lengthscale[0]", "log_variance[0]", "intercept[0]"], ["mean", "sd", "ess_bulk", "r_hat"]])
print(f"all {len(summary)} parameters: largest R-hat {summary['r_hat'].max():.3f}, smallest bulk ESS {summary['ess_bulk'].min():.0f}")
print(f"sampling: {elapsed:.1f} s, mean step size {trace.sample_stats['step_size'].double().mean():.3f}, "
      f"divergences: {sum(trace.divergences)}")

# %%
# Posterior rate and a forecast
# -----------------------------
# Setting new inputs evaluates the basis there. Predictions are trustworthy
# inside the approximation's domain: up to ``t = 25`` here, a quarter of the
# boundary margin beyond the data (shaded); towards ``c + L = 25`` the basis
# functions vanish and the approximate prior variance shrinks.

grid = torch.linspace(0.0, 24.0, 300, dtype=torch.float64)
inputs.set_value(grid)
rates = model.posterior_predicative(trace.thin(2), samples_per_step=1, progress_bar=False)["rate"].reshape(-1, grid.numel())
lower, median, upper = torch.quantile(rates, torch.tensor([0.05, 0.5, 0.95], dtype=rates.dtype), dim=0)

fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
ax.scatter(t, counts, s=2, color="0.7", label="counts")
ax.fill_between(grid, lower, upper, color="tab:blue", alpha=0.3, label="90% band of the rate")
ax.plot(grid, median, color="tab:blue", label="posterior median rate")
ax.plot(grid, torch.exp(true_log_rate(grid)), "k--", linewidth=1, label="true rate")
ax.axvspan(20.0, 24.0, color="0.92", zorder=0)
ax.set(xlabel="t", ylabel="count", title="Poisson rate with a Hilbert-space GP (2000 observations, 30 basis functions)")
ax.legend(loc="upper left", fontsize="small", ncol=2)
plt.show()
