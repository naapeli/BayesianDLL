"""
GP Latent Variable Model: Learning the Inputs
=============================================

Usually the inputs ``X`` of a Gaussian process are observed and only the
outputs are modelled. Every GP class except ``ToeplitzGP`` also accepts
*learnable* inputs: a ``RandomParameter`` or ``DeterministicParameter``
passed as ``inputs`` gets hand-derived gradients through the kernel (and the
mean function), like any other parameter. That covers

* **GP latent variable models** (GPLVM): the inputs are unobserved latent
  coordinates with a prior, and the GP maps them to high-dimensional data,
  a non-linear version of PCA;
* **uncertain inputs**: measured inputs with a prior centred on the
  measurement, ``RandomParameter("x", Normal(x_measured, error))``;
* **input warping**: inputs computed from fixed data by deterministic
  blocks, e.g. ``Affine("x", Data("t", t), loc, scale)`` or a monotone
  warp, so that a stationary kernel fits non-stationary data.

This example is a GPLVM. 200 points lie on a closed curve in 8 dimensions:
each output dimension is a different smooth function of an angle
``t ∈ [0, 2π)`` (first and second harmonics), plus noise. The angle is not
given to the model. We infer a 2-D latent coordinate ``x_i`` per point
(``X`` has shape ``(200, 2)`` with a ``N(0, 1)`` prior) through one
``ExactGP`` with

* ``n_outputs=8``: eight independent GPs, one per output dimension, that
  share ``X``, the inducing inputs ``Z``, the kernel and the noise (the
  observations have shape ``(8, 200)``; the log density is the sum over the
  outputs, computed with a single set of ``m x m`` factors);
* learnable inducing inputs ``Z`` (``(25, 2)``) and the VFE bound, which
  stays a lower bound on the marginal likelihood as ``Z`` moves.

``X``, ``Z``, the signal variance and the noise are fitted jointly by a MAP
estimate (``find_MAP``), starting ``X`` from PCA. The latent space is only
identified up to rotation, reflection and smooth deformation, so success
means that the learned coordinates trace a closed loop in the order of
``t``.
"""

import math
import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import ExactGP, RBF, exact_gp_predictive

torch.manual_seed(0)
dtype = torch.float64

# %%
# Data: a noisy closed curve in eight dimensions
# ----------------------------------------------
# Output ``d`` is ``a_d cos t + b_d sin t + c_d cos 2t + e_d sin 2t`` with
# random coefficients, plus Gaussian noise; every output is then
# standardised. The second harmonics bend the curve out of every plane, so
# the first two principal components show it strongly distorted.

n, n_outputs, m, latent_dim = 200, 8, 25, 2
t = torch.rand(n, dtype=dtype) * 2 * math.pi
harmonics = torch.stack([torch.cos(t), torch.sin(t), torch.cos(2 * t), torch.sin(2 * t)])  # (4, n)
coefficients = torch.randn(n_outputs, 4, dtype=dtype) * torch.tensor([1.0, 1.0, 0.7, 0.7], dtype=dtype)
y = coefficients @ harmonics + 0.1 * torch.randn(n_outputs, n, dtype=dtype)  # (D, n)
y = (y - y.mean(dim=1, keepdim=True)) / y.std(dim=1, keepdim=True)

# PCA with an SVD of the centred data (the outputs are already centred): the
# scores of the first two components, scaled to unit variance to match the
# N(0, 1) prior on X, and the rank-2 reconstruction for comparison.
_, singular_values, right = torch.linalg.svd(y.T, full_matrices=False)
scores = y.T @ right[:latent_dim].T
pca_reconstruction = (scores @ right[:latent_dim]).T
x_pca = scores / scores.std(dim=0)
z_start = x_pca[torch.randperm(n)[:m]].clone()


def angular_agreement(coordinates):
    """Mean resultant length of ``±angle(x_i) - t_i``: 1 when the points go round the centre in the order of ``t``."""
    centred = coordinates - coordinates.mean(dim=0)
    angle = torch.atan2(centred[:, 1], centred[:, 0])
    return max(torch.polar(torch.ones_like(t), s * angle - t).mean().abs().item() for s in (1.0, -1.0))


# %%
# The model
# ---------
# ``X`` is a ``RandomParameter`` with a standard-normal prior, initialised at
# the PCA scores; ``Z`` starts at 25 of them and has a broad prior. The
# signal variance and the noise variance have log-normal priors.
#
# The length-scale is fixed to one. It is redundant with the scale of
# ``X``: only the prior on ``X`` separates the two, and with a learnable
# length-scale the MAP estimate shrinks ``X`` and the length-scale together
# and tears the loop into strands. Fixing it, the usual choice for GPLVMs,
# sets the units of the latent space.

with Model() as model:
    x = RandomParameter("x", Normal(0.0, 1.0), shape=(n, latent_dim), initial_value=x_pca.clone())
    z = RandomParameter("z", Normal(0.0, 3.0), shape=(m, latent_dim), initial_value=z_start)
    log_variance = RandomParameter("log_variance", Normal(0.0, 2.0))
    log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0))
    kernel = RBF(1.0, Exp("variance", log_variance))
    gp = ExactGP("f", x, kernel, inducing_points=z, noise_variance=Exp("noise", log_noise),
                  method="vfe", n_outputs=n_outputs)
    ObservedParameter("y", gp, y)

# %%
# MAP fit
# -------
# Adam on the joint log density (VFE bound plus log priors) of the 400 latent
# coordinates, the 50 inducing coordinates and the two variances. Each step
# costs ``O(D n m²)``, with hand-derived gradients for ``X``
# (``kernel.input_vjp`` of the bound's cotangent of ``K(Z, X)``; the
# diagonal ``k(x, x)`` of the stationary RBF does not depend on ``X``).
#
# The GPLVM objective is not convex and gradient ascent from PCA can stop in
# a local optimum where the loop is cut open (other random data sets do so);
# random restarts or more output dimensions help.

start = time.perf_counter()
bound_before = gp.log_pdf(y).item()
model.find_MAP(lr=0.01, epochs=1500, verbose=False)
elapsed = time.perf_counter() - start
x_learned = model.params["x"].constrained_value.clone()
z_learned = model.params["z"].constrained_value.clone()
variance = model.deterministic_params["variance"].constrained_value.item()
noise = model.deterministic_params["noise"].constrained_value.item()
print(f"MAP fit: {elapsed:.1f} s")
print(f"VFE bound: {bound_before:.1f} at the PCA start, {gp.log_pdf(y).item():.1f} after the fit")
print(f"signal variance {variance:.2f}, noise variance {noise:.4f}")
print(f"angular agreement with t: PCA {angular_agreement(x_pca):.3f}, GPLVM {angular_agreement(x_learned):.3f}")

# %%
# Latent coordinates
# ------------------
# Colour encodes the true angle ``t``, which the model never saw. The PCA
# projection is a distorted loop whose upper part is folded onto itself; the
# GPLVM coordinates form a round loop that goes round once in the order of
# ``t``. The red triangles are the learned inducing inputs.

fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
for ax, coordinates, title in [
    (axes[0], x_pca, f"PCA initialisation (agreement {angular_agreement(x_pca):.2f})"),
    (axes[1], x_learned, f"GPLVM, learned X (agreement {angular_agreement(x_learned):.2f})"),
]:
    points = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=t, cmap="twilight", s=18)
    ax.set(xlabel="latent dimension 1", ylabel="latent dimension 2", title=title, aspect="equal")
axes[1].plot(z_learned[:, 0], z_learned[:, 1], "^", color="tab:red", markersize=7, label="inducing inputs Z")
axes[1].legend(loc="center", fontsize="small")
fig.colorbar(points, ax=axes, label="true angle t", shrink=0.8)
plt.show()

# %%
# Reconstructed data
# ------------------
# The GP posterior at the learned coordinates (``exact_gp_predictive`` with
# the MAP values as a one-draw trace; ``X`` and ``Z`` are read from it)
# reconstructs every output dimension; with ``n_outputs`` the predictive
# draws have shape ``(draws, samples, D, n_new)``. Plotted against the true
# angle, the GPLVM reconstruction follows the data, while the rank-2 PCA
# reconstruction (the best linear map from two coordinates) misses the
# second harmonics.

trace = {
    "variance": torch.tensor(variance, dtype=dtype).reshape(1, 1, 1),
    "noise": torch.tensor(noise, dtype=dtype).reshape(1, 1, 1),
    "x": x_learned.reshape(1, 1, n, latent_dim),
    "z": z_learned.reshape(1, 1, m, latent_dim),
}
draws = exact_gp_predictive(gp, trace, y, x_learned, n_samples=200)[0]  # (samples, D, n)
reconstruction = draws.mean(dim=0)
gplvm_error = (reconstruction - y).square().mean().sqrt().item()
pca_error = (pca_reconstruction - y).square().mean().sqrt().item()
print(f"RMS reconstruction error of the standardised data: PCA {pca_error:.3f}, GPLVM {gplvm_error:.3f}")

order = torch.argsort(t)
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, layout="constrained")
for d, ax in enumerate(axes.reshape(-1)):
    ax.scatter(t, y[d], s=5, color="0.65", label="data")
    ax.plot(t[order], pca_reconstruction[d, order], color="tab:orange", linewidth=1, label="PCA, rank 2")
    ax.plot(t[order], reconstruction[d, order], color="tab:blue", label="GPLVM reconstruction")
    ax.set(title=f"output {d + 1}")
for ax in axes[1]:
    ax.set_xlabel("true angle t")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize="small")
plt.show()
