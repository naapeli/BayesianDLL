"""
Exact GP on a Grid with Kronecker Algebra
=========================================

Images, spatial rasters and sensor arrays put their observations on a
regular grid. Here a smooth surface is measured with noise on a 60 x 60
grid, i.e. 3600 observations. A dense exact GP would factorise a
3600 x 3600 covariance matrix at every NUTS step.

If the kernel is a product of one kernel per axis,
``k((a, b), (a', b')) = k_1(a, a') k_2(b, b')``, the covariance of the grid
is a Kronecker product ``K = K_1 ⊗ K_2`` of two 60 x 60 matrices.
``KroneckerGP`` eigendecomposes the two small factors and evaluates the
*exact* marginal likelihood ``log N(y | 0, K + σ² I)`` and its hand-derived
gradients through mode products with the factors, at a cost of
``O(n_1³ + n_2³ + N (n_1 + n_2))`` instead of ``O(N³)``. Nothing is
approximated: on the same grid it gives the same numbers as ``ExactGP``.

Each axis gets its own length-scale (on the log scale, through ``Exp``),
so the model can learn that the surface varies faster along one axis.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, find_MAP
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import RBF, ExactGP, KroneckerGP, kronecker_gp_predictive


torch.manual_seed(0)


def true_surface(a, b):
    return torch.sin(6.0 * a) * torch.cos(3.0 * b) + 0.5 * a


size = 60
axis_a = torch.linspace(0.0, 1.0, size, dtype=torch.float64)
axis_b = torch.linspace(0.0, 1.0, size, dtype=torch.float64)
truth = true_surface(axis_a[:, None], axis_b[None, :])  # (60, 60), rows = a, columns = b
y = truth + 0.2 * torch.randn_like(truth)

# %%
# The model
# ---------
# ``y`` keeps its grid shape ``(60, 60)``: entry ``[i, j]`` is the
# observation at ``(axis_a[i], axis_b[j])``.

with Model() as model:
    log_lengthscale_a = RandomParameter("log_lengthscale_a", Normal(-1.5, 1.0))
    log_lengthscale_b = RandomParameter("log_lengthscale_b", Normal(-1.5, 1.0))
    log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
    log_noise = RandomParameter("log_noise", Normal(-3.0, 1.0))
    f = KroneckerGP(
        "f",
        grid_inputs=[Data("axis_a", axis_a), Data("axis_b", axis_b)],
        kernels=[
            RBF(Exp("lengthscale_a", log_lengthscale_a), Exp("variance", log_variance)),
            RBF(Exp("lengthscale_b", log_lengthscale_b), 1.0),
        ],
        noise_variance=Exp("noise", log_noise),
    )
    ObservedParameter("y", f, y)

start = time.perf_counter()
trace = model.sample(400, 300, n_chains=2, max_depth=5, delta=0.9, seed=4, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary(include_deterministic=True)
print(summary)
print(f"sampling: {elapsed:.1f} s, divergences: {sum(trace.divergences)}, max R-hat: {summary['r_hat'].max():.3f}")

# %%
# The fitted surface
# ------------------
# ``kronecker_gp_predictive`` predicts on a list of axes (a new grid) or on
# arbitrary points. With ``diagonal=True`` it only computes the pointwise
# variances, which keeps the 3600-point prediction cheap.

draws = kronecker_gp_predictive(f, trace.thin(15), y, [axis_a, axis_b], n_samples=10, diagonal=True)
draws = draws.reshape(-1, size, size)
posterior_mean, posterior_sd = draws.mean(0), draws.std(0)

fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
extent = (0.0, 1.0, 0.0, 1.0)
limits = dict(vmin=float(truth.min()) - 0.3, vmax=float(truth.max()) + 0.3, cmap="viridis", origin="lower", extent=extent)
panels = [(truth, "True surface"), (y, "Noisy data (σ = 0.2)"), (posterior_mean, "Posterior mean")]
for ax, (image, title) in zip(axes, panels):
    shown = ax.imshow(image.T, **limits)
    ax.set(title=title, xlabel="a", ylabel="b")
fig.colorbar(shown, ax=axes[:3], shrink=0.85, label="f(a, b)")
sd_image = axes[3].imshow(posterior_sd.T, cmap="magma", origin="lower", extent=extent)
axes[3].set(title="Posterior sd", xlabel="a", ylabel="b")
fig.colorbar(sd_image, ax=axes[3], shrink=0.85, label="sd")
plt.show()

error = (posterior_mean - truth).abs().mean()
print(f"mean absolute error of the posterior mean: {error:.3f} (noise sd 0.2)")

# %%
# Timing against ExactGP
# ----------------------
# One log density plus gradient on square grids of growing size. The
# ``ExactGP`` uses the equivalent ARD RBF kernel on the flattened grid
# (``exp(-a²/2l_a² - b²/2l_b²)`` is the product of the two axis kernels) and
# is stopped at 30 x 30 = 900 points.


def time_once(build):
    model = build()
    model.joint_grad_log_prob()  # warm-up
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        model.model_log_prob()
        model.joint_grad_log_prob()
        best = min(best, time.perf_counter() - start)
    return best


def kronecker_model(s):
    grid = torch.linspace(0.0, 1.0, s, dtype=torch.float64)
    values = true_surface(grid[:, None], grid[None, :])
    with Model() as timing_model:
        la = RandomParameter("la", Normal(-1.5, 1.0))
        lb = RandomParameter("lb", Normal(-1.5, 1.0))
        g = KroneckerGP("g", [grid, grid], [RBF(Exp("a", la), 1.0), RBF(Exp("b", lb), 1.0)], noise_variance=0.04)
        ObservedParameter("y", g, values)
    return timing_model


def exact_model(s):
    grid = torch.linspace(0.0, 1.0, s, dtype=torch.float64)
    points = torch.cartesian_prod(grid, grid)
    with Model() as timing_model:
        log_l = RandomParameter("log_l", Normal(-1.5, 1.0), shape=2)
        g = ExactGP("g", Data("points", points), RBF(Exp("l", log_l), 1.0), noise_variance=0.04)
        ObservedParameter("y", g, true_surface(points[:, 0], points[:, 1]))
    return timing_model


exact_sizes = [10, 15, 20, 25, 30]
kronecker_sizes = [10, 15, 20, 25, 30, 45, 60, 80, 100]
exact_times = [time_once(lambda: exact_model(s)) for s in exact_sizes]
kronecker_times = [time_once(lambda: kronecker_model(s)) for s in kronecker_sizes]
for s, t in zip(exact_sizes, exact_times):
    print(f"ExactGP     {s:3d} x {s:<3d} (N = {s * s:5d}): {1000 * t:8.1f} ms")
for s, t in zip(kronecker_sizes, kronecker_times):
    print(f"KroneckerGP {s:3d} x {s:<3d} (N = {s * s:5d}): {1000 * t:8.1f} ms")

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
ax.loglog([s * s for s in exact_sizes], exact_times, "o-", label="ExactGP (dense Cholesky)")
ax.loglog([s * s for s in kronecker_sizes], kronecker_times, "s-", label="KroneckerGP")
ax.set(xlabel="number of grid points N", ylabel="seconds per log density + gradient",
       title="Cost of one gradient evaluation")
ax.legend(loc="upper left", fontsize="small")
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %%
# Learnable vs Fixed Grid Comparison
# -----------------------------------
# While stationary Toeplitz models require rigid, equally spaced coordinates,
# ``KroneckerGP`` supports arbitrary and learnable coordinate grids along any axis.
#
# When observations are measured on non-uniform or distorted grids, assuming
# fixed equispaced coordinates leads to model misspecification. Because
# ``KroneckerGP`` evaluates exact reverse-mode vector-Jacobian products with
# respect to each coordinate, the grid coordinates can be treated as
# :class:`~BayesianDLL.RandomParameter` or :class:`~BayesianDLL.DeterministicParameter`
# nodes and estimated jointly with the kernel parameters.
#
# Here, data is generated on a warped axis :math:`a_i = (i / (N - 1))^{1.4}`.
# We compare:
#
# 1. **Fixed Grid Model**: coordinates along axis A are fixed :class:`~BayesianDLL.Data`
#    nodes placed on a uniform grid;
# 2. **Learnable Grid Model**: coordinates along axis A are a :class:`~BayesianDLL.RandomParameter`
#    initialized at the uniform grid, receiving exact gradient feedback from the marginal likelihood.

n_compare = 25
axis_a_true = torch.linspace(0.0, 1.0, n_compare, dtype=torch.float64) ** 1.4
b_axis = torch.linspace(0.0, 1.0, n_compare, dtype=torch.float64)
y_warped = true_surface(axis_a_true[:, None], b_axis[None, :]) + 0.05 * torch.randn(n_compare, n_compare, dtype=torch.float64)
axis_a_init = torch.linspace(0.0, 1.0, n_compare, dtype=torch.float64)

with Model() as fixed_model:
    gp_fixed = KroneckerGP(
        "gp_fixed",
        [Data("axis_a_fixed", axis_a_init), Data("axis_b", b_axis)],
        [RBF(0.3, 1.0), RBF(0.6, 1.0)],
        noise_variance=0.01,
    )
    ObservedParameter("y_fixed", gp_fixed, y_warped)

with Model() as learnable_model:
    axis_a_param = RandomParameter("axis_a_learned", Normal(0.5, 0.5), shape=n_compare, initial_value=axis_a_init.clone())
    gp_learnable = KroneckerGP(
        "gp_learnable",
        [axis_a_param, Data("axis_b", b_axis)],
        [RBF(0.3, 1.0), RBF(0.6, 1.0)],
        noise_variance=0.01,
    )
    ObservedParameter("y_learned", gp_learnable, y_warped)

grads = learnable_model.joint_grad_log_prob()
grad_a = grads["axis_a_learned"]

# Optimize coordinates along the exact likelihood gradient
axis_val = axis_a_init.clone()
for _ in range(40):
    axis_a_param.set_unconstrained_value(axis_val)
    g = learnable_model.joint_grad_log_prob()["axis_a_learned"]
    axis_val = axis_val + 0.005 * (g / (g.abs().max() + 1e-8))
axis_a_param.set_unconstrained_value(axis_val)

mse_initial = (axis_a_init - axis_a_true).square().mean().item()
mse_learned = (axis_val - axis_a_true).square().mean().item()

print(f"Fixed grid parameters:     {list(fixed_model.params.keys())} (coordinates are immutable Data nodes)")
print(f"Learnable grid parameters: {list(learnable_model.params.keys())} (coordinates are RandomParameters)")
print(f"Axis A coordinate MSE to truth: initial (fixed) {mse_initial:.5f} -> learned {mse_learned:.5f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
axes[0].plot(range(n_compare), axis_a_true.numpy(), "k--", lw=2, label="True warped coordinates")
axes[0].plot(range(n_compare), axis_a_init.numpy(), "s:", color="C1", alpha=0.8, label="Fixed / Initial (uniform)")
axes[0].plot(range(n_compare), axis_val.detach().numpy(), "o-", color="C0", lw=2, label="Learned coordinates")
axes[0].set(xlabel="Grid index i", ylabel="Coordinate value $a_i$", title="Axis A: Fixed vs Learned Grid Coordinates")
axes[0].legend(fontsize="small")
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(n_compare), grad_a.numpy(), "D-", color="C2", lw=1.5)
axes[1].axhline(0, color="black", linestyle="--", lw=0.8)
axes[1].set(xlabel="Grid index i", ylabel=r"$\partial \log p(y) / \partial a_i$", title="Marginal Likelihood Gradient w.r.t. Coordinates")
axes[1].grid(True, alpha=0.3)
plt.show()
