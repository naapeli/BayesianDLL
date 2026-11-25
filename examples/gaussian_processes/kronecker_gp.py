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
# variances, which keeps the 3600-point prediction cheap but makes the draws
# independent across points. With ``diagonal=False`` and a new grid it draws
# *joint* posterior samples by Matheron's rule on the union of the training
# and new grids: Kronecker prior draws plus a Kronecker solve, so no
# 3600 x 3600 covariance is formed. Joint draws are needed for any quantity
# that mixes grid points, such as the average of the surface.

thinned = trace.thin(15)
draws = kronecker_gp_predictive(f, thinned, y, [axis_a, axis_b], n_samples=10, diagonal=True)
draws = draws.reshape(-1, size, size)
posterior_mean, posterior_sd = draws.mean(0), draws.std(0)

start = time.perf_counter()
joint = kronecker_gp_predictive(f, thinned, y, [axis_a, axis_b], n_samples=1).reshape(-1, size, size)
print(f"{joint.shape[0]} joint draws of the 60 x 60 surface in {time.perf_counter() - start:.2f} s")

fig, axes = plt.subplots(1, 5, figsize=(20, 4), layout="constrained")
extent = (0.0, 1.0, 0.0, 1.0)
limits = dict(vmin=float(truth.min()) - 0.3, vmax=float(truth.max()) + 0.3, cmap="viridis", origin="lower", extent=extent)
panels = [(truth, "True surface"), (y, "Noisy data (σ = 0.2)"), (posterior_mean, "Posterior mean"), (joint[0], "One joint posterior draw")]
for ax, (image, title) in zip(axes, panels):
    shown = ax.imshow(image.T, **limits)
    ax.set(title=title, xlabel="a", ylabel="b")
fig.colorbar(shown, ax=axes[:4], shrink=0.85, label="f(a, b)")
sd_image = axes[4].imshow(posterior_sd.T, cmap="magma", origin="lower", extent=extent)
axes[4].set(title="Posterior sd", xlabel="a", ylabel="b")
fig.colorbar(sd_image, ax=axes[4], shrink=0.85, label="sd")
plt.show()

error = (posterior_mean - truth).abs().mean()
print(f"mean absolute error of the posterior mean: {error:.3f} (noise sd 0.2)")
average_joint = joint.mean(dim=(1, 2))
average_pointwise = draws.mean(dim=(1, 2))
print(f"sd of the surface average: {average_joint.std():.4f} from joint draws, "
      f"{average_pointwise.std():.4f} from independent pointwise draws (true average {truth.mean():.3f})")

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
# Learnable grid coordinates
# --------------------------
# ``ToeplitzGP`` needs fixed, equally spaced inputs, but the axes of a
# ``KroneckerGP`` can be learned: an axis given as a ``RandomParameter`` gets
# exact gradients of the marginal likelihood, and the grid stays a Cartesian
# product whose points can move along that axis.
#
# Here the rows of a 25 x 25 grid are read at warped positions
# ``a_i = (i / (n - 1))^2.5`` along axis ``a``, but the model is only told the
# row index. On a uniform axis the readings look like a chirp, oscillating
# faster towards the end, which a stationary kernel can only fit with a
# compromise lengthscale. The comparison:
#
# 1. **Fixed axis**: the rows sit on a uniform ``Data`` axis; ``find_MAP``
#    fits the lengthscales, the variance and the noise.
# 2. **Learned axis**: the row positions are a ``RandomParameter`` (prior
#    around the uniform grid) and are fitted jointly with the kernel and noise
#    parameters, starting from the fixed-axis fit.
#
# With a stationary kernel the positions are only identified up to a shift and
# a scale shared with the lengthscale, so the learned axis is compared with the
# truth after rescaling both to ``[0, 1]``.

torch.manual_seed(1)
n_grid = 25
uniform = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64)
warped = uniform ** 2.5


def chirp_surface(a, b):
    return torch.sin(10.0 * a) * torch.cos(3.0 * b) + 0.5 * a


y_warped = chirp_surface(warped[:, None], uniform[None, :]) + 0.05 * torch.randn(n_grid, n_grid, dtype=torch.float64)
hyperparameters = (("log_l_a", -1.5), ("log_l_b", -1.5), ("log_v", 0.0), ("log_noise", -3.0))


def grid_model(learn_axis, start=None):
    with Model() as grid_fit:
        log = {
            name: RandomParameter(name, Normal(prior_mean, 1.0), initial_value=None if start is None else start[name])
            for name, prior_mean in hyperparameters
        }
        if learn_axis:
            axis = RandomParameter("axis_a", Normal(uniform, 0.3), shape=n_grid, initial_value=uniform.clone())
        else:
            axis = Data("axis_a", uniform)
        gp = KroneckerGP(
            "f", [axis, Data("axis_b", uniform)],
            [RBF(Exp("l_a", log["log_l_a"]), Exp("v", log["log_v"])), RBF(Exp("l_b", log["log_l_b"]), 1.0)],
            noise_variance=Exp("noise", log["log_noise"]),
        )
        ObservedParameter("y", gp, y_warped)
    return grid_fit, gp


fixed_fit, fixed_gp = grid_model(learn_axis=False)
find_MAP(fixed_fit, lr=0.03, epochs=400, verbose=False)
start_values = {name: fixed_fit.params[name].constrained_value.clone() for name, _ in hyperparameters}
learned_fit, learned_gp = grid_model(learn_axis=True, start=start_values)
find_MAP(learned_fit, lr=0.01, epochs=600, verbose=False)

for label, grid_fit, gp in (("fixed axis", fixed_fit, fixed_gp), ("learned axis", learned_fit, learned_gp)):
    noise_sd = grid_fit.deterministic_params["noise"].constrained_value.sqrt().item()
    print(f"{label:>12}: log marginal likelihood {gp.log_pdf(y_warped).item():7.1f}, noise sd {noise_sd:.3f} (true 0.05)")


def rescaled(values):
    return (values - values.min()) / (values.max() - values.min())


learned_axis = rescaled(learned_fit.params["axis_a"].constrained_value.detach())
print(f"mean absolute error of the learned (rescaled) positions: {(learned_axis - warped).abs().mean():.3f}")

column = n_grid // 2
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
axes[0].plot(warped, "k--", lw=2, label="true positions")
axes[0].plot(uniform, ":", color="C1", label="fixed uniform axis")
axes[0].plot(learned_axis, "o-", color="C0", ms=4, label="learned axis (rescaled)")
axes[0].set(xlabel="row index i", ylabel="position a_i", title="Axis a: positions of the rows")
axes[0].legend(fontsize="small")
axes[1].plot(uniform, y_warped[:, column], "s:", color="C1", ms=4, label="against the uniform axis")
axes[1].plot(learned_axis, y_warped[:, column], "o-", color="C0", ms=4, label="against the learned axis")
axes[1].set(xlabel="position along a", ylabel="reading", title=f"One column of readings (b = {uniform[column]:.2f})")
axes[1].legend(fontsize="small")
plt.show()
