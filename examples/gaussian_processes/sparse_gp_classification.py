"""
Sparse GP Classification with Learned Inducing Points
=====================================================

Two hundred points in the plane belong to two interleaved half-moons, so
the two classes are separated by a curved boundary. We model the log-odds of
class 1 as an unknown smooth function ``f(x)`` of the 2-D input,

``y_i ~ Bernoulli(sigmoid(f(x_i)))``,

with a Gaussian-process prior on ``f``. The Bernoulli likelihood is not
Gaussian, so ``f`` cannot be integrated out and has to be inferred together
with the hyperparameters.

**A sparse latent GP.** ``SparseLatentGP`` summarises the function by its
values ``u`` at ``m = 16`` *inducing inputs* ``Z`` and interpolates them with
the kernel, ``f(X) = K_XZ K_ZZ⁻¹ u``. This is the DTC ("projected process")
approximation: the prior covariance of ``f`` is
``Q = K_XZ K_ZZ⁻¹ K_ZX`` instead of ``K_XX``, and only ``m x m`` matrices are
ever factorised, so the cost grows linearly with the number of data points.
The inducing values are **whitened**: ``u = L v`` with ``L Lᵀ = K_ZZ`` and
``v ~ Normal(0, I)``, so ``f = K_XZ L⁻ᵀ v``. The latent ``v`` has the same
prior whatever the kernel, which removes the funnel between the function and
its hyperparameters.

**Why learn Z.** Under DTC the function can only vary where there are
inducing inputs: far from all of them ``K_XZ ≈ 0`` and ``f ≈ 0``, i.e.
``P(y = 1) ≈ 0.5``. Here the 16 inducing inputs start badly placed, on a
small grid in the lower-left corner. They are made a ``RandomParameter``
(with a broad Normal prior around the start) and moved by gradient-based
optimisation; their hand-derived gradient goes through the kernel's
``input_vjp``. The workflow is

1. **learn Z** (and a first estimate of everything else) with a MAP
   estimate, ``find_MAP``;
2. **compare Z** by the evidence lower bound of a full-rank variational
   posterior (``AutoMultivariateNormal`` with ``BBVI``) over ``v`` and the
   hyperparameters, once with the learned and once with the initial ``Z``;
3. **sample** ``v`` and the hyperparameters with NUTS, with ``Z`` fixed at
   the learned locations.

Why not treat ``Z`` as a random variable all the way? The inducing inputs
are parameters of the approximation, not quantities of interest. Their
posterior is nearly as broad as their prior (many locations explain the data
equally well, and any permutation of ``Z`` is equivalent), so a Gaussian
guide over ``Z`` spreads them out and blurs the function (in our runs the
guide's standard deviations for ``Z`` grew to about 2). NUTS over ``v``,
``Z`` and the hyperparameters jointly mixes poorly: in our runs (4 x 300
draws) almost every iteration hit the maximum tree depth, there were
divergences and the largest R-hat was above 2, at five times the cost of
everything below. A point estimate
of ``Z`` followed by full inference over the rest is the standard remedy.
"""

import math
import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import AutoMultivariateNormal, Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Bernoulli, Normal
from BayesianDLL.GP import RBF, SparseLatentGP, sparse_latent_gp_predictive
from BayesianDLL.Variational import BBVI


torch.manual_seed(0)
start_time = time.perf_counter()


def two_moons(n, noise=0.15):
    t = torch.rand(n, dtype=torch.float64) * math.pi
    label = (torch.rand(n) < 0.5).to(torch.float64)
    upper = torch.stack([torch.cos(t), torch.sin(t)], dim=-1)
    lower = torch.stack([1.0 - torch.cos(t), 0.5 - torch.sin(t)], dim=-1)
    x = torch.where(label[:, None] == 1, lower, upper) + noise * torch.randn(n, 2, dtype=torch.float64)
    return x, label


n, m = 200, 16
x, y = two_moons(n)
x_test, y_test = two_moons(1000)
corner = torch.linspace(0.0, 0.6, 4, dtype=torch.float64)
z_start = torch.cartesian_prod(corner, corner) + torch.tensor([-1.4, -1.0], dtype=torch.float64)

grid_x = torch.linspace(-1.8, 2.8, 70, dtype=torch.float64)
grid_y = torch.linspace(-1.3, 1.8, 50, dtype=torch.float64)
mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing="xy")
grid = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=-1)


def build(inducing):
    """``inducing`` is a fixed ``(m, 2)`` tensor, or ``None`` for learnable inducing inputs."""
    with Model() as model:
        log_lengthscale = RandomParameter("log_lengthscale", Normal(math.log(0.5), 0.5), initial_value=torch.tensor([math.log(0.5)]))
        log_variance = RandomParameter("log_variance", Normal(math.log(4.0), 0.7), initial_value=torch.tensor([math.log(4.0)]))
        kernel = RBF(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
        if inducing is None:
            inducing = RandomParameter("z", Normal(z_start, 2.0), shape=(m, 2), initial_value=z_start.clone())
        v = RandomParameter("v", Normal(0.0, 1.0), shape=m, initial_value=torch.zeros(m, dtype=torch.float64))
        f = SparseLatentGP("f", Data("x", x), kernel, inducing, v)
        ObservedParameter("y", Bernoulli(logits=f), y)
    return model, f


def scatter_classes(ax, size=14):
    for label, color in ((0, "tab:blue"), (1, "tab:orange")):
        mask = y == label
        ax.scatter(x[mask, 0], x[mask, 1], s=size, color=color, edgecolor="white", linewidth=0.4, label=f"class {label}")


def style(ax, title):
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=title, xlim=(grid_x[0], grid_x[-1]), ylim=(grid_y[0], grid_y[-1]))
    ax.set_aspect("equal")


# %%
# The data and the initial inducing inputs
# ----------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
scatter_classes(ax)
ax.scatter(z_start[:, 0], z_start[:, 1], marker="^", s=60, color="black", label="initial inducing inputs")
style(ax, "Two moons and the initial inducing inputs")
ax.legend(loc="upper right", fontsize="small")
plt.show()

# %%
# Step 1: learning the inducing inputs
# ------------------------------------
# ``find_MAP`` maximises the log joint density over ``v``, ``Z`` and the
# log hyperparameters with Adam. It is run in chunks of 50 steps to record
# the objective and the path of every inducing input. The inducing inputs
# next to the data are pulled onto the two moons within a few hundred steps.
# Those deep in the empty corner stay put: no data point is close enough for
# them to influence ``f``, so their gradient vanishes. They are wasted, but
# harmless.

map_model, map_f = build(None)
steps, objective, paths = [0], [map_model.model_log_prob().item()], [z_start.clone()]
for chunk in range(24):
    map_model.find_MAP(lr=0.03, epochs=50, verbose=False)
    steps.append(50 * (chunk + 1))
    objective.append(map_model.model_log_prob().item())
    paths.append(map_model.params["z"].constrained_value.clone())
paths = torch.stack(paths)  # (n_records, m, 2)
z_learned = paths[-1]
print(f"MAP log joint: {objective[0]:.1f} at the start, {objective[-1]:.1f} after {steps[-1]} steps")
print(f"MAP lengthscale {map_model.deterministic_params['lengthscale'].constrained_value.item():.2f}, "
      f"variance {map_model.deterministic_params['variance'].constrained_value.item():.2f}")

fig, (ax_objective, ax_paths) = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
ax_objective.plot(steps, objective, "o-", color="tab:green", markersize=3)
ax_objective.set(xlabel="Adam step", ylabel="log joint density", title="MAP objective")
scatter_classes(ax_paths, size=8)
for k in range(m):
    ax_paths.plot(paths[:, k, 0], paths[:, k, 1], color="0.3", linewidth=0.8)
    ax_paths.annotate("", xy=tuple(paths[-1, k].tolist()), xytext=tuple(paths[-3, k].tolist()),
                      arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
ax_paths.scatter(z_start[:, 0], z_start[:, 1], marker="^", s=40, color="0.6", label="initial Z")
ax_paths.scatter(z_learned[:, 0], z_learned[:, 1], marker="^", s=60, color="black", label="learned Z")
style(ax_paths, "Trajectories of the inducing inputs")
ax_paths.legend(loc="upper right", fontsize="small", ncol=2)
plt.show()

# %%
# Step 2: comparing learned and initial inducing inputs by the ELBO
# -----------------------------------------------------------------
# With ``Z`` fixed, a full-rank Normal guide over ``v`` and the two log
# hyperparameters is fitted by black-box variational inference. Its ELBO is
# a lower bound on the log evidence ``log p(y | Z)``, so it is a fair way to
# compare the two sets of inducing inputs: the learned ones give a much
# higher bound.

guides, histories, models, functions = {}, {}, {}, {}
for label, inducing in (("initial Z", z_start), ("learned Z", z_learned)):
    models[label], functions[label] = build(inducing)
    guides[label] = AutoMultivariateNormal(models[label], init_scale=0.05)
    histories[label] = BBVI(models[label], guides[label], n_samples=1, epochs=800, lr=0.02, lrd=0.998,
                            callback_frequency=1, verbose=False)
    print(f"{label}: final ELBO {sum(histories[label][-200:]) / 200:.1f}")


def smooth(values, width=50):
    values = torch.tensor(values)
    kernel = torch.ones(width) / width
    return torch.nn.functional.conv1d(values.reshape(1, 1, -1), kernel.reshape(1, 1, -1)).reshape(-1)


fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for (label, history), color in zip(histories.items(), ("tab:red", "tab:green")):
    ax.plot(history, color=color, alpha=0.2, linewidth=0.6)
    ax.plot(range(49, len(history)), smooth(history), color=color, label=f"{label} (moving average)")
ax.set(xlabel="BBVI epoch", ylabel="ELBO estimate", title="Evidence lower bound with Z fixed",
       ylim=(min(sum(h[-200:]) / 200 for h in histories.values()) - 60, max(max(smooth(h)) for h in histories.values()) + 15))
ax.legend(loc="lower right", fontsize="small")
plt.show()

# %%
# Step 3: NUTS with the learned inducing inputs
# ---------------------------------------------
# The whitened posterior of ``v`` and the log hyperparameters (18
# dimensions) is well conditioned, and NUTS samples it with vectorized
# chains.

nuts_model, nuts_f = models["learned Z"], functions["learned Z"]
start = time.perf_counter()
trace = nuts_model.sample(200, 200, n_chains=4, chain_method="vectorized", max_depth=5, seed=1, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary()
print(summary.loc[["log_lengthscale[0]", "log_variance[0]"]])
print(f"NUTS: {elapsed:.1f} s, divergences: {sum(trace.divergences)}, max R-hat: {summary['r_hat'].max():.3f}, "
      f"min bulk ESS: {summary['ess_bulk'].min():.0f}")

# %%
# The predictive probability and its uncertainty
# ----------------------------------------------
# ``sparse_latent_gp_predictive`` evaluates ``f = K_*Z L⁻ᵀ v`` on a grid for
# every posterior draw. The left panel shows the posterior mean of
# ``P(y = 1) = sigmoid(f)`` with the 0.5 decision boundary; the right panel
# the posterior standard deviation of that probability. The uncertainty is
# largest along the boundary and where there are no data.


def probabilities(f, draws, points):
    latent = sparse_latent_gp_predictive(f, draws, points).reshape(-1, points.shape[0])
    return torch.sigmoid(latent)


def test_metrics(f, draws):
    p = probabilities(f, draws, x_test).mean(0).clamp(1e-9, 1 - 1e-9)
    accuracy = ((p > 0.5).to(y_test.dtype) == y_test).double().mean().item()
    log_loss = -(y_test * p.log() + (1 - y_test) * (1 - p).log()).mean().item()
    return accuracy, log_loss


nuts_draws = trace.thin(2)
grid_probability = probabilities(nuts_f, nuts_draws, grid)
mean_surface = grid_probability.mean(0).reshape(mesh_x.shape)
sd_surface = grid_probability.std(0).reshape(mesh_x.shape)

fig, (ax_mean, ax_sd) = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
image = ax_mean.contourf(mesh_x, mesh_y, mean_surface, levels=20, cmap="RdBu_r", vmin=0, vmax=1)
ax_mean.contour(mesh_x, mesh_y, mean_surface, levels=[0.5], colors="black", linewidths=1.5)
fig.colorbar(image, ax=ax_mean, label="posterior mean of P(y = 1)", shrink=0.85)
scatter_classes(ax_mean, size=8)
ax_mean.scatter(z_learned[:, 0], z_learned[:, 1], marker="^", s=50, color="black", label="learned Z")
style(ax_mean, "Predictive probability (black line: P = 0.5)")
ax_mean.legend(loc="upper right", fontsize="x-small", ncol=3)
image = ax_sd.contourf(mesh_x, mesh_y, sd_surface, levels=20, cmap="viridis")
fig.colorbar(image, ax=ax_sd, label="posterior sd of P(y = 1)", shrink=0.85)
ax_sd.scatter(x[:, 0], x[:, 1], s=4, color="white", alpha=0.6)
ax_sd.scatter(z_learned[:, 0], z_learned[:, 1], marker="^", s=50, color="black")
style(ax_sd, "Uncertainty of the probability")
plt.show()

# %%
# Fixed initial inducing inputs give a worse fit
# ----------------------------------------------
# The variational posteriors of step 2 are compared on 1000 fresh test
# points. With the inducing inputs stuck in the corner the function is a
# smooth extrapolation of what the corner sees: the curved boundary cannot be
# represented and the probability fades to 0.5 far from the corner.

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
for ax, (label, guide), z in zip(axes, guides.items(), (z_start, z_learned)):
    draws = guide.sample(300)
    accuracy, log_loss = test_metrics(functions[label], draws)
    print(f"{label} (VI): test accuracy {accuracy:.3f}, test log loss {log_loss:.3f}")
    surface = probabilities(functions[label], draws, grid).mean(0).reshape(mesh_x.shape)
    image = ax.contourf(mesh_x, mesh_y, surface, levels=20, cmap="RdBu_r", vmin=0, vmax=1)
    ax.contour(mesh_x, mesh_y, surface, levels=[0.5], colors="black", linewidths=1.5)
    scatter_classes(ax, size=8)
    ax.scatter(z[:, 0], z[:, 1], marker="^", s=50, color="black", label="inducing inputs")
    style(ax, f"{label}: test accuracy {accuracy:.2f}, log loss {log_loss:.2f}")
axes[0].legend(loc="upper right", fontsize="x-small", ncol=3)
fig.colorbar(image, ax=axes, label="mean of P(y = 1)", shrink=0.85)
plt.show()

accuracy, log_loss = test_metrics(nuts_f, nuts_draws)
print(f"learned Z (NUTS): test accuracy {accuracy:.3f}, test log loss {log_loss:.3f}")
print(f"total runtime: {time.perf_counter() - start_time:.1f} s")
