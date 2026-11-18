"""
Sparse GP Classification with Learned Inducing Points
=====================================================

Three hundred points in the plane lie on two interleaved spirals, one per
class. The decision boundary winds between the arms, so the log-odds
function must change sign several times along every ray from the centre.
We model the log-odds of class 1 as an unknown smooth function ``f(x)`` of
the 2-D input,

``y_i ~ Bernoulli(sigmoid(f(x_i)))``,

with a Gaussian-process prior on ``f``. The Bernoulli likelihood is not
Gaussian, so ``f`` cannot be integrated out and has to be inferred together
with the hyperparameters.

**A sparse latent GP.** ``SparseLatentGP`` summarises the function by its
values ``u`` at ``m = 24`` *inducing inputs* ``Z`` and interpolates them with
the kernel, ``f(X) = K_XZ K_ZZ⁻¹ u``. This is the DTC ("projected process")
approximation: the prior covariance of ``f`` is
``Q = K_XZ K_ZZ⁻¹ K_ZX`` instead of ``K_XX``, and only ``m x m`` matrices are
ever factorised, so the cost grows linearly with the number of data points.
The inducing values are **whitened**: ``u = L v`` with ``L Lᵀ = K_ZZ`` and
``v ~ Normal(0, I)``, so ``f = K_XZ L⁻ᵀ v``. The latent ``v`` has the same
prior whatever the kernel, which removes the funnel between the function and
its hyperparameters.

**Why learn Z.** Under DTC the function can only bend where there are
inducing inputs: it is a combination of ``m`` kernel bumps centred at ``Z``.
Twenty-four bumps are enough to trace the spiral boundary, but only if they
sit in the right places. We start from 24 randomly chosen training points
(plus a little jitter), make ``Z`` a ``RandomParameter`` with a broad Normal
prior, and move it by gradient-based optimisation. Its hand-derived gradient
goes through the kernel's ``input_vjp``. The workflow is

1. **learn Z** (and a first estimate of everything else) with a MAP
   estimate, ``find_MAP``;
2. **compare Z** by the evidence lower bound of a full-rank variational
   posterior (``AutoMultivariateNormal`` with ``BBVI``) over ``v`` and the
   hyperparameters, once with the learned and once with the random initial
   ``Z``;
3. **sample** ``v`` and the hyperparameters with NUTS, with ``Z`` fixed at
   the learned locations.

Why a point estimate of ``Z``? The inducing inputs are parameters of the
approximation, not quantities of interest. Their posterior is nearly as broad
as their prior (many arrangements explain the data equally well, and any
permutation of ``Z`` is equivalent). In our experiments (on a two-moons
version of this example) a Gaussian guide over
``Z`` spread the inducing inputs out (posterior standard deviations of about
2) and blurred the function, and NUTS over ``v``, ``Z`` and the
hyperparameters jointly hit its maximum tree depth almost every iteration
and did not converge (R-hat above 2). We also tried learning ``Z`` by
maximising the ELBO (alternating ``BBVI`` steps on the guide with Adam steps
on ``Z``) and fixing the hyperparameters during a first phase; neither gave a
better ``Z`` than the joint MAP estimate used here.
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


def two_spirals(n, noise=0.08):
    t = torch.rand(n, dtype=torch.float64) * 3 * math.pi + 0.5
    label = (torch.rand(n) < 0.5).to(torch.float64)
    sign = 2 * label - 1
    x = torch.stack([sign * t * torch.cos(t), sign * t * torch.sin(t)], dim=-1) / 5
    return x + noise * torch.randn(n, 2, dtype=torch.float64), label


n, m = 300, 24
x, y = two_spirals(n)
x_test, y_test = two_spirals(2000)
z_start = x[torch.randperm(n)[:m]] + 0.1 * torch.randn(m, 2, dtype=torch.float64)

grid_1d = torch.linspace(-2.3, 2.3, 70, dtype=torch.float64)
mesh_x, mesh_y = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
grid = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=-1)


def build(inducing):
    """``inducing`` is a fixed ``(m, 2)`` tensor, or ``None`` for learnable inducing inputs."""
    with Model() as model:
        log_lengthscale = RandomParameter("log_lengthscale", Normal(math.log(0.3), 0.5), initial_value=torch.tensor([math.log(0.3)]))
        log_variance = RandomParameter("log_variance", Normal(math.log(4.0), 0.5), initial_value=torch.tensor([math.log(4.0)]))
        kernel = RBF(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
        if inducing is None:
            prior_loc = x.mean(0).expand(m, 2).clone()
            inducing = RandomParameter("z", Normal(prior_loc, 2.0), shape=(m, 2), initial_value=z_start.clone())
        v = RandomParameter("v", Normal(0.0, 1.0), shape=m, initial_value=torch.zeros(m, dtype=torch.float64))
        f = SparseLatentGP("f", Data("x", x), kernel, inducing, v)
        ObservedParameter("y", Bernoulli(logits=f), y)
    return model, f


def scatter_classes(ax, size=10):
    for label, color in ((0, "tab:blue"), (1, "tab:orange")):
        mask = y == label
        ax.scatter(x[mask, 0], x[mask, 1], s=size, color=color, edgecolor="white", linewidth=0.3, label=f"class {label}")


def style(ax, title):
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=title, xlim=(grid_1d[0], grid_1d[-1]), ylim=(grid_1d[0], grid_1d[-1]))
    ax.set_aspect("equal")


# %%
# The data and the initial inducing inputs
# ----------------------------------------

fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
scatter_classes(ax)
ax.scatter(z_start[:, 0], z_start[:, 1], marker="^", s=60, color="black", label="initial inducing inputs")
style(ax, "Two spirals and the random initial inducing inputs")
ax.legend(loc="upper left", fontsize="small")
plt.show()

# %%
# Step 1: learning the inducing inputs
# ------------------------------------
# ``find_MAP`` maximises the log joint density over ``v``, ``Z`` and the
# log hyperparameters with Adam. It is run in chunks of 50 steps to record
# the objective and the path of every inducing input.

map_model, map_f = build(None)
steps, objective, paths = [0], [map_model.model_log_prob().item()], [z_start.clone()]
for chunk in range(20):
    map_model.find_MAP(lr=0.03, epochs=50, verbose=False)
    steps.append(50 * (chunk + 1))
    objective.append(map_model.model_log_prob().item())
    paths.append(map_model.params["z"].constrained_value.clone())
paths = torch.stack(paths)  # (n_records, m, 2)
z_learned = paths[-1]
print(f"MAP log joint: {objective[0]:.1f} at the start, {objective[-1]:.1f} after {steps[-1]} steps")
print(f"MAP lengthscale {map_model.deterministic_params['lengthscale'].constrained_value.item():.2f}, "
      f"variance {map_model.deterministic_params['variance'].constrained_value.item():.2f}; "
      f"median distance moved by an inducing input: {(z_learned - z_start).norm(dim=-1).median():.2f}")

fig, (ax_objective, ax_paths) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
ax_objective.plot(steps, objective, "o-", color="tab:green", markersize=3)
ax_objective.set(xlabel="Adam step", ylabel="log joint density", title="MAP objective")
scatter_classes(ax_paths, size=6)
for k in range(m):
    ax_paths.plot(paths[:, k, 0], paths[:, k, 1], color="0.25", linewidth=0.8)
ax_paths.scatter(z_start[:, 0], z_start[:, 1], marker="^", s=35, color="0.65", label="initial Z")
ax_paths.scatter(z_learned[:, 0], z_learned[:, 1], marker="^", s=55, color="black", label="learned Z")
style(ax_paths, "Trajectories of the inducing inputs")
ax_paths.legend(loc="upper left", fontsize="small", ncol=2)
plt.show()

# %%
# Step 2: comparing learned and random inducing inputs by the ELBO
# ----------------------------------------------------------------
# With ``Z`` fixed, a full-rank Normal guide over ``v`` and the two log
# hyperparameters is fitted by black-box variational inference. Its ELBO is
# a lower bound on the log evidence ``log p(y | Z)``, so it is a fair way to
# compare the two sets of inducing inputs.

guides, histories, models, functions = {}, {}, {}, {}
for label, inducing in (("random Z", z_start), ("learned Z", z_learned)):
    models[label], functions[label] = build(inducing)
    guides[label] = AutoMultivariateNormal(models[label], init_scale=0.05)
    histories[label] = BBVI(models[label], guides[label], n_samples=1, epochs=800, lr=0.02, lrd=0.998,
                            callback_frequency=1, verbose=False)
    print(f"{label}: final ELBO {sum(histories[label][-200:]) / 200:.1f}")


def moving_average(values, width=50):
    values = torch.tensor(values)
    return values.unfold(0, width, 1).mean(-1)


fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for (label, history), color in zip(histories.items(), ("tab:red", "tab:green")):
    ax.plot(history, color=color, alpha=0.2, linewidth=0.6)
    ax.plot(range(49, len(history)), moving_average(history), color=color, label=f"{label} (moving average)")
final = [sum(h[-200:]) / 200 for h in histories.values()]
ax.set(xlabel="BBVI epoch", ylabel="ELBO estimate", title="Evidence lower bound with Z fixed",
       ylim=(min(final) - 80, max(final) + 25))
ax.legend(loc="lower right", fontsize="small")
plt.show()

# %%
# Step 3: NUTS with the learned inducing inputs
# ---------------------------------------------
# The whitened posterior of ``v`` and the log hyperparameters (26
# dimensions) is well conditioned, and NUTS samples it with vectorized
# chains.

nuts_model, nuts_f = models["learned Z"], functions["learned Z"]
start = time.perf_counter()
trace = nuts_model.sample(120, 120, n_chains=4, chain_method="vectorized", max_depth=5, seed=1, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary()
print(summary.loc[["log_lengthscale[0]", "log_variance[0]"]])
print(f"NUTS: {elapsed:.1f} s, divergences: {sum(trace.divergences)}, max R-hat: {summary['r_hat'].max():.3f}, "
      f"min bulk ESS: {summary['ess_bulk'].min():.0f}")

# %%
# Are the inducing inputs useful?
# -------------------------------
# Two measures, for the random and the learned ``Z``:
#
# * the fraction of inducing inputs within ``r = 0.25`` of a training point
#   (an inducing input far from all data cannot influence the fit);
# * the number of inducing inputs with a *partner across the boundary*: an
#   inducing input closer than two length-scales (kernel correlation above
#   ``exp(-2) ≈ 0.14``) whose posterior mean of ``f`` has the opposite sign.
#   Such pairs pin the boundary down between them, where ``f`` must change
#   sign quickly.
#
# Random training points are already close to the data, but they are
# scattered unevenly: some arms get several, others none. After learning,
# the inducing inputs line up along the arms, alternating between the
# classes, and almost every one of them has a partner on the neighbouring
# arm across the boundary.


def probabilities(f, draws, points):
    latent = sparse_latent_gp_predictive(f, draws, points).reshape(-1, points.shape[0])
    return torch.sigmoid(latent)


def test_metrics(f, draws):
    p = probabilities(f, draws, x_test).mean(0).clamp(1e-9, 1 - 1e-9)
    accuracy = ((p > 0.5).to(y_test.dtype) == y_test).double().mean().item()
    log_loss = -(y_test * p.log() + (1 - y_test) * (1 - p).log()).mean().item()
    return accuracy, log_loss


def usefulness(f, draws, z):
    """Fraction of ``Z`` near the data, and the partner (or ``-1``) of every inducing input."""
    near_data = (torch.cdist(z, x).min(dim=1).values < 0.25).double().mean().item()
    f_at_z = sparse_latent_gp_predictive(f, draws, z).reshape(-1, m).mean(0)
    lengthscale = draws["lengthscale"].mean().item()
    distance = torch.cdist(z, z).masked_fill(f_at_z[:, None] * f_at_z[None, :] >= 0, float("inf"))
    nearest = distance.min(dim=1)
    partner = torch.where(nearest.values < 2 * lengthscale, nearest.indices, torch.full_like(nearest.indices, -1))
    return near_data, partner


nuts_draws = trace.thin(2)
variational_draws = {label: guide.sample(300) for label, guide in guides.items()}
partners = {}
for label, z in (("random Z", z_start), ("learned Z", z_learned)):
    near_data, partners[label] = usefulness(functions[label], variational_draws[label], z)
    accuracy, log_loss = test_metrics(functions[label], variational_draws[label])
    print(f"{label:>9} (VI): {100 * near_data:.0f}% within 0.25 of the data, "
          f"{int((partners[label] >= 0).sum())}/{m} with a partner across the boundary, "
          f"test accuracy {accuracy:.3f}, test log loss {log_loss:.3f}")
accuracy, log_loss = test_metrics(nuts_f, nuts_draws)
print(f"learned Z (NUTS): test accuracy {accuracy:.3f}, test log loss {log_loss:.3f}")

# %%
# The predictive probability and its uncertainty
# ----------------------------------------------
# ``sparse_latent_gp_predictive`` evaluates ``f = K_*Z L⁻ᵀ v`` on a grid for
# every NUTS draw. The left panel shows the posterior mean of
# ``P(y = 1) = sigmoid(f)`` with the 0.5 decision boundary; grey segments join
# inducing inputs that are partners across the boundary. The right panel
# shows the posterior standard deviation of the probability: it is largest
# along the boundary and away from the data.

grid_probability = probabilities(nuts_f, nuts_draws, grid)
mean_surface = grid_probability.mean(0).reshape(mesh_x.shape)
sd_surface = grid_probability.std(0).reshape(mesh_x.shape)

fig, (ax_mean, ax_sd) = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
image = ax_mean.contourf(mesh_x, mesh_y, mean_surface, levels=20, cmap="RdBu_r", vmin=0, vmax=1)
ax_mean.contour(mesh_x, mesh_y, mean_surface, levels=[0.5], colors="black", linewidths=1.5)
fig.colorbar(image, ax=ax_mean, label="posterior mean of P(y = 1)", shrink=0.85)
scatter_classes(ax_mean, size=6)
for i, j in enumerate(partners["learned Z"].tolist()):
    if j >= 0:
        ax_mean.plot(z_learned[[i, j], 0], z_learned[[i, j], 1], color="0.4", linewidth=2, zorder=3)
ax_mean.scatter(z_learned[:, 0], z_learned[:, 1], marker="^", s=50, color="black", zorder=4, label="learned Z")
style(ax_mean, "Predictive probability (black line: P = 0.5)")
ax_mean.legend(loc="upper left", fontsize="x-small")
image = ax_sd.contourf(mesh_x, mesh_y, sd_surface, levels=20, cmap="viridis")
fig.colorbar(image, ax=ax_sd, label="posterior sd of P(y = 1)", shrink=0.85)
ax_sd.scatter(x[:, 0], x[:, 1], s=3, color="white", alpha=0.6)
ax_sd.scatter(z_learned[:, 0], z_learned[:, 1], marker="^", s=50, color="black")
style(ax_sd, "Uncertainty of the probability")
plt.show()

# %%
# Random fixed inducing inputs give a worse fit
# ---------------------------------------------
# The variational posteriors of step 2, side by side. With the random
# inducing inputs, several arms of the spiral are covered by too few kernel
# bumps, and the boundary cuts across them.

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
for ax, label, z in zip(axes, ("random Z", "learned Z"), (z_start, z_learned)):
    accuracy, log_loss = test_metrics(functions[label], variational_draws[label])
    surface = probabilities(functions[label], variational_draws[label], grid).mean(0).reshape(mesh_x.shape)
    image = ax.contourf(mesh_x, mesh_y, surface, levels=20, cmap="RdBu_r", vmin=0, vmax=1)
    ax.contour(mesh_x, mesh_y, surface, levels=[0.5], colors="black", linewidths=1.5)
    scatter_classes(ax, size=6)
    ax.scatter(z[:, 0], z[:, 1], marker="^", s=50, color="black", label="inducing inputs")
    style(ax, f"{label}: test accuracy {accuracy:.3f}, log loss {log_loss:.3f}")
axes[0].legend(loc="upper left", fontsize="x-small")
fig.colorbar(image, ax=axes, label="mean of P(y = 1)", shrink=0.85)
plt.show()

print(f"total runtime: {time.perf_counter() - start_time:.1f} s")
