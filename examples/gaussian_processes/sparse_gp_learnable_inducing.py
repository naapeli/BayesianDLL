"""
Learning Where to Put Inducing Points
=====================================

A sparse GP is only as good as its inducing inputs ``Z``: the function is
summarised by its values at ``Z``, so regions without inducing inputs are
poorly represented. Choosing ``Z`` by hand is easy in one dimension and
hopeless in many, so here ``Z`` is a model parameter and is *learned*.

The data mix a slow oscillation with a sharp bump near ``x = 7``. Eight
inducing inputs start badly placed, bunched in ``[0, 1.5]``. We compare

* **fixed Z**: the kernel hyperparameters are fitted, ``Z`` stays put;
* **learned Z**: ``Z`` (a ``RandomParameter`` of shape ``(8, 1)`` with a
  broad Normal prior around the start) is fitted jointly with the
  hyperparameters by a MAP estimate: the VFE bound plus the log priors is
  maximised.

The gradient of the bound with respect to ``Z`` is hand-derived: the
kernel's ``input_vjp`` contracts ``∂K_mn/∂Z`` and ``∂K_mm/∂Z`` with the
bound's cotangents without forming any ``m x n x m`` tensor.

Use ``method="vfe"`` for learnable inducing inputs. VFE is a lower bound on
the exact log marginal likelihood, so moving ``Z`` can only make the
approximation tighter; DTC and FITC are not bounds, and optimising ``Z``
under them can overfit (e.g. FITC collapsing inducing points and
underestimating the noise).
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import RBF, ExactGP, SparseGP, sparse_gp_predictive


torch.manual_seed(1)


def true_function(x):
    return torch.sin(0.8 * x) + 1.5 * torch.exp(-0.5 * ((x - 7.0) / 0.35) ** 2)


n, m = 300, 8
x = torch.sort(torch.rand(n, dtype=torch.float64) * 10.0).values
y = true_function(x) + 0.15 * torch.randn(n, dtype=torch.float64)
z_start = torch.linspace(0.0, 1.5, m, dtype=torch.float64).reshape(m, 1)
grid = torch.linspace(-0.3, 10.3, 250, dtype=torch.float64)


def build(inducing):
    """``inducing`` is a fixed tensor, or ``None`` for learnable inducing inputs."""
    with Model() as model:
        inputs = Data("inputs", x)
        log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0))
        if inducing is None:
            inducing = RandomParameter("z", Normal(z_start, 5.0), shape=(m, 1), initial_value=z_start.clone())
        f = SparseGP(
            "f", inputs, RBF(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance)),
            inducing_points=inducing, noise_variance=Exp("noise", log_noise), method="vfe",
        )
        ObservedParameter("y", f, y)
    return model, f


def point_trace(model, f):
    trace = {name: model.deterministic_params[name].constrained_value.reshape(1, 1, -1)
             for name in ("lengthscale", "variance", "noise")}
    if "z" in model.params:
        trace["z"] = model.params["z"].constrained_value.reshape(1, 1, m, 1)
    return trace


# %%
# Fitting
# -------
# Both models are optimised with Adam (``find_MAP``) in chunks of 25 steps,
# recording the VFE bound and the inducing inputs after every chunk.

start = time.perf_counter()
fixed_model, fixed_f = build(z_start)
fixed_model.find_MAP(lr=0.05, epochs=400, verbose=False)
fixed_bound = fixed_f.log_pdf(y).item()

learned_model, learned_f = build(None)
steps, bounds, paths = [0], [learned_f.log_pdf(y).item()], [z_start.reshape(-1).clone()]
for chunk in range(16):
    learned_model.find_MAP(lr=0.05, epochs=25, verbose=False)
    steps.append(25 * (chunk + 1))
    bounds.append(learned_f.log_pdf(y).item())
    paths.append(learned_model.params["z"].constrained_value.reshape(-1).clone())
z_learned = learned_model.params["z"].constrained_value.clone()
paths = torch.stack(paths)
print(f"optimisation: {time.perf_counter() - start:.1f} s")

values = point_trace(learned_model, learned_f)
with Model():
    exact = ExactGP("exact", Data("inputs", x),
                    RBF(values["lengthscale"].reshape(()), values["variance"].reshape(())),
                    noise_variance=values["noise"].reshape(()))
exact_value = exact.log_pdf(y).item()
print(f"VFE bound with fixed Z: {fixed_bound:.1f}; learned Z: {bounds[-1]:.1f}; exact log p(y): {exact_value:.1f}")
print("learned Z:", [round(v, 2) for v in sorted(z_learned.reshape(-1).tolist())])

# %%
# Fits with fixed and learned inducing inputs
# -------------------------------------------
# With the inducing inputs stuck on the left, the fixed model can only
# explain the right part of the data as noise. After learning, the inducing
# inputs spread over the whole range and cluster around the bump.

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, layout="constrained")
for ax, model, f, z, title in [
    (axes[0], fixed_model, fixed_f, z_start, f"Fixed Z (bound {fixed_bound:.0f})"),
    (axes[1], learned_model, learned_f, z_learned, f"Learned Z (bound {bounds[-1]:.0f})"),
]:
    draws = sparse_gp_predictive(f, point_trace(model, f), y, grid, n_samples=300).reshape(-1, grid.numel())
    lower, median, upper = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)
    ax.scatter(x, y, s=4, color="0.65", label="observations")
    ax.fill_between(grid, lower, upper, color="tab:blue", alpha=0.3, label="90% band of f")
    ax.plot(grid, median, color="tab:blue", label="median of f")
    ax.plot(grid, true_function(grid), "k--", linewidth=1, label="true f")
    ax.plot(z.reshape(-1), torch.full((m,), -1.9), "^", color="tab:red", markersize=8, label="inducing inputs")
    ax.set(xlabel="x", title=title)
axes[0].set_ylabel("y")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize="small")
plt.show()

# %%
# How the bound and the inducing inputs evolve
# --------------------------------------------
# The bound rises towards the exact log marginal likelihood (at the learned
# hyperparameters) as the inducing inputs migrate to the right; the first,
# much lower value at step 0 is cut off. The shaded strip marks the bump.

fig, (ax_bound, ax_path) = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
ax_bound.plot(steps, bounds, "o-", color="tab:blue", label="VFE bound, learned Z")
ax_bound.axhline(fixed_bound, color="tab:orange", linestyle="--", label="VFE bound, fixed Z")
ax_bound.axhline(exact_value, color="black", linestyle=":", label="exact log p(y)")
ax_bound.set(xlabel="Adam step", ylabel="log marginal likelihood", title="VFE bound during optimisation",
             ylim=(min(bounds[1], fixed_bound) - 50, exact_value + 30))
ax_bound.legend(loc="center right", fontsize="small")
ax_path.plot(steps, paths, color="tab:red", alpha=0.8)
ax_path.axhspan(6.3, 7.7, color="0.9")
ax_path.text(steps[-1], 7.0, "bump ", va="center", ha="right", fontsize="small")
ax_path.set(xlabel="Adam step", ylabel="inducing input location", title="Trajectories of the inducing inputs")
plt.show()

# %%
# Posterior of the hyperparameters with the learned inducing inputs
# -----------------------------------------------------------------
# Inducing inputs are variational parameters rather than quantities of
# interest, and their posterior is multimodal (any permutation of ``Z`` gives
# the same bound), so we keep the optimised ``Z`` fixed and sample the
# hyperparameters with NUTS.

final_model, final_f = build(z_learned)
start = time.perf_counter()
trace = final_model.sample(300, 300, n_chains=2, max_depth=5, seed=3, progress_bar=False)
elapsed = time.perf_counter() - start
summary = trace.summary(include_deterministic=True)
print(summary)
print(f"sampling: {elapsed:.1f} s, divergences: {sum(trace.divergences)}, max R-hat: {summary['r_hat'].max():.3f}")
