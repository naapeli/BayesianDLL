"""
Bayesian Neural Network with Variational Inference
==================================================

A one-hidden-layer network with ``H = 8`` tanh units is fitted to 40 noisy
points of ``sin(1.3 x)`` that leave a gap around ``x = 0``:

.. math::

    h = \\tanh(x W_1 + b_1), \\quad f(x) = h w_2 + b_2, \\quad
    y \\sim N(f(x), \\sigma).

**Scaling.** ``x`` and ``y`` are standardised, and the weight priors follow
the fan-in of each layer: ``W_1, b_1 ~ N(0, 1)`` (one input) and
``w_2 ~ N(0, 1 / sqrt(H))``, so that the prior function has unit scale
whatever ``H`` is. The noise gets a tight prior on the log scale,
``log sigma ~ N(log 0.2, 0.3)`` with ``sigma = Exp(log sigma)``. The network
is assembled from ``MatMul``, ``Add`` and ``Tanh`` blocks.

**Why variational inference.** The posterior of a neural network is highly
multimodal: permuting the hidden units, or flipping the signs of a unit's
incoming and outgoing weights, gives the same function. NUTS chains
therefore settle in different but equivalent modes, so the R-hat of the
individual weights is meaningless, and the curved, strongly correlated
geometry inside each mode forces very long trajectories. Even with ordered,
sign-constrained hidden units, standardised data and the fan-in priors, a
four-chain NUTS run needed several minutes and still had R-hat of 1.04 for
the predicted function. Variational inference fits one mode in well under
a minute instead, so it is the method of this example.

Two automatic guides are compared: the mean-field
``Guide(model, default_family="mean_field")`` and the full-rank
``Guide(model, default_family="full_rank")`` (26 latents, so the covariance is
cheap), warm-started from the mean-field solution. The figures show the ELBO,
and for each guide the data, the true function, the predictive mean of ``f``
with 50% and 90% bands, and five individual networks drawn from the guide.
Both fit the data. The full-rank guide reaches a higher ELBO and is more
uncertain in the gap and beyond the data, where the mean-field guide is
overconfident. Both describe a single mode, so both understate the
uncertainty of the full posterior away from the data.
"""

import math
import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Guide, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Add, Exp, Linear, MatMul, Tanh
from BayesianDLL.Distributions import Normal
from BayesianDLL.Variational import BBVI


torch.manual_seed(5)

n_observations, n_hidden = 40, 8
x_raw = torch.cat([torch.linspace(-3.0, -0.8, n_observations // 2), torch.linspace(0.8, 3.0, n_observations // 2)]).double()
y_raw = torch.sin(1.3 * x_raw) + 0.15 * torch.randn(n_observations, dtype=torch.float64)
x_mean, x_std = x_raw.mean(), x_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()
x = (x_raw - x_mean) / x_std
y = (y_raw - y_mean) / y_std


def build_model():
    # Random starting weights break the symmetry between the hidden units
    # (the default start, all zeros, makes every unit identical).
    with Model() as model:
        inputs = Data("x", x.unsqueeze(-1))  # (N, 1)
        W1 = RandomParameter("W1", Normal(0.0, 1.0), shape=(1, n_hidden), initial_value=torch.randn(1, n_hidden, dtype=torch.float64))
        b1 = RandomParameter("b1", Normal(0.0, 1.0), shape=(n_hidden,), initial_value=torch.randn(n_hidden, dtype=torch.float64))
        w2 = RandomParameter("w2", Normal(0.0, 1.0 / math.sqrt(n_hidden)), shape=(n_hidden,),
                             initial_value=torch.randn(n_hidden, dtype=torch.float64) / math.sqrt(n_hidden))
        b2 = RandomParameter("b2", Normal(0.0, 1.0))
        log_sigma = RandomParameter("log_sigma", Normal(math.log(0.2), 0.3), initial_value=torch.tensor([math.log(0.2)], dtype=torch.float64))
        sigma = Exp("sigma", log_sigma)
        with plate("observations", n_observations):
            hidden = Tanh("hidden", Add("pre_activation", MatMul("x_W1", inputs, W1), b1))  # (N, H)
            f = Add("f", MatMul("h_w2", hidden, w2), b2)  # (N,)
            ObservedParameter("y", Normal(f, sigma), y)
    return model


# %%
# Fitting both guides
# -------------------
# The mean-field guide is fitted first. The full-rank guide starts from its
# solution (same means, diagonal ``scale_tril``) and then learns the
# correlations between the weights. Started from scratch, the full-rank guide
# tends to get stuck in a poor optimum where the noise explains the data.

torch.manual_seed(0)
model = build_model()
mean_field = Guide(model, default_family="mean_field", init_scale=0.05)
history_mean_field = BBVI(model, mean_field, n_samples=4, epochs=3000, lr=0.02, lrd=0.999, callback_frequency=10, verbose=False)

full_rank = Guide(model, default_family="full_rank")
full_rank.groups[0].loc_parameter.set_new_value(mean_field.groups[0].loc)
full_rank.groups[0].scale_tril_parameter.set_new_value(torch.diag(mean_field.groups[0].scale))
history_full_rank = BBVI(model, full_rank, n_samples=4, epochs=1500, lr=0.005, lrd=0.999, callback_frequency=10, verbose=False)

fits = {
    "mean-field": (mean_field.sample(2000), history_mean_field),
    "full-rank": (full_rank.sample(2000), history_mean_field + history_full_rank),
}
for label, (trace, history) in fits.items():
    print(f"{label:>22}: final ELBO estimate {sum(history[-20:]) / 20:.2f}, "
          f"posterior mean sigma {trace['sigma'].mean().item() * y_std.item():.3f} (true 0.15)")

# %%
# Predictive distribution of f
# ----------------------------
# Every guide draw is one network; evaluating all of them on a grid gives
# the posterior of ``f``. The results are mapped back to the original units.

grid_raw = torch.linspace(-4.5, 4.5, 200, dtype=torch.float64)
grid = (grid_raw - x_mean) / x_std


def networks(trace):
    W1, b1, w2, b2 = (trace[name].flatten(0, 1) for name in ("W1", "b1", "w2", "b2"))
    hidden = torch.tanh(grid.reshape(1, -1, 1) @ W1 + b1.unsqueeze(1))  # (S, G, H)
    f = (hidden @ w2.unsqueeze(-1)).squeeze(-1) + b2  # (S, G)
    return y_mean + y_std * f


colors = {"mean-field": "C1", "full-rank": "C0"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained", gridspec_kw={"width_ratios": [0.8, 1, 1]})
steps = range(10, 10 * len(fits["full-rank"][1]) + 1, 10)
axes[0].plot(steps[:len(history_mean_field)], history_mean_field, color=colors["mean-field"], alpha=0.8, label="mean-field")
axes[0].plot(steps[len(history_mean_field):], history_full_rank, color=colors["full-rank"], alpha=0.8,
             label="full-rank (warm start)")
axes[0].set_ylim(history_full_rank[-1] - 120, max(history_full_rank) + 10)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("ELBO estimate")
axes[0].set_title("Optimisation")
axes[0].legend(fontsize="small", loc="lower right")

quantiles = torch.tensor([0.05, 0.25, 0.75, 0.95], dtype=torch.float64)
for ax, (label, (trace, history)) in zip(axes[1:], fits.items()):
    f = networks(trace)
    q05, q25, q75, q95 = f.quantile(quantiles, dim=0)
    ax.fill_between(grid_raw, q05, q95, color=colors[label], alpha=0.2, label="90% band")
    ax.fill_between(grid_raw, q25, q75, color=colors[label], alpha=0.4, label="50% band")
    ax.plot(grid_raw, f[:5].T, color=colors[label], lw=0.7, alpha=0.8)
    ax.plot(grid_raw, f.mean(0), color=colors[label], lw=2, label="predictive mean")
    ax.plot(grid_raw, torch.sin(1.3 * grid_raw), color="black", linestyle="--", label="true function")
    ax.scatter(x_raw, y_raw, color="black", s=12, zorder=3, label="data")
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"{label} (thin lines: 5 networks)")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize="small")
plt.show()

# %%
# A deeper network: where mean-field VI breaks down
# -------------------------------------------------
# Neural network architectures can be constructed using deterministic layer
# blocks by composing :class:`~BayesianDLL.Deterministic.Linear` and
# :class:`~BayesianDLL.Deterministic.Tanh` blocks.
#
# Here a four-hidden-layer MLP (1 -> 8 -> 8 -> 8 -> 8 -> 1) is fitted to a composite
# signal :math:`\sin(1.5 x) + 0.5 \sin(3.5 x)`. The weights use fan-in priors with scale
# :math:`2 / \sqrt{H}` so the network has sufficient expressive range to capture
# the higher-frequency harmonic. Both methods start from a MAP estimate:
#
# * **Mean-field VI**, run until the ELBO levels off. With 241 weights the
#   ELBO-optimal factorised guide prunes hidden units (their weights revert
#   to the prior) and smooths the harmonic away, so its band is wide but
#   misses the signal. Stopping it early with a tiny initial scale instead
#   leaves every standard deviation near its starting value, and the band
#   then reflects the optimiser, not the posterior.
# * **NUTS from the MAP estimate**, two chains. Each chain explores one of the
#   equivalent modes (see above), but the predicted functions agree, and the
#   band is narrow where the data are and widens beyond them.

h1_dim, h2_dim, h3_dim, h4_dim = 8, 8, 8, 8

def complex_signal(t):
    return torch.sin(1.5 * t) + 0.5 * torch.sin(3.5 * t)

torch.manual_seed(42)
n_deep = 100
x_deep_raw = torch.linspace(-3.0, 3.0, n_deep, dtype=torch.float64)
y_deep_raw = complex_signal(x_deep_raw) + 0.15 * torch.randn(n_deep, dtype=torch.float64)

x_deep_mean, x_deep_std = x_deep_raw.mean(), x_deep_raw.std()
y_deep_mean, y_deep_std = y_deep_raw.mean(), y_deep_raw.std()
x_deep = (x_deep_raw - x_deep_mean) / x_deep_std
y_deep = (y_deep_raw - y_deep_mean) / y_deep_std

with Model() as deep_bnn:
    x_in = Data("x_in", x_deep.unsqueeze(-1))
    w1 = RandomParameter("w1", Normal(0.0, 2.0), shape=(1, h1_dim),
                         initial_value=2.0 * torch.randn(1, h1_dim, dtype=torch.float64))
    b1 = RandomParameter("b1", Normal(0.0, 2.0), shape=(h1_dim,),
                         initial_value=torch.randn(h1_dim, dtype=torch.float64) * 0.1)
    w2 = RandomParameter("w2", Normal(0.0, 2.0 / math.sqrt(h1_dim)), shape=(h1_dim, h2_dim),
                         initial_value=2.0 * torch.randn(h1_dim, h2_dim, dtype=torch.float64) / math.sqrt(h1_dim))
    b2 = RandomParameter("b2", Normal(0.0, 2.0), shape=(h2_dim,),
                         initial_value=torch.randn(h2_dim, dtype=torch.float64) * 0.1)
    w3 = RandomParameter("w3", Normal(0.0, 2.0 / math.sqrt(h2_dim)), shape=(h2_dim, h3_dim),
                         initial_value=2.0 * torch.randn(h2_dim, h3_dim, dtype=torch.float64) / math.sqrt(h2_dim))
    b3 = RandomParameter("b3", Normal(0.0, 2.0), shape=(h3_dim,),
                         initial_value=torch.randn(h3_dim, dtype=torch.float64) * 0.1)
    w4 = RandomParameter("w4", Normal(0.0, 2.0 / math.sqrt(h3_dim)), shape=(h3_dim, h4_dim),
                         initial_value=2.0 * torch.randn(h3_dim, h4_dim, dtype=torch.float64) / math.sqrt(h3_dim))
    b4 = RandomParameter("b4", Normal(0.0, 2.0), shape=(h4_dim,),
                         initial_value=torch.randn(h4_dim, dtype=torch.float64) * 0.1)
    w5 = RandomParameter("w5", Normal(0.0, 2.0 / math.sqrt(h4_dim)), shape=(h4_dim,),
                         initial_value=2.0 * torch.randn(h4_dim, dtype=torch.float64) / math.sqrt(h4_dim))
    b5 = RandomParameter("b5", Normal(0.0, 2.0), initial_value=torch.zeros(1, dtype=torch.float64))
    sigma_deep = Exp("sigma_deep", RandomParameter("log_sigma_deep", Normal(math.log(0.15), 0.3),
                                                   initial_value=torch.tensor([math.log(0.15)], dtype=torch.float64)))

    h1 = Tanh("h1", Linear("h1_lin", x_in, coefficients=w1, intercept=b1))
    h2 = Tanh("h2", Linear("h2_lin", h1, coefficients=w2, intercept=b2))
    h3 = Tanh("h3", Linear("h3_lin", h2, coefficients=w3, intercept=b3))
    h4 = Tanh("h4", Linear("h4_lin", h3, coefficients=w4, intercept=b4))
    f_mlp = Linear("f_mlp", h4, coefficients=w5, intercept=b5)
    with plate("deep_obs", n_deep):
        ObservedParameter("y_deep", Normal(f_mlp, sigma_deep), y_deep)

# Start at the MAP mode: random initial weights are far from the posterior mass.
deep_bnn.find_MAP(lr=0.03, epochs=1500, verbose=False)
map_values = {name: parameter.constrained_value.clone() for name, parameter in deep_bnn.params.items()}

deep_guide = Guide(deep_bnn, default_family="mean_field", init_scale=0.05)
deep_history = BBVI(deep_bnn, deep_guide, n_samples=4, epochs=800, lr=0.01, lrd=0.999, verbose=False)
deep_draws = {"mean-field VI": {k: v.squeeze(0) for k, v in deep_guide.sample(200).items()}}
print(f"Deep BNN mean-field VI: final ELBO {sum(deep_history[-20:]) / 20:.1f}")

deep_bnn.set_params(map_values)
start = time.perf_counter()
deep_nuts = deep_bnn.sample(100, 100, n_chains=2, init_scale=0.0, max_depth=6, seed=0, progress_bar=False,
                            chain_method="vectorized", check_convergence=False)
deep_draws["NUTS from the MAP"] = {k: v.flatten(0, 1) for k, v in deep_nuts.trace.items()}
print(f"Deep BNN NUTS: {time.perf_counter() - start:.1f} s, divergences: {sum(deep_nuts.divergences)}")


def network_outputs(draws):
    """The network on the grid (original scale) for every draw of the weights."""
    inputs = ((grid_raw - x_deep_mean) / x_deep_std).unsqueeze(-1)
    outputs = []
    for i in range(draws["w1"].shape[0]):
        h = inputs
        for layer in (1, 2, 3, 4):
            h = torch.tanh(h @ draws[f"w{layer}"][i] + draws[f"b{layer}"][i])
        outputs.append((h @ draws["w5"][i] + draws["b5"][i]) * y_deep_std + y_deep_mean)
    return torch.stack(outputs)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, layout="constrained")
in_range = grid_raw.abs() <= 3.0
for ax, (label, draws), color in zip(axes, deep_draws.items(), ("C1", "C4")):
    f = network_outputs(draws)
    q05, q95 = f.quantile(torch.tensor([0.05, 0.95], dtype=torch.float64), dim=0)
    error = (f.mean(0) - complex_signal(grid_raw))[in_range].square().mean().sqrt()
    print(f"{label:>18}: RMS error of the predictive mean in the data range {error:.3f}")
    ax.fill_between(grid_raw, q05, q95, color=color, alpha=0.25, label="90% band")
    ax.plot(grid_raw, f.mean(0), color=color, lw=2, label="predictive mean")
    ax.plot(grid_raw, complex_signal(grid_raw), "k--", label="true function")
    ax.scatter(x_deep_raw, y_deep_raw, color="black", s=10, zorder=3, label="data")
    ax.set(title=label, xlabel="x", ylim=(-2.8, 2.8))
axes[0].set_ylabel("f(x)")
axes[1].legend(fontsize="small", loc="lower right")
fig.suptitle("Four-hidden-layer Bayesian neural network (Linear & Tanh layers)")
plt.show()
