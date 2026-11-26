r"""
KL Annealing for Variational Inference
======================================

The ELBO is the expected log likelihood plus :math:`E_q[\log p(\theta) - \log q(\theta)]`.
The second term pulls the guide towards the prior and keeps it wide. In a
Bayesian neural network started from scratch this is fatal: at a random
initialisation the likelihood has little to say about any single weight,
while the entropy term immediately inflates the standard deviations of the
guide. The noise this injects into every layer swamps the signal, the
likelihood gradient for the means stays weak and the network settles on a
nearly constant function whose hidden units have reverted to the prior.

``BBVI(..., kl_annealing=W)`` follows, at epoch :math:`t`, the gradient of

.. math::

    E_q[\log p(y \mid \theta)] + w_t \, E_q[\log p(\theta) - \log q(\theta)],
    \qquad w_t = \min(1, t / W),

a linear warm-up of the KL weight over ``W`` epochs (a callable
``epoch -> weight`` in :math:`(0, 1]` gives any other schedule). The
likelihood shapes the guide first and the prior and entropy come in
gradually. From epoch ``W`` on this is plain BBVI, and the recorded ELBO is
always the true one (:math:`w = 1`), also during the warm-up.

Here a network with two hidden layers of 30 tanh units is fitted to the
signal :math:`\sin(1.5 x) + 0.5 \sin(3.5 x)` by mean-field VI from a random
initialisation, with and without a 2500-epoch warm-up (same seeds, learning
rate and number of epochs). Without it the guide collapses to a nearly
constant mean; with it the fit follows the signal. The collapsed run ends
with the *higher* ELBO; the last section shows why. Annealing is a way to
reach a different, better-fitting optimum, not a way to raise the ELBO.
"""

import math
import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Guide, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Exp, Linear, Tanh
from BayesianDLL.Distributions import Normal
from BayesianDLL.Variational import BBVI

widths = [1, 30, 30, 1]
n_observations = 200
torch.manual_seed(0)
signal = lambda t: torch.sin(1.5 * t) + 0.5 * torch.sin(3.5 * t)
x_raw = torch.linspace(-3.0, 3.0, n_observations, dtype=torch.float64)
y_raw = signal(x_raw) + 0.05 * torch.randn(n_observations, dtype=torch.float64)
x_mean, x_std = x_raw.mean(), x_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()
x = (x_raw - x_mean) / x_std
y = (y_raw - y_mean) / y_std
n_layers = len(widths) - 1


def build_model():
    """The network with a random initial value for every weight."""
    with Model() as model:
        h = Data("x", x.unsqueeze(-1))
        for layer in range(1, n_layers + 1):
            fan_in, fan_out = widths[layer - 1], widths[layer]
            last = layer == n_layers  # the output layer has a vector of weights and a scalar bias
            w_shape = (fan_in,) if last else (fan_in, fan_out)
            W = RandomParameter(f"w{layer}", Normal(0.0, 1.0), shape=w_shape,
                                initial_value=torch.randn(*w_shape, dtype=torch.float64) / math.sqrt(fan_in))
            b = RandomParameter(f"b{layer}", Normal(0.0, 1.0), **({} if last else {"shape": (fan_out,)}),
                                initial_value=0.1 * torch.randn(*((1,) if last else (fan_out,)), dtype=torch.float64))
            h = Linear(f"lin{layer}", h, coefficients=W, intercept=b)
            if not last:
                h = Tanh(f"h{layer}", h)
        log_sigma = RandomParameter("log_sigma", Normal(math.log(0.3), 0.5), initial_value=torch.tensor([math.log(0.3)], dtype=torch.float64))
        with plate("observations", n_observations):
            ObservedParameter("y", Normal(h, Exp("sigma", log_sigma)), y)
    return model

# %%
# Fitting with and without a warm-up
# ----------------------------------
# Both runs start from the same random initialisation (not from a MAP
# estimate, which would already have fitted the signal) and use the same
# seeds. A hidden unit of the last layer is *active* when the posterior mean
# of its outgoing weight exceeds 0.1; a pruned unit has a mean of about zero.
# The RMS error compares the predictive mean with the true signal.

epochs, warmup = 3500, 2500
grid_raw = torch.linspace(-3.0, 3.0, 200, dtype=torch.float64)
grid = (grid_raw - x_mean) / x_std

results, guides = {}, {}
print(f"{'':>20} {'ELBO':>8} {'RMS error':>10} {'active units':>13} {'time':>6}")
for label, annealing in (("no annealing", None), (f"{warmup}-epoch warm-up", warmup)):
    torch.manual_seed(1)
    model = build_model()
    torch.manual_seed(2)
    guide = Guide(model, default_family="mean_field", init_scale=0.05)
    start = time.perf_counter()
    history = BBVI(model, guide, n_samples=2, epochs=epochs, lr=0.02, lrd=0.9995, callback_frequency=10, verbose=False, kl_annealing=annealing)
    elapsed = time.perf_counter() - start
    draws = {k: v.flatten(0, 1) for k, v in guide.sample(200).items()}
    outputs = []
    for i in range(200):
        a = grid.unsqueeze(-1)
        for layer in range(1, n_layers):
            a = torch.tanh(a @ draws[f"w{layer}"][i] + draws[f"b{layer}"][i])
        outputs.append((a @ draws[f"w{n_layers}"][i] + draws[f"b{n_layers}"][i]) * y_std + y_mean)
    f = torch.stack(outputs)
    rms = (f.mean(0) - signal(grid_raw)).pow(2).mean().sqrt().item()
    out_weights = draws[f"w{n_layers}"].mean(0)
    active = int((out_weights.abs() > 0.1).sum())
    elbo = sum(history[-20:]) / 20
    results[label] = (history, f, out_weights)
    guides[label] = guide
    print(f"{label:>20} {elbo:8.1f} {rms:10.3f} {active:>9d}/{widths[-2]} {elapsed:5.1f}s")

# %%
# The fits and the optimisation
# -----------------------------
# Top: the predictive mean and 90% band of the network function for both
# runs. Bottom left: the recorded ELBO (the true ELBO, also during the
# warm-up), smoothed, together with the KL weight. Bottom right: the posterior means of
# the outgoing weights of the last hidden layer.

colors = {"no annealing": "C1", f"{warmup}-epoch warm-up": "C0"}
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
for ax, (label, (history, f, weights)) in zip(axes[0], results.items()):
    lower, upper = torch.quantile(f, torch.tensor([0.05, 0.95], dtype=torch.float64), dim=0)
    ax.fill_between(grid_raw, lower, upper, color=colors[label], alpha=0.3, label="90% band")
    ax.plot(grid_raw, f.mean(0), color=colors[label], lw=2, label="predictive mean")
    ax.plot(grid_raw, signal(grid_raw), "k--", label="true signal")
    ax.scatter(x_raw, y_raw, color="black", s=8, zorder=3, label="data")
    ax.set(xlabel="x", ylabel="f(x)", ylim=(-2.2, 2.2), title=label)
    ax.legend(fontsize="small", loc="lower right")
epoch_axis = range(10, epochs + 1, 10)
for label, (history, _, weights) in results.items():
    smooth = torch.tensor(history, dtype=torch.float64).unfold(0, 10, 1).mean(1)  # moving average over 100 epochs
    axes[1, 0].plot(epoch_axis[9:], smooth, color=colors[label], label=label)
    axes[1, 1].plot(weights, "o", color=colors[label], label=label)
axes[1, 0].set(xlabel="epoch", ylabel="ELBO (100-epoch average)", ylim=(-700, -300), title="Optimisation")
axes[1, 0].legend(fontsize="small", loc="lower right")
weight_axis = axes[1, 0].twinx()
weight_axis.plot(epoch_axis, [min(1.0, t / warmup) for t in epoch_axis], color="gray", ls=":")
weight_axis.set(ylabel="KL weight (dotted)", ylim=(0, 1.05))
axes[1, 1].axhline(0.0, color="gray", ls=":")
axes[1, 1].set(xlabel="unit of the last hidden layer", ylabel="posterior mean of its output weight", title="Output weights")
axes[1, 1].legend(fontsize="small")
plt.show()

# %%
# Why the collapsed guide has the higher ELBO
# -------------------------------------------
# The ELBO is split into the expected log likelihood and the KL divergence
# from the prior, layer by layer, from 2000 draws of each guide (the guides
# are Gaussian in every weight, so each weight's KL follows from its mean and
# standard deviation).
#
# A mean-field guide gives every weight an independent variance. The exact
# posterior can pin down the *function* while single weights stay uncertain,
# because their errors cancel; a mean-field guide cannot express this, so it
# can only make the function precise by shrinking the standard deviation of
# every weight it uses, at a cost of about :math:`\log(1 / s)` nats each. The
# remaining weight noise still makes every sampled function wobble, so the
# learned noise scale stays well above the true one, which caps what the
# better fit gains in likelihood. With 200 observations the fit gains less
# likelihood than it pays in KL, so the ELBO prefers the prior: annealing
# reaches the better fit, but the ELBO scores it lower. The likelihood grows
# with the number of observations and the KL does not; with 2000 observations
# both guides fit the signal and the better fit also has the higher ELBO. This
# is a known weakness of mean-field Bayesian neural networks (Coker et al.,
# 2022, "Wide Mean-Field Bayesian Neural Networks Ignore the Data").

groups = {"first layer": ("w1", "b1"), "hidden layer": ("w2", "b2"), "output layer": (f"w{n_layers}", f"b{n_layers}"),
          "noise scale": ("log_sigma",)}
print(f"\n{'':>20} {'E[log lik]':>11} " + " ".join(f"{name:>13}" for name in groups) + f" {'ELBO':>8} {'noise sd':>9}")
for label, guide in guides.items():
    torch.manual_seed(3)
    draws = {k: v.flatten(0, 1) for k, v in guide.sample(2000).items()}
    a = x.reshape(1, -1, 1)
    for layer in range(1, n_layers):
        a = torch.tanh(a @ draws[f"w{layer}"] + draws[f"b{layer}"].unsqueeze(1))
    f_data = (a @ draws[f"w{n_layers}"].unsqueeze(-1)).squeeze(-1) + draws[f"b{n_layers}"].reshape(-1, 1)
    sigma = draws["log_sigma"].reshape(-1, 1).exp()
    log_likelihood = torch.distributions.Normal(f_data, sigma).log_prob(y.reshape(1, -1)).sum(-1).mean().item()
    kl = {}
    for name, members in groups.items():
        total = 0.0
        for member in members:
            prior_mean, prior_sd = (math.log(0.3), 0.5) if member == "log_sigma" else (0.0, 1.0)
            values = draws[member].reshape(draws[member].shape[0], -1)
            mean, sd = values.mean(0), values.std(0)
            total += (torch.log(prior_sd / sd) + (sd ** 2 + (mean - prior_mean) ** 2) / (2 * prior_sd ** 2) - 0.5).sum().item()
        kl[name] = total
    elbo = log_likelihood - sum(kl.values())
    print(f"{label:>20} {log_likelihood:11.1f} " + " ".join(f"{value:13.1f}" for value in kl.values())
          + f" {elbo:8.1f} {sigma.mean().item() * y_std.item():9.3f}")
print(f"(KL per group in nats; true noise sd {0.05:.3f})")
