r"""
KL Annealing for Variational Inference
======================================

The ELBO is the expected log likelihood plus :math:`E_q[\log p(\theta) - \log q(\theta)]`.
The second term pulls the guide towards the prior and keeps it wide. In a
deep Bayesian neural network most weights are weakly determined by the data,
so early in training this pull competes with a poor fit, and units whose
weights the guide lets revert to the prior tend to stay there: the network
ends up smaller than intended.

``BBVI(..., kl_annealing=W)`` follows, at epoch :math:`t`, the gradient of

.. math::

    E_q[\log p(y \mid \theta)] + w_t \, E_q[\log p(\theta) - \log q(\theta)],
    \qquad w_t = \min(1, t / W),

a linear warm-up of the KL weight over ``W`` epochs (a callable
``epoch -> weight`` in :math:`(0, 1]` gives any other schedule). The
likelihood shapes the guide first and the prior and entropy come in
gradually. From epoch ``W`` on this is plain BBVI, and the recorded ELBO is
always the true one (:math:`w = 1`), also during the warm-up.

Here a four-hidden-layer network (1 -> 8 -> 8 -> 8 -> 8 -> 1) is fitted to
the signal :math:`\sin(1.5 x) + 0.5 \sin(3.5 x)` by mean-field VI from the
same MAP estimate, with and without a 400-epoch warm-up. The warm-up
lowers the RMS error of the predictive mean (0.364 to 0.292 in the run
below), but both guides keep the same number of active units in the last
layer, and after 800 epochs, with both runs still improving, the annealed
one has the lower ELBO (the ELBO scores the approximation of the posterior,
not the fit to the signal). Annealing is a cheap option to try when a
mean-field fit looks over-smoothed, not a guaranteed gain.
"""

import math
import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Guide, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Exp, Linear, Tanh
from BayesianDLL.Distributions import Normal
from BayesianDLL.Variational import BBVI

widths = [1, 8, 8, 8, 8, 1]
n_observations = 100
torch.manual_seed(42)
signal = lambda t: torch.sin(1.5 * t) + 0.5 * torch.sin(3.5 * t)
x_raw = torch.linspace(-3.0, 3.0, n_observations, dtype=torch.float64)
y_raw = signal(x_raw) + 0.15 * torch.randn(n_observations, dtype=torch.float64)
x_mean, x_std = x_raw.mean(), x_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()
x = (x_raw - x_mean) / x_std
y = (y_raw - y_mean) / y_std
n_layers = len(widths) - 1


def prior_scale(layer):
    return 2.0 / math.sqrt(widths[layer - 1]) if layer > 1 else 2.0


with Model() as model:
    h = Data("x", x.unsqueeze(-1))
    for layer in range(1, n_layers + 1):
        fan_in, fan_out = widths[layer - 1], widths[layer]
        last = layer == n_layers  # the output layer has a vector of weights and a scalar bias
        W = RandomParameter(f"w{layer}", Normal(0.0, prior_scale(layer)), shape=(fan_in,) if last else (fan_in, fan_out),
                            initial_value=2.0 * torch.randn(*((fan_in,) if last else (fan_in, fan_out)), dtype=torch.float64) / math.sqrt(fan_in))
        b = RandomParameter(f"b{layer}", Normal(0.0, 2.0), **({} if last else {"shape": (fan_out,)}),
                            initial_value=0.1 * torch.randn(*((1,) if last else (fan_out,)), dtype=torch.float64))
        h = Linear(f"lin{layer}", h, coefficients=W, intercept=b)
        if not last:
            h = Tanh(f"h{layer}", h)
    log_sigma = RandomParameter("log_sigma", Normal(math.log(0.15), 0.3), initial_value=torch.tensor([math.log(0.15)], dtype=torch.float64))
    with plate("observations", n_observations):
        ObservedParameter("y", Normal(h, Exp("sigma", log_sigma)), y)

# %%
# Fitting with and without a warm-up
# ----------------------------------
# Both runs start from the same MAP estimate and use the same seed. A hidden
# unit of the last layer is *active* when the posterior mean of its outgoing
# weight exceeds 0.1; a pruned unit has a mean of about zero.

torch.manual_seed(0)
model.find_MAP(lr=0.03, epochs=1500, verbose=False)
map_values = {name: parameter.constrained_value.clone() for name, parameter in model.params.items()}

epochs, warmup = 800, 400
grid_raw = torch.linspace(-3.0, 3.0, 200, dtype=torch.float64)
grid = (grid_raw - x_mean) / x_std

results = {}
print(f"{'':>20} {'ELBO':>8} {'RMS error':>10} {'active units':>13} {'time':>6}")
for label, annealing in (("no annealing", None), (f"{warmup}-epoch warm-up", warmup)):
    model.set_params(map_values)
    torch.manual_seed(1)
    guide = Guide(model, default_family="mean_field", init_scale=0.05)
    start = time.perf_counter()
    history = BBVI(model, guide, n_samples=4, epochs=epochs, lr=0.01, lrd=0.999, callback_frequency=10, verbose=False, kl_annealing=annealing)
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
    print(f"{label:>20} {elbo:8.1f} {rms:10.3f} {active:>9d}/{widths[-2]} {elapsed:5.1f}s")

# %%
# The ELBO and the fits
# ---------------------
# Left: the recorded ELBO of both runs. Middle: the posterior means of the
# outgoing weights of the last hidden layer. Right: the predictive means
# against the true signal.

colors = {"no annealing": "C1", f"{warmup}-epoch warm-up": "C0"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
for label, (history, f, weights) in results.items():
    axes[0].plot(range(10, epochs + 1, 10), history, color=colors[label], alpha=0.8, label=label)
    axes[1].plot(weights, "o", color=colors[label], label=label)
    axes[2].plot(grid_raw, f.mean(0), color=colors[label], lw=2, label=label)
best = max(max(h[-20:]) for h, _, _ in results.values())
axes[0].set(xlabel="epoch", ylabel="ELBO estimate", ylim=(best - 400, best + 20), title="Optimisation")
axes[0].legend(fontsize="small", loc="lower right")
axes[1].axhline(0.0, color="gray", ls=":")
axes[1].set(xlabel="unit of the last hidden layer", ylabel="posterior mean of its output weight", title="Output weights")
axes[2].plot(grid_raw, signal(grid_raw), "k--", label="true signal")
axes[2].scatter(x_raw, y_raw, color="black", s=10, zorder=3, label="data")
axes[2].set(xlabel="x", ylabel="f(x)", ylim=(-2.2, 2.2), title="Predictive mean")
axes[2].legend(fontsize="small", loc="lower right")
plt.show()
