"""
Variational Inference for GP Hyperparameters
============================================

Eighteen noisy observations of ``sin(2 pi x)`` are modelled with an exact
Gaussian process, ``y ~ GP(0, k_RBF) + N(0, noise_variance)``. ``ExactGP``
integrates the latent function out analytically, so the only latents are
the three kernel hyperparameters: lengthscale, signal variance and noise
variance.

All three are positive and span orders of magnitude, so the model puts
Normal priors on their **logarithms** and recovers them with ``Exp`` blocks.
In these coordinates the posterior is close to Gaussian, but the
hyperparameters are correlated (a longer lengthscale can be traded against
more noise), so the guide is ``Guide(model, default_family="full_rank")``,
one full-rank Normal over the three log-hyperparameters.

NUTS on the same model serves as the reference. The figures show the ELBO,
the marginal posteriors of the hyperparameters under both methods, and the
posterior predictive mean and 95% band of the latent function from both,
over the data.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Guide, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import RBF, ExactGP
from BayesianDLL.Variational import BBVI


torch.manual_seed(0)

n_observations = 18
x = torch.linspace(0.0, 1.0, n_observations, dtype=torch.float64)
y = torch.sin(2 * torch.pi * x) + 0.08 * torch.randn_like(x)

with Model() as model:
    inputs = Data("inputs", x)
    log_lengthscale = RandomParameter("log_lengthscale", Normal(-1.0, 0.7))
    log_variance = RandomParameter("log_variance", Normal(0.0, 0.7))
    log_noise_variance = RandomParameter("log_noise_variance", Normal(-2.5, 0.7))
    lengthscale = Exp("lengthscale", log_lengthscale)
    variance = Exp("variance", log_variance)
    noise_variance = Exp("noise_variance", log_noise_variance)
    gp = ExactGP("f", inputs, RBF(lengthscale, variance), noise_variance=noise_variance)
    with plate("observations", inputs):
        ObservedParameter("y", gp, y)

# %%
# Fitting the guide and the reference
# -----------------------------------

guide = Guide(model, default_family="full_rank")
history = BBVI(model, guide, n_samples=4, epochs=800, lr=0.03, lrd=0.998, callback_frequency=5, verbose=False)
variational = guide.sample(1200)
print(f"final ELBO estimate: {sum(history[-20:]) / 20:.2f}")

nuts = model.sample(n_samples=300, warmup_length=300, n_chains=4, progress_bar=False, seed=0, delta=0.9)
summary = nuts.summary()
print(summary)
print(f"NUTS: {sum(nuts.divergences)} divergences, largest R-hat {summary['r_hat'].max():.3f}")

# %%
# Hyperparameters and predictions
# -------------------------------
# For every hyperparameter draw, ``exact_gp_predictive`` samples the latent
# function on a grid from its Gaussian conditional given the data, so the
# band combines the hyperparameter uncertainty and the GP's own uncertainty.
# It widens outside the data range ``[0, 1]``.

traces = {"full-rank guide": variational, "NUTS": nuts}
predictive_traces = {label: trace.thin(4) for label, trace in traces.items()}  # 300 draws each
colors = {"full-rank guide": "C0", "NUTS": "black"}
x_new = torch.linspace(-0.2, 1.2, 120, dtype=torch.float64)

fig = plt.figure(figsize=(13, 8), layout="constrained")
top = fig.subfigures(2, 1)
upper_axes = top[0].subplots(1, 4)
upper_axes[0].plot(range(5, 5 * len(history) + 1, 5), history, color="C0")
upper_axes[0].set_ylim(history[-1] - 30, history[-1] + 8)
upper_axes[0].set_xlabel("epoch")
upper_axes[0].set_ylabel("ELBO estimate")
upper_axes[0].set_title("Optimisation")
for ax, name in zip(upper_axes[1:], ("lengthscale", "variance", "noise_variance")):
    for label, trace in traces.items():
        draws = trace[name].flatten().log10()
        ax.hist(draws, bins=40, density=True, histtype="step", lw=1.8, color=colors[label], label=label)
    ax.set_xlabel(f"log10 {name}")
    ax.set_title(name.replace("_", " "))
upper_axes[1].legend(fontsize="small")

ax = top[1].subplots()
for label, trace in predictive_traces.items():
    draws = model.posterior_predicative(trace, inputs=x_new, samples_per_step=4, progress_bar=False)["y"].reshape(-1, x_new.numel())
    lower, upper = draws.quantile(torch.tensor([0.025, 0.975], dtype=torch.float64), dim=0)
    ax.plot(x_new, draws.mean(0), color=colors[label], label=f"{label} mean")
    ax.fill_between(x_new, lower, upper, color=colors[label], alpha=0.15, label=f"{label} 95% band")
ax.plot(x_new, torch.sin(2 * torch.pi * x_new), color="C2", linestyle=":", label="true function")
ax.scatter(x, y, color="C3", zorder=3, s=20, label="data")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Posterior predictive of the latent function")
ax.legend(fontsize="small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.show()
