r"""
Extrapolating Quasi-Periodic Signals with a Spectral Mixture Kernel
===================================================================

A stationary kernel is the Fourier transform of a spectral density, and
the spectral mixture kernel (Wilson & Adams, 2013) models that density as a
sum of :math:`Q` Gaussians:

.. math::

    k(\tau) = \sum_{q=1}^{Q} w_q \exp\left(-\frac{\tau^2}{2 l_q^2}\right) \cos(2 \pi \mu_q \tau).

Component :math:`q` is a quasi-periodic oscillation with frequency
:math:`\mu_q` (period :math:`1 / \mu_q`) whose phase coherence decays over the
lengthscale :math:`l_q`. Because the frequencies are learned from the data,
the GP can continue the oscillations beyond the observations. An RBF kernel
only encodes smoothness, so its forecast reverts to the mean.

Here a signal made of two oscillations is observed on ``[0, 10]`` and
forecast on ``[10, 14]``. The hyperparameters are found by type-II MAP,
starting from the strongest peaks of the periodogram (the likelihood is
multimodal in the frequencies, so the starting point matters).
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import RBF, ExactGP, SpectralMixture


torch.manual_seed(0)


def true_function(x):
    return torch.sin(2 * torch.pi * 0.35 * x) + 0.6 * torch.sin(2 * torch.pi * 1.1 * x + 1.0)


n = 120
x = torch.sort(torch.rand(n, dtype=torch.float64) * 10.0).values
y = true_function(x) + 0.2 * torch.randn_like(x)

# %%
# Initial frequencies from the periodogram
# ----------------------------------------
# The periodogram of the irregularly spaced data is computed by least squares
# at a grid of frequencies. Its two largest well-separated peaks initialise
# the component frequencies.

freqs = torch.linspace(0.05, 2.0, 400, dtype=torch.float64)
centred = y - y.mean()
power = torch.empty_like(freqs)
for i, mu in enumerate(freqs):
    design = torch.stack([torch.cos(2 * torch.pi * mu * x), torch.sin(2 * torch.pi * mu * x)], dim=1)
    coefficients = torch.linalg.lstsq(design, centred[:, None]).solution
    power[i] = (design @ coefficients).square().mean()

peaks = [i for i in range(1, len(freqs) - 1) if power[i] > power[i - 1] and power[i] >= power[i + 1]]
peaks = sorted(peaks, key=lambda i: -power[i])
first = peaks[0]
second = next(i for i in peaks if abs(freqs[i] - freqs[first]) > 0.2)
initial_frequencies = freqs[[first, second]]
print("periodogram peaks:", ", ".join(f"{mu:.2f}" for mu in initial_frequencies), "(true 0.35 and 1.10)")

# %%
# Two models
# ----------
# Both models share the noise prior and fit the centred data. The spectral
# mixture has ``Q = 2`` components, each with a weight, a frequency and a
# lengthscale, so six kernel hyperparameters against two for the RBF kernel.

y_mean = y.mean()


def build(kind):
    with Model() as model:
        inputs = Data("x", x)
        log_noise = RandomParameter("log_noise", Normal(-3.0, 1.0), initial_value=torch.tensor([-3.0], dtype=torch.float64))
        if kind == "spectral":
            log_weights = RandomParameter("log_weights", Normal(-0.5, 1.0), shape=2, initial_value=torch.full((2,), -0.5, dtype=torch.float64))
            log_frequencies = RandomParameter("log_frequencies", Normal(0.0, 1.5), shape=2, initial_value=initial_frequencies.log())
            log_lengthscales = RandomParameter("log_lengthscales", Normal(2.0, 1.0), shape=2, initial_value=torch.full((2,), 2.0, dtype=torch.float64))
            kernel = SpectralMixture(Exp("weights", log_weights), Exp("frequencies", log_frequencies), Exp("lengthscales", log_lengthscales))
        else:
            log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0))
            log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
            kernel = RBF(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance))
        f = ExactGP("f", inputs, kernel, noise_variance=Exp("noise", log_noise))
        ObservedParameter("y", f, y - y_mean)
    return model


models = {"spectral": build("spectral"), "rbf": build("rbf")}

# %%
# Type-II MAP and forecast
# ------------------------
# ``model.predict`` conditions on the observations and draws the function at
# the new inputs for the MAP values. The grid covers the data and a forecast
# of four time units.

grid = torch.linspace(0.0, 14.0, 280, dtype=torch.float64)
bands = {}
for name, model in models.items():
    start = time.perf_counter()
    result = model.find_MAP(lr=0.05, epochs=300, verbose=False)
    elapsed = time.perf_counter() - start
    model.data["x"].set_value(grid)
    draws = model.predict(result.mode(), samples_per_step=300)["f"].reshape(-1, grid.numel()) + y_mean
    bands[name] = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)
    model.data["x"].set_value(x)
    print(f"{name}: MAP in {elapsed:.1f} s, noise sd {model.deterministic_params['noise'].constrained_value.item() ** 0.5:.3f} (true 0.2)")

spectral = models["spectral"]
learned = {key: spectral.deterministic_params[key].constrained_value.reshape(-1) for key in ("weights", "frequencies", "lengthscales")}
print("learned frequencies:", ", ".join(f"{v:.3f}" for v in learned["frequencies"]))
print("learned lengthscales:", ", ".join(f"{v:.1f}" for v in learned["lengthscales"]))

forecast = grid > 10.0
truth = true_function(grid)
for name, (lower, median, upper) in bands.items():
    rmse = (median[forecast] - truth[forecast]).square().mean().sqrt()
    print(f"{name}: forecast RMSE {rmse:.3f}")

# %%
# Forecasts
# ---------
# The right panel shows the learned spectral density of the mixture against
# the true frequencies.

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained", gridspec_kw={"width_ratios": [2, 2, 1.3]})
for ax, (name, title, color) in zip(axes[:2], (("spectral", "Spectral mixture, Q = 2", "tab:blue"), ("rbf", "RBF kernel", "tab:red"))):
    lower, median, upper = bands[name]
    ax.fill_between(grid, lower, upper, color=color, alpha=0.25, label="90% band of f")
    ax.plot(grid, median, color=color, label="median of f")
    ax.plot(grid, truth, "k--", linewidth=1, label="true f")
    ax.scatter(x, y, color="black", s=8, zorder=3, label="observations")
    ax.axvspan(10.0, 14.0, color="0.9", zorder=0, label="forecast")
    ax.set(xlabel="x", ylabel="y", title=title, ylim=(-2.6, 2.6))
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize="small")

omega = torch.linspace(0.0, 2.0, 400, dtype=torch.float64)
density = torch.zeros_like(omega)
for w, mu, l in zip(learned["weights"], learned["frequencies"], learned["lengthscales"]):
    sd = 1.0 / (2 * torch.pi * l)
    density += w * l * torch.exp(-0.5 * ((omega - mu) / sd) ** 2)
ax = axes[2]
ax.plot(omega, density / density.max(), color="tab:blue", label="learned density")
for mu in (0.35, 1.1):
    ax.axvline(mu, color="k", linestyle="--", linewidth=1)
ax.plot([], [], "k--", linewidth=1, label="true frequencies")
ax.set(xlabel="frequency", ylabel="relative density", title="Learned spectral density")
ax.legend(loc="upper right", fontsize="small")
plt.show()
