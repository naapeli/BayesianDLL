"""
Regularly Sampled Time Series with a Toeplitz GP
================================================

A sensor reports a value every time step: 2000 equally spaced, noisy
readings of a slowly drifting signal. With a *stationary* kernel the
covariance between two readings depends only on how many steps apart they
are, so ``K + σ² I`` is a symmetric Toeplitz matrix, fully described by its
first column.

``ToeplitzGP`` exploits this with Levinson's recursion: the exact
marginal likelihood and the solve ``(K + σ² I)⁻¹ y`` take ``O(n²)`` time
and ``O(n)`` memory instead of the ``O(n³)`` and ``O(n²)`` of a dense
Cholesky factorisation. The hand-derived gradients reuse the recursion's
solution ``T⁻¹ e_0`` (Gohberg-Semencul formula) and FFT correlations.

The recursion is sequential (one Python-level step per observation), so a
multi-threaded dense Cholesky factorisation stays competitive up to one or
two thousand points; beyond that the recursion wins, and its memory use is
linear throughout. With thousands of points a
gradient still costs a noticeable fraction of a second, so this example
computes a type-II MAP estimate of the hyperparameters instead of running
MCMC: ``find_MAP`` maximises the exact marginal likelihood (``f``
integrated out) times the log-normal priors of the hyperparameters, working
on their log scale.

The kernel is a Matérn-5/2 (twice differentiable sample paths).

Note on Learnable Grids
-----------------------
``ToeplitzGP`` requires fixed, equally spaced 1-D inputs. A learnable uniform
spacing ``Δt`` would be mathematically redundant with the kernel lengthscale
``l`` (since stationary kernels depend only on ``|t_i - t_j| / l = |i - j| Δt / l``),
while non-uniform or independently learnable grid locations destroy the Toeplitz
Toeplitz structure. For models where grid coordinates are learnable parameters,
use :class:`~BayesianDLL.GP.KroneckerGP` (which supports learnable axes on Cartesian
grids) or :class:`~BayesianDLL.GP.ExactGP` / :class:`~BayesianDLL.GP.SparseGP`.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import ExactGP, Matern52, ToeplitzGP, toeplitz_gp_predictive


torch.manual_seed(2)


def true_signal(t):
    return torch.sin(t / 8.0) + 0.5 * torch.sin(t / 3.0 + 1.0)


n = 2000
t = torch.arange(n, dtype=torch.float64) * 0.05  # one reading every 0.05 time units
y = true_signal(t) + 0.3 * torch.randn(n, dtype=torch.float64)


def build(cls, inputs, observations):
    with Model() as model:
        log_lengthscale = RandomParameter("log_lengthscale", Normal(1.0, 1.0))
        log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
        log_noise = RandomParameter("log_noise", Normal(-2.0, 1.0))
        f = cls(
            "f", Data("t", inputs),
            Matern52(Exp("lengthscale", log_lengthscale), Exp("variance", log_variance)),
            noise_variance=Exp("noise", log_noise),
        )
        ObservedParameter("y", f, observations)
    return model, f


model, f = build(ToeplitzGP, t, y)

# %%
# Type-II MAP estimate of the hyperparameters
# -------------------------------------------

start = time.perf_counter()
model.find_MAP(lr=0.1, epochs=60, verbose=False)
elapsed = time.perf_counter() - start
fitted = {name: model.deterministic_params[name].constrained_value.reshape(1, 1, -1)
          for name in ("lengthscale", "variance", "noise")}
print(f"type-II MAP estimate in {elapsed:.1f} s: lengthscale {fitted['lengthscale'].item():.2f}, "
      f"variance {fitted['variance'].item():.2f}, noise sd {fitted['noise'].item() ** 0.5:.3f} (true 0.3)")

# %%
# Predictive distribution on a window and a forecast
# --------------------------------------------------
# ``toeplitz_gp_predictive`` also uses Levinson solves (one right-hand side
# per prediction point), so no ``n x n`` matrix is formed. The band widens
# beyond the last reading.

new_t = torch.linspace(80.0, 110.0, 150, dtype=torch.float64)
draws = toeplitz_gp_predictive(f, fitted, y, new_t, n_samples=300).reshape(-1, new_t.numel())
lower, median, upper = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)

# %%
# Timing against ExactGP
# ----------------------


def time_once(cls, size):
    timing_model, _ = build(cls, t[:size], y[:size])
    timing_model.joint_grad_log_prob()
    best = float("inf")
    for _ in range(1):
        start = time.perf_counter()
        timing_model.model_log_prob()
        timing_model.joint_grad_log_prob()
        best = min(best, time.perf_counter() - start)
    return best


sizes = [250, 500, 1000, 2000]
timings = {cls.__name__: [time_once(cls, s) for s in sizes] for cls in (ExactGP, ToeplitzGP)}
for name, values in timings.items():
    print(name, ", ".join(f"n={s}: {1000 * v:.0f} ms" for s, v in zip(sizes, values)))

fig, (ax, ax_time) = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained", gridspec_kw={"width_ratios": [2, 1]})
window = t >= 80.0
ax.scatter(t[window], y[window], s=6, color="0.6", label="readings")
ax.fill_between(new_t, lower, upper, color="tab:purple", alpha=0.3, label="90% band of f")
ax.plot(new_t, median, color="tab:purple", label="median of f")
ax.plot(new_t, true_signal(new_t), "k--", linewidth=1, label="true signal")
ax.axvline(float(t[-1]), color="0.4", linestyle=":")
ax.set(xlabel="time", ylabel="value", title="Last readings and forecast")
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize="small")
for name, values in timings.items():
    ax_time.loglog(sizes, values, "o-", label=name)
ax_time.set(xlabel="n", ylabel="seconds per log density + gradient", title="Cost per gradient")
ax_time.legend(loc="upper left", fontsize="small")
ax_time.grid(True, which="both", alpha=0.3)
plt.show()
