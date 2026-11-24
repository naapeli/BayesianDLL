r"""
Regularly Sampled Time Series with a Toeplitz GP
================================================

A sensor reports a value every time step: equally spaced, noisy readings of a
slowly drifting signal. With a *stationary* kernel the covariance between two
readings depends only on their time separation, making the covariance matrix
symmetric Toeplitz.

``ToeplitzGP`` computes the exact marginal likelihood and solve via Levinson
recursion in :math:`\mathcal{O}(N^2)` time and :math:`\mathcal{O}(N)` memory,
compared to :math:`\mathcal{O}(N^3)` and :math:`\mathcal{O}(N^2)` for dense
Cholesky factorisation. The recursion is sequential (one Python-level step per
observation), so a multi-threaded dense factorisation stays competitive up to one
or two thousand points; beyond that the recursion wins, and its memory use stays
linear. With thousands of points a gradient still takes a noticeable fraction of
a second, so this example computes a type-II MAP estimate of the hyperparameters
instead of running MCMC.
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import ExactGP, Matern52, ToeplitzGP


torch.manual_seed(2)


def true_signal(t):
    return torch.sin(t / 8.0) + 0.5 * torch.sin(t / 3.0 + 1.0)


n_total = 4800
n_train = 1200
t_all = torch.arange(n_total, dtype=torch.float64) * 0.05
y_all = true_signal(t_all) + 0.3 * torch.randn(n_total, dtype=torch.float64)
t, y = t_all[:n_train], y_all[:n_train]


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
map_result = model.find_MAP(lr=0.1, epochs=35, verbose=False)
elapsed = time.perf_counter() - start
fitted = {name: model.deterministic_params[name].constrained_value.reshape(1, 1, -1)
          for name in ("lengthscale", "variance", "noise")}
print(f"type-II MAP estimate in {elapsed:.1f} s: lengthscale {fitted['lengthscale'].item():.2f}, "
      f"variance {fitted['variance'].item():.2f}, noise sd {fitted['noise'].item() ** 0.5:.3f} (true 0.3)")

# %%
# Predictive distribution on a window and a forecast
# --------------------------------------------------
# ``model.predict`` conditions on the training readings with Levinson solves
# (one right-hand side per new input, so no ``n x n`` matrix is formed) and
# draws the function from its posterior at the new inputs. The predictive
# covariance over the new inputs is dense, so predict on a window. The band
# widens beyond the last reading.

new_t = torch.linspace(45.0, 75.0, 60, dtype=torch.float64)
model.data["t"].set_value(new_t)
draws = model.predict(map_result.mode(), samples_per_step=300)["f"].squeeze(0)
lower, median, upper = torch.quantile(draws, torch.tensor([0.05, 0.5, 0.95], dtype=draws.dtype), dim=0)

# %%
# Timing against ExactGP
# ----------------------
# One log density and gradient evaluation per size, after a warm-up call (best
# of two runs). Where the two curves cross depends on the machine: a dense
# Cholesky factorisation uses all cores, the recursion is one sequential loop.


def time_evaluation(cls, size):
    timing_model, _ = build(cls, t_all[:size], y_all[:size])
    timing_model.joint_grad_log_prob()
    best = float("inf")
    for _ in range(2):
        start = time.perf_counter()
        timing_model.model_log_prob()
        timing_model.joint_grad_log_prob()
        best = min(best, time.perf_counter() - start)
    return best


sizes = [600, 1200, 2400, 4800]
timings = {cls.__name__: [time_evaluation(cls, s) for s in sizes] for cls in (ExactGP, ToeplitzGP)}
for name, values in timings.items():
    print(name, ", ".join(f"n={s}: {1000 * v:.0f} ms" for s, v in zip(sizes, values)))

fig, (ax, ax_time) = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained", gridspec_kw={"width_ratios": [2, 1]})
window = t >= 45.0
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
