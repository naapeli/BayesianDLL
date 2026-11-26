r"""
Smooth Trend with Missing Observations
======================================

:class:`~BayesianDLL.TimeSeries.LinearGaussianStateSpace` is the distribution of
an observed series :math:`y_t` driven by hidden states,

.. math::

    y_t = Z \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, H), \qquad
    \alpha_{t+1} = T \alpha_t + R \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q).

Its log density is the exact Kalman-filter likelihood with the states
integrated out, so the only parameters NUTS sees are the system matrices. Here
the states are a *level* and a *slope*, :math:`\alpha_t = (\mu_t, \beta_t)`,

.. math::

    \mu_{t+1} = \mu_t + \beta_t, \qquad \beta_{t+1} = \beta_t + \eta_t,

with :math:`T = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}`, :math:`Z = (1, 0)`
and :math:`R = (0, 1)^\top`, so the scalar disturbance :math:`\eta_t \sim
\mathcal{N}(0, \sigma_\beta^2)` enters the slope only (the *smooth trend*
special case of the local linear trend). :math:`H = \sigma_y^2`, and
:math:`\sigma_\beta` and :math:`\sigma_y` are the only unknowns. Missing
observations (``nan``) are integrated out exactly: the filter simply predicts
through them.

After sampling, ``filter`` returns the one-step predictions and the
innovations of the Kalman filter for every posterior draw, from which the
filtered level follows, and ``forecast`` simulates the continuation of the
series exactly. The figure shows both, with the true level.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Deterministic import Multiply
from BayesianDLL.Distributions import HalfNormal
from BayesianDLL.Evaluation import summary
from BayesianDLL.TimeSeries import LinearGaussianStateSpace

torch.manual_seed(3)
dtype = torch.float64

# %%
# Simulated series with a gap
# ---------------------------
# The states are simulated step by step so that the true level is available
# for comparison. The series starts at level 0 with slope 0.1; a gap of twelve
# steps and a few isolated values are removed.
T_obs = 80
sd_true = dict(slope=0.04, obs=0.6)
transition = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=dtype)
design = torch.tensor([[1.0, 0.0]], dtype=dtype)
selection = torch.tensor([[0.0], [1.0]], dtype=dtype)

state = torch.tensor([0.0, 0.1], dtype=dtype)
levels, y = [], []
for _ in range(T_obs):
    levels.append(state[0])
    y.append(state[0] + sd_true["obs"] * torch.randn((), dtype=dtype))
    state = transition @ state + selection[:, 0] * sd_true["slope"] * torch.randn((), dtype=dtype)
level_true, y = torch.stack(levels), torch.stack(y)

missing = torch.zeros(T_obs, dtype=torch.bool)
missing[40:52] = True
missing[torch.randperm(T_obs)[:6]] = True
y_obs = torch.where(missing, torch.full_like(y, float("nan")), y)
observed = ~missing
print(f"{int(missing.sum())} of {T_obs} observations are missing")

# %%
# The model and its posterior
# ---------------------------
# The initial state has a wide prior (``init_cov``, mean zero), since the series
# is not stationary. ``Multiply`` squares the standard deviations.
eye = torch.eye(2, dtype=dtype)
with Model() as model:
    slope_sd = RandomParameter("slope_sd", HalfNormal(0.5), shape=(1, 1))
    obs_sd = RandomParameter("obs_sd", HalfNormal(2.0), shape=(1, 1))
    Q = Multiply("Q", slope_sd, slope_sd)
    H = Multiply("H", obs_sd, obs_sd)
    ObservedParameter(
        "y",
        LinearGaussianStateSpace(transition, design, Q, obs_cov=H, selection=selection, init_cov=25.0 * eye),
        y_obs.unsqueeze(-1),
    )

result = model.sample(150, 200, n_chains=4, seed=1, chain_method="vectorized", progress_bar=False)
print(summary({"slope_sd": result["slope_sd"], "obs_sd": result["obs_sd"]}).to_string())
print(f"true values: slope sd {sd_true['slope']}, observation sd {sd_true['obs']}")

# %%
# Filtered level and forecast
# ---------------------------
# The filter is run for 200 posterior draws at once by giving the system
# matrices a leading batch dimension. The filtered level is
# :math:`a_{t|t} = a_t + P_t Z^\top F_t^{-1} v_t` with the predicted state
# :math:`a_t`, innovation :math:`v_t` and its variance :math:`F_t`, and the
# filtered variance subtracts :math:`P_t Z^\top F_t^{-1} Z P_t` at observed times.
# The filter only uses the observations up to time :math:`t`, so in the gap it
# extrapolates the last level and slope, and its band widens until the next
# observation arrives.
slope_sd = result["slope_sd"].reshape(-1)
obs_sd = result["obs_sd"].reshape(-1)
keep = torch.linspace(0, len(obs_sd) - 1, 200).long()
slope_sd, obs_sd = slope_sd[keep], obs_sd[keep]
posterior = LinearGaussianStateSpace(
    transition, design, (slope_sd ** 2).reshape(-1, 1, 1),
    obs_cov=(obs_sd ** 2).reshape(-1, 1, 1), selection=selection, init_cov=25.0 * eye,
)

out = posterior.filter(y_obs.unsqueeze(-1))
a, P = out["predicted_mean"][:, :-1], out["predicted_cov"][:, :-1]  # (draws, N, 2), (draws, N, 2, 2)
v, F = out["innovations"][..., 0], out["innovation_cov"][..., 0, 0]
gain = P[..., 0, 0] / F
filtered_level = a[..., 0] + torch.where(observed, gain * v, torch.zeros_like(v))
filtered_var = P[..., 0, 0] - torch.where(observed, P[..., 0, 0] ** 2 / F, torch.zeros_like(F))
mean_level = filtered_level.mean(0)
sd_level = (filtered_var.mean(0) + filtered_level.var(0)).sqrt()  # law of total variance
err = mean_level - level_true
inside = ((level_true - mean_level).abs() < 1.645 * sd_level).double().mean()
print(f"rmse of the filtered level at the observed times: {err[observed].pow(2).mean().sqrt():.3f} "
      f"(raw observations: {(y_obs - level_true)[observed].pow(2).mean().sqrt():.3f}); "
      f"inside the gap: {err[40:52].pow(2).mean().sqrt():.3f}")
print(f"filtered sd of the level: {sd_level[observed].mean():.2f} where observed, "
      f"{sd_level[40:52].max():.2f} at most inside the gap; true level inside the 90% band at {100 * inside:.0f}% of the times")

steps = 15
future = posterior.forecast(y_obs.unsqueeze(-1), steps, n_samples=1)[0, ..., 0]  # (draws, steps)
q05, q50, q95 = future.quantile(torch.tensor([0.05, 0.5, 0.95], dtype=dtype), dim=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained", gridspec_kw={"width_ratios": [2, 1]})
ax = axes[0]
t = torch.arange(T_obs)
ax.axvspan(40, 51, color="gray", alpha=0.15, linewidth=0, label="gap")
ax.scatter(t[observed], y_obs[observed], s=8, color="gray", label="observations")
ax.fill_between(t, mean_level - 1.645 * sd_level, mean_level + 1.645 * sd_level, color="C0", alpha=0.3, linewidth=0)
ax.plot(t, mean_level, color="C0", label="filtered level (90% band)")
ax.plot(t, level_true, "k--", label="true level")
ax.set_xlabel("t")
ax.set_title("Filtered level")
ax.legend(frameon=False, fontsize="small", loc="upper left")

ax = axes[1]
tf = torch.arange(T_obs, T_obs + steps)
ax.scatter(t[-30:], y_obs[-30:], s=8, color="gray", label="observations")
ax.fill_between(tf, q05, q95, color="C1", alpha=0.3, linewidth=0)
ax.plot(tf, q50, color="C1", label="forecast median (90% band)")
ax.set_xlabel("t")
ax.set_title("Forecast of the observations")
ax.legend(frameon=False, fontsize="small", loc="upper left")
plt.show()
