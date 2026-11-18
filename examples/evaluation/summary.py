"""
Posterior Summary Table
=======================

``summary(result)`` (or ``result.summary()``) turns posterior draws into a
table with one row per scalar component: the mean and standard deviation, an
equal-tailed (``ci_kind="eti"``) or highest-density (``"hdi"``) credible
interval, the bulk and tail effective sample sizes, R-hat and the Monte Carlo
standard errors of the mean and of the standard deviation. Deterministic
nodes are added with ``include_deterministic=True``. A plain dict of draws
with shape ``(n_chains, n_draws, ...)`` works as well.

The draws here are simulated from known distributions and wrapped in a
``SamplingResult``, so the table can be checked against the true values and
against ``arviz.summary``. The figure shows the 94% intervals from both
libraries next to the true means.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import torch

from BayesianDLL.Samplers._result import SamplingResult


torch.manual_seed(42)
n_chains, n_draws = 4, 1000


def normal_draws(mean, sd, *shape):
    mean = torch.as_tensor(mean, dtype=torch.float64)
    return mean + sd * torch.randn(n_chains, n_draws, *shape, dtype=torch.float64)


# sigma is |Normal(0.8, 0.2)|, whose mean is 0.8 to three decimals.
true_means = {"intercept": 1.5, "slope": 2.5, "weights[0]": 0.5, "weights[1]": -0.5, "sigma": 0.8, "R2": 0.85}
trace = {
    "intercept": normal_draws(1.5, 0.2),
    "slope": normal_draws(2.5, 0.1),
    "weights": normal_draws([0.5, -0.5], 0.3, 2),
    "sigma": normal_draws(0.8, 0.2).abs(),
}
result = SamplingResult(
    trace=trace,
    divergences=[0] * n_chains,
    acceptance_probabilities=[[0.8]] * n_chains,
    step_sizes=[[0.05]] * n_chains,
    deterministic_trace={"R2": normal_draws(0.85, 0.02)},
)

table = result.summary(hdi_prob=0.94, include_deterministic=True)
print("BayesianDLL summary")
print(table.to_string())

draws = {**result.trace, **result.deterministic_trace}
idata = az.convert_to_dataset({name: value.numpy() for name, value in draws.items()})
reference = az.summary(idata, ci_prob=0.94, ci_kind="eti", round_to="none")
print("\nArviZ summary of the same draws")
print(reference.to_string())

raw = result.summary(hdi_prob=0.94, round_to=None, include_deterministic=True)
columns = {"mean": "mean", "sd": "sd", "eti_3%": "eti94_lb", "eti_97%": "eti94_ub", "ess_bulk": "ess_bulk",
           "ess_tail": "ess_tail", "r_hat": "r_hat", "mcse_mean": "mcse_mean", "mcse_sd": "mcse_sd"}
print("\nlargest absolute difference to ArviZ per column")
for ours, theirs in columns.items():
    difference = np.max(np.abs(raw[ours].to_numpy(float) - reference[theirs].to_numpy(float)))
    print(f"  {ours:10s} {difference:.2e}")

fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
rows = list(raw.index)
positions = np.arange(len(rows))
ax.errorbar(raw["mean"], positions - 0.12, xerr=[raw["mean"] - raw["eti_3%"], raw["eti_97%"] - raw["mean"]],
            fmt="o", capsize=3, label="BayesianDLL")
ax.errorbar(reference["mean"], positions + 0.12,
            xerr=[reference["mean"] - reference["eti94_lb"], reference["eti94_ub"] - reference["mean"]],
            fmt="s", capsize=3, label="ArviZ")
truth = [true_means.get(row, true_means.get(row.split("[")[0], np.nan)) for row in rows]
ax.plot(truth, positions, "k|", markersize=16, label="true mean")
ax.set_yticks(positions, rows)
ax.invert_yaxis()
ax.set_xlabel("value")
ax.set_title("Posterior means with 94% equal-tailed intervals")
ax.legend(frameon=False, fontsize="small", loc="lower right")
plt.show()
