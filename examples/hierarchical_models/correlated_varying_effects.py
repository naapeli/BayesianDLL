"""
Correlated Varying Intercepts and Slopes
========================================

Every group ``j`` has its own intercept and slope, and the two are correlated
across groups:

.. math::

    (\\alpha_j, \\beta_j) \\sim N(\\mu, \\Sigma), \\quad
    \\Sigma = \\operatorname{diag}(\\sigma) \\, R \\, \\operatorname{diag}(\\sigma), \\quad
    y_i \\sim N(\\alpha_{g[i]} + \\beta_{g[i]} x_i, \\sigma_y).

The correlation matrix gets an LKJ prior on its Cholesky factor
``L_R ~ LKJCholesky(2, eta)`` and ``CholeskyCovariance`` turns it into the
Cholesky factor ``L = diag(sigma) L_R`` of ``Sigma``. The group effects are
non-centered: standard normal ``Z`` (shape ``2 x J``) is mapped to
``mu + L @ Z`` with ``MatMul``. (The centered alternative is a
``RandomParameter`` with ``MultivariateNormal(mu, scale_tril=L)``.)
``Index`` selects the effects of every observation's group.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Add, CholeskyCovariance, Index, MatMul, Multiply, Sum
from BayesianDLL.Distributions import HalfNormal, LKJCholesky, Normal


torch.manual_seed(11)

n_groups, per_group = 12, 12
true_mu = torch.tensor([1.0, 0.5], dtype=torch.float64)
true_sigma = torch.tensor([1.0, 0.7], dtype=torch.float64)
true_rho = -0.7
true_R = torch.tensor([[1.0, true_rho], [true_rho, 1.0]], dtype=torch.float64)
true_cov = torch.diag(true_sigma) @ true_R @ torch.diag(true_sigma)
true_effects = true_mu + torch.randn(n_groups, 2, dtype=torch.float64) @ torch.linalg.cholesky(true_cov).T

group = torch.arange(n_groups).repeat_interleave(per_group)
x = torch.randn(n_groups * per_group, dtype=torch.float64)
y = true_effects[group, 0] + true_effects[group, 1] * x + 0.4 * torch.randn(len(x), dtype=torch.float64)

with Model() as model:
    group_data = Data("group", group)
    design = Data("design", torch.stack([torch.ones_like(x), x]))  # rows: intercept, slope

    mu = RandomParameter("mu", Normal(0.0, 2.0), shape=(2, 1))
    sigma = RandomParameter("sigma", HalfNormal(1.5), shape=(2,))
    L_corr = RandomParameter("L_corr", LKJCholesky(2, 2.0))
    L = CholeskyCovariance("L", L_corr, sigma)

    Z = RandomParameter("Z", Normal(0.0, 1.0), shape=(2, n_groups))
    effects = Add("effects", mu, MatMul("L_Z", L, Z))  # (2, J): row 0 intercepts, row 1 slopes
    sigma_y = RandomParameter("sigma_y", HalfNormal(1.0))

    with plate("observations", len(x)):
        effects_obs = Index("effects_obs", effects, group_data, dim=1)  # (2, N)
        mean = Sum("mean", Multiply("terms", effects_obs, design), dim=0)
        ObservedParameter("y", Normal(mean, sigma_y), y)

trace = model.sample(n_samples=300, warmup_length=300, n_chains=2, progress_bar=False, seed=0)
print(trace.summary())
print("divergences:", sum(trace.divergences))

# The correlation is R[1, 0] = L_corr[1, 0] * L_corr[0, 0] = L_corr[1, 0].
rho = trace["L_corr"][..., 1, 0].flatten()
print(f"posterior mean of rho: {rho.mean().item():.2f} (true value {true_rho})")

effects_mean = trace["effects"].flatten(0, 1).mean(0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
axes[0].hist(rho, bins=40, density=True, alpha=0.7)
axes[0].axvline(true_rho, color="black", linestyle="--", label="true value")
axes[0].set_xlabel(r"correlation $\rho$ of intercepts and slopes")
axes[0].set_ylabel("posterior density")
axes[0].legend()

axes[1].scatter(true_effects[:, 0], true_effects[:, 1], color="black", label="true")
axes[1].scatter(effects_mean[0], effects_mean[1], color="C1", label="posterior mean")
for j in range(n_groups):
    axes[1].plot([true_effects[j, 0], effects_mean[0, j]], [true_effects[j, 1], effects_mean[1, j]], color="grey", lw=0.8)
axes[1].set_xlabel("intercept")
axes[1].set_ylabel("slope")
axes[1].set_title("Group effects")
axes[1].legend()
plt.show()
