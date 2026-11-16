"""
Slice Sampling
==============

Samples posteriors with the coordinate-wise slice sampler (Neal, 2003), which
needs no gradients and no acceptance-rate tuning: every update steps an interval
out until it covers the slice and shrinks it until a uniform draw lands inside.
During warmup the interval width of every coordinate adapts to the scale of the
posterior.

The example samples a skewed Gamma posterior of an exponential rate, a bounded
Beta distribution (through its logit transform) and a model whose location is
sampled with NUTS and whose scale with the slice sampler.
"""

import torch
import matplotlib.pyplot as plt

from BayesianDLL import Model, RandomParameter, ObservedParameter, sample
from BayesianDLL.Distributions import Gamma, Exponential, Beta, Normal


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


# %%
# A skewed Gamma posterior
# ------------------------
# With a ``Gamma(2, 1)`` prior on the rate of exponential observations, the
# posterior is ``Gamma(2 + n, 1 + sum(y))``. The slice sampler is selected with
# ``sampler="slice"``; the initial width can be set with ``width=...``.
y = torch.tensor([0.5, 1.5, 0.3])
with Model() as gamma_model:
    rate = RandomParameter("rate", Gamma(2.0, 1.0), sampler="slice", width=0.5)
    ObservedParameter("y", Exponential(rate), y)
gamma_result = sample(2000, 500, n_chains=2, model=gamma_model, progress_bar=False, seed=0)

alpha, beta = 2.0 + len(y), 1.0 + y.sum().item()
print(f"posterior mean {gamma_result['rate'].mean():.3f} (exact {alpha / beta:.3f})")
print(f"posterior variance {gamma_result['rate'].var():.3f} (exact {alpha / beta ** 2:.3f})")
stats = gamma_result.sample_stats[0]
print(f"mean width {stats['step_size'].mean():.3f}, log density evaluations per draw {stats['n_evals'].mean():.1f}")


# %%
# A bounded Beta distribution
# ---------------------------
# The slice sampler moves on the unconstrained (logit) scale, so the draws
# always stay inside (0, 1).
with Model() as beta_model:
    RandomParameter("p", Beta(2.0, 5.0), sampler="slice")
beta_result = sample(2000, 500, n_chains=2, model=beta_model, progress_bar=False, seed=1)


# %%
# NUTS and slice in one Gibbs scheme
# ----------------------------------
# Every parameter can choose its sampler; the blocks are updated in turn.
data = 1.0 + 0.5 * torch.randn(50)
with Model() as mixed_model:
    mu = RandomParameter("mu", Normal(0.0, 10.0))
    sigma = RandomParameter("sigma", Gamma(2.0, 2.0), sampler="slice")
    ObservedParameter("data", Normal(mu, sigma), data)
mixed_result = sample(1000, 500, n_chains=2, model=mixed_model, progress_bar=False, seed=2)
print("samplers per block:", mixed_result.sample_stats.samplers)


fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
x = torch.linspace(1e-3, 6, 200).unsqueeze(-1)
axes[0].hist(gamma_result["rate"].flatten().numpy(), bins=50, density=True, alpha=0.6, label="slice")
axes[0].plot(x.squeeze().numpy(), Gamma(alpha, beta).pdf(x).squeeze().numpy(), label="exact")
axes[0].set_title("Gamma posterior of the rate")
axes[0].legend()

x = torch.linspace(1e-3, 1 - 1e-3, 200).unsqueeze(-1)
axes[1].hist(beta_result["p"].flatten().numpy(), bins=50, density=True, alpha=0.6, label="slice")
axes[1].plot(x.squeeze().numpy(), Beta(2.0, 5.0).pdf(x).squeeze().numpy(), label="exact")
axes[1].set_title("Beta(2, 5)")
axes[1].legend()

axes[2].scatter(mixed_result["mu"].flatten().numpy(), mixed_result["sigma"].flatten().numpy(), s=2, alpha=0.3)
axes[2].set_xlabel("mu (NUTS)")
axes[2].set_ylabel("sigma (slice)")
axes[2].set_title("Joint posterior")
plt.tight_layout()
plt.show()
