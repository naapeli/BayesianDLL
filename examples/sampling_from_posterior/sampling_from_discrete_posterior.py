"""
Posterior of a Discrete Parameter
=================================

Eight counts are drawn from ``Binomial(n, 0.5)`` with an unknown number of
trials ``n``. With a ``DiscreteUniform(0, 15)`` prior, ``n`` is a discrete
parameter with a finite support, and the sampler updates it with exact Gibbs
steps: each step evaluates the posterior at all sixteen values and draws from
it.

Values of ``n`` below the largest observed count are impossible, and larger
values become less likely because they predict larger counts than observed.
Because the support is small, the exact posterior is easy to compute by
enumeration. The figure compares the sampled posterior with the exact one
and with the prior, and the observed counts with the posterior predictive
distribution.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import Binomial, DiscreteUniform


torch.manual_seed(2)

n_true, p = 6, 0.5
n_obs, n_max = 8, 15
counts = torch.distributions.Binomial(n_true, torch.tensor(p)).sample((n_obs,)).double()
print("observed counts:", counts.long().tolist())

with Model() as model:
    n = RandomParameter("n", DiscreteUniform(0, n_max))
    with plate("observations", n_obs):
        ObservedParameter("counts", Binomial(n, p), counts)

result = model.sample(2500, 500, n_chains=4, seed=0, chain_method="vectorized", progress_bar=False)
print("sampler:", result.sample_stats.samplers)

support = torch.arange(0, n_max + 1)
log_likelihood = torch.stack([
    torch.distributions.Binomial(int(value), torch.tensor(p)).log_prob(counts).sum() if value >= counts.max()
    else torch.tensor(-torch.inf, dtype=torch.float64) for value in support
])
exact = torch.softmax(log_likelihood, 0)
draws = result["n"].reshape(-1).long()
sampled = torch.bincount(draws, minlength=n_max + 1).double() / draws.numel()
print("value  sampled  exact")
for value in support[exact > 1e-3]:
    print(f"{int(value):5d}  {sampled[value]:.3f}    {exact[value]:.3f}")

values = torch.arange(0, n_max + 1, dtype=torch.float64)
predictive = sum(exact[m] * torch.distributions.Binomial(int(m), torch.tensor(p)).log_prob(values.clamp(max=m)).exp()
                 * (values <= m) for m in support if exact[m] > 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
ax = axes[0]
ax.bar(support - 0.2, torch.full((n_max + 1,), 1 / (n_max + 1)), width=0.4, color="lightgray", label="prior")
ax.bar(support + 0.2, sampled, width=0.4, color="C0", label="Gibbs draws")
ax.plot(support + 0.2, exact, "ko", markersize=4, label="exact posterior")
ax.axvline(n_true, color="C3", linestyle="--", label="true n")
ax.set_xlabel("number of trials n")
ax.set_ylabel("probability")
ax.set_title("Posterior of the number of trials")
ax.legend(frameon=False, fontsize="small")

ax = axes[1]
observed = torch.bincount(counts.long(), minlength=n_max + 1).double() / n_obs
ax.bar(values, observed, color="lightgray", label="observed counts")
ax.plot(values, predictive, "o-", color="C0", label="posterior predictive")
ax.set_xlim(-0.5, n_max + 0.5)
ax.set_xlabel("count")
ax.set_ylabel("relative frequency")
ax.set_title("Data and posterior predictive")
ax.legend(frameon=False, fontsize="small")
plt.show()
