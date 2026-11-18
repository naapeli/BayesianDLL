"""
Writing a Variational Guide by Hand
===================================

The target is a one-dimensional, two-component Gaussian mixture
``0.3 N(-2, 0.5) + 0.7 N(2, 0.8)``, and the guide is a single Normal
``q(x) = N(m, s)``. The guide is written by hand in a ``Guide``
block: a ``RandomParameter`` with the same name as the model's latent,
whose distribution has the ``VariationalParameter`` objects ``m`` and ``s``.
``s`` is created with ``log_scale=True``, so the optimiser updates
``log(s)``: the standard deviation stays positive and can change by large
factors without overshooting.

``BBVI`` maximises the evidence lower bound, which is the same as minimising
``KL(q || p)``. This divergence penalises guide mass where the target has
little density much more than target mass the guide misses, so a single
Normal settles on **one** mode instead of spreading over both. Which mode it
finds depends on where it starts. The guide is therefore fitted twice, from
starting means of -3 and 4.

The figure shows both fitted guides over the exact target density and a
histogram of exact draws from the mixture, together with the ELBO of both
runs. The final ELBO tells which local optimum is better: it equals
``-KL(q || p)`` here, because the target is normalised. When a guide matches
one well-separated component exactly, this KL is ``-log(weight)`` of that
component: ``-log 0.3 = 1.20`` and ``-log 0.7 = 0.36``.
"""

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Guide, Model, RandomParameter, VariationalParameter
from BayesianDLL.Distributions import Mixture, Normal
from BayesianDLL.Variational import BBVI


torch.manual_seed(0)

target = Mixture([Normal(-2.0, 0.5), Normal(2.0, 0.8)], [0.3, 0.7])
with Model() as model:
    RandomParameter("x", target)


def normal_guide(initial_mean):
    with Guide() as guide:
        m = VariationalParameter("m", torch.tensor([initial_mean], dtype=torch.float64))
        s = VariationalParameter("s", torch.tensor([1.0], dtype=torch.float64), log_scale=True)
        RandomParameter("x", Normal(m, s))
    return guide, m, s


fits = {}
for start in (-3.0, 4.0):
    guide, m, s = normal_guide(start)
    history = BBVI(model, guide, n_samples=8, epochs=1500, lr=0.02, lrd=0.999, callback_frequency=5, verbose=False)
    final_elbo = sum(history[-40:]) / 40
    fits[start] = (m.value.item(), s.value.item(), history, final_elbo)
    print(f"start {start:+.0f}: m = {m.value.item():+.3f}, s = {s.value.item():.3f}, ELBO = -KL(q||p) = {final_elbo:.3f}")

# %%
# Fitted guides against the target
# --------------------------------
# The guide started at -3 locks onto the smaller, narrower component and the
# one started at 4 onto the larger component, whose final ELBO is higher. Neither
# covers both modes.

exact_draws = target.sample(20000).flatten()
grid = torch.linspace(-4.0, 6.0, 400, dtype=torch.float64)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
axes[0].hist(exact_draws, bins=100, density=True, color="grey", alpha=0.35, label="exact draws")
axes[0].plot(grid, target.pdf(grid).flatten(), color="black", lw=2, label="target density")
for (start, (m, s, history, final_elbo)), color in zip(fits.items(), ("C0", "C1")):
    axes[0].plot(grid, Normal(m, s).pdf(grid).flatten(), color=color, lw=2, linestyle="--",
                 label=f"guide from {start:+.0f}")
    axes[1].plot(range(5, 5 * len(history) + 1, 5), history, color=color, alpha=0.8,
                 label=f"from {start:+.0f} (final {final_elbo:.2f})")
axes[0].set_xlabel("x")
axes[0].set_ylabel("density")
axes[0].set_title("Single-Normal guides fitted to a mixture")
axes[0].legend(fontsize="small")
axes[1].set_ylim(-3, 0.2)
axes[1].set_xlabel("epoch")
axes[1].set_ylabel(r"ELBO $= -\mathrm{KL}(q\,\|\,p)$")
axes[1].set_title("Optimisation")
axes[1].legend(fontsize="small", loc="lower right")
plt.show()
