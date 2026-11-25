"""
Low-Rank Plus Diagonal Guides
=============================

Mean-field VI ignores posterior correlations; full-rank VI captures all of
them with :math:`d(d+1)/2` covariance parameters and :math:`O(d^2)` work per
draw. ``Guide(model, default_family="low_rank", rank=k)`` is in between:
:math:`q = N(\\mu, D^2 + V V^\\top)` with a diagonal :math:`D` and a
:math:`d \\times k` factor :math:`V`, i.e. :math:`d(k+1)` covariance
parameters and :math:`O(dk)` work per draw (the entropy uses the matrix
determinant lemma).

It pays off when the posterior correlations are spread over many latents
along a few directions of large posterior variance. Here a regression with
:math:`d = 40` predictors that (nearly) sum to zero determines only the
differences between the coefficients: their common level is weakly identified,
which gives the covariance one large direction, :math:`\\mathbf{1} \\mathbf{1}^\\top`,
involving every latent. The posterior is Gaussian, so it is known exactly and the
approximations can be scored against it by their KL divergence
:math:`\\mathrm{KL}(q \\,\\|\\, p)`.

When the correlations are confined to a few known latents, a small full-rank
:class:`~BayesianDLL.Group` for them is the better tool; and a low-rank guide
does not cure every failure of mean-field VI (e.g. the pruning of hidden units
in a deep Bayesian neural network is a property of the ELBO's optimum, not of
the correlation structure).
"""

import time

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Guide, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import Normal
from BayesianDLL.Variational import BBVI

torch.manual_seed(0)

n, d, noise = 100, 40, 0.5
raw = torch.randn(n, d, dtype=torch.float64)
design = raw - 0.97 * raw.mean(1, keepdim=True)  # the predictors nearly sum to zero
true_beta = torch.randn(d, dtype=torch.float64) * 0.5
y = design @ true_beta + noise * torch.randn(n, dtype=torch.float64)

with Model() as model:
    X = Data("X", design)
    beta = RandomParameter("beta", Normal(0.0, 3.0), shape=(d,), initial_value=torch.zeros(d, dtype=torch.float64))
    mu = Linear("mu", X, coefficients=beta)
    with plate("obs", n):
        ObservedParameter("y", Normal(mu, noise), y)

# The exact Gaussian posterior.
precision = design.T @ design / noise**2 + torch.eye(d, dtype=torch.float64) / 9.0
exact_cov = torch.linalg.inv(precision)
exact_mean = exact_cov @ design.T @ y / noise**2


def kl_to_exact(mean, cov):
    """KL(q || p) between two Gaussians."""
    diff = mean - exact_mean
    return 0.5 * (torch.trace(precision @ cov) + diff @ precision @ diff - d
                  + torch.logdet(exact_cov) - torch.logdet(cov)).item()


guides = {
    "mean-field": Guide(model, default_family="mean_field"),
    "low-rank, rank 1": Guide(model, default_family="low_rank", rank=1),
    "low-rank, rank 3": Guide(model, default_family="low_rank", rank=3),
    "full-rank": Guide(model, default_family="full_rank"),
}
fitted = {}
print(f"{'guide':18s} {'cov. parameters':>15s} {'KL(q || exact)':>15s} {'time':>7s}")
for label, guide in guides.items():
    torch.manual_seed(1)
    start = time.perf_counter()
    BBVI(model, guide, n_samples=4, epochs=1000, lr=0.03, lrd=0.998, verbose=False)
    elapsed = time.perf_counter() - start
    moments = guide.moments()
    fitted[label] = moments
    group = guide.groups[0]
    n_parameters = {"mean_field": d, "low_rank": d * (group.rank + 1) if group.rank else 0, "full_rank": d * (d + 1) // 2}[group.family]
    print(f"{label:18s} {n_parameters:15d} {kl_to_exact(moments.mean(), moments.matrix):15.2f} {elapsed:6.1f}s")

# %%
# The exact posterior sd of every coefficient against the approximations, and
# the exact correlation of the first coefficient with the others.
exact_sd = exact_cov.diagonal().sqrt()
exact_corr = exact_cov[0] / (exact_sd[0] * exact_sd)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for (label, moments), color in zip(fitted.items(), ("C1", "C2", "C4", "C0")):
    sd = moments.matrix.diagonal().sqrt()
    corr = moments.matrix[0] / (sd[0] * sd)
    axes[0].plot(sd, "o", color=color, ms=4, label=label)
    axes[1].plot(corr, "o", color=color, ms=4, label=label)
axes[0].plot(exact_sd, "k_", ms=10, label="exact")
axes[1].plot(exact_corr, "k_", ms=10, label="exact")
axes[0].set(xlabel="coefficient", ylabel="posterior sd")
axes[1].set(xlabel="coefficient $j$", ylabel=r"corr($\beta_0$, $\beta_j$)")
axes[0].legend(fontsize=8)
plt.show()
