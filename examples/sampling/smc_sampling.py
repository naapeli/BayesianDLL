"""
Sequential Monte Carlo
======================

Sequential Monte Carlo (SMC) moves a population of particles from the prior to
the posterior through the tempered posteriors ``p(theta) p(y | theta)^beta``.
The inverse temperature ``beta`` grows adaptively from 0 to 1: every stage
reweights the particles, resamples them and moves them with a few MCMC steps.

Because the particles start from the prior, SMC explores every mode the prior
covers, which a single MCMC chain cannot do for a multimodal posterior. The
weights also give an estimate of the marginal likelihood ``p(y)``, which is
used here to compute a Bayes factor between two models.
"""

import math

import torch
import matplotlib.pyplot as plt

from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, sample, sample_smc
from BayesianDLL.Distributions import Normal, StudentT, Gamma


torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


# %%
# A bimodal posterior
# -------------------
# The observations only depend on ``|theta|``, so the posterior has two
# symmetric modes at about ``±2.5``.
y = 2.5 + torch.randn(20)
with Model() as bimodal:
    theta = RandomParameter("theta", Normal(0.0, 3.0))
    magnitude = DeterministicParameter("magnitude", lambda t: t.abs(), lambda t: {"theta": torch.sign(t)}, [theta])
    ObservedParameter("y", Normal(magnitude, 1.0), y)

smc_result = sample_smc(bimodal, n_particles=1000, kernel="metropolis", seed=0, progress_bar=False)
nuts_result = sample(1000, 300, n_chains=1, model=bimodal, progress_bar=False, check_convergence=False, seed=0)

print("inverse temperatures:", [round(beta, 4) for beta in smc_result.beta_schedule])
print(f"fraction of positive draws: SMC {(smc_result['theta'] > 0).double().mean():.2f}, "
      f"NUTS {(nuts_result['theta'] > 0).double().mean():.2f}")


# %%
# Bayes factor via the log evidence
# ---------------------------------
# Data with a few outliers is described by a Normal and by a Student-t model.
# The ratio of their marginal likelihoods, the Bayes factor, measures how much
# more the data supports one model than the other.
data = torch.cat([torch.randn(40), torch.tensor([6.0, -5.5, 7.0])])


def location_scale_model(likelihood):
    with Model() as model:
        mu = RandomParameter("mu", Normal(0.0, 5.0))
        sigma = RandomParameter("sigma", Gamma(2.0, 1.0))
        ObservedParameter("data", likelihood(mu, sigma), data)
    return model


normal_model = location_scale_model(lambda mu, sigma: Normal(mu, sigma))
student_model = location_scale_model(lambda mu, sigma: StudentT(3.0, mu, sigma))
normal_fit = sample_smc(normal_model, n_particles=1000, seed=1, progress_bar=False)
student_fit = sample_smc(student_model, n_particles=1000, seed=2, progress_bar=False)

log_bayes_factor = student_fit.log_marginal_likelihood - normal_fit.log_marginal_likelihood
print(f"log p(y | Normal) = {normal_fit.log_marginal_likelihood:.2f}")
print(f"log p(y | Student-t) = {student_fit.log_marginal_likelihood:.2f}")
print(f"Bayes factor Student-t : Normal = {math.exp(log_bayes_factor):.3g}")


fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
bins = torch.linspace(-4, 4, 80).numpy()
axes[0].hist(smc_result["theta"].flatten().numpy(), bins=bins, density=True, alpha=0.6, label="SMC")
axes[0].hist(nuts_result["theta"].flatten().numpy(), bins=bins, density=True, alpha=0.6, label="NUTS (1 chain)")
axes[0].set_title("Bimodal posterior of theta")
axes[0].legend()

axes[1].plot(smc_result.beta_schedule, marker="o")
axes[1].set_xlabel("stage")
axes[1].set_ylabel("inverse temperature")
axes[1].set_title("Adaptive tempering")

axes[2].hist(normal_fit["mu"].flatten().numpy(), bins=40, density=True, alpha=0.6, label="Normal")
axes[2].hist(student_fit["mu"].flatten().numpy(), bins=40, density=True, alpha=0.6, label="Student-t")
axes[2].set_title(f"Posterior of mu (log BF = {log_bayes_factor:.1f})")
axes[2].legend()
plt.tight_layout()
plt.show()
