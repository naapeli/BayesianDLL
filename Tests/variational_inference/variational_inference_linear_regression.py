import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfCauchy, Exponential
from BayesianDLL import Model, MeanFieldGuide, RandomParameter, ObservedParameter, DeterministicParameter, VariationalParameter, plate
from BayesianDLL.Variational import BBVI


torch.manual_seed(7)

# Generate synthetic data
N = 500  # 10
true_intercept = 1.0
true_slope = 2.5
true_variance = 0.5
x = torch.linspace(0, 1, N).double()
y = true_intercept + true_slope * x + torch.normal(0, true_variance ** 0.5, size=(N,))

with Model() as linear_model:
    # Priors
    prior_intercept = RandomParameter("intercept", Normal(0, 20))
    prior_slope = RandomParameter("slope", Normal(0, 20))
    prior_sigma = RandomParameter("sigma", HalfCauchy(10))
    # prior_sigma = 0.5

    # make the transform for the predicted line
    mu = DeterministicParameter("mu", lambda b, m: m * x + b, lambda b, m: {"slope": x, "intercept": torch.ones_like(x)}, [prior_intercept, prior_slope])
    
    with plate("data", N):
        likelihood = ObservedParameter("likelihood", Normal(mu, prior_sigma), y)

with MeanFieldGuide() as guide:
    RandomParameter("intercept", Normal(VariationalParameter("mean", torch.full((1,), 0).double()), VariationalParameter("variance", torch.full((1,), 1).double(), min=1e-8)))
    RandomParameter("slope", Normal(VariationalParameter("mean", torch.full((1,), 0).double()), VariationalParameter("variance", torch.full((1,), 1).double(), min=1e-8)))
    RandomParameter("sigma", Exponential(VariationalParameter("scale", torch.full((1,), 1).double(), min=1e-8)))

history = BBVI(linear_model, guide, n_samples=5, epochs=2000, callback_frequency=10, lr=1e-2)
plt.figure()
plt.semilogy([-elbo for elbo in history])

print(guide.params["intercept"].distribution.variational_parameters)
print(guide.params["slope"].distribution.variational_parameters)
print(guide.params["sigma"].distribution.variational_parameters)


samples = guide.sample(n_samples=1000, warmup_length=1000, n_chains=4)  # sample from the fitted guide

y_preds = samples["slope"].flatten(0, 1)[:, None] * x[None, :] + samples["intercept"].flatten(0, 1)[:, None]
y_preds = y_preds.squeeze()
y_mean = y_preds.mean(dim=0)
y_lower = y_preds.quantile(0.025, dim=0)
y_upper = y_preds.quantile(0.975, dim=0)
std = samples["sigma"].mean().sqrt()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="Observed data", alpha=0.6)
plt.plot(x, y_mean, label="Posterior mean", color="black")
plt.fill_between(x, y_lower, y_upper, color="blue", alpha=0.2, label="95% CI for mean")
plt.fill_between(x, y_mean - 1.96 * std, y_mean + 1.96 * std, color="blue", alpha=0.2, label="95% CI for data points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()

plt.show()
