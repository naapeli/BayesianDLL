import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfCauchy, Independent
from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, sample, sample_posterior_predicative
from BayesianDLL.Evaluation import Graphics


torch.manual_seed(7)

# Generate synthetic data
N = 500
true_intercept = 1.0
true_slope = 2.5
true_variance = 0.5
x = torch.linspace(0, 1, N).double().unsqueeze(1)
y = true_intercept + true_slope * x + torch.normal(0, true_variance ** 0.5, size=(N, 1))

with Model() as linear_model:
    # Priors
    prior_intercept = RandomParameter("intercept", Normal(0, 20), torch.tensor([0], dtype=torch.float64), sampler="auto", delta=0.4)
    prior_slope = RandomParameter("slope", Normal(0, 20), torch.tensor([0], dtype=torch.float64), sampler="auto", delta=0.4)
    prior_sigma = RandomParameter("sigma", HalfCauchy(10), torch.tensor([1], dtype=torch.float64), sampler="auto")

    # make the transform for the predicted line
    mu = DeterministicParameter("mu", lambda b, m: m * x + b, lambda b, m: {"slope": x, "intercept": torch.ones_like(x)}, [prior_intercept, prior_slope])
    
    likelihood = ObservedParameter("likelihood", Independent(Normal(mu, prior_sigma), dims=0), y)  # TODO: cannot use Independent as we do not want to sum over the samples during posterior predicative sampling
    
    samples = sample(1000, 1000, n_chains=2)
    posterior_predicative_samples = sample_posterior_predicative(n_samples=20, warmup_length=100, samples_per_step=500, warmup_per_sample=500)
    # print(posterior_predicative_samples, posterior_predicative_samples["likelihood"].shape)
    # plt.plot(posterior_predicative_samples["likelihood"][:, :, 0].T)
    # plt.show()


plt.figure()
Graphics.plot_model(linear_model)

plt.figure()
Graphics.plot_posterior(samples)

plt.figure()
Graphics.plot_predicative_distribution(posterior_predicative_samples, y, method="kde")


x = x.squeeze()
y_preds = samples["slope"][0, :, None] * x[None, :] + samples["intercept"][0, :, None]
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
