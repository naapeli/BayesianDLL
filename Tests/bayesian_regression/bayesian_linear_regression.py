import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfCauchy
from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, plate, find_MAP
from BayesianDLL.Evaluation import Graphics


torch.manual_seed(7)

# Generate synthetic data
N = 500
true_intercept = 1.0
true_slope = 2.5
true_variance = 0.5
x = torch.linspace(0, 1, N).double()
y = true_intercept + true_slope * x + torch.normal(0, true_variance ** 0.5, size=(N,))

with Model() as linear_model:
    # Priors
    prior_intercept = RandomParameter("intercept", Normal(0, 20), sampler="auto", delta=0.4)
    prior_slope = RandomParameter("slope", Normal(0, 20), sampler="auto", delta=0.4)
    prior_sigma = RandomParameter("sigma", HalfCauchy(10), sampler="auto", max_depth=4)

    # make the transform for the predicted line
    mu = DeterministicParameter("mu", lambda b, m: m * x + b, lambda b, m: {"slope": x, "intercept": torch.ones_like(x)}, [prior_intercept, prior_slope])
    
    with plate("data", N):
        likelihood = ObservedParameter("likelihood", Normal(mu, prior_sigma), y)
    
    predicative_distribution = linear_model.sample_posterior_predicative(20, 1000, samples_per_step=10, warmup_per_sample=100)
    plt.figure()
    Graphics.plot_predicative_distribution(predicative_distribution, y, kind="pdf")
    plt.show()

    find_MAP(linear_model, verbose=False)
    samples = linear_model.sample(1000, 500, blocks=[["slope", "intercept", "sigma"]], delta=0.7)


plt.figure()
Graphics.plot_model(linear_model)

plt.figure()
Graphics.plot_posterior(samples)


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
