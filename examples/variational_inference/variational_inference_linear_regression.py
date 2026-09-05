import torch
import matplotlib.pyplot as plt


from BayesianDLL.Distributions import Normal, HalfCauchy, Exponential
from BayesianDLL import Model, MeanFieldGuide, RandomParameter, ObservedParameter, DeterministicParameter, VariationalParameter, plate
from BayesianDLL.Variational import BBVI
from BayesianDLL.Evaluation.Graphics import plot_posterior


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
    prior_variance = RandomParameter("sigma_squared", HalfCauchy(10))
    # prior_variance = 0.5

    # make the transform for the predicted line
    mu = DeterministicParameter("mu", lambda b, m: m * x + b, lambda b, m: {"slope": x, "intercept": torch.ones_like(x)}, [prior_intercept, prior_slope])
    
    with plate("data", N):
        likelihood = ObservedParameter("likelihood", Normal(mu, prior_variance), y)

with MeanFieldGuide() as guide:
    RandomParameter("intercept", Normal(VariationalParameter("mean", torch.full((1,), 0).double()), VariationalParameter("variance", torch.full((1,), 1).double(), min=1e-8)))
    RandomParameter("slope", Normal(VariationalParameter("mean", torch.full((1,), 0).double()), VariationalParameter("variance", torch.full((1,), 1).double(), min=1e-8)))
    # RandomParameter("sigma_squared", Exponential(VariationalParameter("scale", torch.full((1,), 1).double(), min=1e-8)))  # this is less stable and by reparametrizing, we get a more stable algorithm (cannot use as high of a learning rate if this is used)
    log_sigma_squared = RandomParameter("log_sigma_squared", Normal(VariationalParameter("mean", torch.full((1,), 0).double()), VariationalParameter("variance", torch.full((1,), 1).double(), min=1e-8)))
    DeterministicParameter("sigma_squared", lambda log_sigma_squared: torch.exp(log_sigma_squared), lambda log_sigma_squared: {"log_sigma_squared": torch.exp(log_sigma_squared)}, [log_sigma_squared])

history = BBVI(linear_model, guide, n_samples=10, epochs=500, lr=1e-2)  # n_samples can be lowered for faster optimization or increased for lower variance.
plt.figure()
plt.semilogy([-elbo for elbo in history])

samples = guide.sample(n_samples=1000, warmup_length=1000, n_chains=4)  # sample from the fitted guide
plot_posterior(samples, parameters=["intercept", "slope", "sigma_squared"])

y_preds = samples["slope"].flatten(0, 1)[:, None] * x[None, :] + samples["intercept"].flatten(0, 1)[:, None]
y_preds = y_preds.squeeze()
y_mean = y_preds.mean(dim=0)
y_lower = y_preds.quantile(0.025, dim=0)
y_upper = y_preds.quantile(0.975, dim=0)
std = samples["sigma_squared"].mean().sqrt()

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
