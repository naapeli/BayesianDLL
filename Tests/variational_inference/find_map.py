import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfCauchy
from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, find_MAP
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
    prior_intercept = RandomParameter("intercept", Normal(0, 20))
    prior_slope = RandomParameter("slope", Normal(0, 20))
    prior_sigma = RandomParameter("sigma", HalfCauchy(10))

    # make the transform for the predicted line
    mu = DeterministicParameter("mu", lambda b, m: m * x + b, lambda b, m: {"slope": x, "intercept": torch.ones_like(x)}, [prior_intercept, prior_slope])
    
    likelihood = ObservedParameter("likelihood", Normal(mu, prior_sigma), y)

    history = find_MAP(linear_model, lr=1e-2, epochs=1000, callback_frequency=100)

intercept = linear_model.params["intercept"].constrained_value.squeeze()
slope = linear_model.params["slope"].constrained_value.squeeze()
sigma = linear_model.params["sigma"].constrained_value.squeeze()

plt.figure()
plt.plot(history)

plt.figure()
Graphics.plot_model(linear_model)


x = x.squeeze()
y_preds = slope * x + intercept
std = sigma.sqrt()

plt.figure(figsize=(10, 6))
plt.plot(x, y.squeeze(), 'o', label="Observed data", alpha=0.6)
plt.plot(x, y_preds, label="Posterior mean", color="black")
plt.fill_between(x, y_preds - 1.96 * std, y_preds + 1.96 * std, color="blue", alpha=0.2, label="95% CI for data points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()

plt.show()
