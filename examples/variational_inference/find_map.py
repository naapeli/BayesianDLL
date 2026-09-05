import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfCauchy
from BayesianDLL import Data, Model, RandomParameter, ObservedParameter, find_MAP, plate
from BayesianDLL.Deterministic import Linear
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
    prior_intercept = RandomParameter("intercept", Normal(0, 20))
    prior_slope = RandomParameter("slope", Normal(0, 20))
    prior_sigma = RandomParameter("sigma", HalfCauchy(10))

    x_data = Data("x", x)
    mu = Linear("mu", x_data, slope=prior_slope, intercept=prior_intercept)
    
    with plate("data", N):
        likelihood = ObservedParameter("likelihood", Normal(mu, prior_sigma), y)

    history = find_MAP(linear_model, lr=1e-2, epochs=1000, callback_frequency=100)

intercept = linear_model.params["intercept"].constrained_value.squeeze()
slope = linear_model.params["slope"].constrained_value.squeeze()
sigma = linear_model.params["sigma"].constrained_value.squeeze()

plt.figure()
plt.plot(history)

Graphics.plot_model(linear_model)


y_preds = slope * x + intercept
std = sigma.sqrt()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="Observed data", alpha=0.6)
plt.plot(x, y_preds, label="Posterior mean", color="black")
plt.fill_between(x, y_preds - 1.96 * std, y_preds + 1.96 * std, color="blue", alpha=0.2, label="95% CI for data points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()

plt.show()
