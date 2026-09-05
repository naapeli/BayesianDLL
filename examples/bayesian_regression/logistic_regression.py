import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear, Sigmoid
from BayesianDLL.Distributions import Bernoulli, Normal
from BayesianDLL.Evaluation import Graphics


torch.manual_seed(7)

n_observations = 500
x = torch.linspace(-3.0, 3.0, n_observations, dtype=torch.float64)
true_logits = -0.4 + 1.6 * x
y = torch.bernoulli(torch.sigmoid(true_logits))

with Model() as model:
    inputs = Data("inputs", x)
    intercept = RandomParameter("intercept", Normal(0.0, 2.0))
    slope = RandomParameter("slope", Normal(0.0, 2.0))

    logits = Linear("logits", inputs, slope=slope, intercept=intercept)
    probability = Sigmoid("probability", logits)

    with plate("observations", x):
        ObservedParameter("observations", Bernoulli(probability), y)

    Graphics.plot_model(model)
    plt.show()


trace = model.sample(
    n_samples=1000,
    warmup_length=500,
    n_chains=2,
    delta=0.8,
)
print(trace.summary())

x_prediction = torch.linspace(x.min(), x.max(), 200, dtype=x.dtype)
slope_draws = trace["slope"].reshape(-1, 1)
intercept_draws = trace["intercept"].reshape(-1, 1)
probability_draws = torch.sigmoid(intercept_draws + slope_draws * x_prediction.reshape(1, -1))
probability_mean = probability_draws.mean(dim=0)
probability_lower = probability_draws.quantile(0.025, dim=0)
probability_upper = probability_draws.quantile(0.975, dim=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, color="C0", alpha=0.65, label="Observed outcome")
ax.plot(x_prediction, probability_mean, color="black", label="Posterior mean")
ax.fill_between(
    x_prediction,
    probability_lower,
    probability_upper,
    color="C0",
    alpha=0.2,
    label="95% credible interval",
)
ax.set_title("Bayesian logistic regression")
ax.set_xlabel("x")
ax.set_ylabel("P(y = 1)")
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.2)
ax.legend()
plt.show()
