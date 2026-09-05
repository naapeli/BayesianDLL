import torch
import matplotlib.pyplot as plt

from BayesianDLL.Distributions import Normal, HalfCauchy
from BayesianDLL import Data, Model, RandomParameter, ObservedParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Evaluation import Graphics, summary


torch.manual_seed(7)

N = 15
true_intercept = 1.0
true_slope = 2.5
true_variance = 0.5
x = torch.linspace(0, 1, N).double()
y = true_intercept + true_slope * x + torch.normal(0, true_variance ** 0.5, size=(N,))

indices = torch.randperm(N)
split = int(0.8 * N)
train_indices = indices[:split]
test_indices = indices[split:]
x_train, y_train = x[train_indices], y[train_indices]
x_test, y_test = x[test_indices], y[test_indices]

with Model() as linear_model:
    x_data = Data("x", x_train)
    y_data = Data("y", y_train)

    prior_intercept = RandomParameter("intercept", Normal(0, 20), sampler="auto", delta=0.4)
    prior_slope = RandomParameter("slope", Normal(0, 20), sampler="auto", delta=0.4)
    prior_sigma = RandomParameter("sigma", HalfCauchy(1), sampler="auto", max_depth=4)

    mu = Linear("mu", x_data, slope=prior_slope, intercept=prior_intercept)
    
    with plate("data", x_data):
        likelihood = ObservedParameter("likelihood", Normal(mu, prior_sigma), y_data)

    Graphics.plot_model(linear_model)
    
    linear_model.find_MAP(verbose=False)
    samples = linear_model.sample(1000, 500, blocks=[["slope", "intercept", "sigma"]], delta=0.7)
    print(summary(samples))

    x_prediction = torch.linspace(x.min(), x.max(), 1000, dtype=x.dtype)
    x_data.set_value(x_prediction)
    predicative_distribution = linear_model.posterior_predicative(samples, n_samples=20, samples_per_step=50, warmup_per_sample=50)


Graphics.plot_posterior(samples)

Graphics.plot_predicative_distribution(predicative_distribution, y_test, kind="pdf")


y_preds = samples["slope"].reshape(-1, 1) * x_prediction[None, :] + samples["intercept"].reshape(-1, 1)
y_mean = y_preds.mean(dim=0)
y_mean_standard_error = y_preds.std(dim=0)

mean_variance = y_preds.var(dim=0)
noise_variance = samples["sigma"].reshape(-1).mean()
predictive_standard_deviation = torch.sqrt(mean_variance + noise_variance)
data_credible_lower = y_mean - 1.96 * predictive_standard_deviation
data_credible_upper = y_mean + 1.96 * predictive_standard_deviation

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(x_prediction, data_credible_lower, data_credible_upper, color="C0", alpha=0.12, linewidth=0, label="95% credible interval for data")
ax.fill_between(x_prediction, y_mean - y_mean_standard_error, y_mean + y_mean_standard_error, color="C0", alpha=0.28, linewidth=0, label="Mean +/- standard error")
ax.plot(x_prediction, y_mean, color="black", linewidth=2, label="Fitted posterior mean")
ax.scatter(x_train, y_train, s=46, color="C0", alpha=0.8, edgecolors="white", linewidths=0.7, label="Training data", zorder=3)
ax.scatter(x_test, y_test, s=54, color="C1", marker="D", alpha=0.9, edgecolors="white", linewidths=0.7, label="Test data", zorder=3)
ax.set_title("Posterior predictive distribution", pad=12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(alpha=0.2)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncols=2)
fig.tight_layout()

plt.show()
