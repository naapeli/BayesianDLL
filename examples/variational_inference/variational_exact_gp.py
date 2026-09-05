import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, MeanFieldGuide, Model, ObservedParameter, RandomParameter, VariationalParameter, plate
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import ExactGP, RBF, exact_gp_predictive
from BayesianDLL.Variational import BBVI
from BayesianDLL.Evaluation import Graphics


def variational_normal(name, dtype):
    mean = VariationalParameter(f"{name}_mean", torch.zeros(1, dtype=dtype))
    variance = VariationalParameter(f"{name}_variance", torch.ones(1, dtype=dtype), min=1e-6)
    return RandomParameter(name, Normal(mean, variance))


torch.manual_seed(0)

n_observations = 18
x = torch.linspace(0.0, 1.0, n_observations, dtype=torch.float64)
y = torch.sin(2 * torch.pi * x) + 0.08 * torch.randn_like(x)

with Model() as model:
    inputs = Data("inputs", x)
    log_lengthscale = RandomParameter("log_lengthscale", Normal(-1.0, 0.5))
    log_variance = RandomParameter("log_variance", Normal(0.0, 0.5))
    log_noise_variance = RandomParameter("log_noise_variance", Normal(-2.5, 0.5))
    lengthscale = Exp("lengthscale", log_lengthscale)
    variance = Exp("variance", log_variance)
    noise_variance = Exp("noise_variance", log_noise_variance)
    function = ExactGP(
        "function", inputs, RBF(lengthscale, variance), noise_variance=noise_variance
    )
    with plate("observations", x):
        ObservedParameter("observations", function, y)

with MeanFieldGuide() as guide:
    q_log_lengthscale = variational_normal("log_lengthscale", x.dtype)
    q_log_variance = variational_normal("log_variance", x.dtype)
    q_log_noise_variance = variational_normal("log_noise_variance", x.dtype)
    Exp("lengthscale", q_log_lengthscale)
    Exp("variance", q_log_variance)
    Exp("noise_variance", q_log_noise_variance)

history = BBVI(model, guide, n_samples=4, epochs=1000, lr=5e-3, callback_frequency=50)

posterior = guide.sample(
    n_samples=300,
    warmup_length=300,
    n_chains=4,
)
Graphics.plot_posterior(posterior)
Graphics.plot_posterior(posterior, vars="deterministic")

x_prediction = torch.linspace(-0.1, 1.1, 120, dtype=x.dtype)
predictions = exact_gp_predictive(
    function, posterior, y, x_prediction, n_samples=8
)
predictions = predictions.reshape(-1, x_prediction.numel())
prediction_mean = predictions.mean(dim=0)
prediction_lower = predictions.quantile(0.025, dim=0)
prediction_upper = predictions.quantile(0.975, dim=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history)
axes[0].set_title("Exact-GP ELBO")
axes[0].set_xlabel("Optimization epoch")
axes[0].set_ylabel("ELBO")
axes[1].scatter(x, y, color="C0", label="Observed data")
axes[1].plot(x_prediction, prediction_mean, color="black", label="Posterior mean")
axes[1].fill_between(
    x_prediction, prediction_lower, prediction_upper,
    color="C0", alpha=0.2, label="95% credible interval"
)
axes[1].set_title("Variational exact GP")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].legend()
fig.tight_layout()
plt.show()
