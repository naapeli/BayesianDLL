import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, MeanFieldGuide, Model, ObservedParameter, RandomParameter, VariationalParameter, condition, plate
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.GP import LatentGP, RBF, gp_predictive
from BayesianDLL.Variational import BBVI
from BayesianDLL.Evaluation import Graphics


def variational_normal(name, shape, dtype, mean=0.0, variance=1.0):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    mean_parameter = VariationalParameter(
        f"{name}_mean", torch.full(shape, mean, dtype=dtype)
    )
    variance_parameter = VariationalParameter(
        f"{name}_variance", torch.full(shape, variance, dtype=dtype), min=1e-6
    )
    return RandomParameter(name, Normal(mean_parameter, variance_parameter), shape=shape)


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
    function = LatentGP("function", inputs, RBF(lengthscale, variance))
    with plate("observations", x):
        ObservedParameter("observations", Normal(function, noise_variance), y)

condition_hyperparameters = True

if condition_hyperparameters:
    fixed_log_hyperparameters = {
        "log_lengthscale": torch.tensor([-1.0], dtype=x.dtype),
        "log_variance": torch.tensor([0.0], dtype=x.dtype),
        "log_noise_variance": torch.tensor([-2.5], dtype=x.dtype),
    }
    model = condition(model, fixed_log_hyperparameters)
    function = model.deterministic_params["function"]
    fixed_lengthscale = torch.exp(fixed_log_hyperparameters["log_lengthscale"]).item()
    fixed_variance = torch.exp(fixed_log_hyperparameters["log_variance"]).item()

with MeanFieldGuide() as guide:
    guide_inputs = Data("inputs", x)
    if condition_hyperparameters:
        guide_kernel = RBF(fixed_lengthscale, fixed_variance)
    else:
        q_log_lengthscale = variational_normal("log_lengthscale", 1, x.dtype)
        q_log_variance = variational_normal("log_variance", 1, x.dtype)
        q_log_noise_variance = variational_normal("log_noise_variance", 1, x.dtype)
        q_lengthscale = Exp("lengthscale", q_log_lengthscale)
        q_variance = Exp("variance", q_log_variance)
        Exp("noise_variance", q_log_noise_variance)
        guide_kernel = RBF(q_lengthscale, q_variance)

    q_function = LatentGP(
        "function",
        guide_inputs,
        guide_kernel,
        latent=variational_normal("function_white", n_observations, x.dtype),
    )

history = BBVI(model, guide, n_samples=1, epochs=5000, lr=1e-3, callback_frequency=50)

posterior = guide.sample(
    n_samples=300,
    warmup_length=300,
    n_chains=4,
)

Graphics.plot_posterior(posterior, aggregate=True)
plt.show()

x_prediction = torch.linspace(-0.1, 1.1, 120, dtype=x.dtype)
predictions = gp_predictive(function, posterior, x_prediction, n_samples=8)
predictions = predictions.reshape(-1, x_prediction.numel())
prediction_mean = predictions.mean(dim=0)
prediction_lower = predictions.quantile(0.025, dim=0)
prediction_upper = predictions.quantile(0.975, dim=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history)
axes[0].set_title("Latent-GP ELBO")
axes[0].set_xlabel("Optimization epoch")
axes[0].set_ylabel("ELBO")
axes[1].scatter(x, y, color="C0", label="Observed data")
axes[1].plot(x_prediction, prediction_mean, color="black", label="Posterior mean")
axes[1].fill_between(
    x_prediction, prediction_lower, prediction_upper,
    color="C0", alpha=0.2, label="95% credible interval"
)
axes[1].set_title("Variational latent GP")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")
axes[1].legend()
fig.tight_layout()
plt.show()
