import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Uniform, Normal, Exponential
from BayesianDLL.Evaluation import Graphics
from BayesianDLL.GP import ExactGP, Periodic, exact_gp_predictive


torch.manual_seed(0)

x = torch.linspace(0.0, 3.0, 24, dtype=torch.float64)
y = torch.sin(2 * torch.pi * x) + 0.2 ** 0.5 * torch.randn_like(x)

with Model() as model:
    inputs = Data("inputs", x)

    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0))
    log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
    # log_period = RandomParameter("log_period", Normal(0.0, 1.0))
    lengthscale = Exp("lengthscale", log_lengthscale)
    variance = Exp("variance", log_variance)
    # period = Exp("period", log_period)
    period = RandomParameter("period", Uniform(0.5, 1.5))

    # noise_variance = 0.05
    noise_variance = RandomParameter("noise_variance", Exponential(1.0))

    f = ExactGP(
        "f",
        inputs,
        Periodic(lengthscale=lengthscale, variance=variance, period=period),
        noise_variance=noise_variance,
    )
    with plate("observations", x):
        ObservedParameter("y", f, y)

    Graphics.plot_model(model)
    plt.show()


trace = model.sample(
    200,
    400,
    start_point_variance=0.1,
    max_depth=6,
    delta=0.8,
)
print(trace.summary())

new_inputs = torch.linspace(-0.1, 3.1, 100, dtype=x.dtype)
function_samples = exact_gp_predictive(
    f,
    trace,
    y,
    x,
    n_samples=1,
).squeeze(1)
predictions = exact_gp_predictive(
    f,
    trace,
    y,
    new_inputs,
    n_samples=20,
)

Graphics.plot_posterior(trace, method="kde", aggregate=True)

function_quantiles = torch.quantile(
    function_samples,
    torch.tensor([0.05, 0.5, 0.95], dtype=function_samples.dtype),
    dim=0,
)
predictive_samples = predictions.reshape(-1, new_inputs.numel())
predictive_quantiles = torch.quantile(
    predictive_samples,
    torch.tensor([0.05, 0.5, 0.95], dtype=predictive_samples.dtype),
    dim=0,
)

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.fill_between(
    x,
    function_quantiles[0],
    function_quantiles[2],
    color="tab:blue",
    alpha=0.2,
    label="Latent 90% interval",
)
ax.plot(x, function_quantiles[1], color="tab:blue", label="Latent posterior median")
ax.fill_between(
    new_inputs,
    predictive_quantiles[0],
    predictive_quantiles[2],
    color="tab:orange",
    alpha=0.2,
    label="Predictive 90% interval",
)
ax.plot(new_inputs, predictive_quantiles[1], color="tab:orange", label="Predictive median")
ax.scatter(x, y, color="black", zorder=3, label="Observations")
ax.set(title="Exact periodic GP posterior and predictive distribution", xlabel="x", ylabel="y")
ax.legend()
plt.show()
