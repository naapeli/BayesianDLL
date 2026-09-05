# BayesianDLL

BayesianDLL is a Python library for Bayesian machine learning with probability
distributions, graphical models, MCMC, and variational inference. Examples are
in [examples](examples), and tests are in [Tests](Tests).

## Environment

Requires Python >= 3.13 and `uv`.

From the project root:

```sh
uv sync --dev
```

Run examples as modules:

```sh
uv run python -m examples.bayesian_regression.logistic_regression
uv run python -m examples.variational_inference.variational_exact_gp
```

To run the tests:

```sh
uv run pytest
```

## Examples

### Logistic regression

`Linear` creates the linear predictor and `Sigmoid` converts it to a
probability for a Bernoulli likelihood:

```python
import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear, Sigmoid
from BayesianDLL.Distributions import Bernoulli, Normal
from BayesianDLL.Evaluation import Graphics, summary


x = torch.linspace(-3.0, 3.0, 80)
y = torch.bernoulli(torch.sigmoid(-0.4 + 1.6 * x))

with Model() as model:
    inputs = Data("inputs", x)
    intercept = RandomParameter("intercept", Normal(0.0, 2.0))
    slope = RandomParameter("slope", Normal(0.0, 2.0))

    logits = Linear("logits", inputs, slope=slope, intercept=intercept)
    probability = Sigmoid("probability", logits)

    with plate("observations", x):
        ObservedParameter("observations", Bernoulli(probability), y)

Graphics.plot_model(model)
trace = model.sample(500, 500)
print(summary(trace))
Graphics.plot_posterior(trace, parameters=["intercept", "slope"])
plt.show()
```

### Variational exact Gaussian Process

`ExactGP` analytically integrates out the latent function for a Gaussian
likelihood. Variational inference then optimizes a guide over the GP
hyperparameters:

```python
import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, MeanFieldGuide, Model, ObservedParameter, RandomParameter, VariationalParameter, plate
from BayesianDLL.Deterministic import Exp
from BayesianDLL.Distributions import Normal
from BayesianDLL.Evaluation import Graphics, summary
from BayesianDLL.GP import ExactGP, RBF, exact_gp_predictive
from BayesianDLL.Variational import BBVI


x = torch.linspace(0.0, 1.0, 18, dtype=torch.float64)
y = torch.sin(2 * torch.pi * x) + 0.08 * torch.randn_like(x)

with Model() as model:
    inputs = Data("inputs", x)
    log_lengthscale = RandomParameter("log_lengthscale", Normal(0.0, 1.0))
    log_variance = RandomParameter("log_variance", Normal(0.0, 1.0))
    log_noise = RandomParameter("log_noise", Normal(0.0, 1.0))

    lengthscale = Exp("lengthscale", log_lengthscale)
    variance = Exp("variance", log_variance)
    noise_variance = Exp("noise_variance", log_noise)

    function = ExactGP(
        "function",
        inputs,
        RBF(lengthscale, variance),
        noise_variance=noise_variance,
    )
    with plate("observations", x):
        ObservedParameter("observations", function, y)

Graphics.plot_model(model)

with MeanFieldGuide() as guide:
    def q_normal(name):
        return RandomParameter(
            name,
            Normal(
                VariationalParameter(f"{name}_mean", torch.zeros(1, dtype=x.dtype)),
                VariationalParameter(f"{name}_variance", torch.ones(1, dtype=x.dtype), min=1e-6)
            ),
        )

    q_log_lengthscale = q_normal("log_lengthscale")
    q_log_variance = q_normal("log_variance")
    q_log_noise = q_normal("log_noise")
    Exp("lengthscale", q_log_lengthscale)
    Exp("variance", q_log_variance)
    Exp("noise_variance", q_log_noise)

BBVI(model, guide, n_samples=4, epochs=200, lr=5e-3)
trace = guide.sample(200, 200, n_chains=2)
print(summary(trace, include_deterministic=True))
Graphics.plot_posterior(trace, vars="random")
Graphics.plot_posterior(trace, vars="deterministic")
predictions = exact_gp_predictive(
    function,
    trace,
    y,
    torch.linspace(-0.1, 1.1, 100, dtype=x.dtype),
)
plt.show()
```
