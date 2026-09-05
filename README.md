A Bayesian machine learning library capable of performing MCMC and variational inference for fitting complex models.

Demonstrations live in [examples](examples). Run them as modules from the project root, for example:

```sh
uv run python -m examples.bayesian_regression.bayesian_linear_regression
```

The [Tests](Tests) folder contains the pytest suite. Install the development dependencies and run all tests with:

```sh
uv sync --dev
uv run pytest
```

Tests cover distributions and gradients, transforms and state spaces, model construction and plates, MCMC and predictive sampling, MAP and variational inference, diagnostics, result containers, and plotting. Randomized tests use fixed seeds; plots use a noninteractive backend. Longer sampling and optimization checks are marked `integration`:

```sh
uv run pytest -m "not integration"
uv run pytest -m integration
```

The original demonstrations (including exploratory gradient scripts) are preserved in `examples`; pytest only discovers tests in `Tests` by default.

Use `Data` for inputs you want to replace before MAP estimation or posterior
sampling without rebuilding the model:

```python
from BayesianDLL import Data, Model, RandomParameter, ObservedParameter
from BayesianDLL.Distributions import Normal

with Model() as model:
    observations = Data("observations", [1.0, 2.0, 3.0])
    mean = RandomParameter("mean", Normal(0.0, 4.0))
    ObservedParameter("likelihood", Normal(mean, 1.0), observations)

observations.set_value([2.0, 3.0, 4.0])
samples = model.sample(1000, 500)
```

`Data` can also be a distribution argument or an entry in
`DeterministicParameter.inputs`; the forward and derivative functions receive
its current tensor. See the [linear regression example](examples/bayesian_regression/bayesian_linear_regression.py)
for mutable predictors and observations. Access registered inputs through
`model.data[name]`. Data are fixed during inference and are not sampled.
Updates preserve the event shape, dtype, and device while allowing batch shapes
to change. By default, `event_ndim=0` treats every element as a scalar event.
For vector events, use `Data("features", values, event_ndim=1)`: values with shape
`(100, 3)` can be replaced by `(50, 3)`, but not `(50, 4)`. Use `event_ndim=2`
for matrix events. Rebuild the model to change the event shape. Use floating-point
initial values for continuous data.

Model graphs use the standard graphical-model distinction between latent,
observed, deterministic, and data nodes, and show distributions, data shapes,
and plate sizes by default:

```python
from BayesianDLL.Evaluation.Graphics import plot_model

plot_model(model)
plot_model(model, include_data=False)  # compact stochastic/generative view
```

The optional `ax`, `show_distributions`, `show_plates`, and `legend` arguments
can be used to embed or simplify the plot. `plot_model` returns the Matplotlib
axes it draws into.

All plotting helpers can be embedded in existing Matplotlib figures. Predictive
plots accept and return one `ax`; posterior plots accept and return an axes grid
with one row per scalar parameter component and density/trace columns:

```python
import matplotlib.pyplot as plt
from BayesianDLL.Evaluation import Graphics

fig, ax = plt.subplots()
Graphics.plot_predicative_distribution(predictions, data=y, ax=ax)

fig, axes = plt.subplots(2, 2)
Graphics.plot_posterior(samples, parameters=["intercept", "slope"], axes=axes)
# For vector-valued parameters, put all components in one density/trace row.
Graphics.plot_posterior(samples, parameters=["function_white"], aggregate=True)
```

## Latent Gaussian processes

`BayesianDLL.GP` provides explicit, whitened latent-function Gaussian
processes for MCMC and variational inference. This keeps the latent function
in the model instead of integrating it out into a collapsed GP likelihood.

```python
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Distributions import Exponential, Normal
from BayesianDLL.GP import LatentGP, RBF, gp_predictive

x = torch.linspace(0.0, 1.0, 20)
y = torch.sin(6 * x)

with Model() as model:
    inputs = Data("inputs", x)
    z = RandomParameter("z", Normal(0.0, 1.0), shape=x.numel())
    lengthscale = RandomParameter("lengthscale", Exponential(1.0))
    variance = RandomParameter("variance", Exponential(1.0))
    function = LatentGP(
        "function", inputs, RBF(lengthscale=lengthscale, variance=variance), latent=z
    )
    with plate("observations", x):
        ObservedParameter("observations", Normal(function, 0.05), y)

trace = model.sample(500, 500)
```

The latent vector can be defined explicitly, as above, or omitted; in the
latter case the process creates a standard-normal vector named
`function_white`. It is transformed through the kernel Cholesky factor. Use
`gp_predictive(function, trace, new_inputs)` to sample latent function values
at new inputs. The generic `model.posterior_predicative(...)` method samples
observed likelihood sites at their existing inputs; GP prediction at new
inputs uses the conditional GP distribution and therefore goes through
`gp_predictive`. Supported kernels include `RBF`, `Periodic`, `Matern32`, `Matern52`,
`Linear`, `Constant`, and `WhiteNoise`; kernels can be added or multiplied
together.

## Ready-made deterministic transformations

Reusable deterministic nodes are available from `BayesianDLL.Deterministic`.
They register their named inputs in the model graph and provide derivatives for
MCMC and optimization:

```python
from BayesianDLL.Deterministic import Exp, Linear, Log

positive_scale = Exp("scale", log_scale)
log_scale = Log("log_scale", positive_scale)
mean = Linear("mean", inputs, slope=slope, intercept=intercept)
```

`Linear` also accepts a design matrix and a coefficient vector, using
`inputs @ coefficients + intercept`.

Variational GP examples are available for both formulations:
[explicit latent GP](examples/variational_inference/variational_latent_gp.py)
and [collapsed exact GP](examples/variational_inference/variational_exact_gp.py).

For binary outcomes, compose `Linear` with `Sigmoid` and use the result as the
probability of a `Bernoulli` likelihood. See the
[logistic regression example](examples/bayesian_regression/logistic_regression.py).

## Exact Gaussian processes

For a Gaussian likelihood, `ExactGP` integrates the latent
function out analytically and evaluates the multivariate-normal GP marginal
likelihood. Kernel and noise hyperparameters remain ordinary model parameters,
so they can be sampled with MCMC without adding one latent variable per input:

```python
from BayesianDLL.Distributions import Uniform
from BayesianDLL.GP import ExactGP, Periodic, exact_gp_predictive

with Model() as model:
    inputs = Data("inputs", x)
    lengthscale = RandomParameter("lengthscale", Uniform(0.05, 1.0))
    variance = RandomParameter("variance", Uniform(0.25, 2.0))
    period = RandomParameter("period", Uniform(0.5, 1.5))
    noise_variance = RandomParameter("noise_variance", Uniform(0.001, 0.1))
    function = ExactGP(
        "function", inputs,
        Periodic(lengthscale, variance, period),
        noise_variance=noise_variance,
    )
    with plate("observations", x):
        ObservedParameter("y", function, y)

trace = model.sample(500, 500)
predictions = exact_gp_predictive(function, trace, y, new_inputs)
```

`exact_gp_predictive` samples latent or noisy predictions after inference from
the exact conditional GP distribution. The exact implementation costs
`O(N^3)` per likelihood evaluation and is intended for Gaussian observations;
use `LatentGP` when the latent function must remain in the model or the
likelihood is non-Gaussian. The previous names `GaussianProcess` and
`ExactGaussianProcess` remain available as compatibility aliases.
