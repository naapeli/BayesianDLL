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
