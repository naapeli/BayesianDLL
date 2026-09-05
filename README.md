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
