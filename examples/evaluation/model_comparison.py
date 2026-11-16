"""
Model Comparison with LOO, WAIC and Posterior Predictive Checks
===============================================================

Fits a linear and a quadratic regression to data with curvature and compares
them by their estimated expected log predictive density (ELPD):

* :func:`~BayesianDLL.Evaluation.loo` estimates leave-one-out cross-validation
  with Pareto smoothed importance sampling (PSIS-LOO) and reports the Pareto
  shape diagnostic k-hat for every observation,
* :func:`~BayesianDLL.Evaluation.waic` computes the widely applicable
  information criterion,
* :func:`~BayesianDLL.Evaluation.compare` ranks the models, computes the
  standard error of their ELPD difference and stacking weights.

Both use the pointwise log-likelihood that :func:`~BayesianDLL.sample`
records for every draw. Finally, posterior predictive p-values of a test
statistic show *where* the linear model fails.
"""

import warnings

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Model, RandomParameter, ObservedParameter, DeterministicParameter, plate
from BayesianDLL.Distributions import Normal, HalfCauchy
from BayesianDLL.Evaluation import loo, waic, compare, ppc_pvalue, Graphics


torch.manual_seed(1)

n = 60
x = torch.linspace(-2, 2, n, dtype=torch.float64)
y = 0.5 + 1.5 * x - 0.7 * x ** 2 + 0.6 * torch.randn(n, dtype=torch.float64)


def regression_model(degree):
    """Polynomial regression of the given degree with a Normal(mu, sigma) likelihood."""
    with Model() as model:
        coefficients = [RandomParameter(f"c{d}", Normal(0.0, 5.0)) for d in range(degree + 1)]
        sigma = RandomParameter("sigma", HalfCauchy(1.0))
        mu = DeterministicParameter(
            "mu",
            lambda *c: sum(c_d * x ** d for d, c_d in enumerate(c)),
            lambda *c: {f"c{d}": x ** d for d in range(degree + 1)},
            coefficients,
        )
        with plate("data", n):
            ObservedParameter("y", Normal(mu, sigma), y)
    return model


models = {"linear": regression_model(1), "quadratic": regression_model(2)}

# %%
# Sampling records ``log p(y_i | theta)`` for every observation and draw in
# ``result.log_likelihood`` (pass ``log_likelihood=False`` to skip it).
results = {name: model.sample(500, 400, n_chains=2, seed=0, progress_bar=False) for name, model in models.items()}
print("Pointwise log-likelihood shape:", tuple(results["linear"].log_likelihood["y"].shape))

# %%
# PSIS-LOO and WAIC of each model. Large k-hat values (above
# ``min(1 - 1 / log10(S), 0.7)``) would flag observations for which the
# importance sampling approximation is unreliable.
for name, result in results.items():
    loo_result = loo(result, pointwise=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # WAIC warns when it is unreliable
        waic_result = waic(result)
    print(f"\n{name} model")
    print(loo_result)
    print(f"elpd_waic = {waic_result.elpd:.2f} (SE {waic_result.se:.2f}), p_waic = {waic_result.p:.2f}")
    print(f"largest k-hat: {loo_result.pareto_k.max().item():.2f}")

# %%
# ``compare`` ranks the models by ``elpd_loo``. ``elpd_diff`` is the
# difference to the best model and ``dse`` its standard error, computed from
# the pointwise differences on the shared observations. The stacking weights
# put most of the weight on the quadratic model.
table = compare(results, ic="loo")
print(table.to_string())

fig, ax = plt.subplots(layout="constrained")
ax.errorbar(table["elpd"], range(len(table)), xerr=table["se"], fmt="o", color="black", capsize=4, label="elpd_loo")
ax.set_yticks(range(len(table)), table.index)
ax.set_xlabel("elpd_loo")
ax.set_title("Estimated expected log predictive density")
ax.invert_yaxis()

# %%
# Posterior predictive checks. The linear model cannot reproduce the
# curvature of the data: for the statistic "mean at both ends minus mean in
# the centre" its p-value is extreme, whereas the quadratic model passes.
ends = (x.abs() > 1.4)
centre = (x.abs() < 0.6)


def curvature(dataset):
    return dataset[ends].mean() - dataset[centre].mean()


fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, (name, model) in zip(axes, models.items()):
    predictive = model.posterior_predicative(results[name], n_samples=40, samples_per_step=10, warmup_per_sample=30, progress_bar=False)
    check = ppc_pvalue(predictive, y, curvature)
    print(f"{name}: P(T(y_rep) >= T(y)) = {check.p_value:.3f}")
    Graphics.plot_ppc_statistic(check, ax=ax)
    ax.set_title(f"{name} model: p = {check.p_value:.3f}")

plt.show()
