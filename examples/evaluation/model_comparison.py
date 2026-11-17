"""
Model Comparison With LOO, WAIC and Predictive Checks
=====================================================

A linear and a quadratic regression are fitted to data with curvature and
compared by how well they would predict new observations, measured by the
expected log pointwise predictive density (ELPD):

* :func:`~BayesianDLL.Evaluation.loo` estimates leave-one-out
  cross-validation with Pareto-smoothed importance sampling (PSIS-LOO); its
  Pareto ``k`` values flag observations for which the estimate is unreliable;
* :func:`~BayesianDLL.Evaluation.waic` computes the widely applicable
  information criterion;
* :func:`~BayesianDLL.Evaluation.compare` ranks the models and gives the
  standard error of each ELPD difference and stacking weights;
* :func:`~BayesianDLL.Evaluation.ppc_pvalue` tests whether replicated data
  reproduce a chosen statistic of the observed data.

All of them use the pointwise log-likelihood that ``sample`` stores in
``result.log_likelihood``.

Both models use ``Linear`` on a design matrix of centred features: ``x`` and
``x^2 - mean(x^2)``. Centring the squared term removes its strong posterior
correlation with the intercept. The figures show both fits over the data,
the ELPD comparison, and the predictive checks of a curvature statistic.
"""

import warnings

import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, Model, ObservedParameter, RandomParameter, plate
from BayesianDLL.Deterministic import Linear
from BayesianDLL.Distributions import HalfNormal, Normal
from BayesianDLL.Evaluation import Graphics, compare, loo, ppc_pvalue, waic


torch.manual_seed(1)

n_obs = 60
x = torch.linspace(-2.0, 2.0, n_obs, dtype=torch.float64)


def true_curve(x):
    return 0.5 + 1.5 * x - 0.7 * x ** 2


y = true_curve(x) + 0.6 * torch.randn(n_obs, dtype=torch.float64)
features = torch.stack([x, x ** 2 - (x ** 2).mean()], dim=1)


def regression(degree):
    with Model() as model:
        # A single feature is passed as a vector: Linear treats a one-element
        # coefficient as a scalar slope.
        design = Data("design", features[:, 0] if degree == 1 else features[:, :degree])
        intercept = RandomParameter("intercept", Normal(0.0, 5.0))
        coefficients = RandomParameter("coefficients", Normal(0.0, 2.5), shape=degree)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        mu = Linear("mu", design, coefficients=coefficients, intercept=intercept)
        with plate("observations", n_obs):
            ObservedParameter("y", Normal(mu, sigma), y)
    return model


models = {"linear": regression(1), "quadratic": regression(2)}
results = {name: model.sample(1000, 500, n_chains=4, seed=0, delta=0.85, chain_method="vectorized", progress_bar=False)
           for name, model in models.items()}
print("shape of the pointwise log-likelihood (chains, draws, observations):",
      tuple(results["linear"].log_likelihood["y"].shape))

# %%
# PSIS-LOO and WAIC
# -----------------
for name, result in results.items():
    loo_result = loo(result, pointwise=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # WAIC warns when its variance terms are large
        waic_result = waic(result)
    print(f"\n{name} model ({sum(result.divergences)} divergences)")
    print(loo_result)
    print(f"elpd_waic {waic_result.elpd:.2f} (SE {waic_result.se:.2f}), p_waic {waic_result.p:.2f}, "
          f"largest Pareto k {loo_result.pareto_k.max().item():.2f}")

# %%
# Ranking
# -------
table = compare(results, ic="loo")
print("\n" + table.to_string())

# %%
# Posterior predictive check of the curvature
# -------------------------------------------
# The statistic is the mean of the outer observations minus the mean of the
# central ones. Replicated data from the linear model cannot reproduce it.
outer, centre = x.abs() > 1.4, x.abs() < 0.6


def curvature(dataset):
    return dataset[outer].mean() - dataset[centre].mean()


checks = {}
for name, model in models.items():
    predictive = model.posterior_predicative(results[name], n_samples=100, samples_per_step=10, progress_bar=False)
    checks[name] = ppc_pvalue(predictive, y, curvature)
    print(f"{name}: P(T(y_rep) >= T(y)) = {checks[name].p_value:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")

ax = axes[0, 0]
ax.scatter(x, y, s=14, color="gray", label="data")
for (name, result), color in zip(results.items(), ("C0", "C1")):
    mu = result["mu"].reshape(-1, n_obs)
    ax.fill_between(x, mu.quantile(0.025, 0), mu.quantile(0.975, 0), color=color, alpha=0.3, linewidth=0)
    ax.plot(x, mu.mean(0), color=color, label=f"{name} fit (95% band)")
ax.plot(x, true_curve(x), "k--", label="true curve")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Fitted mean curves")
ax.legend(frameon=False, fontsize="small")

ax = axes[0, 1]
ax.errorbar(table["elpd"], range(len(table)), xerr=table["se"], fmt="o", color="black", capsize=4)
ax.set_yticks(range(len(table)), table.index)
ax.invert_yaxis()
ax.margins(y=0.4)
ax.set_xlabel("elpd_loo (higher is better)")
ax.set_title("Estimated predictive accuracy with standard errors")

for ax, (name, check) in zip(axes[1], checks.items()):
    Graphics.plot_ppc_statistic(check, ax=ax)
    ax.set_title(f"{name} model: curvature check, p = {check.p_value:.3f}")
    ax.set_xlabel("outer mean minus central mean")
plt.show()
