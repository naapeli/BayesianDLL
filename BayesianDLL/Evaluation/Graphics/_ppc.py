import matplotlib.pyplot as plt
import numpy as np

from .._ppc import PPCResult, ppc_pvalue


def plot_ppc_statistic(
    predicative_distribution,
    data=None,
    statistic="mean",
    var_name=None,
    bins=30,
    ax=None,
):
    """Plot the posterior predictive distribution of a test statistic.

    Draws a histogram of ``T(y_rep)`` over the replicated datasets, marks the
    observed ``T(y)`` with a vertical line and puts the posterior predictive
    p-value ``P(T(y_rep) >= T(y))`` in the title. ``predicative_distribution``
    is a PredicativeResult (then ``data`` is the observed data) or the
    :class:`~BayesianDLL.Evaluation.PPCResult` of
    :func:`~BayesianDLL.Evaluation.ppc_pvalue`. ``statistic`` is a callable
    or the name of a built-in statistic, see
    :func:`~BayesianDLL.Evaluation.ppc_pvalue`.
    """
    if isinstance(predicative_distribution, PPCResult):
        result = predicative_distribution
    else:
        if data is None:
            raise ValueError("The observed data are needed to compute the test statistic.")
        result = ppc_pvalue(predicative_distribution, data, statistic=statistic, var_name=var_name)

    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    t_rep = result.t_rep.detach().cpu().numpy()
    finite = t_rep[np.isfinite(t_rep)]
    if np.unique(finite).size <= bins and np.allclose(finite, np.round(finite)):
        # Integer valued statistics (e.g. the number of zeros): one bar per value.
        low, high = finite.min(), finite.max()
        edges = np.arange(low - 0.5, high + 1.5)
    else:
        edges = bins
    ax.hist(finite, bins=edges, density=True, color="lightblue", edgecolor="white", label=r"$T(y_{rep})$")
    ax.axvline(result.t_observed, color="black", linewidth=2, label=r"$T(y)$")
    ax.set_xlabel(f"T = {result.statistic}")
    ax.set_ylabel("Density")
    ax.set_title(f"{result.statistic} - {result.var_name}: p = {result.p_value:.3f}")
    ax.legend(loc="upper right")
    return ax
