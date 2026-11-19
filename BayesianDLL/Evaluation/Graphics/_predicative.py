import matplotlib.pyplot as plt
import numpy as np
import torch

from ...Samplers._result import PredicativeResult
from .._numerics import gaussian_kde


def plot_predicative_distribution(
    predicative_distribution: PredicativeResult,
    data=None,
    kind="pdf",
    method="kde",
    bins=30,
    ax=None,
):
    """Plot predictive densities or CDFs on one axes.

    For every parameter draw ``i`` the ``samples_per_step`` replicated values
    (all trailing dimensions flattened) give one light curve; their pointwise
    average is drawn in orange and the observed data in black. The curves are
    evaluated on a 500-point grid spanning the predictive draws and the data.
    The variables are assumed univariate.

    Parameters
    ----------
    predicative_distribution : PredicativeResult
        Predictive draws of shape ``(n_samples, samples_per_step, *obs_shape)``
        per observed node, e.g. from ``posterior_predicative`` or
        ``sample_prior_predicative``. All its variables are drawn on the same axes.
    data : Tensor, optional
        Observed data to overlay (flattened); the same tensor is used for
        every variable.
    kind : {"pdf", "cdf"}, default "pdf"
        Draw densities, or CDFs obtained by cumulatively summing the density
        values on the grid and normalising the last value to one.
    method : {"kde", "hist"}, default "kde"
        Gaussian kernel density estimate (Scott's bandwidth) or a normalised
        histogram interpolated through the bin centres.
    bins : int, default 30
        Number of histogram bins (``method="hist"`` only).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if kind not in ["pdf", "cdf"]:
        raise ValueError('kind should be in ["pdf", "cdf"].')
    if method not in ["kde", "hist"]:
        raise ValueError('method should be in ["kde", "hist"].')

    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    for name, predicative_samples in predicative_distribution.items():
        # predicative_samples shape: (n_parameter_samples, n_predictive_samples, *batch_shape, *event_shape)
        # We assume for now that we are plotting univariate distributions.
        # We flatten everything after the first dimension (parameter samples) to get the predictive distribution.
        
        left_boarder = (min(data.min(), predicative_samples.min()) if data is not None else predicative_samples.min()).item()
        right_boarder = (max(data.max(), predicative_samples.max()) if data is not None else predicative_samples.max()).item()
        x_grid = np.linspace(left_boarder, right_boarder, 500)

        values = []


        for i in range(len(predicative_samples)):
            samples = predicative_samples[i].detach().cpu().reshape(-1).numpy()
            # calculate the pdf
            if method == "kde":
                result = gaussian_kde(torch.from_numpy(samples), torch.from_numpy(x_grid)).numpy()
            elif method == "hist":
                hist, bin_edges = np.histogram(samples, bins=bins, range=(left_boarder, right_boarder), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                result = np.interp(x_grid, bin_centers, hist)

            if kind == "pdf":
                values.append(result)
            elif kind == "cdf":
                result = np.cumsum(result)
                result /= result[-1]
                values.append(result)
            ax.plot(x_grid, result, color="lightblue", alpha=0.2)

        values = np.stack(values, axis=0)
        mean_values = values.mean(axis=0)
        ax.plot(x_grid, mean_values, color="orange", linewidth=2, label="Predicative mean")

        if data is not None:

            obs = data.flatten().cpu().numpy()

            # calculate the pdf
            if method == "kde":
                obs_values = gaussian_kde(torch.from_numpy(obs), torch.from_numpy(x_grid)).numpy()
            elif method == "hist":
                hist, bin_edges = np.histogram(obs, bins=bins, range=(left_boarder, right_boarder), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                obs_values = np.interp(x_grid, bin_centers, hist)

            if kind == "cdf":
                obs_values = np.cumsum(obs_values)
                obs_values /= obs_values[-1]
            ax.plot(x_grid, obs_values, color="black", label="Observed")

        ax.set_title(f"{kind.upper()} - {name}")
        ax.legend(loc="upper right")

    return ax
