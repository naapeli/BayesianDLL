from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from scipy.stats import gaussian_kde

from ...Samplers._result import SamplingResult


def plot_posterior(
    trace: SamplingResult,
    method: str = "kde",
    bins: int = 30,
    parameters: None | list[str] = None,
    axes=None,
    aggregate: bool = False,
):
    """Plot posterior densities and traces.

    By default, each scalar parameter component occupies one row: its marginal
    density is drawn in the first column and its MCMC trace in the second. If
    ``aggregate`` is true, all components of each parameter are drawn in the
    same row instead. If ``axes`` is supplied, it must contain exactly two
    axes per plotted row. The returned axes array always has shape
    ``(n_rows, 2)``.
    """
    if method not in ["kde", "hist"]:
        raise ValueError('method should be in ["kde", "hist"].')
    linestyles = ['-', '--', '-.', ':']

    selected = [
        (name, samples)
        for name, samples in trace.trace.items()
        if parameters is None or name in parameters
    ]
    total_rows = 0
    for _, samples in selected:
        param_shape = samples.shape[2:]
        n_features = np.prod(param_shape, dtype=int) if param_shape else 1
        total_rows += 1 if aggregate else n_features

    if total_rows == 0:
        raise ValueError("No posterior parameter components were selected for plotting.")

    if axes is None:
        _, axes_array = plt.subplots(
            total_rows,
            2,
            squeeze=False,
            figsize=(10, max(3.0, 2.8 * total_rows)),
            layout="constrained",
        )
    else:
        axes_array = np.asarray(axes, dtype=object)
        if axes_array.size != total_rows * 2:
            raise ValueError(
                f"axes must contain exactly {total_rows * 2} axes "
                f"({total_rows} rows by 2 columns); got {axes_array.size}."
            )
        axes_array = axes_array.reshape(total_rows, 2)

    row = 0
    for name, samples in selected:
        n_chains, trace_length = samples.shape[:2]
        param_shape = samples.shape[2:]
        n_features = np.prod(param_shape, dtype=int) if param_shape else 1
        reshaped_samples = samples.reshape(n_chains, trace_length, n_features)

        feature_rows = [range(n_features)] if aggregate else [[feature] for feature in range(n_features)]
        for features in feature_rows:
            density_ax, trace_ax = axes_array[row]
            for feature in features:
                feature_samples = reshaped_samples[:, :, feature]
                if aggregate:
                    cmap = colormaps["viridis"].resampled(n_features + 2)
                    colors = [cmap(feature + 1)] * n_chains
                    repeated_linestyles = [linestyles[i % len(linestyles)] for i in range(n_chains)]
                else:
                    cmap = colormaps["Blues"].resampled(n_chains + 2)
                    colors = [cmap(i + 1) for i in range(n_chains)]
                    repeated_linestyles = [linestyles[i % len(linestyles)] for i in range(n_chains)]
                prop_cycle = cycler("color", colors) + cycler("linestyle", repeated_linestyles)

                density_ax.set_prop_cycle(prop_cycle)
                x_grid = np.linspace(feature_samples.min().item(), feature_samples.max().item(), 500)

                for i, chain in enumerate(feature_samples):
                    chain_bins = bins if not isinstance(bins, dict) else bins[name]
                    chain_values = chain.detach().cpu().numpy()

                    if method == "kde":
                        est = gaussian_kde(chain_values)
                        pdf = est(x_grid)
                    else:
                        hist, bin_edges = np.histogram(
                            chain_values,
                            bins=chain_bins,
                            range=(x_grid.min(), x_grid.max()),
                            density=True,
                        )
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        pdf = np.interp(x_grid, bin_centers, hist)
                    if aggregate:
                        label = f"{name}[{feature}]" if i == 0 else "_nolegend_"
                    else:
                        label = f"Chain {i + 1}"
                    density_ax.plot(x_grid, pdf, alpha=0.3, label=label)

                trace_ax.set_prop_cycle(prop_cycle)
                for i, chain in enumerate(feature_samples):
                    if aggregate:
                        label = f"{name}[{feature}]" if i == 0 else "_nolegend_"
                    else:
                        label = f"Chain {i + 1}"
                    trace_ax.plot(chain.detach().cpu().numpy(), alpha=0.7, label=label)

            title = name if aggregate else f"{name}[{features[0]}]"
            density_ax.set_title(title)
            density_ax.legend(loc="upper right")
            trace_ax.set_title(title)
            trace_ax.legend(loc="upper right")
            row += 1

    return axes_array
