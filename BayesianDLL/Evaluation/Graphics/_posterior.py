import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from cycler import cycler
import numpy as np
import matplotlib.cm as cm

from ...Samplers._result import SamplingResult


def plot_posterior(trace: SamplingResult, method: str = "kde", bins: int = 30, parameters: None | list[str] = None):
    if method not in ["kde", "hist"]:
        raise ValueError('method should be in ["kde", "hist"].')
    # linestyles = ["-", "--", "-.", ":", (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5))]
    linestyles = ['-', '--', '-.', ':']

    row = 0
    total_rows = 0
    for name, samples in trace.trace.items():
        # samples shape: (n_chains, trace_length, *param_shape)
        param_shape = samples.shape[2:]
        n_features = np.prod(param_shape, dtype=int) if param_shape else 1
        total_rows += n_features

    for name, samples in trace.trace.items():
        if parameters is not None and name not in parameters:
            continue
        # Flatten all dimensions except chains and trace_length for iteration
        n_chains, trace_length = samples.shape[:2]
        param_shape = samples.shape[2:]
        n_features = np.prod(param_shape, dtype=int) if param_shape else 1
        
        reshaped_samples = samples.reshape(n_chains, trace_length, n_features)
        
        for feature in range(n_features):
            feature_samples = reshaped_samples[:, :, feature]

            n_chains = len(feature_samples)
            cmap = cm.get_cmap("Blues", n_chains + 2)
            colors = [cmap(i + 1) for i in range(n_chains)]
            repeated_linestyles = [linestyles[i % len(linestyles)] for i in range(n_chains)]
            prop_cycle = cycler("color", colors) + cycler("linestyle", repeated_linestyles)

            plt.subplot(total_rows, 2, 2 * row + 1)
            plt.gca().set_prop_cycle(prop_cycle)

            x_grid = np.linspace(feature_samples.min(), feature_samples.max(), 500)

            # mean_pdf = np.zeros_like(x_grid)

            for i, chain in enumerate(feature_samples):
                chain_bins = bins if not isinstance(bins, dict) else bins[name]

                if method == "kde":
                    est = gaussian_kde(chain)
                    pdf = est(x_grid)
                elif method == "hist":
                    hist, bin_edges = np.histogram(chain, bins=chain_bins, range=(x_grid.min(), x_grid.max()), density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    pdf = np.interp(x_grid, bin_centers, hist)

                # mean_pdf += pdf
                plt.plot(x_grid, pdf, alpha=0.3, label=f"Chain {i+1}")
            
            # mean_pdf /= n_chains
            # plt.plot(x_grid, mean_pdf, color="orange", linewidth=2, label="Posterior mean")
            plt.title(f"{name}[{feature}]")
            plt.legend(loc="upper right")

            plt.subplot(total_rows, 2, 2 * row + 2)
            plt.gca().set_prop_cycle(prop_cycle)  # TODO: make sure the traces and histograms have the same colors (currently not the case)
            for i, chain in enumerate(feature_samples):
                plt.plot(chain, alpha=0.7, label=f"Chain {i+1}")
            plt.title(f"{name}[{feature}]")
            plt.legend(loc="upper right")
            plt.tight_layout()
            row += 1
