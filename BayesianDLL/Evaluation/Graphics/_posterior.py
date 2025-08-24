import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from cycler import cycler
import numpy as np
import matplotlib.cm as cm


def plot_posterior(trace, method="kde", bins=30):
    if method not in ["kde", "hist"]:
        raise ValueError('method should be in ["kde", "hist"].')
    # linestyles = ["-", "--", "-.", ":", (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5))]
    linestyles = ['-', '--', '-.', ':']

    for j, (name, samples) in enumerate(trace.items()):
        if samples.ndim != 3 or samples.size(2) != 1:
            raise NotImplementedError()
        samples = samples.squeeze(2).numpy()

        n_chains = len(samples)
        cmap = cm.get_cmap("Blues", n_chains + 2)
        colors = [cmap(i + 1) for i in range(n_chains)]
        repeated_linestyles = [linestyles[i % len(linestyles)] for i in range(n_chains)]
        prop_cycle = cycler("color", colors) + cycler("linestyle", repeated_linestyles)

        plt.subplot(len(trace), 2, 2 * j + 1)
        plt.gca().set_prop_cycle(prop_cycle)

        x_grid = np.linspace(samples.min(), samples.max(), 500)

        # mean_pdf = np.zeros_like(x_grid)

        for i, chain in enumerate(samples):
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
        plt.title(name)
        plt.legend(loc="upper right")

        plt.subplot(len(trace), 2, 2 * j + 2)
        plt.gca().set_prop_cycle(prop_cycle)  # TODO: make sure the traces and histograms have the same colors (currently not the case)
        for i, chain in enumerate(samples):
            plt.plot(chain, alpha=0.7, label=f"Chain {i+1}")
        plt.title(name)
        plt.legend(loc="upper right")
        plt.tight_layout()
