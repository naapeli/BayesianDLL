import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np


def plot_predicative_distribution(predicative_distribution, data=None, kind="pdf", method="kde", bins=30):
    if kind not in ["pdf", "cdf"]:
        raise ValueError('kind should be in ["pdf", "cdf"].')
    if method not in ["kde", "hist"]:
        raise ValueError('method should be in ["kde", "hist"].')
    
    for name, predicative_samples in predicative_distribution.items():
        if predicative_samples.ndim != 3 or predicative_samples.size(2) != 1:
            raise NotImplementedError()

        left_boarder = (min(data.min(), predicative_samples.min()) if data is not None else predicative_samples.min()).item()
        right_boarder = (max(data.max(), predicative_samples.max()) if data is not None else predicative_samples.max()).item()
        x_grid = np.linspace(left_boarder, right_boarder, 500)

        values = []

        plt.figure()

        for i in range(len(predicative_samples)):
            samples = predicative_samples[i].reshape(-1).numpy()
            # calculate the pdf
            if method == "kde":
                est = gaussian_kde(samples)
                result = est(x_grid)
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
            plt.plot(x_grid, result, color="lightblue", alpha=0.2)

        values = np.stack(values, axis=0)
        mean_values = values.mean(axis=0)
        plt.plot(x_grid, mean_values, color="orange", linewidth=2, label="Predicative mean")

        if data is not None:
            if data.ndim != 2 or data.size(1) != 1:
                raise NotImplementedError()

            obs = data.reshape(-1).numpy()

            # calculate the pdf
            if method == "kde":
                est = gaussian_kde(obs)
                obs_values = est(x_grid)
            elif method == "hist":
                hist, bin_edges = np.histogram(obs, bins=bins, range=(left_boarder, right_boarder), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                obs_values = np.interp(x_grid, bin_centers, hist)

            if kind == "cdf":
                obs_values = np.cumsum(obs_values)
                obs_values /= obs_values[-1]
            plt.plot(x_grid, obs_values, color="black", label="Observed")

        plt.title(f"{kind.upper()} - {name}")
        plt.legend(loc="upper right")
