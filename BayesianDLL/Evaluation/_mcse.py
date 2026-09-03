import torch
import numpy as np
from ._effective_sample_size import _ess_mean


def _mcse_mean(ary: np.ndarray) -> float:
    """
    Compute Monte Carlo Standard Error (MCSE) for the posterior mean.
    Formula: sd / sqrt(ess_mean)
    """
    ary = np.asarray(ary, dtype=float)
    if ary.ndim < 2:
        ary = np.expand_dims(ary, axis=0)
    ess = _ess_mean(ary)
    sd = float(np.std(ary, ddof=1)) if ary.size > 1 else 0.0
    return float(sd / np.sqrt(ess)) if ess > 0 else float("nan")


def _mcse_sd(ary: np.ndarray) -> float:
    """
    Compute Monte Carlo Standard Error (MCSE) for the posterior standard deviation.
    Formula: Delta-method on 4th central moment (sample kurtosis).
    """
    ary = np.asarray(ary, dtype=float)
    if ary.ndim < 2:
        ary = np.expand_dims(ary, axis=0)
    flat = ary.flatten()
    ess = _ess_mean(ary)
    if ess <= 0 or len(flat) <= 1:
        return float("nan")

    sims_c2 = (flat - flat.mean()) ** 2
    evar = float(sims_c2.mean())
    if evar <= 0:
        return 0.0
    varvar = float(((sims_c2 ** 2).mean() - evar ** 2) / ess)
    varsd = varvar / evar / 4.0
    return float(np.sqrt(max(0.0, varsd)))


def mcse(samples: dict[str, torch.Tensor], method: str = "mean") -> dict[str, torch.Tensor]:
    """
    Compute Markov Chain Standard Error (MCSE).

    Parameters:
    -----------
    samples : dict[str, Tensor]
        Mapping from parameter name to tensor of shape (n_chains, n_samples, ...)
    method : str, default "mean"
        - "mean": MCSE for the posterior mean.
        - "sd": MCSE for the posterior standard deviation.

    Returns:
    --------
    dict[str, Tensor]:
        Mapping from parameter name to tensor of MCSE values.
    """
    results = {}
    for name, chains in samples.items():
        if chains.ndim < 2:
            raise NotImplementedError("Each parameter tensor must have at least 2 dimensions: (n_chains, n_samples, ...)")

        n_chains, n_samples = chains.shape[:2]
        feature_shape = chains.shape[2:] if chains.ndim > 2 else ()
        chains_np = chains.detach().cpu().numpy().reshape(n_chains, n_samples, -1)
        n_features = chains_np.shape[2]

        mcse_vals = torch.zeros(n_features, dtype=torch.float64)
        for f in range(n_features):
            feat = chains_np[:, :, f]
            if method == "mean":
                mcse_vals[f] = _mcse_mean(feat)
            elif method == "sd":
                mcse_vals[f] = _mcse_sd(feat)
            else:
                raise ValueError(f"Unknown method '{method}'. Must be 'mean' or 'sd'.")

        if len(feature_shape) > 0:
            results[name] = mcse_vals.reshape(feature_shape)
        else:
            results[name] = mcse_vals
    return results


def mcse_mean(samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute MCSE for posterior mean."""
    return mcse(samples, method="mean")


def mcse_sd(samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute MCSE for posterior standard deviation."""
    return mcse(samples, method="sd")

