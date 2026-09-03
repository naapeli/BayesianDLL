import torch
import numpy as np
from scipy.stats import rankdata, norm


def _rhat_core(chains: torch.Tensor) -> float:
    """
    Classical Gelman-Rubin R-hat (Gelman & Rubin, 1992; BDA3).
    chains: 2D tensor of shape (n_chains, n_samples)
    """
    n_chains, n_samples = chains.shape
    if n_chains < 2 or n_samples < 2:
        return 1.0

    chain_vars = chains.var(dim=1, unbiased=True)  # ddof=1
    W = chain_vars.mean()
    chain_means = chains.mean(dim=1)
    grand_mean = chain_means.mean()
    B = (n_samples / (n_chains - 1)) * ((chain_means - grand_mean) ** 2).sum()

    var_hat = ((n_samples - 1) / n_samples) * W + (1.0 / n_samples) * B

    if W == 0 or torch.isnan(W) or torch.isnan(B):
        return 1.0

    rhat = torch.sqrt(var_hat / W)
    return float(rhat)


def _split_rhat_core(chains: torch.Tensor) -> float:
    """
    Split R-hat (Gelman et al., 2013; Stan; BDA3):
    Splits each chain in half, producing 2*n_chains chains of length n_samples // 2.
    Detects non-stationarity within chains.
    """
    _, n_samples = chains.shape
    if n_samples % 2 != 0:
        chains = chains[:, :-1]
        n_samples -= 1

    half = n_samples // 2
    if half < 2:
        return 1.0

    split_chains = torch.cat([chains[:, :half], chains[:, half:]], dim=0)
    return _rhat_core(split_chains)


def _z_scale(chains: torch.Tensor) -> torch.Tensor:
    """
    Rank-normalize values across all chains and draws using Blom's transform:
    z = Phi^{-1}((rank - 3/8) / (S + 1/4))
    """
    flat = chains.detach().cpu().numpy().flatten()
    S = len(flat)
    ranks = rankdata(flat, method="average")
    u = (ranks - 0.375) / (S + 0.25)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    z = norm.ppf(u)
    return torch.tensor(z, dtype=chains.dtype, device=chains.device).reshape(chains.shape)


def _rank_normalized_rhat_core(chains: torch.Tensor) -> float:
    """
    Rank-normalized folded split R-hat (Vehtari et al., 2021).
    Computes both bulk R-hat and folded tail R-hat and takes the maximum.
    """
    z_bulk = _z_scale(chains)
    rhat_bulk = _split_rhat_core(z_bulk)

    median = torch.tensor(np.median(chains.detach().cpu().numpy()), dtype=chains.dtype, device=chains.device)
    folded = torch.abs(chains - median)
    z_tail = _z_scale(folded)
    rhat_tail = _split_rhat_core(z_tail)

    return max(rhat_bulk, rhat_tail)


def gelman_rubin(samples: dict[str, torch.Tensor], method: str = "rank") -> dict[str, torch.Tensor]:
    """
    Compute Gelman-Rubin convergence diagnostic R-hat for all parameters.

    Parameters:
    -----------
    samples : dict[str, Tensor]
        Mapping from parameter name to tensor of shape (n_chains, n_samples, ...)
    method : str, default "rank"
        One of:
        - "classical": Original Gelman & Rubin (1992)
        - "split": Split R-hat (Gelman et al., 2013 / BDA3)
        - "rank": Rank-normalized folded split R-hat (Vehtari et al., 2021; ArviZ default)

    Returns:
    --------
    dict[str, Tensor]:
        Mapping from parameter name to tensor of R-hat values.
    """
    results = {}
    for name, chains in samples.items():
        if chains.ndim < 2:
            raise NotImplementedError("Each parameter tensor must have at least 2 dimensions: (n_chains, n_samples, ...)")

        n_chains, n_samples = chains.shape[:2]
        feature_shape = chains.shape[2:] if chains.ndim > 2 else ()
        chains_reshaped = chains.reshape(n_chains, n_samples, -1)
        n_features = chains_reshaped.shape[2]

        rhat_vals = torch.zeros(n_features, dtype=torch.float64)
        for f in range(n_features):
            feat = chains_reshaped[:, :, f]
            if method == "classical":
                rhat_vals[f] = _rhat_core(feat)
            elif method == "split":
                rhat_vals[f] = _split_rhat_core(feat)
            elif method == "rank":
                rhat_vals[f] = _rank_normalized_rhat_core(feat)
            else:
                raise ValueError(f"Unknown method '{method}'. Must be 'classical', 'split', or 'rank'.")

        if len(feature_shape) > 0:
            results[name] = rhat_vals.reshape(feature_shape)
        else:
            results[name] = rhat_vals
    return results
