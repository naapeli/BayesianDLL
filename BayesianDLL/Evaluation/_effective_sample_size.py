import torch
import numpy as np
from scipy.stats import rankdata, norm
from scipy.fftpack import next_fast_len


def _split_chains(ary: np.ndarray) -> np.ndarray:
    _, n_draws = ary.shape
    half = n_draws // 2
    return np.vstack((ary[:, :half], ary[:, -half:]))


def _z_scale(ary: np.ndarray) -> np.ndarray:
    size = ary.size
    ranks = rankdata(ary.flatten(), method="average")
    u = (ranks - 0.375) / (size + 0.25)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    z = norm.ppf(u)
    return z.reshape(ary.shape)


def _autocov(ary: np.ndarray, axis: int = 1) -> np.ndarray:
    n = ary.shape[axis]
    m = next_fast_len(2 * n)
    ary_centered = ary - ary.mean(axis=axis, keepdims=True)

    ifft_ary = np.fft.rfft(ary_centered, n=m, axis=axis)
    ifft_ary *= np.conjugate(ifft_ary)

    cov = np.fft.irfft(ifft_ary, n=m, axis=axis)[:, :n]
    cov /= n
    return cov


def _ess_core(ary: np.ndarray) -> float:
    ary = np.asarray(ary, dtype=float)
    if (np.max(ary) - np.min(ary)) < np.finfo(float).resolution:
        return float(ary.size)

    n_chain, n_draw = ary.shape
    acov = _autocov(ary, axis=1)
    chain_mean = ary.mean(axis=1)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += np.var(chain_mean, axis=None, ddof=1)

    rho_hat_t = np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd

    t = 1
    while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even

    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    ess = n_chain * n_draw
    tau_hat = (
        -1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1]) + np.sum(rho_hat_t[max_t + 1 : max_t + 2])
    )
    tau_hat = max(tau_hat, 1.0 / np.log10(ess))
    ess = ess / tau_hat
    if np.isnan(rho_hat_t).any():
        ess = float("nan")
    return float(ess)


def _ess_mean(chains: np.ndarray) -> float:
    """Compute split mean ESS for posterior expectation."""
    split_feat = _split_chains(chains)
    return _ess_core(split_feat)


def _ess_bulk(chains: np.ndarray) -> float:
    """Compute rank-normalized split bulk ESS."""
    split_feat = _split_chains(chains)
    z_feat = _z_scale(split_feat)
    return _ess_core(z_feat)


def _ess_tail(chains: np.ndarray, prob: float = 0.89) -> float:
    """
    Compute tail ESS using quantile indicators matching ArviZ / Stan.
    prob: coverage probability (default 0.89 matching ArviZ's stats.ci_prob).
    """
    if prob is None:
        prob = 0.89
    if not isinstance(prob, (list, tuple)):
        prob = sorted((prob, 1.0 - prob))

    prob_low, prob_high = prob
    flat = chains.flatten()
    q_low = float(np.quantile(flat, prob_low))
    q_high = float(np.quantile(flat, prob_high))
    split_low = _split_chains((chains <= q_low).astype(float))
    split_high = _split_chains((chains <= q_high).astype(float))
    ess_low = _ess_core(split_low)
    ess_high = _ess_core(split_high)
    return float(min(ess_low, ess_high))


def effective_sample_size(samples: dict[str, torch.Tensor], method: str = "bulk", prob: float = 0.89) -> dict[str, torch.Tensor]:
    """
    Compute Effective Sample Size (ESS).

    Parameters:
    -----------
    samples : dict[str, Tensor]
        Mapping from parameter name to tensor of shape (n_chains, n_samples, ...)
    method : str, default "bulk"
        - "bulk": Rank-normalized split ESS.
        - "tail": Tail ESS computed on upper and lower quantile indicators.
        - "mean": Raw split ESS for posterior mean.
    prob : float or tuple, default 0.89
        Coverage probability or quantile pair for tail ESS (only used when method="tail").

    Returns:
    --------
    dict[str, Tensor]:
        Mapping from parameter name to tensor of ESS values.
    """
    results = {}
    for name, chains in samples.items():
        if chains.ndim < 2:
            raise NotImplementedError("Each parameter tensor must have at least 2 dimensions: (n_chains, n_samples, ...)")

        n_chains, n_samples = chains.shape[:2]
        feature_shape = chains.shape[2:] if chains.ndim > 2 else ()
        chains_np = chains.detach().cpu().numpy().reshape(n_chains, n_samples, -1)
        n_features = chains_np.shape[2]

        ess_vals = torch.zeros(n_features, dtype=torch.float64)
        for f in range(n_features):
            feat = chains_np[:, :, f]
            if method == "mean":
                ess_vals[f] = _ess_mean(feat)
            elif method == "bulk":
                ess_vals[f] = _ess_bulk(feat)
            elif method == "tail":
                ess_vals[f] = _ess_tail(feat, prob=prob)
            else:
                raise ValueError(f"Unknown method '{method}'. Must be 'bulk', 'tail', or 'mean'.")

        if len(feature_shape) > 0:
            results[name] = ess_vals.reshape(feature_shape)
        else:
            results[name] = ess_vals
    return results


def ess_bulk(samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute rank-normalized bulk ESS."""
    return effective_sample_size(samples, method="bulk")


def ess_tail(samples: dict[str, torch.Tensor], prob: float = 0.89) -> dict[str, torch.Tensor]:
    """Compute tail ESS using quantile indicators at coverage `prob` (default 0.89)."""
    return effective_sample_size(samples, method="tail", prob=prob)


def ess_mean(samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute split mean ESS for posterior mean."""
    return effective_sample_size(samples, method="mean")
