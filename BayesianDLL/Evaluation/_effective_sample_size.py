import torch
import numpy as np

from ._numerics import autocov, z_scale


def _split_chains(ary: np.ndarray) -> np.ndarray:
    _, n_draws = ary.shape
    half = n_draws // 2
    return np.vstack((ary[:, :half], ary[:, -half:]))


def _z_scale(ary: np.ndarray) -> np.ndarray:
    return z_scale(torch.from_numpy(np.ascontiguousarray(ary, dtype=np.float64))).numpy()


def _autocov(ary: np.ndarray, axis: int = 1) -> np.ndarray:
    return autocov(torch.from_numpy(np.ascontiguousarray(ary, dtype=np.float64)), dim=axis).numpy()


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
    r"""Effective sample size (ESS) of every parameter component (Vehtari et al., 2021).

    All methods split each chain into two halves (``2 M`` chains of
    ``N' = N // 2`` draws, the middle draw dropped for odd ``N``) and compute

    .. math::

       \hat\rho_t = 1 - \frac{W - \frac{1}{M'} \sum_m \hat\gamma_{m,t}}{\widehat{\mathrm{var}}^+},
       \qquad
       \widehat{\mathrm{var}}^+ = \frac{N' - 1}{N'} W + \frac{B}{N'},
       \qquad
       \mathrm{ESS} = \frac{M' N'}{\hat\tau},
       \quad \hat\tau = -1 + 2 \sum_{t=0}^{2k+1} \hat\rho_t,

    with the biased within-chain autocovariances :math:`\hat\gamma_{m,t}` (FFT),
    ``W``/``B`` as in :func:`gelman_rubin`, and the sum truncated by Geyer's
    initial positive sequence and made monotone (initial monotone sequence).
    :math:`\hat\tau` is bounded below by ``1 / log10(M' N')``, so the ESS can
    exceed the number of draws for antithetic chains. A constant input has an
    ESS equal to its number of draws. This matches ArviZ.

    Parameters
    ----------
    samples : dict[str, Tensor]
        Mapping from parameter name to a tensor of shape
        ``(n_chains, n_samples, *event_shape)``.
    method : {"bulk", "tail", "mean"}, default "bulk"
        - ``"bulk"``: the ESS of the rank-normalised split chains,
          :math:`z = \Phi^{-1}((r - 3/8) / (S + 1/4))`.
        - ``"tail"``: the minimum of the ESS of the indicators
          :math:`1[x \le \hat q_{p_\mathrm{low}}]` and
          :math:`1[x \le \hat q_{p_\mathrm{high}}]` of the pooled empirical
          quantiles.
        - ``"mean"``: the ESS of the split chains themselves, the one that
          enters the MCSE of the posterior mean.
    prob : float or tuple of float, default 0.89
        Only used by ``method="tail"``. A float ``p`` gives the quantile pair
        ``sorted((p, 1 - p))`` (so the default uses the 0.11 and 0.89
        quantiles, matching ArviZ's ``ci_prob``), a pair ``(p_low, p_high)``
        is used as given.

    Returns
    -------
    dict[str, Tensor]
        Mapping from parameter name to a float64 tensor of shape ``event_shape``.

    Raises
    ------
    ValueError
        For an unknown ``method``.
    NotImplementedError
        If a tensor has fewer than two dimensions.
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

