import math

import torch
import numpy as np

from ._numerics import z_scale


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

    if torch.isnan(W) or torch.isnan(B):
        return float("nan")
    if W == 0:
        # No within-chain variation: undefined if all chains are the same
        # constant, infinite if they are stuck at different values.
        return float("nan") if B == 0 else float("inf")

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
    return z_scale(chains.detach()).to(dtype=chains.dtype, device=chains.device)


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

    # The larger of the two, ignoring an undefined (NaN) one; NaN only when
    # both are undefined (all draws equal).
    finite = [value for value in (rhat_bulk, rhat_tail) if not math.isnan(value)]
    return max(finite) if finite else float("nan")


def gelman_rubin(samples: dict[str, torch.Tensor], method: str = "rank") -> dict[str, torch.Tensor]:
    r"""Gelman-Rubin potential scale reduction factor R-hat for every parameter component.

    For ``M`` chains of ``N`` draws with chain means :math:`\bar x_m`, grand
    mean :math:`\bar x` and within-chain sample variances :math:`s_m^2`
    (``ddof=1``), the classical statistic is

    .. math::

       W = \frac{1}{M} \sum_m s_m^2, \qquad
       B = \frac{N}{M - 1} \sum_m (\bar x_m - \bar x)^2, \qquad
       \hat R = \sqrt{\frac{\frac{N - 1}{N} W + \frac{1}{N} B}{W}}.

    Values close to 1 indicate that the chains mix; 1.01 is a common threshold
    for the rank-normalised version.

    Parameters
    ----------
    samples : dict[str, Tensor]
        Mapping from parameter name to a tensor of shape
        ``(n_chains, n_samples, *event_shape)``.
    method : {"classical", "split", "rank"}, default "rank"
        - ``"classical"``: the formula above on the chains as given
          (Gelman & Rubin, 1992).
        - ``"split"``: the formula on ``2 M`` half chains of ``N // 2`` draws
          (Gelman et al., 2013, BDA3); with an odd ``N`` the last draw is dropped.
          Detects non-stationarity within chains.
        - ``"rank"``: rank-normalised split R-hat (Vehtari et al., 2021):
          the maximum of the split R-hat of the rank-normalised draws
          :math:`z = \Phi^{-1}((r - 3/8) / (S + 1/4))` (bulk) and of the
          rank-normalised folded draws :math:`|x - \mathrm{median}(x)|` (tail).
          The ArviZ default.

    Returns
    -------
    dict[str, Tensor]
        Mapping from parameter name to a float64 tensor of shape ``event_shape``.
        It is 1 with fewer than two (split) chains or fewer than two draws per
        (split) chain. With zero within-chain variance ``W`` it is ``inf``
        when the chains differ (``B > 0``, e.g. chains stuck at different
        constants) and NaN when every chain is the same constant
        (``W = B = 0``). For ``"rank"`` an undefined (NaN) bulk or tail
        R-hat is ignored in the maximum.

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
