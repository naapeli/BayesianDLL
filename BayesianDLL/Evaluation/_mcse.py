import math

import torch
import numpy as np

from ._effective_sample_size import _ess_mean
from ._numerics import _betaincinv


# The Beta quantiles at Phi(-1) and Phi(1): the interval they bound holds the
# central ~68.27% of the mass, matching +-1 standard error of a normal.
_ONE_SIGMA_PROBS = (0.1586553, 0.8413447)

_METHODS = ("mean", "sd", "median", "quantile")


def _mcse_mean(ary: np.ndarray) -> float:
    """MCSE of the posterior mean: ``sd(x) / sqrt(ESS_mean(x))``."""
    ary = np.asarray(ary, dtype=float)
    if ary.ndim < 2:
        ary = np.expand_dims(ary, axis=0)
    ess = _ess_mean(ary)
    sd = float(np.std(ary, ddof=1)) if ary.size > 1 else 0.0
    return float(sd / np.sqrt(ess)) if ess > 0 else float("nan")


def _mcse_sd(ary: np.ndarray) -> float:
    """MCSE of the posterior standard deviation, by the delta method.

    With ``c_s = (x_s - x̄)^2`` and ``v̂ = mean(c)``, the MCSE of ``v̂`` is
    ``sqrt((mean(c^2) - v̂^2) / ESS_mean(c))`` and the delta method for
    ``sd = sqrt(v)`` divides it by ``2 sqrt(v̂)``.
    """
    ary = np.asarray(ary, dtype=float)
    if ary.ndim < 2:
        ary = np.expand_dims(ary, axis=0)
    if ary.size <= 1:
        return float("nan")

    sims_c2 = (ary - ary.mean()) ** 2
    evar = float(sims_c2.mean())
    if evar <= 0:
        return 0.0
    ess = _ess_mean(sims_c2)
    if not ess > 0:
        return float("nan")
    varvar = float(((sims_c2 ** 2).mean() - evar ** 2) / ess)
    varsd = varvar / evar / 4.0
    return float(np.sqrt(max(0.0, varsd)))


def _mcse_quantile(ary: np.ndarray, prob: float) -> float:
    """MCSE of the posterior ``prob`` quantile (Vehtari et al., 2021)."""
    ary = np.asarray(ary, dtype=float)
    if ary.ndim < 2:
        ary = np.expand_dims(ary, axis=0)
    flat = ary.ravel()
    quantile = float(np.quantile(flat, prob))
    ess = _ess_mean((ary <= quantile).astype(float))
    if not math.isfinite(ess) or ess <= 0:
        return float("nan")

    alpha, beta = ess * prob + 1.0, ess * (1.0 - prob) + 1.0
    lower, upper = (_betaincinv(alpha, beta, p) for p in _ONE_SIGMA_PROBS)
    sorted_draws = np.sort(flat)
    size = sorted_draws.size
    th1 = sorted_draws[int(math.floor(max(lower * size - 1.0, 0.0)))]
    th2 = sorted_draws[int(math.ceil(min(upper * size - 1.0, size - 1.0)))]
    return float((th2 - th1) / 2.0)


def mcse(samples: dict[str, torch.Tensor], method: str = "mean", prob: float | None = None) -> dict[str, torch.Tensor]:
    r"""Monte Carlo standard error (MCSE) of a posterior summary.

    The MCSE is the standard deviation of the Monte Carlo error of an estimate
    computed from (autocorrelated) MCMC draws: it says how far the reported
    number may be from the value an infinitely long run would give. ``method``
    selects the posterior quantity (the estimand) whose estimate the error
    refers to. For example ``mcse(samples, method="sd")`` is the MCSE *of the
    posterior standard deviation estimate*, not the standard deviation of an
    MCSE.

    With ``S`` pooled draws ``x_1, ..., x_S`` over all chains, and
    :math:`\mathrm{ESS}(y)` the split-chain effective sample size of the
    sequence ``y`` (``effective_sample_size(..., method="mean")`` of ``y``),
    the methods compute

    - ``"mean"``: MCSE of the posterior mean,
      :math:`\hat\sigma / \sqrt{\mathrm{ESS}(x)}` with the sample standard
      deviation :math:`\hat\sigma` (``ddof=1``).
    - ``"sd"``: MCSE of the posterior standard deviation, by the delta method.
      With :math:`c_s = (x_s - \bar x)^2` and :math:`\hat v = \bar c`,

      .. math::

         \mathrm{MCSE}(\hat v) = \sqrt{\frac{\overline{c^2} - \hat v^2}{\mathrm{ESS}(c)}},
         \qquad
         \mathrm{MCSE}(\sqrt{\hat v}) = \frac{\mathrm{MCSE}(\hat v)}{2 \sqrt{\hat v}}.

      The ESS is that of the squared deviations :math:`c`, because
      :math:`\hat v` is their mean (Vehtari et al., 2021). ArviZ 1.x uses :math:`\mathrm{ESS}(x)` here
      instead, so its value differs by the factor
      :math:`\sqrt{\mathrm{ESS}(x) / \mathrm{ESS}(c)}`.
    - ``"quantile"``: MCSE of the posterior ``prob`` quantile
      :math:`\hat q_p`. With the indicators :math:`I_s = 1[x_s \le \hat q_p]`,
      their effective sample size :math:`E = \mathrm{ESS}(I)`, and
      :math:`a, b` the 0.1586553 and 0.8413447 quantiles (:math:`\Phi(\mp 1)`)
      of :math:`\mathrm{Beta}(E p + 1, E (1 - p) + 1)`,

      .. math::

         \mathrm{MCSE}(\hat q_p) = (x_{(\lceil b S - 1 \rceil)} - x_{(\lfloor a S - 1 \rfloor)}) / 2,

      half the width of the interval of order statistics :math:`x_{(i)}`
      (0-based, clipped to ``0 .. S-1``) that covers :math:`\pm 1` standard
      error of the quantile's probability. :math:`\hat q_p` itself is
      ``numpy.quantile``'s linear interpolation of the pooled draws. This is the
      computation of ArviZ, which takes the order statistics with floor/ceil
      rounding rather than interpolating between them.
    - ``"median"``: the ``"quantile"`` method with ``prob=0.5``.

    The MCSE of the posterior expectation of any function ``g`` is
    ``mcse({"g": g(x)}, method="mean")``, with ``g`` applied to the draws.

    Parameters
    ----------
    samples : dict[str, Tensor]
        Mapping from parameter name to a tensor of shape
        ``(n_chains, n_samples, *event_shape)``.
    method : {"mean", "sd", "median", "quantile"}, default "mean"
        The posterior quantity whose estimate the error refers to.
    prob : float, optional
        The quantile probability ``p``, with ``0 < p < 1``. Required for
        ``method="quantile"`` and not accepted by the other methods.

    Returns
    -------
    dict[str, Tensor]
        Mapping from parameter name to a float64 tensor of shape
        ``event_shape`` holding one MCSE per scalar component.

    Raises
    ------
    ValueError
        For an unknown ``method``, a missing or out-of-range ``prob`` with
        ``method="quantile"``, or a ``prob`` given with another method.
    NotImplementedError
        If a tensor has fewer than two dimensions.

    References
    ----------
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Bürkner, P.-C.
    (2021). Rank-normalization, folding, and localization: an improved
    R-hat for assessing convergence of MCMC. *Bayesian Analysis* 16(2),
    667-718.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Must be one of {', '.join(repr(m) for m in _METHODS)}.")
    if method == "quantile":
        if prob is None:
            raise ValueError("method='quantile' requires prob, the quantile probability in (0, 1).")
        prob = float(prob)
        if not 0.0 < prob < 1.0:
            raise ValueError(f"prob must lie strictly between 0 and 1, got {prob}.")
    elif prob is not None:
        raise ValueError(f"prob is only used by method='quantile', not method='{method}'.")
    if method == "median":
        prob = 0.5

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
                mcse_vals[f] = _mcse_quantile(feat, prob)

        if len(feature_shape) > 0:
            results[name] = mcse_vals.reshape(feature_shape)
        else:
            results[name] = mcse_vals
    return results
