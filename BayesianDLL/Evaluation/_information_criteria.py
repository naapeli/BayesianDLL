"""Predictive information criteria: PSIS-LOO, WAIC and model comparison.

References
----------
Vehtari, Gelman and Gabry (2017). Practical Bayesian model evaluation using
leave-one-out cross-validation and WAIC. Statistics and Computing.

Vehtari, Simpson, Gelman, Yao and Gabry (2024). Pareto smoothed importance
sampling. Journal of Machine Learning Research.

Zhang and Stephens (2009). A new and efficient estimation method for the
generalized Pareto distribution. Technometrics.

Yao, Vehtari, Simpson and Gelman (2018). Using stacking to average Bayesian
predictive distributions. Bayesian Analysis.
"""

import math
from dataclasses import dataclass, field
from warnings import warn

import numpy as np
import torch
from scipy.special import logsumexp

from ._effective_sample_size import effective_sample_size


@dataclass
class ELPDResult:
    """Estimate of the expected log pointwise predictive density (ELPD).

    Attributes
    ----------
    kind : ``"loo"`` or ``"waic"``.
    elpd : the estimated ELPD (``elpd_loo`` or ``elpd_waic``), on the log scale.
    se : its standard error, ``sqrt(n) * sd(elpd_i)``.
    p : the effective number of parameters (``p_loo`` or ``p_waic``).
    n_samples, n_data_points : number of posterior draws and of observations.
    elpd_i : pointwise ELPD values (a tensor of the observation shape) when
        ``pointwise=True``, else ``None``.
    pareto_k : Pareto shape estimates k-hat per observation (LOO only).
    good_k : the sample-size dependent k-hat threshold ``min(1 - 1/log10(S), 0.7)``.
    warning : whether a reliability warning was issued.
    """

    kind: str
    elpd: float
    se: float
    p: float
    n_samples: int
    n_data_points: int
    elpd_i: torch.Tensor | None = None
    pareto_k: torch.Tensor | None = None
    p_i: torch.Tensor | None = None
    good_k: float | None = None
    warning: bool = False
    var_name: str | None = None
    _elpd_i_flat: np.ndarray | None = field(default=None, repr=False)

    @property
    def lppd(self):
        return self.elpd + self.p

    def __getitem__(self, key):
        aliases = {f"elpd_{self.kind}": "elpd", f"p_{self.kind}": "p", f"{self.kind}_i": "elpd_i"}
        return getattr(self, aliases.get(key, key))

    def __repr__(self):
        lines = [
            f"Computed from {self.n_samples} posterior samples and {self.n_data_points} observations.",
            f"           Estimate     SE",
            f"elpd_{self.kind:<5s} {self.elpd:10.2f} {self.se:6.2f}",
            f"p_{self.kind:<8s} {self.p:10.2f}      -",
        ]
        if self.kind == "loo" and self.pareto_k is not None:
            k = self.pareto_k.reshape(-1).cpu().numpy()
            n_bad = int(np.sum(~(k <= self.good_k)))
            lines.append(f"Pareto k-hat > {self.good_k:.2f} (unreliable): {n_bad} of {k.size} observations")
        return "\n".join(lines)


# ------------------------------------------------------------------ inputs
def _log_likelihood_arrays(result, var_name=None):
    """Return ``(name, array of shape (chains, draws, n), observation shape, dtype, device)``."""
    if isinstance(result, torch.Tensor):
        log_lik = {"y": result}
        unavailable = {}
    elif isinstance(result, dict):
        log_lik = result
        unavailable = {}
    else:
        log_lik = getattr(result, "log_likelihood", None)
        unavailable = getattr(result, "log_likelihood_unavailable", {}) or {}
        if log_lik is None:
            raise TypeError(f"Expected a SamplingResult, a dict of log-likelihood tensors or a tensor, got {type(result).__name__}.")

    if var_name is not None:
        if var_name in unavailable:
            raise ValueError(unavailable[var_name])
        if var_name not in log_lik:
            raise KeyError(f"No pointwise log-likelihood for '{var_name}'. Available: {list(log_lik)}.")
        names = [var_name]
    else:
        if unavailable:
            raise ValueError(" ".join(unavailable.values()) + " LOO and WAIC cannot be computed for such models.")
        names = list(log_lik)
        if not names:
            raise ValueError(
                "The result holds no pointwise log-likelihood. Sample with log_likelihood=True "
                "(the default) from a model with observed parameters."
            )

    arrays, shapes = [], []
    first = log_lik[names[0]]
    for name in names:
        values = torch.as_tensor(log_lik[name])
        if values.ndim < 2:
            raise ValueError(f"The log-likelihood of '{name}' must have shape (chains, draws, ...), got {tuple(values.shape)}.")
        shapes.append(tuple(values.shape[2:]))
        arrays.append(values.detach().cpu().to(torch.float64).numpy().reshape(values.shape[0], values.shape[1], -1))
    shape = shapes[0] if len(names) == 1 else (sum(a.shape[2] for a in arrays),)
    label = names[0] if len(names) == 1 else "+".join(names)
    return label, np.concatenate(arrays, axis=2), shape, first.dtype, first.device


def _to_tensor(values, shape, dtype, device):
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float64
    return torch.as_tensor(np.asarray(values).reshape(shape), dtype=dtype, device=device)


def _se(elpd_i):
    return float(np.sqrt(elpd_i.size * np.var(elpd_i)))


# -------------------------------------------------------------------- WAIC
def waic(result, var_name=None, pointwise=False):
    """Widely applicable information criterion (Watanabe 2010).

    Parameters
    ----------
    result : SamplingResult (sampled with ``log_likelihood=True``), a dict
        mapping observed names to log-likelihood tensors of shape
        ``(chains, draws, *obs_shape)``, or such a tensor.
    var_name : the observed node to use. By default all observed nodes are
        used and their observations are concatenated (the ELPD is the sum
        over all of them).
    pointwise : also return the pointwise ``elpd_waic_i`` and ``p_waic_i``.

    Returns
    -------
    ELPDResult with ``elpd`` (elpd_waic), ``se``, ``p`` (p_waic) and, if
    requested, ``elpd_i``. A warning is issued when the posterior variance of
    the log-likelihood of an observation exceeds 0.4, in which case WAIC is
    unreliable and PSIS-LOO is preferable.
    """
    name, log_lik, shape, dtype, device = _log_likelihood_arrays(result, var_name)
    n_chains, n_draws, n_obs = log_lik.shape
    n_samples = n_chains * n_draws
    flat = log_lik.reshape(n_samples, n_obs)

    lppd_i = logsumexp(flat, axis=0) - math.log(n_samples)
    p_waic_i = np.var(flat, axis=0, ddof=1)
    elpd_i = lppd_i - p_waic_i

    warning = bool(np.any(p_waic_i > 0.4))
    if warning:
        warn(
            "For one or more observations the posterior variance of the log predictive density "
            "exceeds 0.4, so WAIC may be unreliable. Use PSIS-LOO (loo) instead.",
            stacklevel=2,
        )

    return ELPDResult(
        kind="waic",
        elpd=float(np.sum(elpd_i)),
        se=_se(elpd_i),
        p=float(np.sum(p_waic_i)),
        n_samples=n_samples,
        n_data_points=n_obs,
        elpd_i=_to_tensor(elpd_i, shape, dtype, device) if pointwise else None,
        p_i=_to_tensor(p_waic_i, shape, dtype, device) if pointwise else None,
        warning=warning,
        var_name=name,
        _elpd_i_flat=elpd_i,
    )


# --------------------------------------------------------------------- PSIS
def _gpd_fit(x):
    """Zhang & Stephens (2009) empirical Bayes fit of a generalized Pareto distribution.

    ``x`` holds the sorted (ascending), positive exceedances over the threshold.
    The shape estimate is regularised with the weakly informative prior of
    Vehtari et al. (2024): a normal prior centred at 0.5 worth 10
    observations. Returns ``(k, sigma)``.
    """
    n = len(x)
    prior_bs = 3.0
    prior_k = 10.0
    m = 30 + int(math.sqrt(n))

    x_star = x[int(n / 4 + 0.5) - 1]  # first quartile
    if x_star <= x[0]:
        return math.nan, math.nan

    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1, dtype=np.float64) - 0.5))
    b = b / (prior_bs * x_star) + 1.0 / x[-1]

    k = np.mean(np.log1p(-b[:, None] * x[None, :]), axis=1)
    log_lik = n * (np.log(-b / k) - k - 1.0)
    weights = np.exp(log_lik - logsumexp(log_lik))
    keep = weights >= 10 * np.finfo(np.float64).eps
    b, weights = b[keep], weights[keep]
    weights = weights / weights.sum()

    b_post = np.sum(b * weights)
    k_hat = np.mean(np.log1p(-b_post * x))
    sigma = -k_hat / b_post
    k_hat = (n * k_hat + prior_k * 0.5) / (n + prior_k)
    if np.isnan(k_hat):
        return math.inf, math.nan
    return float(k_hat), float(sigma)


def _gpd_quantile(p, k, sigma, mu):
    if sigma <= 0:
        return np.full_like(p, np.nan)
    if k == 0:
        return mu - sigma * np.log1p(-p)
    return mu + sigma * np.expm1(-k * np.log1p(-p)) / k


def psis_tail_length(n_samples, r_eff):
    """Number of largest importance ratios smoothed: ``ceil(min(0.2 S, 3 sqrt(S / r_eff)))``."""
    return int(math.ceil(min(0.2 * n_samples, 3.0 * math.sqrt(n_samples / r_eff))))


def psis(log_ratios, r_eff=1.0, truncate=True):
    """Pareto smoothed importance sampling of one vector of log importance ratios.

    The ``M = ceil(min(0.2 S, 3 sqrt(S / r_eff)))`` largest ratios are
    replaced by the expected order statistics (quantiles at ``(j - 1/2) / M``)
    of a generalized Pareto distribution fitted to them, but never by more
    than the largest raw ratio. With ``truncate`` the weights are then
    truncated at ``S^(3/4)`` times their mean.

    Returns ``(log_weights, k_hat)`` with log weights normalised to sum to one.
    """
    log_ratios = np.asarray(log_ratios, dtype=np.float64).reshape(-1)
    n_samples = log_ratios.size
    log_weights = log_ratios - np.max(log_ratios)
    tail_length = psis_tail_length(n_samples, r_eff)
    k_hat = math.nan
    if 5 <= tail_length < n_samples:
        order = np.argsort(log_weights, kind="stable")
        tail_ids = order[n_samples - tail_length:]
        log_tail = log_weights[tail_ids]
        log_cutoff = log_weights[order[n_samples - tail_length - 1]]
        max_tail = log_tail[-1]
        if max_tail - log_tail[0] > np.finfo(np.float64).tiny:
            cutoff = math.exp(log_cutoff)
            k_hat, sigma = _gpd_fit(np.exp(log_tail) - cutoff)
            if math.isfinite(k_hat):
                p = (np.arange(tail_length) + 0.5) / tail_length
                with np.errstate(divide="ignore", invalid="ignore"):
                    smoothed = np.log(_gpd_quantile(p, k_hat, sigma, cutoff))
                smoothed = np.minimum(smoothed, max_tail)
                log_weights[tail_ids] = smoothed
    elif tail_length < 5:
        warn(f"The PSIS tail has only {tail_length} draws (at least 5 are needed); the importance ratios are not smoothed.", stacklevel=3)

    if truncate:
        log_mean = logsumexp(log_weights) - math.log(n_samples)
        log_weights = np.minimum(log_weights, 0.75 * math.log(n_samples) + log_mean)
    log_weights = log_weights - logsumexp(log_weights)
    return log_weights, k_hat


def relative_efficiency(log_lik):
    """Relative effective sample size of ``exp(log_lik)`` per observation.

    ``log_lik`` has shape ``(chains, draws, n)``. Uses the split-chain mean
    ESS of :func:`~BayesianDLL.Evaluation.effective_sample_size`; returns an
    array of shape ``(n,)`` (ones for a single chain of fewer than 4 draws).
    """
    n_chains, n_draws, _ = log_lik.shape
    if n_draws < 4:
        return np.ones(log_lik.shape[2])
    likelihood = np.exp(log_lik - log_lik.max(axis=(0, 1), keepdims=True))
    ess = effective_sample_size({"lik": torch.from_numpy(likelihood)}, method="mean")["lik"].numpy()
    r_eff = ess / (n_chains * n_draws)
    return np.where(np.isfinite(r_eff) & (r_eff > 0), r_eff, 1.0)


def _good_k(n_samples):
    return min(1.0 - 1.0 / math.log10(n_samples), 0.7) if n_samples > 1 else 0.7


def loo(result, var_name=None, pointwise=False, reff=None):
    """Pareto smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).

    Parameters
    ----------
    result : SamplingResult (sampled with ``log_likelihood=True``), a dict
        mapping observed names to log-likelihood tensors of shape
        ``(chains, draws, *obs_shape)``, or such a tensor.
    var_name : the observed node to use. By default all observed nodes are
        used and their observations are concatenated.
    pointwise : also return ``elpd_i`` (and ``p_i``) per observation.
    reff : relative effective sample size of the draws, a scalar or one
        value per observation. By default it is estimated per observation as
        the relative mean ESS of the likelihood ``exp(log p(y_i | theta))``.

    Returns
    -------
    ELPDResult with ``elpd`` (elpd_loo), ``se``, ``p`` (p_loo) and
    ``pareto_k`` (k-hat per observation, always returned). k-hat above
    ``good_k = min(1 - 1/log10(S), 0.7)`` (S posterior draws) means that the
    importance sampling estimate of that observation is unreliable, and a
    warning is issued.
    """
    name, log_lik, shape, dtype, device = _log_likelihood_arrays(result, var_name)
    n_chains, n_draws, n_obs = log_lik.shape
    n_samples = n_chains * n_draws

    if reff is None:
        r_eff = relative_efficiency(log_lik)
    else:
        r_eff = np.broadcast_to(np.asarray(torch.as_tensor(reff).detach().cpu().numpy(), dtype=np.float64).reshape(-1), (n_obs,))

    flat = log_lik.reshape(n_samples, n_obs)
    elpd_i = np.empty(n_obs)
    k_hat = np.empty(n_obs)
    for i in range(n_obs):
        log_weights, k_hat[i] = psis(-flat[:, i], r_eff=float(r_eff[i]))
        elpd_i[i] = logsumexp(log_weights + flat[:, i])
    lppd_i = logsumexp(flat, axis=0) - math.log(n_samples)
    p_loo_i = lppd_i - elpd_i

    good_k = _good_k(n_samples)
    warning = False
    bad = ~(k_hat <= good_k)
    if np.any(bad):
        warning = True
        n_very_bad = int(np.sum(~(k_hat <= 0.7)))
        warn(
            f"Estimated Pareto shape k-hat exceeds {good_k:.2f} for {int(np.sum(bad))} of {n_obs} observations "
            f"({n_very_bad} above 0.7). The PSIS-LOO estimate is unreliable for them: the leave-one-out "
            "posteriors differ too much from the full posterior. Consider more draws, a more robust "
            "model or exact cross-validation for these observations.",
            stacklevel=2,
        )

    return ELPDResult(
        kind="loo",
        elpd=float(np.sum(elpd_i)),
        se=_se(elpd_i),
        p=float(np.sum(p_loo_i)),
        n_samples=n_samples,
        n_data_points=n_obs,
        elpd_i=_to_tensor(elpd_i, shape, dtype, device) if pointwise else None,
        pareto_k=_to_tensor(k_hat, shape, dtype, device),
        p_i=_to_tensor(p_loo_i, shape, dtype, device) if pointwise else None,
        good_k=good_k,
        warning=warning,
        var_name=name,
        _elpd_i_flat=elpd_i,
    )


# ----------------------------------------------------------------- compare
def _project_to_simplex(v):
    """Euclidean projection onto the probability simplex (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u) - 1.0
    index = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - cumulative / index > 0)[0][-1]
    theta = cumulative[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def stacking_weights(elpd_i, max_iter=10000, tol=1e-12):
    """Stacking weights of predictive distributions (Yao et al. 2018).

    Maximises ``sum_i log sum_k w_k exp(elpd_i[i, k])`` over the simplex with
    projected gradient ascent and a backtracking (Armijo) line search; the
    objective is concave, so this converges to the global optimum.
    ``elpd_i`` has shape ``(n_observations, n_models)``.
    """
    elpd_i = np.asarray(elpd_i, dtype=np.float64)
    n_obs, n_models = elpd_i.shape
    # Subtracting a per-observation constant leaves the maximiser unchanged.
    likelihood = np.exp(elpd_i - elpd_i.max(axis=1, keepdims=True))

    def objective(w):
        return np.sum(np.log(likelihood @ w))

    def gradient(w):
        return (likelihood / (likelihood @ w)[:, None]).sum(axis=0)

    w = np.full(n_models, 1.0 / n_models)
    value = objective(w)
    step = 1.0 / n_obs
    for _ in range(max_iter):
        g = gradient(w)
        while True:
            candidate = _project_to_simplex(w + step * g)
            with np.errstate(divide="ignore"):
                candidate_value = objective(candidate)
            if candidate_value >= value + 1e-4 * g @ (candidate - w) or step < 1e-20:
                break
            step *= 0.5
        change = np.max(np.abs(candidate - w))
        w, improvement, value = candidate, candidate_value - value, candidate_value
        step *= 2.0
        if change < tol or (0 <= improvement < tol * max(1.0, abs(value)) and change < 1e-9):
            break
    return w


def compare(results, ic="loo", method="stacking", var_name=None):
    """Compare models by their estimated expected log predictive density.

    Parameters
    ----------
    results : dict mapping a model name to a SamplingResult (with pointwise
        log-likelihood), or to an ``ELPDResult`` computed with ``pointwise=True``.
    ic : ``"loo"`` (PSIS-LOO) or ``"waic"``.
    method : model weights, ``"stacking"`` (Yao et al. 2018) or
        ``"pseudo-BMA"`` (Akaike-type weights ``exp(elpd_k)``, normalised).
    var_name : observed node passed to :func:`loo` / :func:`waic`.

    Returns
    -------
    pandas.DataFrame indexed by model name and sorted by ``elpd`` (best
    first), with the columns ``rank``, ``elpd``, ``p``, ``elpd_diff``
    (``elpd`` minus the ELPD of the best model, so zero or negative, as in
    ArviZ and the R package loo), ``weight``, ``se``, ``dse`` (standard
    error of the pointwise ELPD difference to the best model), ``warning``
    and ``good_k`` / ``n_bad_k`` for LOO.
    """
    import pandas as pd

    if not isinstance(results, dict) or len(results) < 2:
        raise ValueError("compare needs a dict with at least two models.")
    ic = ic.lower()
    if ic not in ("loo", "waic"):
        raise ValueError(f"ic must be 'loo' or 'waic', got {ic!r}.")
    if method.lower() not in ("stacking", "pseudo-bma"):
        raise ValueError(f"method must be 'stacking' or 'pseudo-BMA', got {method!r}.")

    estimates = {}
    for name, result in results.items():
        if isinstance(result, ELPDResult):
            if result.kind != ic:
                raise ValueError(f"The estimate of model '{name}' is {result.kind!r}, but ic={ic!r}.")
            if result._elpd_i_flat is None:
                raise ValueError(f"The estimate of model '{name}' holds no pointwise values.")
            estimates[name] = result
        else:
            function = loo if ic == "loo" else waic
            estimates[name] = function(result, var_name=var_name, pointwise=True)

    n_points = {name: e.n_data_points for name, e in estimates.items()}
    if len(set(n_points.values())) != 1:
        raise ValueError(f"All models must be evaluated on the same observations, but the numbers of observations differ: {n_points}.")

    names = sorted(estimates, key=lambda n: estimates[n].elpd, reverse=True)
    pointwise = np.stack([estimates[n]._elpd_i_flat for n in names], axis=1)

    if method.lower() == "stacking":
        weights = stacking_weights(pointwise)
    else:
        elpd = np.array([estimates[n].elpd for n in names])
        weights = np.exp(elpd - elpd.max())
        weights = weights / weights.sum()

    best = pointwise[:, 0]
    rows = []
    for rank, name in enumerate(names):
        estimate = estimates[name]
        difference = pointwise[:, rank] - best
        row = {
            "rank": rank,
            "elpd": estimate.elpd,
            "p": estimate.p,
            "elpd_diff": float(np.sum(difference)),
            "weight": float(weights[rank]),
            "se": estimate.se,
            "dse": _se(difference) if rank > 0 else 0.0,
            "warning": estimate.warning,
        }
        if ic == "loo":
            k = estimate.pareto_k.reshape(-1).cpu().numpy()
            row["n_bad_k"] = int(np.sum(~(k <= estimate.good_k)))
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(names, name="model"))
