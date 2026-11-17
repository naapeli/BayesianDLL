"""Posterior predictive check statistics."""

from dataclasses import dataclass

import torch


def _flat(y):
    return y.reshape(y.shape[0], -1)


_STATISTICS = {
    "mean": lambda y: _flat(y).mean(dim=1),
    "std": lambda y: _flat(y).std(dim=1),
    "min": lambda y: _flat(y).min(dim=1).values,
    "max": lambda y: _flat(y).max(dim=1).values,
    "median": lambda y: _flat(y).quantile(0.5, dim=1),
    "n_zeros": lambda y: (_flat(y) == 0).sum(dim=1).to(torch.float64),
    "skew": lambda y: _skew(_flat(y)),
    "q05": lambda y: _flat(y).quantile(0.05, dim=1),
    "q95": lambda y: _flat(y).quantile(0.95, dim=1),
}


def _skew(y):
    centred = y - y.mean(dim=1, keepdim=True)
    m2 = (centred ** 2).mean(dim=1)
    m3 = (centred ** 3).mean(dim=1)
    return m3 / m2 ** 1.5


@dataclass
class PPCResult:
    """Result of :func:`ppc_pvalue`.

    ``p_value`` is the share of replicated datasets with ``T(y_rep) >=
    T(y)``; ``t_rep`` holds ``T(y_rep)`` per replicated dataset and
    ``t_observed`` is ``T(y)``.
    """

    statistic: str
    var_name: str
    p_value: float
    t_observed: float
    t_rep: torch.Tensor

    def __repr__(self):
        return (
            f"PPCResult(statistic={self.statistic!r}, var_name={self.var_name!r}, p_value={self.p_value:.3f}, "
            f"t_observed={self.t_observed:.4g}, n_replications={self.t_rep.numel()})"
        )


def _replicated_datasets(predictive_result, observed, var_name):
    samples = predictive_result.samples if hasattr(predictive_result, "samples") else predictive_result
    if isinstance(samples, torch.Tensor):
        samples = {"y": samples}
    if var_name is None:
        if len(samples) != 1:
            raise ValueError(f"The predictive result holds several variables {list(samples)}; choose one with var_name.")
        var_name = next(iter(samples))
    if var_name not in samples:
        raise KeyError(f"'{var_name}' is not in the predictive result. Available: {list(samples)}.")
    replicated = samples[var_name]
    if isinstance(observed, dict):
        observed = observed[var_name]
    observed = torch.as_tensor(observed, device=replicated.device)

    # Replicated datasets have shape (n_samples, samples_per_step, *obs_shape);
    # every (parameter draw, replication) pair is one replicated dataset.
    extra = replicated.ndim - observed.ndim
    if extra == 2:
        replicated = replicated.flatten(0, 1)
    elif extra != 1:
        raise ValueError(
            f"The replicated data of '{var_name}' have shape {tuple(replicated.shape)}, which does not "
            f"match observations of shape {tuple(observed.shape)} with one or two leading sample dimensions."
        )
    if tuple(replicated.shape[1:]) != tuple(observed.shape):
        if replicated[0].numel() != observed.numel():
            raise ValueError(f"Replicated datasets of shape {tuple(replicated.shape[1:])} do not match the observations of shape {tuple(observed.shape)}.")
        replicated = replicated.reshape(replicated.shape[0], *observed.shape)
    return var_name, replicated, observed


def ppc_pvalue(predictive_result, observed, statistic="mean", var_name=None):
    """Posterior predictive p-value ``P(T(y_rep) >= T(y) | y)`` of a test statistic.

    Parameters
    ----------
    predictive_result : PredicativeResult (e.g. from
        :func:`~BayesianDLL.posterior_predicative`), whose samples have shape
        ``(n_samples, samples_per_step, *obs_shape)``. Every replication is
        treated as one replicated dataset.
    observed : the observed data ``y`` (a tensor, or a dict by variable name).
    statistic : a callable mapping one dataset (tensor of the observation
        shape) to a scalar, or one of ``"mean"``, ``"std"`` (with Bessel's
        correction), ``"min"``, ``"max"``, ``"median"``, ``"n_zeros"``,
        ``"skew"`` (moment coefficient of skewness), ``"q05"``, ``"q95"``.
    var_name : the predicted variable; required when there are several.

    Returns
    -------
    PPCResult with ``p_value``, ``t_observed`` and ``t_rep``. P-values close
    to 0 or 1 indicate that the model does not reproduce the statistic.
    """
    var_name, replicated, observed = _replicated_datasets(predictive_result, observed, var_name)
    if isinstance(statistic, str):
        if statistic not in _STATISTICS:
            raise ValueError(f"Unknown statistic {statistic!r}. Use a callable or one of {list(_STATISTICS)}.")
        function = _STATISTICS[statistic]
        dtype = replicated.dtype if torch.is_floating_point(replicated) else torch.float64
        t_rep = function(replicated.to(dtype))
        t_observed = function(observed.to(dtype).unsqueeze(0))[0]
        name = statistic
    elif callable(statistic):
        t_rep = torch.stack([torch.as_tensor(statistic(dataset)) for dataset in replicated]).reshape(-1)
        t_observed = torch.as_tensor(statistic(observed)).reshape(())
        name = getattr(statistic, "__name__", "statistic")
    else:
        raise TypeError("statistic must be a callable or the name of a built-in statistic.")
    t_rep = t_rep.to(torch.float64)
    t_observed = float(t_observed)
    p_value = float((t_rep >= t_observed).to(torch.float64).mean())
    return PPCResult(statistic=name, var_name=var_name, p_value=p_value, t_observed=t_observed, t_rep=t_rep)
