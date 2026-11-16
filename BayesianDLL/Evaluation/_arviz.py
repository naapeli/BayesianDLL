"""Conversion of sampling results to ArviZ."""

import torch


_SAMPLE_STAT_NAMES = {
    "energy": "energy",
    "diverging": "diverging",
    "tree_depth": "tree_depth",
    "step_size": "step_size",
    "accept_stat": "acceptance_rate",
    "lp": "lp",
    "n_leapfrog": "n_steps",
}


def _numpy(value):
    return torch.as_tensor(value).detach().cpu().numpy()


def _sample_stats(sample_stats):
    """Map the per-block sampler statistics to ArviZ's sample_stats names.

    A statistic recorded by one block keeps the ArviZ name; when several
    blocks record it, ``diverging`` is combined (any block diverged) and the
    others get a ``_block{i}`` suffix.
    """
    blocks = getattr(sample_stats, "blocks", [])
    stats = {}
    for name, arviz_name in _SAMPLE_STAT_NAMES.items():
        recording = [(i, block["stats"][name]) for i, block in enumerate(blocks) if name in block["stats"]]
        if not recording:
            continue
        if name == "diverging":
            stats[arviz_name] = _numpy(torch.stack([v.bool() for _, v in recording]).any(dim=0))
        elif len(recording) == 1:
            value = recording[0][1]
            stats[arviz_name] = _numpy(value.long() if name == "tree_depth" else value)
        else:
            for i, value in recording:
                stats[f"{arviz_name}_block{i}"] = _numpy(value)
    return stats


def _dims(name, shape):
    return [f"{name}_dim_{i}" for i in range(len(shape))]


def to_arviz(result, predictive=None, observed=None):
    """Convert a SamplingResult to an ArviZ ``DataTree`` (``InferenceData`` for ArviZ < 1).

    Parameters
    ----------
    result : SamplingResult. Its random (and enumerated) parameters and
        deterministic nodes form the ``posterior`` group, the sampler
        statistics the ``sample_stats`` group (``energy``, ``diverging``,
        ``tree_depth``, ``step_size``, ``acceptance_rate``, ``lp``,
        ``n_steps``) and the pointwise log-likelihood the ``log_likelihood``
        group.
    predictive : optional PredicativeResult, stored as
        ``posterior_predictive`` with one chain whose draws are all
        replicated datasets (shape ``(1, n_samples * samples_per_step,
        *obs_shape)``).
    observed : optional observed data, a dict by observed variable name or a
        tensor when the model has a single observed variable; stored as
        ``observed_data``.

    ArviZ is imported here only; it is not a dependency of BayesianDLL.
    """
    try:
        import arviz as az
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("to_arviz requires ArviZ: pip install arviz") from error

    groups = {}
    dims = {}

    posterior = {name: _numpy(value) for name, value in result.trace.items()}
    for name, value in result.deterministic_trace.items():
        posterior.setdefault(name, _numpy(value))
    groups["posterior"] = posterior

    sample_stats = _sample_stats(result.sample_stats)
    if sample_stats:
        groups["sample_stats"] = sample_stats

    log_likelihood = {name: _numpy(value) for name, value in getattr(result, "log_likelihood", {}).items()}
    if log_likelihood:
        groups["log_likelihood"] = log_likelihood
        for name, value in log_likelihood.items():
            dims[name] = _dims(name, value.shape[2:])

    if predictive is not None:
        samples = predictive.samples if hasattr(predictive, "samples") else predictive
        posterior_predictive = {}
        for name, value in samples.items():
            value = torch.as_tensor(value)
            value = value.reshape(1, value.shape[0] * value.shape[1], *value.shape[2:])
            posterior_predictive[name] = _numpy(value)
            dims.setdefault(name, _dims(name, value.shape[2:]))
        groups["posterior_predictive"] = posterior_predictive

    if observed is not None:
        if not isinstance(observed, dict):
            names = list(log_likelihood) or (list(groups.get("posterior_predictive", {})))
            if len(names) != 1:
                raise ValueError("Pass observed as a dict {name: values} when the model has several (or no recorded) observed variables.")
            observed = {names[0]: observed}
        observed_data = {}
        for name, value in observed.items():
            value = _numpy(getattr(value, "value", value))
            observed_data[name] = value
            dims.setdefault(name, _dims(name, value.shape))
        groups["observed_data"] = observed_data

    major = int(az.__version__.split(".")[0])
    if major >= 1:
        return az.from_dict(groups, dims=dims)
    return az.from_dict(**groups, dims=dims)  # pragma: no cover - ArviZ 0.x API
