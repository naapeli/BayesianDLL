"""Pointwise (elementwise) log-likelihood of the observed nodes of a model.

Used by :func:`BayesianDLL.sample` to record ``log p(y_i | theta)`` for every
kept draw, which :func:`~BayesianDLL.Evaluation.loo` and
:func:`~BayesianDLL.Evaluation.waic` need.
"""

import torch


def pointwise_log_likelihood_plan(model):
    """Decide how the pointwise log-likelihood of every observed node is computed.

    Returns ``(plan, unavailable)``. ``plan`` maps an observed node name to
    ``None`` (evaluate ``log_pdf`` of the node directly) or to the
    enumeration group whose per-coordinate log marginal is the pointwise
    term. ``unavailable`` maps the names of the nodes whose pointwise
    marginal log-likelihood cannot be computed to the reason.

    An observed node that depends on an enumerated (summed out) parameter
    ``z`` has the pointwise term ``log sum_k p(z_i = k) p(y_i | z_i = k,
    theta)``. This is available when the group of ``z`` is summed out
    coordinatewise (the ``"factorised"`` mode), the node is the group's only
    term besides the prior of ``z``, and the node's elementwise log density
    has the shape of ``z`` (one ``z_i`` per observation ``y_i``).
    """
    if not model._compiled:
        model.compile()
    group_of = model._enumeration_group_of
    plan, unavailable = {}, {}
    for name in model.observed_params:
        group = group_of.get(name)
        if group is None:
            plan[name] = None
            continue
        group.prepare(model)
        reason = None
        if group.mode != "factorised":
            reason = (
                f"it depends on the enumerated parameters {group.enumerated_names}, which are summed "
                "out jointly over all coordinates, so the marginal likelihood does not split into "
                "per-observation terms"
            )
        elif group.random_terms or group.observed_terms != [name]:
            others = [t for t in group.random_terms + group.observed_terms if t != name]
            reason = (
                f"the enumerated parameter {group.enumerated_names[0]!r} also affects {others}, so "
                "its marginal does not split into per-observation likelihood terms"
            )
        else:
            parameter = model.enumerated_params[group.enumerated_names[0]]
            observed = model.observed_params[name]
            term_shape = tuple(observed.distribution.log_pdf(observed.observed_values).shape)
            if term_shape != tuple(parameter.constrained_value.shape):
                reason = (
                    f"the enumerated parameter {parameter.name!r} of shape "
                    f"{tuple(parameter.constrained_value.shape)} is not one coordinate per observation "
                    f"of {name!r} (log density shape {term_shape})"
                )
        if reason is None:
            plan[name] = group
        else:
            unavailable[name] = f"The pointwise log-likelihood of '{name}' is unavailable: {reason}."
    return plan, unavailable


def pointwise_log_likelihood(model, plan):
    """Elementwise log-likelihood of every planned observed node at the model's current values."""
    values = {}
    n_chains = getattr(model, "n_chains", None)
    if n_chains is not None and model._batch_mode == "loop":
        return model._loop_over_chains(lambda: pointwise_log_likelihood(model, plan))
    for name, group in plan.items():
        if group is None:
            if hasattr(model, "_elementwise_term"):
                value = model._elementwise_term(name)
                if n_chains is not None and not model._term_is_batched(name):
                    value = value.unsqueeze(0).expand(n_chains, *value.shape)
                values[name] = value
            else:
                observed = model.observed_params[name]
                values[name] = observed.distribution.log_pdf(observed.observed_values)
        else:
            from .._batching import batched_evaluation
            with batched_evaluation(n_chains):
                log_terms = group.evaluate(model)["log_terms"]
            values[name] = torch.logsumexp(log_terms, dim=0)
    return values
