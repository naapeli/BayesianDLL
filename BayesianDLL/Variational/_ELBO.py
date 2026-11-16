"""Monte Carlo estimates of the evidence lower bound and its gradient.

Guide semantics
---------------
A guide (a :class:`MeanFieldGuide`) contains one ``RandomParameter`` per
latent of the model, with the same name, whose distribution has
:class:`VariationalParameter` arguments. What a guide value means depends on
the support of the guide distribution:

* A guide distribution on the whole real line (``Normal``, ``Cauchy``,
  ``Laplace``, ``StudentT``, ...) describes the latent's **unconstrained**
  value, i.e. the value the samplers work with (``log sigma`` for a positive
  ``sigma``). This is the common case.
* A guide distribution with a restricted support (``LogNormal``, ``Gamma``,
  ``Exponential``, ``Beta``, ...) describes the latent's **constrained**
  value. Its support must match the support of the model latent (e.g. a
  ``LogNormal`` guide for a ``HalfCauchy`` latent).
* A guide latent that is not a model latent is auxiliary. A guide
  ``DeterministicParameter`` named like a model latent gives that latent's
  constrained value as an elementwise function of one auxiliary latent,
  e.g. ``Exp("sigma", log_sigma)``.

In every case the ELBO is ``E_q[log p(theta, y) - log q(theta)]`` with both
densities taken with respect to the same variables: the guide variables
``u``. The model density is evaluated in the unconstrained space and
transformed with the Jacobian ``log|d eta / d u|`` of the map from ``u`` to
the unconstrained model value ``eta``, so ``log p`` and ``log q`` always
match.

Gradients
---------
Guide distributions with ``has_rsample`` use the reparameterisation trick;
the model gradient comes from the hand-written ``model.joint_grad_log_prob``
and is converted from the unconstrained to the guide variables with the
transform derivatives. Other guide distributions use the score-function
(REINFORCE) estimator with a baseline control variate (leave-one-out mean of
the other draws, or a running mean of earlier calls for ``n_samples=1``).
"""

import math

import torch

from .._model import Model, _sum_to_shape
from ..Distributions._state_space import ContinuousReal
from ..Samplers._ancestral import draw_shaped


_BASELINE_DECAY = 0.9


def variational_parameters(guide):
    """``{key: VariationalParameter}`` of a guide, with the keys used by :func:`elbo`."""
    if not isinstance(guide, Model) and hasattr(guide, "variational_parameters"):
        return dict(guide.variational_parameters())
    table = {}
    for name, parameter in guide.params.items():
        for parameter_name, variational_parameter in parameter.distribution.variational_parameters.items():
            table[f"{name}_{parameter_name}"] = variational_parameter
    return table


def scaled_log_joint(model, scales=None):
    """``log p`` in the unconstrained space with observed terms ``name`` scaled by ``scales[name]``."""
    log_p = model.model_log_prob()
    for name, scale in (scales or {}).items():
        observed = model.observed_params[name]
        log_p = log_p + (scale - 1) * observed.distribution.log_pdf(observed.observed_values).sum()
    return log_p


def scaled_log_joint_grad(model, names, scales=None):
    """Hand-written gradient of :func:`scaled_log_joint` w.r.t. the unconstrained ``names``."""
    grads = model.joint_grad_log_prob(list(names))
    if scales:
        by_scale = {}
        for name, scale in scales.items():
            by_scale.setdefault(scale, set()).add(name)
        for scale, terms in by_scale.items():
            extra = model._reverse_pass(list(names), active_terms=terms)
            for name in names:
                grads[name] = grads[name] + (scale - 1) * extra[name]
    return grads


def _is_real(distribution):
    return isinstance(distribution.state_space, ContinuousReal)


def _bounds_are_constant(transform):
    for attribute in ("border", "low", "high"):
        value = getattr(transform, attribute, None)
        if value is not None and not isinstance(value, int | float | torch.Tensor):
            return False
    for attribute in ("first", "second"):
        inner = getattr(transform, attribute, None)
        if inner is not None and not _bounds_are_constant(inner):
            return False
    return True


class _Plan:
    """How each guide latent maps onto the model."""

    def __init__(self, model, guide):
        if not model._compiled:
            model.compile()
        if not guide._compiled:
            guide.compile()
        guide_nodes = set(guide.params) | set(guide.deterministic_params) | set(guide.observed_params)
        enumerated = getattr(model, "enumerated_params", {})

        self.modes = {}
        for name, parameter in guide.params.items():
            distribution = parameter.distribution
            if set(distribution.parameters) & guide_nodes:
                raise NotImplementedError(
                    f"The guide distribution of '{name}' depends on other guide nodes. Only mean-field guides "
                    "(distributions whose arguments are VariationalParameters, constants or Data) are supported."
                )
            if name in enumerated:
                raise ValueError(f"'{name}' is enumerated (summed out) in the model and must not appear in the guide.")
            if name not in model.params:
                self.modes[name] = "auxiliary"
                continue
            latent = model.params[name]
            if _is_real(distribution) and latent.state_space.is_continuous():
                expected = latent.unconstrained_value.shape
                self.modes[name] = "unconstrained"
            else:
                if type(distribution.state_space) is not type(latent.state_space):
                    raise ValueError(
                        f"The guide distribution of '{name}' has support {type(distribution.state_space).__name__}, "
                        f"but the model latent has support {type(latent.state_space).__name__}. Use a guide with "
                        "the same support (it then describes the constrained value) or a guide on the real line "
                        "(it then describes the unconstrained value)."
                    )
                if not _bounds_are_constant(latent.distribution.transform):
                    raise NotImplementedError(
                        f"The support of '{name}' depends on other latents; use a guide on the real line for it."
                    )
                expected = latent.constrained_value.shape
                self.modes[name] = "constrained"
            if tuple(parameter.constrained_value.shape) != tuple(expected):
                raise ValueError(
                    f"The guide value of '{name}' has shape {tuple(parameter.constrained_value.shape)}, "
                    f"expected {tuple(expected)}."
                )

        # Guide deterministic nodes that give the constrained value of a model latent.
        self.targets = {}
        used_sources = set()
        for name, deterministic in guide.deterministic_params.items():
            if name not in model.params:
                continue
            sources = [getattr(i, "name", None) for i in deterministic.inputs if getattr(i, "name", None) in guide.params]
            if len(sources) != 1 or self.modes[sources[0]] != "auxiliary" or sources[0] in used_sources:
                raise NotImplementedError(
                    f"The guide deterministic '{name}' must be a function of exactly one auxiliary guide latent "
                    "that feeds no other model latent."
                )
            used_sources.add(sources[0])
            self.targets[name] = (deterministic, sources[0])
        self.sources = {source: target for target, (_, source) in self.targets.items()}

        covered = {n for n, mode in self.modes.items() if mode != "auxiliary"} | set(self.targets)
        missing = [name for name in model.params if name not in covered]
        if missing:
            raise ValueError(f"The guide has no distribution for the model latent(s) {missing}.")

        # Auxiliary latents that feed no model latent do not enter the ELBO.
        self.active = [n for n, mode in self.modes.items() if mode != "auxiliary" or n in self.sources]
        self.reparametrized = {
            name: bool(getattr(guide.params[name].distribution, "has_rsample", False))
            and guide.params[name].distribution.state_space.is_continuous()
            for name in self.active
        }


def _set_guide_values(guide, values):
    saved = {name: guide.params[name].unconstrained_value for name in values}
    for name, value in values.items():
        guide.params[name].constrained_value = value
        guide.params[name].unconstrained_value = guide.params[name].distribution.transform.forward(value)
    return saved


def _restore_guide_values(guide, saved):
    for name, value in saved.items():
        guide.params[name].set_unconstrained_value(value)


def _derivative_of_source(deterministic, source, like):
    derivative = deterministic.derivative(source)
    derivative = torch.as_tensor(derivative, dtype=like.dtype, device=like.device)
    if derivative.shape != like.shape:
        if derivative.numel() == like.numel() and derivative.ndim <= like.ndim:
            return derivative.reshape(like.shape)
        raise NotImplementedError(f"The guide deterministic '{deterministic.name}' must be an elementwise function of its latent.")
    return derivative


def _guide_elbo(model, guide, n_samples, scales=None):
    plan = _Plan(model, guide)
    table = variational_parameters(guide)

    draws, dz = {}, {}
    for name in plan.active:
        parameter = guide.params[name]
        shape = parameter.constrained_value.shape
        if plan.reparametrized[name]:
            try:
                draws[name], dz[name] = draw_shaped(parameter.distribution, n_samples, shape, reparametrized=True)
            except NotImplementedError:
                plan.reparametrized[name] = False
        if not plan.reparametrized[name]:
            draws[name] = draw_shaped(parameter.distribution, n_samples, shape)

    grads = {key: torch.zeros_like(value.value) for key, value in table.items()}
    reinforce = [name for name in plan.active if not plan.reparametrized[name]]
    gradient_names = [
        plan.sources[name] if plan.modes[name] == "auxiliary" else name
        for name in plan.active if plan.reparametrized[name]
    ]
    values = []
    scores = []

    for i in range(n_samples):
        u = {name: draws[name][i] for name in plan.active}
        saved = _set_guide_values(guide, u) if plan.targets else {}
        try:
            eta = {}
            log_jacobian = 0.0
            chain = {}  # d eta / d u (elementwise) and d log|d eta/du| / d u for reparametrized latents
            for name in plan.active:
                mode = plan.modes[name]
                if mode == "unconstrained":
                    eta[name] = u[name]
                elif mode == "constrained":
                    transform = model.params[name].distribution.transform
                    eta[name] = transform.forward(u[name])
                    log_jacobian = log_jacobian - transform.log_abs_det_jacobian(eta[name])
            for target, (deterministic, source) in plan.targets.items():
                transform = model.params[target].distribution.transform
                value = deterministic.constrained_value
                eta[target] = transform.forward(value)
                derivative = _derivative_of_source(deterministic, source, u[source])
                log_jacobian = log_jacobian + torch.log(derivative.abs()).sum() - transform.log_abs_det_jacobian(eta[target])
                if plan.reparametrized[source]:
                    # d/du log|f'(u)| by central differences of the hand-written
                    # elementwise derivative f'.
                    x = u[source]
                    step = torch.finfo(x.dtype).eps ** (1 / 3) * (1 + x.abs())
                    logs = []
                    for sign in (1, -1):
                        _set_guide_values(guide, {source: x + sign * step})
                        logs.append(torch.log(_derivative_of_source(deterministic, source, x).abs()))
                    _set_guide_values(guide, {source: x})
                    chain[source] = (logs[0] - logs[1]) / (2 * step)

            with model.temporarily_set_many(eta):
                log_p = scaled_log_joint(model, scales)
                model_grads = scaled_log_joint_grad(model, gradient_names, scales) if gradient_names else {}
                eta_derivatives = {}
                for name in plan.active:
                    if not plan.reparametrized[name] or plan.modes[name] == "unconstrained":
                        continue
                    target = name if plan.modes[name] == "constrained" else plan.sources[name]
                    transform = model.params[target].distribution.transform
                    eta_value = model.params[target].unconstrained_value
                    jacobian = transform.derivative(eta_value)
                    if jacobian.shape != eta_value.shape:
                        raise NotImplementedError(
                            f"Reparameterised gradients of a constrained-space guide for '{target}' need an elementwise transform."
                        )
                    eta_derivatives[name] = (model_grads[target] - transform.grad_log_abs_det_jacobian(eta_value), jacobian)
            if plan.targets:
                derivatives = {
                    source: _derivative_of_source(deterministic, source, u[source])
                    for deterministic, source in plan.targets.values()
                }
        finally:
            if saved:
                _restore_guide_values(guide, saved)

        log_q = 0.0
        for name in plan.active:
            log_q = log_q + guide.params[name].distribution.log_pdf(u[name]).sum()
        value = log_p + log_jacobian - log_q
        values.append(value)

        for name in plan.active:
            distribution = guide.params[name].distribution
            param_grads = distribution.log_pdf_param_grads(u[name])
            keys = {f"{name}_{vp_name}": vp for vp_name, vp in distribution.variational_parameters.items()}
            if not plan.reparametrized[name]:
                scores.append((i, {key: _sum_to_shape(param_grads[vp.name], vp.value.shape) for key, vp in keys.items()}))
                continue
            mode = plan.modes[name]
            if mode == "unconstrained":
                grad_u = model_grads[name]
            elif mode == "constrained":
                outer, jacobian = eta_derivatives[name]
                grad_u = outer / jacobian
            else:
                outer, jacobian = eta_derivatives[name]
                grad_u = outer * derivatives[name] / jacobian + chain[name]
            grad_u = grad_u.reshape(u[name].shape) - distribution.log_pdf_grad(u[name])
            for key, vp in keys.items():
                grad = -_sum_to_shape(param_grads[vp.name], vp.value.shape)
                if vp.name in dz[name]:
                    grad = grad + _sum_to_shape(grad_u * dz[name][vp.name][i], vp.value.shape)
                grads[key] = grads[key] + grad

    total = sum(values) / n_samples

    if reinforce:
        detached = torch.stack([torch.as_tensor(v).detach().reshape(()) for v in values])
        if n_samples > 1:
            baselines = (detached.sum() - detached) / (n_samples - 1)
        else:
            previous = getattr(guide, "_reinforce_baseline", None)
            baselines = torch.full_like(detached, 0.0 if previous is None else previous)
            guide._reinforce_baseline = detached.mean().item() if previous is None else (
                _BASELINE_DECAY * previous + (1 - _BASELINE_DECAY) * detached.mean().item()
            )
        for i, score in scores:
            advantage = detached[i] - baselines[i]
            for key, grad in score.items():
                grads[key] = grads[key] + advantage * grad

    for key in grads:
        grads[key] = grads[key] / n_samples
    return total, grads


def elbo(model: Model, guide, n_samples=1):
    """Monte Carlo ELBO and its gradient w.r.t. the guide's variational parameters.

    Returns ``(elbo, grads)`` with ``grads[f"{latent}_{variational_parameter}"]``
    the gradient of the ELBO (to be *maximised*). See the module docstring
    for how guide values map onto the model latents. ``guide`` is a
    :class:`MeanFieldGuide` or an automatic guide (``AutoNormal``,
    ``AutoMultivariateNormal``).
    """
    return _elbo(model, guide, n_samples)


def _elbo(model, guide, n_samples, scales=None):
    if not isinstance(guide, Model) and hasattr(guide, "_elbo"):
        return guide._elbo(model, n_samples, scales=scales)
    return _guide_elbo(model, guide, n_samples, scales=scales)
