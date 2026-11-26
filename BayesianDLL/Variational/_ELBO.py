"""Monte Carlo estimates of the evidence lower bound and its gradient.

Guide semantics
---------------
A :class:`Guide` is a product of independent pieces: Gaussian
:class:`Group` objects over unconstrained coordinates, and hand-written
entries. A hand-written entry is a ``RandomParameter`` with the name of a
model latent whose distribution has :class:`VariationalParameter`
arguments. What an entry's value means depends on the support of its
distribution:

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
:func:`estimate_elbo` evaluates the model's ``log p`` and its hand-written
gradient once per joint draw and sums the ``log q`` and log-Jacobians of all
pieces. Groups always use the reparameterisation trick. Hand-written guide
distributions with ``has_rsample`` use the reparameterisation trick;
the model gradient comes from the hand-written ``model.joint_grad_log_prob``
and is converted from the unconstrained to the guide variables with the
transform derivatives. Other guide distributions use the score-function
(REINFORCE) estimator with a baseline control variate (leave-one-out mean of
the other draws, or a running mean of earlier calls for ``n_samples=1``).
"""

import torch

from .._model import Model, _sum_to_shape
from ..Samplers._ancestral import draw_shaped


_BASELINE_DECAY = 0.9


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


def _draw_entry(name, distribution, n, shape):
    """Exact draws of the hand-written entry ``name``; a clear error when it has no sampler."""
    try:
        return draw_shaped(distribution, n, shape)
    except NotImplementedError:
        raise NotImplementedError(
            f"The guide entry '{name}' ({type(distribution).__name__}) has no exact sampler: its sample() is not "
            "implemented. A guide is only ever sampled exactly (never with MCMC), so without a sampler it cannot "
            "be fitted with score-function gradients, sampled, or summarised by Monte Carlo. Give the entry a "
            "distribution with a working sample()."
        ) from None


class HandWrittenEntries:
    """The hand-written ``RandomParameter`` entries of a :class:`Guide`, as one piece of the ELBO estimator.

    ``plan`` (a validated ``_Plan``) says how each entry maps onto the model:
    a real-line entry gives the latent's unconstrained value, a
    support-matched entry its constrained value, and a guide deterministic
    named like a model latent its constrained value as a function of one
    auxiliary entry. Entries with ``has_rsample`` use the reparameterisation
    trick; the others use the score function with the full ELBO sample as
    their learning signal (see :func:`estimate_elbo`). The entries are
    mutually independent and independent of the groups.
    """

    def __init__(self, guide, plan, model):
        self.guide = guide
        self.plan = plan
        self.model = model
        self._saved = {}

    def variational_parameters(self):
        table = {}
        for name in self.plan.active:
            for parameter_name, variational_parameter in self.guide.params[name].distribution.variational_parameters.items():
                table[f"{name}_{parameter_name}"] = variational_parameter
        return table

    def gradient_names(self):
        plan = self.plan
        return [
            plan.sources[name] if plan.modes[name] == "auxiliary" else name
            for name in plan.active if plan.reparametrized[name]
        ]

    def draw(self, n, reparametrized=True):
        """Draw ``n`` values of every entry (with the sample derivatives when ``reparametrized``)."""
        plan, guide = self.plan, self.guide
        self._draws, self._dz, self._scores = {}, {}, []
        for name in plan.active:
            parameter = guide.params[name]
            shape = parameter.constrained_value.shape
            if reparametrized and plan.reparametrized[name]:
                try:
                    self._draws[name], self._dz[name] = draw_shaped(parameter.distribution, n, shape, reparametrized=True)
                    continue
                except NotImplementedError:
                    plan.reparametrized[name] = False
            self._draws[name] = _draw_entry(name, parameter.distribution, n, shape)

    def to_model(self, i, jacobian=True):
        """Unconstrained model values of draw ``i`` and ``log|d eta / d u|``.

        The guide values stay set (for the guide deterministics) until :meth:`release`.
        """
        plan, guide, model = self.plan, self.guide, self.model
        u = {name: self._draws[name][i] for name in plan.active}
        self._u = u
        self._saved = _set_guide_values(guide, u) if plan.targets else {}
        eta = {}
        log_jacobian = 0.0
        self._chain = {}  # d/du log|f'(u)| of the deterministic targets
        for name in plan.active:
            mode = plan.modes[name]
            if mode == "unconstrained":
                eta[name] = u[name]
            elif mode == "constrained":
                transform = model.params[name].distribution.transform
                eta[name] = transform.forward(u[name])
                if jacobian:
                    log_jacobian = log_jacobian - transform.log_abs_det_jacobian(eta[name])
        for target, (deterministic, source) in plan.targets.items():
            transform = model.params[target].distribution.transform
            eta[target] = transform.forward(deterministic.constrained_value)
            if not jacobian:
                continue
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
                self._chain[source] = (logs[0] - logs[1]) / (2 * step)
        return eta, log_jacobian

    def in_model(self, i, model_grads):
        """Transform derivatives that need the model set to draw ``i`` (called inside the model context)."""
        plan, model = self.plan, self.model
        self._eta_derivatives = {}
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
            self._eta_derivatives[name] = (model_grads[target] - transform.grad_log_abs_det_jacobian(eta_value), jacobian)
        self._derivatives = {
            source: _derivative_of_source(deterministic, source, self._u[source])
            for deterministic, source in plan.targets.values()
        }

    def release(self, i):
        """Restore the guide values set by :meth:`to_model`."""
        if self._saved:
            _restore_guide_values(self.guide, self._saved)
        self._saved = {}

    def log_q(self, i):
        log_q = 0.0
        for name in self.plan.active:
            log_q = log_q + self.guide.params[name].distribution.log_pdf(self._u[name]).sum()
        return log_q

    def accumulate(self, i, model_grads, grads):
        """Add the pathwise gradients of draw ``i`` to ``grads``; record the score terms."""
        plan, u = self.plan, self._u
        for name in plan.active:
            distribution = self.guide.params[name].distribution
            param_grads = distribution.log_pdf_param_grads(u[name])
            keys = {f"{name}_{vp_name}": vp for vp_name, vp in distribution.variational_parameters.items()}
            if not plan.reparametrized[name]:
                self._scores.append((i, {key: _sum_to_shape(param_grads[vp.name], vp.value.shape) for key, vp in keys.items()}))
                continue
            mode = plan.modes[name]
            if mode == "unconstrained":
                grad_u = model_grads[name]
            elif mode == "constrained":
                outer, jacobian = self._eta_derivatives[name]
                grad_u = outer / jacobian
            else:
                outer, jacobian = self._eta_derivatives[name]
                grad_u = outer * self._derivatives[name] / jacobian + self._chain[name]
            grad_u = grad_u.reshape(u[name].shape) - distribution.log_pdf_grad(u[name])
            for key, vp in keys.items():
                grad = -_sum_to_shape(param_grads[vp.name], vp.value.shape)
                if vp.name in self._dz[name]:
                    grad = grad + _sum_to_shape(grad_u * self._dz[name][vp.name][i], vp.value.shape)
                grads[key] = grads[key] + grad

    def score_terms(self):
        """``[(i, {key: d log q / d parameter})]`` of the score-function entries."""
        return self._scores


def _prior_terms(model):
    """The latents whose prior terms form ``log p(theta)`` (latents inside an enumeration group excluded)."""
    grouped = model._enumeration_group_of
    return [parameter for name, parameter in model.params.items() if name not in grouped]


def estimate_elbo(model, pieces, n_samples, scales=None, baseline_owner=None, kl_weight=1.0):
    """Monte Carlo ELBO and its gradient for a guide made of independent ``pieces``.

    Every piece (a :class:`Group` or the :class:`HandWrittenEntries`)
    implements ``draw(n)``, ``to_model(i)`` (its unconstrained model values
    and log-Jacobian), ``log_q(i)`` and ``accumulate(i, model_grads, grads)``
    (its gradient step). For each joint draw the model's ``log p`` and its
    hand-written gradient are evaluated **once**, and the ``log q`` and
    log-Jacobians of all pieces are summed. Reparameterised pieces add their
    pathwise gradients; score-function entries use the full ELBO sample
    minus a baseline (leave-one-out mean of the other draws, or a running
    mean of earlier calls stored on ``baseline_owner`` when
    ``n_samples=1``) as their learning signal.

    **KL annealing.** With ``0 < kl_weight = w < 1`` the gradient is that of
    the annealed objective

    ``E_q[log p(y | theta) + w (log p(theta) + log|d eta / d u| - log q(u))]``,

    where ``log p(theta)`` is the sum of the latents' prior terms in the
    unconstrained space (an enumeration group's log marginal counts as
    likelihood) and ``log p(y | theta)`` everything else, including the
    minibatch scaling. It equals ``w`` times the ELBO of the model whose
    likelihood is raised to the power ``1 / w``, which is how it is computed:
    the pieces receive the tempered ``log p`` and its gradient
    ``grad log p(theta) + grad log p(y | theta) / w``, and their pathwise
    gradients are multiplied by ``w``; the score-function signal is the
    annealed sample. The prior gradient takes one extra reverse pass over the
    prior terms. The returned value is always the ELBO estimate
    (``kl_weight = 1``) of the same draws.
    """
    annealed = kl_weight != 1
    if annealed:
        if not 0 < kl_weight <= 1:
            raise ValueError(f"kl_weight must be in (0, 1], got {kl_weight}.")
        prior_parameters = _prior_terms(model)
        prior_names = {parameter.name for parameter in prior_parameters}
    grads = {}
    for piece in pieces:
        for key, parameter in piece.variational_parameters().items():
            if key in grads:
                raise ValueError(f"Two parts of the guide use the variational parameter name '{key}'.")
            grads[key] = torch.zeros_like(parameter.value)
    for piece in pieces:
        piece.draw(n_samples)
    names = []
    for piece in pieces:
        names += [name for name in piece.gradient_names() if name not in names]

    values, signals = [], []
    for i in range(n_samples):
        eta, log_jacobian = {}, 0.0
        try:
            for piece in pieces:
                values_i, jacobian_i = piece.to_model(i)
                eta.update(values_i)
                log_jacobian = log_jacobian + jacobian_i
            with model.temporarily_set_many(eta):
                log_p = scaled_log_joint(model, scales)
                model_grads = scaled_log_joint_grad(model, names, scales) if names else {}
                tempered = log_p
                if annealed:
                    # log p(theta) + log p(y | theta) / w and its gradient.
                    log_prior = model._log_prob_of_terms(prior_parameters, [])
                    tempered = log_prior + (log_p - log_prior) / kl_weight
                    if names:
                        prior_grads = model._reverse_pass(names, active_terms=prior_names)
                        model_grads = {
                            name: prior_grads[name] + (model_grads[name] - prior_grads[name]) / kl_weight
                            for name in names
                        }
                for piece in pieces:
                    piece.in_model(i, model_grads)
        finally:
            for piece in pieces:
                piece.release(i)
        log_q = 0.0
        for piece in pieces:
            log_q = log_q + piece.log_q(i)
        values.append(log_p + log_jacobian - log_q)
        signals.append(kl_weight * (tempered + log_jacobian - log_q) if annealed else values[-1])
        for piece in pieces:
            piece.accumulate(i, model_grads, grads)

    total = sum(values) / n_samples
    if annealed:
        for key in grads:
            grads[key] = kl_weight * grads[key]

    scores = [term for piece in pieces for term in piece.score_terms()]
    if scores:
        detached = torch.stack([torch.as_tensor(v).detach().reshape(()) for v in signals])
        if n_samples > 1:
            baselines = (detached.sum() - detached) / (n_samples - 1)
        else:
            previous = getattr(baseline_owner, "_reinforce_baseline", None)
            baselines = torch.full_like(detached, 0.0 if previous is None else previous)
            if baseline_owner is not None:
                baseline_owner._reinforce_baseline = detached.mean().item() if previous is None else (
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

    Returns ``(elbo, grads)`` with ``grads[key]`` the gradient of the ELBO (to
    be *maximised*) w.r.t. the VariationalParameter ``key`` of
    ``guide.variational_parameters()``. See the module docstring for how
    guide values map onto the model latents.
    """
    return guide._elbo(model, n_samples)
