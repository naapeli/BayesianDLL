"""Variational guides: the shared guide interface and hand-written guides.

A guide is the variational family ``q`` that :func:`BBVI` fits to a model's
posterior. There are two kinds, with the same interface (:class:`BaseGuide`):

* :class:`Guide`, written by hand: you choose each latent's variational
  family (``Normal``, ``LogNormal``, ``Gamma``, ``Beta``, ...).
* the automatic Gaussian guides :class:`AutoNormal` and
  :class:`AutoMultivariateNormal`, built from the model.
"""

from abc import ABC, abstractmethod

import torch

from .._model import Model
from ..Distributions._state_space import ContinuousReal
from ..Samplers._ancestral import draw_shaped
from ..Samplers._result import SamplingResult
from ._ELBO import _guide_elbo, _restore_guide_values, _set_guide_values


class BaseGuide(ABC):
    """Interface of every guide that :func:`BBVI` and :func:`elbo` accept."""

    @abstractmethod
    def variational_parameters(self):
        """``{key: VariationalParameter}`` optimised by :func:`BBVI`; the keys of the ELBO gradient."""

    @abstractmethod
    def _elbo(self, model, n_samples, scales=None):
        """Monte Carlo ``(elbo, grads)`` for ``model`` with observed terms scaled by ``scales``."""

    @abstractmethod
    def sample(self, n_samples=1000, n_chains=1):
        """Independent draws from the guide as a :class:`SamplingResult` in the model's constrained space."""


def trace_from_values(model, values, n_chains, n_samples):
    """A :class:`SamplingResult` from ``n_chains * n_samples`` dicts of unconstrained latent values.

    The trace holds constrained values of the latents and the values of the
    deterministic nodes, like the result of :func:`BayesianDLL.sample`.
    """
    trace = {name: [] for name in model.params}
    det_trace = {name: [] for name in model.deterministic_params}
    for value in values:
        with model.temporarily_set_many(value):
            for name, parameter in model.params.items():
                trace[name].append(parameter.constrained_value.clone())
            for name, parameter in model.deterministic_params.items():
                det_trace[name].append(torch.as_tensor(parameter.constrained_value).clone())

    def stack(draws):
        stacked = torch.stack(draws, dim=0)
        return stacked.reshape(n_chains, n_samples, *stacked.shape[1:])

    return SamplingResult(
        {name: stack(draws) for name, draws in trace.items() if draws},
        [0] * n_chains, [[] for _ in range(n_chains)], [[] for _ in range(n_chains)],
        deterministic_trace={name: stack(draws) for name, draws in det_trace.items() if draws},
    )


def trace_from_unconstrained(model, layout, draws):
    """A :class:`SamplingResult` from flat unconstrained draws of shape ``(n_chains, n_samples, D)``."""
    n_chains, n_samples = draws.shape[:2]
    values = (layout.unpack(flat) for flat in draws.reshape(n_chains * n_samples, layout.dim))
    return trace_from_values(model, values, n_chains, n_samples)


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
    """How each guide latent maps onto the model (built and checked by :meth:`Guide.validate`)."""

    def __init__(self, model, guide):
        if not model._compiled:
            model.compile()
        if not guide._compiled:
            guide.compile()
        if guide.enumerated_params:
            raise ValueError(
                f"The guide contains the enumerated latent(s) {list(guide.enumerated_params)}; "
                "a guide cannot sum out latents."
            )
        guide_nodes = set(guide.params) | set(guide.deterministic_params) | set(guide.observed_params)
        enumerated = model.enumerated_params

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


class Guide(Model, BaseGuide):
    """A hand-written mean-field guide: you choose each latent's variational family.

    Build it like a model, with one ``RandomParameter`` per model latent (same
    name) whose distribution takes :class:`VariationalParameter` arguments::

        with Guide() as guide:
            RandomParameter("sigma", LogNormal(VariationalParameter("m", 0.0), VariationalParameter("s", 1.0, min=1e-3)))

    Any family with the support of the latent describes its constrained value
    (``LogNormal``, ``Gamma``, ``Exponential`` or ``Beta`` guides, which
    :class:`AutoNormal` cannot express); a family on the real line describes
    its unconstrained value. A guide ``DeterministicParameter`` named like a
    model latent gives that latent's constrained value as an elementwise
    function of one auxiliary guide latent, e.g. ``Exp("sigma", log_sigma)``.
    The guide must be mean-field (no guide distribution depends on another
    guide node) and must not contain enumerated latents.

    ``model`` is the model the guide approximates. It can be omitted: the
    guide is bound to the model it is checked against by :meth:`validate`,
    which :func:`BBVI` and :func:`elbo` call.
    """

    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def variational_parameters(self):
        """``{f"{latent}_{name}": VariationalParameter}`` of every guide distribution."""
        table = {}
        for name, parameter in self.params.items():
            for parameter_name, variational_parameter in parameter.distribution.variational_parameters.items():
                table[f"{name}_{parameter_name}"] = variational_parameter
        return table

    def validate(self, model=None):
        """Check the guide against ``model`` (default: the bound model), bind it and return the plan.

        Raises ``ValueError`` when a model latent has no guide distribution,
        when a guide support matches neither the latent's support nor the
        real line, or when the guide contains enumerated latents, and
        ``NotImplementedError`` for a guide that is not mean-field.
        """
        model = self.model if model is None else model
        if model is None:
            raise ValueError("The guide is not bound to a model: pass it as Guide(model) or to validate(model).")
        plan = _Plan(model, self)
        self.model = model
        return plan

    def _elbo(self, model, n_samples, scales=None):
        return _guide_elbo(model, self, self.validate(model), n_samples, scales=scales)

    def sample(self, n_samples=1000, n_chains=1):
        """Draw ``n_chains * n_samples`` independent draws from the guide (no MCMC).

        Every guide latent is drawn from its distribution's ``sample`` and
        mapped onto the model; the result is a :class:`SamplingResult` in the
        model's constrained space, including the model's deterministic nodes.
        """
        plan = self.validate()
        model = self.model
        total = n_chains * n_samples
        draws = {
            name: draw_shaped(self.params[name].distribution, total, self.params[name].constrained_value.shape)
            for name in plan.active
        }
        values = []
        for i in range(total):
            u = {name: draws[name][i] for name in plan.active}
            eta = {}
            for name in plan.active:
                if plan.modes[name] == "unconstrained":
                    eta[name] = u[name]
                elif plan.modes[name] == "constrained":
                    eta[name] = model.params[name].distribution.transform.forward(u[name])
            if plan.targets:
                saved = _set_guide_values(self, u)
                try:
                    for target, (deterministic, _) in plan.targets.items():
                        eta[target] = model.params[target].distribution.transform.forward(deterministic.constrained_value)
                finally:
                    _restore_guide_values(self, saved)
            values.append(eta)
        return trace_from_values(model, values, n_chains, n_samples)
