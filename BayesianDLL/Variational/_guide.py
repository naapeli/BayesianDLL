"""Variational guides: :class:`Guide`, a product of Gaussian groups and hand-written entries."""

from abc import ABC, abstractmethod

import torch

from .._model import _GUIDE_CONDITION_MESSAGE, Model
from .._parameters import RandomParameter
from ..Distributions._state_space import ContinuousReal
from ..Distributions._transforms import IdentityTransform
from ..Samplers._result import SamplingResult
from ._ELBO import HandWrittenEntries, estimate_elbo
from ._groups import FAMILIES, Group, _discrete_message
from ._structured import StructuredPiece
from ._moments import GuideCovariance, mc_moments, transform_batch


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
    """How each hand-written guide entry maps onto the model (built and checked by :meth:`Guide.validate`).

    ``covered`` are the model latents that the guide's groups cover.
    """

    def __init__(self, model, guide, covered=(), exclude=()):
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
            if name in exclude:
                continue
            distribution = parameter.distribution
            if set(distribution.parameters) & guide_nodes:
                raise NotImplementedError(
                    f"The guide distribution of '{name}' depends on other guide nodes. Only mean-field guides "
                    "(distributions whose arguments are VariationalParameters, constants or Data) are supported."
                )
            if name in enumerated:
                raise ValueError(f"'{name}' is enumerated (summed out) in the model and must not appear in the guide.")
            if name in model.observed_params:
                raise ValueError(
                    f"'{name}' is observed in the model (e.g. fixed by condition()); the guide must not cover it. "
                    "Remove its guide entry."
                )
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
            if name not in model.params or name in exclude:
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

        mine = {n for n, mode in self.modes.items() if mode != "auxiliary"} | set(self.targets)
        missing = [name for name in model.params if name not in mine and name not in covered]
        if missing:
            raise ValueError(f"The guide has no distribution for the model latent(s) {missing}.")
        # The model latents described by the entries, in model order.
        self.latents = [name for name in model.params if name in mine]

        # Auxiliary latents that feed no model latent do not enter the ELBO.
        self.active = [n for n, mode in self.modes.items() if mode != "auxiliary" or n in self.sources]
        self.reparametrized = {
            name: bool(getattr(guide.params[name].distribution, "has_rsample", False))
            and guide.params[name].distribution.state_space.is_continuous()
            for name in self.active
        }


def _distribution_moments(distribution, shape):
    """Closed-form ``(flat mean, covariance matrix)`` of a guide distribution whose draws have ``shape``.

    Raises ``NotImplementedError`` or ``ValueError`` when there is no closed form.
    """
    shape = tuple(shape)
    mean = torch.as_tensor(distribution.mean())
    covariance = getattr(distribution, "covariance", None)
    if callable(covariance) and len(distribution.event_shape) == 1:
        matrix = covariance()
        k = matrix.shape[-1]
        mean = mean.expand(shape)
        blocks = matrix.expand(*shape[:-1], k, k).reshape(-1, k, k)
        return mean.reshape(-1), torch.block_diag(*blocks)
    variance = torch.as_tensor(distribution.variance())
    mean = mean.expand(shape).reshape(-1)
    variance = variance.to(mean.dtype).expand(shape).reshape(-1)
    return mean, torch.diag(variance)


class Guide(Model, BaseGuide):
    """The variational family ``q`` for :func:`BBVI`: independent Gaussian groups and hand-written entries.

    A guide is bound to the model it approximates::

        guide = Guide(model, default_family="full_rank")    # one full-rank MVN over the whole model
        guide = Guide(model, default_family="mean_field")   # an independent Normal per coordinate

        with Guide(model, default_family="mean_field") as guide:
            Group(["mu", "beta"], family="full_rank")         # dependent latents share a group
            RandomParameter("sigma", LogNormal(VariationalParameter("sigma_loc", 0.0),
                                               VariationalParameter("sigma_scale", 0.5, transform=LogTransform())))
        BBVI(model, guide, ...)
        trace = guide.sample(1000)

    The ``with`` block is only needed to add :class:`Group` objects or
    hand-written entries. The joint ``q`` is the product of independent
    pieces:

    * **Groups** (:class:`Group`): Gaussians over the unconstrained values of
      their latents, mean-field or full-rank, or structured groups
      (``with Group([...]):``) whose hand-written latents may depend on each
      other within the group.
    * **Hand-written entries**: a ``RandomParameter`` named like a model
      latent whose distribution has :class:`VariationalParameter` arguments.
      A family on the real line describes the latent's unconstrained value,
      a family with the latent's support (``LogNormal``, ``Gamma``,
      ``Beta``, a discrete family, ...) its constrained value. A guide
      ``DeterministicParameter`` named like a model latent gives that
      latent's constrained value as an elementwise function of one
      auxiliary entry, e.g. ``Exp("sigma", log_sigma)``. Entries without a
      reparameterised sampler (discrete or ``Gamma`` entries, mixtures) are
      fitted with score-function gradients. Entries must not depend on other
      guide nodes and must not be enumerated.
    * **The default group**: with ``default_family="mean_field"`` or
      ``"full_rank"``, every latent that no group or entry covers is
      collected into one group of that family (created at the first
      :meth:`validate`, and on :attr:`groups`). When it covers the whole
      model its VariationalParameters are unprefixed (``{latent}_loc`` /
      ``{latent}_scale``, or ``loc`` / ``scale_tril``); otherwise they are
      prefixed with ``default_``. With ``default_family=None`` (the default)
      an uncovered latent is an error.

    Every latent belongs to exactly one group or entry. ``init_scale`` is
    the initial standard deviation of the default group's coordinates.

    A guide cannot be conditioned or marginalised (``condition(guide, ...)``
    raises ``TypeError``): to fix a latent to a known value, condition the
    model, ``Guide(condition(model, tau=0.5), ...)``; the conditioned latent
    is then observed and the guide must not cover it.

    Moments
    -------
    :meth:`mean` and :meth:`covariance` (and the same methods of every
    group) give the moments of ``q`` in the ``"unconstrained"`` or the
    ``"constrained"`` space. Pieces of the guide are independent, so the
    covariance is block diagonal: every block between two groups or entries
    is exactly zero. Within a block:

    ==============================  =====================  =====================================
    piece                           unconstrained space    constrained space
    ==============================  =====================  =====================================
    Gaussian group                  exact: ``loc``,        identity transform: exact;
                                    ``diag(scale^2)`` or   elementwise ``log``: exact log-normal
                                    ``L L^T``              moments (full-rank: exact
                                                           cross-covariances too); any other
                                                           transform (logit, stick-breaking,
                                                           ...): Monte Carlo
    real-line entry                 exact if the family    exact if the latent's transform is
                                    has closed-form        the identity and the moments are
                                    moments, else MC       closed-form, else MC
    support-matched entry           exact if the           exact if the family has closed-form
                                    transform is the       moments, else MC
                                    identity, else MC
    deterministic (auxiliary) entry Monte Carlo            Monte Carlo
    structured group                Monte Carlo            Monte Carlo
    ==============================  =====================  =====================================

    The result lists the Monte Carlo blocks in ``estimated`` and their
    standard errors in ``standard_error``.

    Accuracy and cost
    -----------------
    * **Exact Gaussian moments, unconstrained space.** No sampling error.
      The cost is ``O(d)`` for a mean-field group and ``O(d^2)`` for a
      full-rank group (forming ``L L^T``): essentially free.
    * **Exact closed forms, constrained space** (log-normal moments
      ``exp(mu + s^2/2)``, ``(exp(s^2) - 1) exp(2 mu + s^2)`` and
      ``cov(e^x_i, e^x_j) = exp(mu_i + mu_j + (S_ii + S_jj)/2) (exp(S_ij) - 1)``,
      and the closed-form moments of hand-written distributions). No
      sampling error, ``O(d)`` or ``O(d^2)``: also essentially free. They
      are exact moments of the approximating distribution ``q``, not of the
      true posterior; how close ``q`` is to the posterior is a separate
      question, which the ELBO and a comparison with MCMC answer.
    * **Monte Carlo.** The error of an estimated mean is ``O(sd / sqrt(n))``
      per coordinate (reported as ``standard_error``); the relative error
      of an estimated variance is about ``sqrt(2 / n)`` for Gaussian-like
      draws and larger for heavy-tailed transforms (e.g. ``exp`` of a wide
      Normal). The cost is ``n`` draws times (the transform cost plus
      ``O(d^2)`` for accumulating the covariance), linear in ``n``. The
      default ``n = 4000`` gives about 1.6% relative standard error on a
      mean in units of its sd (``1 / sqrt(4000)``) and about 2.2% on a
      variance (``sqrt(2 / 4000)``). The draws come from a local generator
      seeded with ``seed``, so the global random state is untouched and the
      result is reproducible.
    * **The draws are exact and independent.** Every group is sampled
      exactly (``z = loc + L eps`` for full-rank, independent Normals for
      mean-field), every hand-written entry with its distribution's own
      ``sample()`` and every structured group by ancestral sampling in
      topological order; the guide is never sampled with MCMC (an entry without a
      working ``sample()`` raises an error). So ``n`` draws are ``n``
      effective samples, with no autocorrelation and no warm-up. Strong
      correlation inside a full-rank group does not change this, because
      it lives in ``L``; its only consequence is that covariance estimates
      along near-singular directions are noisier. This is unlike
      :func:`~BayesianDLL.sample` on the posterior, which runs MCMC and
      gives autocorrelated draws whose effective sample size is smaller
      than their number.
    * **Constrained vs unconstrained.** The unconstrained moments describe
      ``q`` directly. Constrained moments are moments of the transformed
      variable and generally cannot be obtained by transforming the
      unconstrained mean: ``exp(E[x]) != E[exp(x)]``. For a log transform
      ``exp(E[x])`` is the median of the log-normal, and it underestimates
      the mean by the factor ``exp(-s^2/2)`` (Jensen's inequality); a logit
      transform pulls ``sigmoid(E[x])`` away from the mean towards the
      nearer boundary.
    * **When to use which.** Use the unconstrained moments to diagnose the
      optimisation (the scales and correlations that BBVI actually fits,
      e.g. to see whether a full-rank group is needed). Use the
      constrained moments to report the approximation in the model's own
      units. For anything else (quantiles, intervals, functions of several
      latents, deterministic nodes) draw ``guide.sample(n)`` and use
      :func:`~BayesianDLL.Evaluation.summary`: it covers every case with the
      same exact, independent draws.
    """

    _is_guide = True

    def condition(self, *args, **kwargs):
        """Guides cannot be conditioned: condition the model instead (raises ``TypeError``)."""
        raise TypeError(_GUIDE_CONDITION_MESSAGE)

    def marginalize(self, *args, **kwargs):
        """Guides cannot be marginalised (raises ``TypeError``)."""
        raise TypeError(_GUIDE_CONDITION_MESSAGE)

    def __init__(self, model=None, default_family=None, init_scale=0.1):
        if not isinstance(model, Model) or isinstance(model, Guide):
            raise TypeError(
                "Guide(model, default_family=...) needs the model it approximates as its first argument, "
                f"got {type(model).__name__}."
            )
        if default_family is not None and default_family not in FAMILIES:
            raise ValueError(f"default_family must be None, 'mean_field' or 'full_rank', got {default_family!r}.")
        super().__init__()
        if not model._compiled:
            model.compile()
        self.model = model
        self.default_family = default_family
        self.init_scale = float(init_scale)
        self._groups = []
        self._default_group = None
        self._reinforce_baseline = None

    # --- groups -------------------------------------------------------------
    def _structured_nodes(self):
        return {name for group in self._groups for name in group.nodes}

    def _hand_written_latents(self, model):
        """Model latents described by hand-written entries outside structured groups."""
        owned = self._structured_nodes()
        return {
            name for name in list(self.params) + list(self.deterministic_params)
            if name in model.params and name not in owned
        }

    def _group_names(self, params):
        """Check the latents of a new :class:`Group` and return their names."""
        if isinstance(params, str | RandomParameter):
            params = [params]
        names = []
        for parameter in params:
            if isinstance(parameter, RandomParameter):
                name = parameter.name
            elif isinstance(parameter, str):
                name = parameter
            else:
                raise TypeError(f"Group params must be latent names or RandomParameter objects, got {type(parameter).__name__}.")
            if name in names:
                raise ValueError(f"'{name}' appears twice in the Group.")
            names.append(name)
        if not names:
            raise ValueError("A Group needs at least one latent.")
        model = self.model
        for name in names:
            if name in model.enumerated_params:
                raise ValueError(f"'{name}' is enumerated (summed out) in the model and must not be in a Group.")
            if name in model.observed_params:
                raise ValueError(
                    f"'{name}' is observed in the model (e.g. fixed by condition()); the guide must not cover it."
                )
            if name not in model.params:
                raise ValueError(f"'{name}' is not a latent of the model; its latents are {list(model.params)}.")
            if not model.params[name].distribution.transformed_state_space.is_continuous():
                raise ValueError(_discrete_message(name, "Groups"))
            for group in self._groups:
                if name in group.names:
                    raise ValueError(
                        f"'{name}' is already in the group '{group.name}'; every latent belongs to exactly one group "
                        "or hand-written entry."
                    )
            if name in self._hand_written_latents(model):
                raise ValueError(f"'{name}' already has a hand-written guide entry; it cannot also be in a Group.")
        return names

    def _register_group(self, group):
        if any(other.name == group.name for other in self._groups):
            raise ValueError(f"A group named '{group.name}' already exists in this guide.")
        self._groups.append(group)

    def _finalize(self, model):
        """Check the coverage of ``model``'s latents and (re)build the default group; returns the pieces."""
        if not model._compiled:
            model.compile()
        for group in self._groups:
            if group.family != "structured":
                group.check(model)
            elif not group.nodes:
                raise ValueError(f"The structured group '{group.name}' defines no guide latents.")
        in_groups = {name: group for group in self._groups for name in group.names}
        hand_written = self._hand_written_latents(model)
        overlap = sorted(hand_written & set(in_groups))
        if overlap:
            raise ValueError(
                f"The latent(s) {overlap} have a hand-written guide entry and are also in a Group; every latent "
                "belongs to exactly one group or hand-written entry."
            )
        uncovered = [name for name in model.params if name not in in_groups and name not in hand_written]
        if uncovered and self.default_family is None:
            raise ValueError(
                f"The guide has no distribution for the model latent(s) {uncovered}. Put them in a Group, write "
                "hand-written entries for them inside `with Guide(model) as guide:`, or pass "
                "default_family='mean_field' or 'full_rank' to collect them into one default group."
            )
        if not uncovered:
            self._default_group = None
        else:
            default = self._default_group
            whole = len(uncovered) == len(model.params)
            if (default is None or default.names != uncovered or default.model is not model
                    or default.family != self.default_family or (default.prefix == "") != whole):
                self._default_group = Group._default(model, uncovered, self.default_family, self.init_scale, prefixed=not whole)
        pieces = [StructuredPiece(group, self, model) if group.family == "structured" else group for group in self._groups]
        if self._default_group is not None:
            pieces.append(self._default_group)
        owned = self._structured_nodes()
        if any(name not in owned for name in list(self.params) + list(self.deterministic_params)):
            plan = _Plan(model, self, covered=set(in_groups) | set(uncovered), exclude=owned)
            if plan.active:
                pieces.append(HandWrittenEntries(self, plan, model))
        return pieces

    @property
    def groups(self):
        """The groups: the explicit ones in creation order, then the default group (if any)."""
        self._finalize(self.model)
        return list(self._groups) + ([self._default_group] if self._default_group is not None else [])

    def validate(self, model=None):
        """Check the guide against ``model`` (default: the bound model), bind it and return its pieces.

        Raises ``ValueError`` when a model latent has no group or entry (and
        ``default_family`` is ``None``), when a latent is in two pieces, when
        a hand-written support matches neither the latent's support nor the
        real line, or when the guide contains enumerated latents, and
        ``NotImplementedError`` for a hand-written entry that depends on other
        guide nodes. :func:`BBVI` calls it.
        """
        model = self.model if model is None else model
        pieces = self._finalize(model)
        self.model = model
        return pieces

    def variational_parameters(self):
        """``{key: VariationalParameter}`` of every group and hand-written entry."""
        table = {}
        for piece in self.validate():
            for key, parameter in piece.variational_parameters().items():
                if key in table:
                    raise ValueError(f"Two parts of the guide use the variational parameter name '{key}'.")
                table[key] = parameter
        return table

    def _elbo(self, model, n_samples, scales=None):
        return estimate_elbo(model, self.validate(model), n_samples, scales=scales, baseline_owner=self)

    # --- sampling -------------------------------------------------------------
    def sample(self, n_samples=1000, n_chains=1):
        """Draw ``n_chains * n_samples`` exact, independent draws from the guide (never MCMC).

        Every group is sampled exactly (``z = loc + L eps`` for a full-rank
        group, independent Normals for a mean-field group; one
        ``randn(n, d)`` block per group, in group order) and every
        hand-written entry with its distribution's ``sample()``; an entry
        without a working ``sample()`` raises an error instead of falling
        back to MCMC. The draws are independent, so ``n`` draws are ``n``
        effective samples, however strongly a full-rank group correlates its
        latents (that correlation lives in ``L``). Contrast
        :func:`~BayesianDLL.sample` on the posterior, whose MCMC draws are
        autocorrelated.

        Returns a :class:`SamplingResult` in the model's constrained space,
        including the model's deterministic nodes.
        """
        pieces = self.validate()
        total = n_chains * n_samples
        for piece in pieces:
            piece.draw(total, reparametrized=False)
        values = []
        for i in range(total):
            eta = {}
            try:
                for piece in pieces:
                    eta.update(piece.to_model(i, jacobian=False)[0])
            finally:
                for piece in pieces:
                    piece.release(i)
            values.append(eta)
        return trace_from_values(self.model, values, n_chains, n_samples)

    # --- moments --------------------------------------------------------------
    def mean(self, space="unconstrained", n_samples=4000, seed=0):
        """Return posterior means ``{latent: mean}`` under the guide.

        Convenience shortcut that evaluates the guide's joint moments via
        :meth:`covariance` and returns the reshaped means per latent parameter.

        Parameters
        ----------
        space : {"unconstrained", "constrained"}, default "unconstrained"
            Coordinate space in which the moments are computed.
        n_samples : int, default 4000
            Number of Monte Carlo draws for latents whose moments have no closed
            form.
        seed : int, default 0
            Seed for Monte Carlo moment estimates.

        Returns
        -------
        dict[str, Tensor]
            Mapping from latent parameter name to its mean tensor.
        """
        return self.covariance(space, n_samples, seed)._means()

    def covariance(self, space="unconstrained", n_samples=4000, seed=0):
        """Compute the joint moments (mean and block-diagonal covariance) of the guide.

        Evaluates the first and second moments of the guide's parameters.
        Returns a :class:`~BayesianDLL.Variational.GuideCovariance` containing
        the full covariance matrix (``.matrix``), block accessors
        (``.block(latent)``), per-latent variances (``.variances()``), and
        covariances (``.covariance()`` or ``.covariances()``). For parameter
        means, use :meth:`mean` on the guide or group directly.

        Parameters
        ----------
        space : {"unconstrained", "constrained"}, default "unconstrained"
            Coordinate space in which the moments are computed.
        n_samples : int, default 4000
            Number of Monte Carlo draws for latents whose moments have no closed
            form.
        seed : int, default 0
            Seed for Monte Carlo moment estimates.

        Returns
        -------
        GuideCovariance
            A moment container holding the mean and covariance matrix.
        """
        if space not in ("unconstrained", "constrained"):
            raise ValueError(f"space must be 'unconstrained' or 'constrained', got {space!r}.")
        parts = []
        for piece in self.validate():
            if isinstance(piece, Group):
                parts.append(piece.covariance(space, n_samples, seed))
            elif isinstance(piece, StructuredPiece):
                parts.append(self._piece_moments(piece, space, n_samples, seed))
            else:
                parts.append(self._entry_moments(piece, space, n_samples, seed))
        return GuideCovariance.combine(space, parts, list(self.model.params))

    moments = covariance

    def _entry_moments(self, piece, space, n_samples, seed):
        model, plan = self.model, piece.plan
        latents = plan.latents
        attribute = "unconstrained_value" if space == "unconstrained" else "constrained_value"
        shapes = {name: tuple(getattr(model.params[name], attribute).shape) for name in latents}
        reference = model.params[latents[0]].unconstrained_value
        dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
        result = GuideCovariance(space, latents, shapes, dtype=dtype, device=reference.device)

        estimated = []
        for name in latents:
            mode = plan.modes.get(name)
            identity = isinstance(model.params[name].distribution.transform, IdentityTransform)
            if mode is not None and (identity or mode == space):
                try:
                    mean, matrix = _distribution_moments(self.params[name].distribution, shapes[name])
                    result.set_block([name], mean, matrix)
                    continue
                except (NotImplementedError, ValueError):
                    pass
            estimated.append(name)
        if not estimated:
            return result

        # Monte Carlo from exact draws of the entries only; the entries are
        # independent, so every latent gets its own block.
        n = int(n_samples)
        draws = self._mc_draws(piece, estimated, space, n, seed, shapes, dtype, reference.device)
        for name in estimated:
            mean, matrix, standard_error = mc_moments(draws[name])
            result.set_block([name], mean, matrix)
            result.set_standard_errors([name], standard_error, [name], n)
        result.estimated = estimated
        return result

    def _mc_draws(self, piece, names, space, n, seed, shapes, dtype, device):
        """``{latent: (n, k)}`` exact draws of ``piece`` in ``space``, with the global random state restored."""
        model = self.model
        devices = [device] if device.type == "cuda" else []
        draws = {name: [] for name in names}
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            piece.draw(n, reparametrized=False)
            for i in range(n):
                try:
                    eta = piece.to_model(i, jacobian=False)[0]
                finally:
                    piece.release(i)
                for name in names:
                    draws[name].append(eta[name])
        result = {}
        for name in names:
            values = torch.stack(draws[name]).to(dtype)
            if space == "constrained":
                values = transform_batch(model.params[name].distribution.transform.inverse, values, shapes[name])
            result[name] = values.reshape(n, -1)
        return result

    def _piece_moments(self, piece, space, n_samples, seed):
        """Monte Carlo mean and covariance of a structured group (its latents are dependent: one joint block)."""
        model, latents = self.model, piece.latents
        attribute = "unconstrained_value" if space == "unconstrained" else "constrained_value"
        shapes = {name: tuple(getattr(model.params[name], attribute).shape) for name in latents}
        reference = model.params[latents[0]].unconstrained_value
        dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
        n = int(n_samples)
        draws = self._mc_draws(piece, latents, space, n, seed, shapes, dtype, reference.device)
        mean, matrix, standard_error = mc_moments(torch.cat([draws[name] for name in latents], dim=1))
        result = GuideCovariance(space, latents, shapes, mean=mean, matrix=matrix)
        result.set_standard_errors(latents, standard_error, latents, n)
        result.estimated = list(latents)
        return result
