"""Gaussian groups of a :class:`~BayesianDLL.Variational.Guide`.

A :class:`Group` is a Gaussian over the concatenated unconstrained values of
some of the model's latents: an independent Normal per coordinate
(``family="mean_field"``) or one multivariate Normal with a full covariance
(``family="full_rank"``). Groups are independent of each other and of the
hand-written guide entries, so the joint guide is their product.
"""

import math

import torch

from .._active_model._active_model import get_active_model
from .._parameters import RandomParameter, VariationalParameter
from ..Distributions import IdentityTransform, LogTransform
from ._moments import GuideCovariance, mc_moments, transform_batch


_LOG_2PI = math.log(2 * math.pi)
FAMILIES = ("mean_field", "full_rank")


def _discrete_message(name, purpose):
    return (
        f"'{name}' is a discrete latent. {purpose} need continuous latents; sum it out with discrete "
        f"enumeration instead, e.g. RandomParameter('{name}', ..., enumerate=True) or "
        f"BayesianDLL.marginalize(model, ['{name}']), or give it a hand-written guide entry "
        f"(e.g. RandomParameter('{name}', Bernoulli(VariationalParameter(...))) inside the `with Guide(...)` block), "
        "which uses score-function gradients."
    )


class LatentLayout:
    """Concatenation of (some of) a model's unconstrained latent values into one vector."""

    def __init__(self, model, purpose="Gaussian guides", names=None):
        if not model._compiled:
            model.compile()
        self.names = list(model.params) if names is None else list(names)
        if not self.names:
            raise ValueError("The model has no latent (random) parameters.")
        for name in self.names:
            parameter = model.params[name]
            if not parameter.distribution.transformed_state_space.is_continuous():
                raise ValueError(_discrete_message(name, purpose))
        self.shapes = {}
        self.slices = {}
        offset = 0
        for name in self.names:
            value = model.params[name].unconstrained_value
            self.shapes[name] = value.shape
            self.slices[name] = (offset, offset + value.numel())
            offset += value.numel()
        self.dim = offset
        first = model.params[self.names[0]].unconstrained_value
        dtype = first.dtype if first.is_floating_point() else torch.get_default_dtype()
        self.dtype = dtype
        self.device = first.device

    def check(self, model):
        for name in self.names:
            if name not in model.params or model.params[name].unconstrained_value.shape != self.shapes[name]:
                raise ValueError(f"The model does not match the latent layout of this guide ('{name}').")

    def pack(self, values):
        return torch.cat([values[name].reshape(-1).to(self.dtype) for name in self.names])

    def current(self, model):
        return self.pack({name: model.params[name].unconstrained_value for name in self.names})

    def unpack(self, flat):
        """``{name: value}`` from a flat vector ``(D,)`` or a batch ``(..., D)``."""
        lead = flat.shape[:-1]
        return {name: flat[..., start:end].reshape(*lead, *self.shapes[name]) for name, (start, end) in self.slices.items()}

    def to_constrained(self, model, flat):
        return {name: model.params[name].distribution.transform.inverse(value) for name, value in self.unpack(flat).items()}


def _lower(matrix, diagonal=0):
    """``torch.tril`` by masking (elementwise, cheap for small matrices)."""
    n, m = matrix.shape[-2:]
    rows = torch.arange(n, device=matrix.device).unsqueeze(-1)
    cols = torch.arange(m, device=matrix.device).unsqueeze(0)
    return matrix * (cols - rows <= diagonal).to(matrix.dtype)


class ScaleTrilParameter(VariationalParameter):
    """A lower-triangular matrix with a positive diagonal, optimised in log-diagonal coordinates.

    The optimiser works on ``tril(L, -1) + diag(log diag L)``: the strictly
    upper triangle is never touched (its gradient is masked out) and the
    diagonal stays positive.
    """

    def __init__(self, name, value):
        value = torch.as_tensor(value)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("scale_tril must be a square matrix.")
        if not bool((torch.diagonal(value) > 0).all()):
            raise ValueError("The diagonal of scale_tril must be positive.")
        super().__init__(name, _lower(value))

    @property
    def unconstrained_value(self):
        return _lower(self.value, -1) + torch.diag_embed(torch.log(torch.diagonal(self.value)))

    def unconstrained_grad(self, grad):
        diagonal = torch.diagonal(grad) * torch.diagonal(self.value)
        return _lower(grad, -1) + torch.diag_embed(diagonal)

    def set_unconstrained_value(self, unconstrained_value):
        unconstrained_value = unconstrained_value.reshape(self.value.shape)
        value = _lower(unconstrained_value, -1) + torch.diag_embed(torch.exp(torch.diagonal(unconstrained_value)))
        self.set_new_value(value)


def _elementwise_kind(transform):
    """``("identity", None)``, ``("log", (border, sign))`` or ``("other", None)`` for exact constrained moments."""
    if isinstance(transform, IdentityTransform):
        return "identity", None
    if isinstance(transform, LogTransform) and isinstance(transform.border, int | float | torch.Tensor):
        return "log", (transform.border, 1.0 if transform.side == "larger" else -1.0)
    return "other", None


class Group:
    """A Gaussian over the unconstrained values of some latents, registered with the active :class:`Guide`.

    Create it inside a ``with Guide(model, ...)`` block (it registers itself
    with that guide, as a ``RandomParameter`` registers with the active
    model)::

        with Guide(model, default_family="mean_field") as guide:
            Group(["mu", "beta"], family="full_rank")

    **Structured groups.** Used as a context manager without ``family``, a
    group holds hand-written guide latents that may depend on each other::

        with Guide(model, default_family="mean_field") as guide:
            with Group(["mu", "theta"]):
                mu = RandomParameter("mu", Normal(VariationalParameter("m", 0.0),
                                                  VariationalParameter("s", 1.0, log_scale=True)))
                RandomParameter("theta", Normal(Affine("theta_loc", mu, loc=VariationalParameter("b", 0.0),
                                                       scale=VariationalParameter("a", 1.0)),
                                                VariationalParameter("t", 1.0, log_scale=True)))

    ``q`` of the group then factorises along the guide graph,
    ``q(mu) q(theta | mu)``. Distributions and deterministic blocks inside the
    group may take other latents of the same group, VariationalParameters,
    constants and Data as arguments, but no latent of another group (groups
    stay independent). ``params`` must equal the model latents defined inside
    the block; entries describe unconstrained or constrained values and
    auxiliary latents give a model latent through a guide deterministic,
    exactly as for single hand-written entries. Sampling is ancestral and
    exact; gradients are reparameterised through the chain when every
    distribution has a reparameterised sampler, and use the score function
    for the whole group otherwise (see :mod:`BayesianDLL.Variational._structured`).
    Its ``mean`` and ``covariance`` are Monte Carlo estimates in both spaces.

    Args:
        params: the latents of the group, as names or as the model's
            ``RandomParameter`` objects. Every latent belongs to exactly one
            group or hand-written guide entry; unknown, enumerated and
            discrete latents are rejected (sum discrete latents out with
            enumeration, or give them a hand-written guide entry).
        family: ``None`` (default): ``"mean_field"``, or a structured group
            when the group is used as a context manager.
            ``"mean_field"``: an independent Normal per unconstrained
            coordinate, with VariationalParameters ``{prefix}{latent}_loc`` and
            ``{prefix}{latent}_scale`` (optimised on the log scale).
            ``"full_rank"``: one multivariate Normal ``z = loc + L eps`` over
            the group's concatenated unconstrained coordinates, with
            VariationalParameters ``{prefix}loc`` and ``{prefix}scale_tril``
            (lower triangular, optimised in log-diagonal coordinates).
        init_scale: initial standard deviation of every unconstrained
            coordinate (default ``0.1``); the means start at the model's
            current values.
        name: prefix of the group's VariationalParameter names (``prefix =
            name + "_"``). Default ``group0``, ``group1``, ... in creation
            order.

    The reparameterised ELBO gradients are hand-written: with ``g`` the
    gradient of the unconstrained log joint at the draw ``z``,
    ``d/dloc = g``, ``d/dscale = g eps + 1 / scale`` (mean-field) and
    ``d/dL = tril(g eps^T) + diag(1 / L_ii)`` (full-rank), where the last
    terms are the entropy gradients.

    Attributes:
        names: the latents of the group, in the order of the flat vector.
        loc: the mean of the group in the unconstrained space (flat vector).
        scale: standard deviations (mean-field only, flat vector).
        scale_tril: the Cholesky factor ``L`` (full-rank only).
        locs, scales: ``{latent: VariationalParameter}`` (mean-field only).
        loc_parameter, scale_tril_parameter: the VariationalParameters
            (full-rank only).
    """

    def __init__(self, params, family=None, init_scale=0.1, name=None):
        guide = get_active_model()
        if guide is None or not hasattr(guide, "_register_group"):
            raise RuntimeError("Create a Group inside a `with Guide(model, ...) as guide:` block.")
        names = guide._group_names(params)
        if name is None:
            name = f"group{len(guide._groups)}"
        self.guide = guide
        self._family_given = family is not None
        self.nodes = []
        self._setup(guide.model, names, "mean_field" if family is None else family, init_scale, name, prefixed=True)
        guide._register_group(self)

    # --- structured groups ----------------------------------------------------
    def __enter__(self):
        if self._family_given:
            raise ValueError(
                f"The group '{self.name}' has family={self.family!r}, a Gaussian over its latents. Use "
                "`with Group([...]):` without a family for a structured group of hand-written guide latents."
            )
        if self.nodes:
            raise ValueError(f"The structured group '{self.name}' was already defined.")
        self.family = "structured"
        for attribute in ("locs", "scales", "loc_parameter", "scale_tril_parameter"):
            self.__dict__.pop(attribute, None)
        self._before = set(self.guide.params) | set(self.guide.deterministic_params)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False
        guide, model = self.guide, self.model
        self.nodes = [
            name for name in list(guide.params) + list(guide.deterministic_params) if name not in self._before
        ]
        described = [name for name in self.nodes if name in model.params]
        if set(described) != set(self.names):
            raise ValueError(
                f"The structured group '{self.name}' lists {sorted(self.names)} but defines the model latents "
                f"{sorted(described)}; the two must be equal."
            )
        return False

    def _structured_piece(self):
        from ._structured import StructuredPiece
        return StructuredPiece(self, self.guide, self.guide.model)

    @classmethod
    def _default(cls, model, names, family, init_scale, prefixed):
        group = cls.__new__(cls)
        group._setup(model, names, family, init_scale, "default", prefixed=prefixed)
        return group

    def _setup(self, model, names, family, init_scale, name, prefixed):
        if family not in FAMILIES:
            raise ValueError(f"family must be 'mean_field' or 'full_rank', got {family!r}.")
        self.family = family
        self.name = name
        self.prefix = f"{name}_" if prefixed else ""
        self.model = model
        self.init_scale = float(init_scale)
        self.layout = LatentLayout(model, purpose="Gaussian guide groups", names=names)
        self.names = list(self.layout.names)
        initial = self.layout.current(model)
        if family == "mean_field":
            self.locs, self.scales = {}, {}
            for latent, value in self.layout.unpack(initial).items():
                self.locs[latent] = VariationalParameter(f"{self.prefix}{latent}_loc", value.clone())
                self.scales[latent] = VariationalParameter(
                    f"{self.prefix}{latent}_scale", torch.full_like(value, self.init_scale), log_scale=True
                )
        else:
            dim, dtype, device = self.layout.dim, self.layout.dtype, self.layout.device
            self.loc_parameter = VariationalParameter(f"{self.prefix}loc", initial.clone())
            self.scale_tril_parameter = ScaleTrilParameter(
                f"{self.prefix}scale_tril", self.init_scale * torch.eye(dim, dtype=dtype, device=device)
            )

    def __repr__(self):
        return f"Group({self.names}, family={self.family!r}, name={self.name!r})"

    # --- parameters -----------------------------------------------------------
    def variational_parameters(self):
        """``{key: VariationalParameter}`` of the group (the keys of the ELBO gradient)."""
        if self.family == "structured":
            return self._structured_piece().variational_parameters()
        if self.family == "mean_field":
            table = {}
            for latent in self.names:
                table[self.locs[latent].name] = self.locs[latent]
                table[self.scales[latent].name] = self.scales[latent]
            return table
        return {self.loc_parameter.name: self.loc_parameter, self.scale_tril_parameter.name: self.scale_tril_parameter}

    def _flat(self, parameters):
        return torch.cat([parameters[latent].value.reshape(-1) for latent in self.names])

    @property
    def loc(self):
        """Mean of the group in the unconstrained space, as a flat vector."""
        if self.family == "mean_field":
            return self._flat(self.locs)
        return self.loc_parameter.value

    @property
    def scale(self):
        """Standard deviations of the unconstrained coordinates (mean-field groups), as a flat vector."""
        if self.family != "mean_field":
            raise AttributeError("scale is defined for mean-field groups; a full-rank group has scale_tril.")
        return self._flat(self.scales)

    @property
    def scale_tril(self):
        """The lower-triangular Cholesky factor ``L`` of a full-rank group."""
        if self.family != "full_rank":
            raise AttributeError("scale_tril is defined for full-rank groups; a mean-field group has scale.")
        return self.scale_tril_parameter.value

    def _transform(self, eps):
        """Unconstrained draws ``(n, D)`` from standard normal noise ``(n, D)``."""
        if self.family == "mean_field":
            return self.loc + eps * self.scale
        return self.loc_parameter.value + eps @ self.scale_tril.transpose(-2, -1)

    def _noise(self, n, generator=None):
        return torch.randn(n, self.layout.dim, dtype=self.layout.dtype, device=self.layout.device, generator=generator)

    def sample_unconstrained(self, n, generator=None):
        """``n`` draws ``(n, D)`` of the group's flat unconstrained vector."""
        return self._transform(self._noise(n, generator))

    # --- pieces of the shared ELBO estimator ----------------------------------
    def check(self, model):
        self.layout.check(model)

    def gradient_names(self):
        return list(self.names)

    def draw(self, n, reparametrized=True):
        """Draw ``n`` values (one ``randn(n, D)`` block) for the ELBO estimator or for sampling."""
        self._eps = self._noise(n)
        self._z = self._transform(self._eps)

    def to_model(self, i, jacobian=True):
        """Unconstrained model values of draw ``i`` and the log-Jacobian (zero: the group lives in that space)."""
        return self.layout.unpack(self._z[i]), 0.0

    def in_model(self, i, model_grads):
        pass

    def release(self, i):
        pass

    def log_q(self, i):
        eps = self._eps[i]
        if self.family == "mean_field":
            return (-0.5 * eps ** 2 - torch.log(self.scale) - 0.5 * _LOG_2PI).sum()
        diagonal = torch.diagonal(self.scale_tril)
        return -0.5 * (eps ** 2).sum() - torch.log(diagonal).sum() - 0.5 * self.layout.dim * _LOG_2PI

    def accumulate(self, i, model_grads, grads):
        """Add the reparameterised gradient of ``log p - log q`` at draw ``i`` to ``grads``."""
        g = self.layout.pack(model_grads)
        eps = self._eps[i]
        if self.family == "mean_field":
            grad_scale = g * eps + 1 / self.scale
            for latent, (start, end) in self.layout.slices.items():
                shape = self.locs[latent].value.shape
                grads[self.locs[latent].name] += g[start:end].reshape(shape)
                grads[self.scales[latent].name] += grad_scale[start:end].reshape(shape)
        else:
            grads[self.loc_parameter.name] += g
            grads[self.scale_tril_parameter.name] += _lower(torch.outer(g, eps)) + torch.diag_embed(1 / torch.diagonal(self.scale_tril))

    def score_terms(self):
        return []

    # --- moments ----------------------------------------------------------------
    def _unconstrained_covariance(self):
        if self.family == "mean_field":
            return torch.diag(self.scale ** 2)
        L = self.scale_tril
        return L @ L.transpose(-2, -1)

    def mean(self, space="unconstrained", n_samples=4000, seed=0):
        """``{latent: mean}`` of the group in ``space`` (see :meth:`covariance`)."""
        return self.covariance(space, n_samples, seed).means()

    def covariance(self, space="unconstrained", n_samples=4000, seed=0):
        """Mean and covariance of the group in ``"unconstrained"`` or ``"constrained"`` space.

        Returns a :class:`GuideCovariance`. In the unconstrained space the
        moments are exact (``loc`` and ``diag(scale^2)`` or ``L L^T``). In the
        constrained space they are exact for latents whose transform is the
        identity or an elementwise ``log`` (log-normal moments; for a
        full-rank group also the cross-covariances, e.g.
        ``cov(e^x_i, e^x_j) = exp(mu_i + mu_j + (S_ii + S_jj)/2) (exp(S_ij) - 1)``);
        every block that involves another transform (logit, stick-breaking,
        ...) is a Monte Carlo estimate from ``n_samples`` draws of this group,
        made with a local generator seeded with ``seed``, and is listed in
        ``estimated``. See the "Accuracy and cost" section of :class:`Guide`.
        """
        if space not in ("unconstrained", "constrained"):
            raise ValueError(f"space must be 'unconstrained' or 'constrained', got {space!r}.")
        if self.family == "structured":
            return self.guide._piece_moments(self._structured_piece(), space, n_samples, seed)
        loc = self.loc
        cov = self._unconstrained_covariance()
        if space == "unconstrained":
            return GuideCovariance(space, self.names, dict(self.layout.shapes), loc, cov)

        kinds = {latent: _elementwise_kind(self.model.params[latent].distribution.transform) for latent in self.names}
        exact = [latent for latent in self.names if kinds[latent][0] != "other"]
        estimated = [latent for latent in self.names if kinds[latent][0] == "other"]

        # Exact moments of the identity / log coordinates.
        exact_mean = exact_cov = None
        if exact:
            index = torch.cat([torch.arange(*self.layout.slices[latent], device=loc.device) for latent in exact])
            mu = loc[index]
            sigma = cov[index.unsqueeze(-1), index.unsqueeze(0)]

            def per_coordinate(value_of):
                return torch.cat([
                    torch.as_tensor(value_of(latent), dtype=loc.dtype, device=loc.device)
                    .expand(self.layout.shapes[latent]).reshape(-1)
                    for latent in exact
                ])

            is_log = per_coordinate(lambda latent: float(kinds[latent][0] == "log")) > 0
            border = per_coordinate(lambda latent: kinds[latent][1][0] if kinds[latent][0] == "log" else 0.0)
            sign = per_coordinate(lambda latent: kinds[latent][1][1] if kinds[latent][0] == "log" else 1.0)
            variances = torch.diagonal(sigma)
            # E[e^x] for the log coordinates; the identity coordinates use x itself.
            expected_exp = torch.exp(mu + variances / 2)
            exact_mean = torch.where(is_log, border + sign * expected_exp, mu)
            if self.family == "mean_field":
                log_variance = torch.expm1(variances) * torch.exp(2 * mu + variances)
                exact_cov = torch.diag(torch.where(is_log, log_variance, variances))
            else:
                # cov(y_i, y_j) for y = x (identity) or y = border + sign e^x (log):
                #   id-id:   S_ij
                #   id-log:  sign_j S_ij E[e^x_j]                 (Stein's lemma)
                #   log-log: sign_i sign_j E[e^x_i] E[e^x_j] (exp(S_ij) - 1)
                a = torch.where(is_log, sign * expected_exp, torch.ones_like(mu))
                log_i, log_j = is_log.unsqueeze(-1), is_log.unsqueeze(-2)
                both = (a.unsqueeze(-1) * a.unsqueeze(-2)) * torch.expm1(sigma)
                one = a.unsqueeze(-1) * a.unsqueeze(-2) * sigma
                exact_cov = torch.where(log_i & log_j, both, one)

        shapes = {}
        for latent in self.names:
            if kinds[latent][0] == "other":
                shapes[latent] = tuple(self.model.params[latent].constrained_value.shape)
            else:
                shapes[latent] = tuple(self.layout.shapes[latent])
        result = GuideCovariance(space, self.names, shapes, dtype=loc.dtype, device=loc.device)
        if estimated:
            # Monte Carlo for the blocks that involve another transform, from
            # draws of this group only (a mean-field group's latents are
            # independent, so only the estimated latents are needed).
            generator = torch.Generator(device=self.layout.device).manual_seed(int(seed))
            draws = self.layout.unpack(self.sample_unconstrained(int(n_samples), generator))
            mc_names = estimated if self.family == "mean_field" else self.names
            columns = []
            for latent in mc_names:
                transform = self.model.params[latent].distribution.transform
                columns.append(transform_batch(transform.inverse, draws[latent], shapes[latent]).reshape(int(n_samples), -1))
            mc_mean, mc_cov, standard_error = mc_moments(torch.cat(columns, dim=1))
            result.set_block(mc_names, mc_mean, mc_cov)
            result.set_standard_errors(mc_names, standard_error, estimated, n_samples)
            result.estimated = list(estimated)
        if exact:
            result.set_block(exact, exact_mean, exact_cov)
        return result
