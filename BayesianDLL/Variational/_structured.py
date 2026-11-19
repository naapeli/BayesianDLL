"""Structured groups: hand-written guide latents with dependencies inside one group.

A structured group is a :class:`Group` used as a context manager without a
``family``. The guide nodes created inside its ``with`` block form a small
directed graph, and ``q`` of the group factorises along it::

    q(u) = prod_k q_k(u_k | parents(u_k)),

where the parents are other latents of the same group, reached directly or
through deterministic blocks whose other arguments may be
``VariationalParameter`` objects. Groups remain independent of each other.

Sampling is ancestral (topological order), with each node drawn from its
distribution's exact ``sample()``. ``log q`` is the sum of the conditionals'
log densities; a node with the support of its model latent describes the
constrained value and contributes the log-Jacobian, as for single
hand-written entries.

Gradients
---------
With ``S = log p(eta(u)) + log|d eta / d u| - sum_k log q_k(u_k | pa_k)`` one
ELBO sample, the reparameterised gradient (every node ``has_rsample``) is a
reverse pass over the group's graph, in reverse topological order:

* ``ubar_k``, the total derivative of ``S`` w.r.t. ``u_k`` with all noise
  fixed, starts at ``dS_explicit/du_k - d log q_k/du_k`` (the model gradient
  converted to the guide variable, as for a single entry). Each child ``j``
  then adds ``ubar_j * d u_j / d pa_j`` (its reparameterisation path, from
  the distribution's sample derivatives) and ``- d log q_j / d pa_j`` (its
  log density, from ``log_pdf_param_grads``), carried back to ``u_k``
  through the deterministic blocks between them with their hand-written
  VJPs.
* A VariationalParameter ``phi`` of node ``k``'s distribution gets
  ``ubar_k * d u_k / d phi - d log q_k / d phi``; one inside a deterministic
  block gets the block's VJP of that block's cotangent.

When a node has no reparameterised sampler the whole group uses the score
function: ``grad log q_group(u)`` (the same reverse pass with only the
``log_pdf_param_grads`` terms) times the full ELBO sample minus a baseline.
"""

import torch

from .._model import _sum_to_shape
from ..Samplers._ancestral import draw_shaped
from ._ELBO import _derivative_of_source, _draw_entry, _restore_guide_values, _set_guide_values


def _add(table, key, value):
    table[key] = table[key] + value if key in table else value


class StructuredPiece:
    """The ELBO and sampling piece of a structured :class:`Group` (built by :meth:`Guide.validate`)."""

    def __init__(self, group, guide, model):
        from ._guide import _bounds_are_constant, _is_real

        self.group, self.guide, self.model = group, guide, model
        if not guide._compiled:
            guide.compile()
        nodes = set(group.nodes)
        self.random = [n for n in guide._topo_order if n in nodes and n in guide.params]
        self.deterministic = [n for n in guide._topo_order if n in nodes and n in guide.deterministic_params]
        self.order = [n for n in guide._topo_order if n in nodes]
        guide_nodes = set(guide.params) | set(guide.deterministic_params)

        def check_inputs(name, inputs):
            for parent in inputs:
                if parent in guide_nodes and parent not in nodes:
                    raise ValueError(
                        f"The guide node '{name}' of the group '{group.name}' depends on '{parent}', which is not in "
                        "that group. Groups are independent of each other: define every latent a group depends on "
                        "inside the same group."
                    )

        self.modes = {}
        for name in self.random:
            parameter = guide.params[name]
            distribution = parameter.distribution
            check_inputs(name, distribution.parameters)
            if parameter.enumerated:
                raise ValueError(f"The guide latent '{name}' is enumerated; a guide cannot sum out latents.")
            if name in model.enumerated_params:
                raise ValueError(f"'{name}' is enumerated (summed out) in the model and must not appear in the guide.")
            if name not in model.params:
                self.modes[name] = "auxiliary"
                continue
            latent = model.params[name]
            if _is_real(distribution) and latent.state_space.is_continuous():
                expected, self.modes[name] = latent.unconstrained_value.shape, "unconstrained"
            else:
                if type(distribution.state_space) is not type(latent.state_space):
                    raise ValueError(
                        f"The guide distribution of '{name}' has support {type(distribution.state_space).__name__}, "
                        f"but the model latent has support {type(latent.state_space).__name__}."
                    )
                if not _bounds_are_constant(latent.distribution.transform):
                    raise NotImplementedError(f"The support of '{name}' depends on other latents; use a guide on the real line for it.")
                expected, self.modes[name] = latent.constrained_value.shape, "constrained"
            if tuple(parameter.constrained_value.shape) != tuple(expected):
                raise ValueError(
                    f"The guide value of '{name}' has shape {tuple(parameter.constrained_value.shape)}, expected {tuple(expected)}."
                )
        self.targets = {}
        for name in self.deterministic:
            deterministic = guide.deterministic_params[name]
            check_inputs(name, [getattr(i, "name", None) for i in deterministic.inputs])
            if name not in model.params:
                continue
            sources = [getattr(i, "name", None) for i in deterministic.inputs if getattr(i, "name", None) in guide.params]
            if len(sources) != 1 or self.modes.get(sources[0]) != "auxiliary" or sources[0] in self.targets.values():
                raise NotImplementedError(
                    f"The guide deterministic '{name}' must be a function of exactly one auxiliary guide latent "
                    "that feeds no other model latent."
                )
            self.targets[name] = sources[0]
        self.sources = {source: target for target, source in self.targets.items()}
        for name, mode in self.modes.items():
            if mode == "auxiliary" and name not in self.sources:
                raise ValueError(
                    f"The auxiliary guide latent '{name}' of the structured group '{group.name}' does not give a model "
                    "latent through a guide deterministic. A structured group may only contain latents of the model "
                    "and auxiliary latents that define one."
                )
        described = [n for n in self.random if self.modes[n] != "auxiliary"] + list(self.targets)
        if set(described) != set(group.names):
            raise ValueError(
                f"The structured group '{group.name}' lists {sorted(group.names)} but defines the model latents "
                f"{sorted(described)}; the two must be equal."
            )
        self.latents = [n for n in model.params if n in set(described)]
        self.reparametrized = all(
            bool(getattr(guide.params[n].distribution, "has_rsample", False))
            and guide.params[n].distribution.state_space.is_continuous()
            for n in self.random
        )
        self._saved = {}

    # --- parameters -------------------------------------------------------------
    def variational_parameters(self):
        table = {}
        for name in self.order:
            node = self.guide.params[name] if name in self.guide.params else self.guide.deterministic_params[name]
            owner = node.distribution if name in self.guide.params else node
            for vp_name, vp in getattr(owner, "variational_parameters", {}).items():
                table[f"{name}_{vp_name}"] = vp
        return table

    def gradient_names(self):
        if not self.reparametrized:
            return []
        return [self.sources.get(n, n) for n in self.random]

    # --- sampling ---------------------------------------------------------------
    def draw(self, n, reparametrized=True):
        self._n = n
        self._use_rsample = reparametrized and self.reparametrized
        self._scores = []
        self._pending = {}

    def _value_shape(self, name):
        return self.guide.params[name].constrained_value.shape

    def to_model(self, i, jacobian=True):
        """Ancestral draw ``i`` of the group, its unconstrained model values and log-Jacobian.

        The guide values stay set until :meth:`release`; ``log q`` is evaluated here.
        """
        guide, model = self.guide, self.model
        self._saved = {name: guide.params[name].unconstrained_value for name in self.random}
        u, dz = {}, {}
        for name in self.random:
            distribution = guide.params[name].distribution
            shape = self._value_shape(name)
            if self._use_rsample:
                draws, derivatives = draw_shaped(distribution, 1, shape, reparametrized=True)
                dz[name] = {key: value[0] for key, value in derivatives.items()}
            else:
                draws = _draw_entry(name, distribution, 1, shape)
            u[name] = draws[0]
            _set_guide_values(guide, {name: u[name]})
        self._u, self._dz = u, dz
        log_q = 0.0
        for name in self.random:
            log_q = log_q + guide.params[name].distribution.log_pdf(u[name]).sum()
        self._log_q = log_q

        eta, log_jacobian = {}, 0.0
        self._chain = {}
        for name in self.random:
            mode = self.modes[name]
            if mode == "unconstrained":
                eta[name] = u[name]
            elif mode == "constrained":
                transform = model.params[name].distribution.transform
                eta[name] = transform.forward(u[name])
                if jacobian:
                    log_jacobian = log_jacobian - transform.log_abs_det_jacobian(eta[name])
        for target, source in self.targets.items():
            deterministic = guide.deterministic_params[target]
            transform = model.params[target].distribution.transform
            eta[target] = transform.forward(deterministic.constrained_value)
            if not jacobian:
                continue
            derivative = _derivative_of_source(deterministic, source, u[source])
            log_jacobian = log_jacobian + torch.log(derivative.abs()).sum() - transform.log_abs_det_jacobian(eta[target])
            if self._use_rsample:
                x = u[source]
                step = torch.finfo(x.dtype).eps ** (1 / 3) * (1 + x.abs())
                logs = []
                for sign in (1, -1):
                    _set_guide_values(guide, {source: x + sign * step})
                    logs.append(torch.log(_derivative_of_source(deterministic, source, x).abs()))
                _set_guide_values(guide, {source: x})
                self._chain[source] = (logs[0] - logs[1]) / (2 * step)
        return eta, log_jacobian

    def release(self, i):
        if self._saved:
            _restore_guide_values(self.guide, self._saved)
        self._saved = {}

    def log_q(self, i):
        return self._log_q

    # --- gradients --------------------------------------------------------------
    def _explicit(self, model_grads):
        """``dS_explicit / du_k``: the model gradient (with the Jacobian) in the guide variable of every node."""
        model, explicit = self.model, {}
        for name in self.random:
            mode = self.modes[name]
            target = self.sources.get(name, name)
            if mode == "unconstrained":
                explicit[name] = model_grads[name]
                continue
            transform = model.params[target].distribution.transform
            eta_value = model.params[target].unconstrained_value
            jacobian = transform.derivative(eta_value)
            if jacobian.shape != eta_value.shape:
                raise NotImplementedError(
                    f"Reparameterised gradients of a constrained-space guide for '{target}' need an elementwise transform."
                )
            outer = model_grads[target] - transform.grad_log_abs_det_jacobian(eta_value)
            if mode == "constrained":
                explicit[name] = outer / jacobian
            else:
                derivative = _derivative_of_source(self.guide.deterministic_params[target], name, self._u[name])
                explicit[name] = outer * derivative / jacobian + self._chain[name]
            explicit[name] = explicit[name].reshape(self._u[name].shape)
        return explicit

    def _reverse(self, explicit=None):
        """Reverse pass over the group's graph.

        With ``explicit`` (reparameterised): the gradient of the ELBO sample
        ``S``. Without (score function): the gradient of ``log q_group``.
        """
        guide, reparam = self.guide, explicit is not None
        sign = -1.0 if reparam else 1.0
        nodes = set(self.order)
        cotangents, grads = {}, {}
        if reparam:
            for name in self.random:
                cotangents[name] = explicit[name] - guide.params[name].distribution.log_pdf_grad(self._u[name])

        def shape_of(name):
            node = guide.params[name] if name in guide.params else guide.deterministic_params[name]
            return node.constrained_value.shape

        for name in reversed(self.order):
            if name in guide.params:
                distribution = guide.params[name].distribution
                value = self._u[name]
                param_grads = distribution.log_pdf_param_grads(value)
                bar = cotangents.get(name)
                dz = self._dz.get(name, {}) if reparam else {}
                for vp_name, vp in distribution.variational_parameters.items():
                    grad = sign * param_grads[vp_name] if vp_name in param_grads else torch.zeros_like(value)
                    if reparam and vp_name in dz:
                        grad = grad + bar * dz[vp_name]
                    _add(grads, f"{name}_{vp_name}", _sum_to_shape(grad, vp.value.shape))
                for parent in distribution.parameters:
                    if parent not in nodes:
                        continue
                    grad = sign * param_grads[parent] if parent in param_grads else torch.zeros_like(value)
                    if reparam and parent in dz:
                        grad = grad + bar * dz[parent]
                    _add(cotangents, parent, _sum_to_shape(grad, shape_of(parent)))
            else:
                if name not in cotangents:
                    continue
                deterministic = guide.deterministic_params[name]
                bar = cotangents[name]
                if getattr(deterministic, "vjp_func", None) is not None:
                    input_values = deterministic.input_values()
                    input_grads = deterministic.vjp(bar, input_values)
                    for inp, value in zip(deterministic.inputs, input_values):
                        inp_name = getattr(inp, "name", None)
                        if inp_name in nodes and inp_name in input_grads:
                            _add(cotangents, inp_name, _sum_to_shape(input_grads[inp_name], value.shape))
                            input_grads.pop(inp_name)
                else:
                    local = deterministic.derivatives()
                    for inp in deterministic.inputs:
                        inp_name = getattr(inp, "name", None)
                        if inp_name in nodes and inp_name in local:
                            derivative = torch.as_tensor(local[inp_name], dtype=bar.dtype, device=bar.device)
                            target_shape = shape_of(inp_name)
                            if derivative.ndim == 2 and derivative.shape != bar.shape and derivative.shape[0] == bar.numel():
                                _add(cotangents, inp_name, (bar.reshape(-1) @ derivative).reshape(target_shape))
                            else:
                                _add(cotangents, inp_name, _sum_to_shape(bar * derivative, target_shape))
                variational_vjp = getattr(deterministic, "variational_vjp", None)
                if variational_vjp is not None:
                    for vp_name, grad in variational_vjp(bar).items():
                        _add(grads, f"{name}_{vp_name}", grad)
        return grads

    def in_model(self, i, model_grads):
        """The gradient of draw ``i`` (computed while the guide and the model are set to it)."""
        if self._use_rsample:
            self._pending = self._reverse(self._explicit(model_grads))
        else:
            self._pending = None
            self._score = self._reverse()

    def accumulate(self, i, model_grads, grads):
        if self._pending is None:
            self._scores.append((i, self._score))
            return
        for key, grad in self._pending.items():
            grads[key] = grads[key] + grad

    def score_terms(self):
        return self._scores
