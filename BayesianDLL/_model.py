import torch
from contextlib import contextmanager
import networkx as nx
from networkx import DiGraph
from collections import deque
import copy

from ._active_model._active_model import push_model, pop_model
from ._parameters import RandomParameter, ObservedParameter, DeterministicParameter
from ._enumeration import weight_gradient
from . import _batching
from ._batching import pad_to, sum_to_batched
from ._data import Data


def _sum_to_shape(grad: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    if not isinstance(grad, torch.Tensor):
        grad = torch.as_tensor(grad)
    if grad.shape == target_shape:
        return grad

    if len(target_shape) == 0:
        return grad.sum()

    if target_shape == (1,) and grad.numel() > 1:
        return grad.sum().reshape(1)

    extra_dims = grad.ndim - len(target_shape)
    if extra_dims > 0:
        grad = grad.sum(dim=tuple(range(extra_dims)))

    for dim, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if g_dim != t_dim and t_dim == 1:
            grad = grad.sum(dim=dim, keepdim=True)

    if grad.shape != target_shape:
        grad = grad.reshape(target_shape)

    return grad


def _distribution_dependencies(distribution) -> set[str]:
    """Return named model dependencies, excluding constant arguments.

    Distribution gradient methods use fallback labels such as ``"variance"``
    for literal arguments so they remain useful outside a model. Inside a
    model, those labels must not be mistaken for a RandomParameter with the
    same name.
    """
    return set(getattr(distribution, "parameters", ()))


def _add_grad(cotangents, name, grad):
    cotangents[name] = cotangents[name] + grad if name in cotangents else grad


_DISTRIBUTION_SKIP = {"transform", "state_space", "transformed_state_space", "parameters", "variational_parameters"}


def _distribution_arguments(distribution, _seen=None):
    """Yield ``(argument, event_ndim)`` for every node or tensor a distribution reads.

    Walks the distribution's attributes, those of nested distributions (e.g.
    the components of a ``Mixture``) and of its transform. ``event_ndim`` is
    the number of trailing event dimensions of the argument, declared by the
    distribution class in ``_param_event_ndims`` (zero by default).
    """
    from .Distributions._distributions import Distribution
    from .Distributions._transforms import Transform
    from ._parameters import VariationalParameter
    seen = set() if _seen is None else _seen
    if id(distribution) in seen:
        return
    seen.add(id(distribution))
    event_ndims = getattr(distribution, "_param_event_ndims", {})

    def visit(value, event_ndim):
        if isinstance(value, (Distribution, Transform)):
            # A wrapper may declare event dimensions for a nested distribution
            # (e.g. ``Ordered`` evaluates its base on a vector event); they
            # add to the event dimensions the nested distribution declares.
            for argument, nested_event_ndim in _distribution_arguments(value, seen):
                yield argument, nested_event_ndim + event_ndim
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from visit(item, event_ndim)
        elif isinstance(value, (RandomParameter, DeterministicParameter, ObservedParameter, Data, VariationalParameter, torch.Tensor)):
            yield value, event_ndim

    attributes = vars(distribution) if hasattr(distribution, "__dict__") else {}
    for attribute, value in attributes.items():
        if attribute in _DISTRIBUTION_SKIP:
            continue
        yield from visit(value, event_ndims.get(attribute, 0))
    transform = attributes.get("transform")
    if transform is not None:
        yield from visit(transform, 0)


def _non_batchable_reason(model):
    """Why ``model`` cannot be evaluated as one batched tensor computation (``None``: it can)."""
    for name, parameter in {**model.params, **model.enumerated_params, **model.observed_params}.items():
        stack = [parameter.distribution]
        while stack:
            distribution = stack.pop()
            if getattr(distribution, "_batchable", True) is False:
                return f"the distribution {type(distribution).__name__} of '{name}' does not support batched evaluation"
            for value in vars(distribution).values():
                if hasattr(value, "log_pdf"):
                    stack.append(value)
                elif isinstance(value, (list, tuple)):
                    stack.extend(v for v in value if hasattr(v, "log_pdf"))
    for name, parameter in model.deterministic_params.items():
        if getattr(parameter, "_batchable", True) is False:
            return f"the deterministic node '{name}' ({type(parameter).__name__}) does not support batched evaluation"
    return None


def _stack_chain_results(results):
    first = results[0]
    if isinstance(first, dict):
        return {key: _stack_chain_results([r[key] for r in results]) for key in first}
    if isinstance(first, tuple):
        return tuple(_stack_chain_results([r[i] for r in results]) for i in range(len(first)))
    tensors = [torch.as_tensor(r) for r in results]
    reference = next((t for t in tensors if t.is_floating_point()), tensors[0])
    tensors = [t.to(dtype=reference.dtype, device=reference.device) if t.is_floating_point() else t.to(device=reference.device) for t in tensors]
    shape = torch.broadcast_shapes(*(t.shape for t in tensors))
    return torch.stack([t.expand(shape) for t in tensors], dim=0)


def _move_tensors(root, device, dtype):
    """Move every tensor reachable from ``root`` to ``device`` (floating tensors also to ``dtype``).

    Objects are walked generically: attributes (``__dict__`` and
    ``__slots__``), dict values, list items, tuples (rebuilt), closure cells
    and default arguments of functions (e.g. a tensor captured by a
    ``DeterministicParameter``'s ``forward_func``) and bound methods.
    """
    import types

    def convert(tensor):
        if dtype is not None and tensor.is_floating_point():
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    seen = set()
    atomic = (str, bytes, int, float, complex, bool, type, types.ModuleType, torch.dtype, torch.device,
              types.BuiltinFunctionType, torch.Generator)

    def visit(obj):
        """Return the converted object (only tensors and tuples are replaced)."""
        if isinstance(obj, torch.Tensor):
            return convert(obj)
        if obj is None or isinstance(obj, atomic):
            return obj
        if id(obj) in seen:
            return obj
        seen.add(id(obj))
        if isinstance(obj, tuple):
            items = [visit(item) for item in obj]
            if any(a is not b for a, b in zip(items, obj)):
                try:
                    return type(obj)(*items) if hasattr(obj, "_fields") else type(obj)(items)
                except TypeError:
                    return obj
            return obj
        if isinstance(obj, list):
            for index, item in enumerate(obj):
                obj[index] = visit(item)
            return obj
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                obj[key] = visit(obj[key])
            return obj
        if isinstance(obj, (set, frozenset)):
            return obj
        if isinstance(obj, types.FunctionType):
            for cell in obj.__closure__ or ():
                try:
                    contents = cell.cell_contents
                except ValueError:
                    continue
                new = visit(contents)
                if new is not contents:
                    cell.cell_contents = new
            if obj.__defaults__:
                obj.__defaults__ = tuple(visit(d) for d in obj.__defaults__)
            return obj
        if isinstance(obj, types.MethodType):
            visit(obj.__func__)
            visit(obj.__self__)
            return obj
        attributes = getattr(obj, "__dict__", None)
        if isinstance(attributes, dict):
            for key in list(attributes.keys()):
                value = attributes[key]
                new = visit(value)
                if new is not value:
                    attributes[key] = new
        for cls in type(obj).__mro__:
            for slot in cls.__dict__.get("__slots__", ()):
                if slot in ("__dict__", "__weakref__"):
                    continue
                try:
                    value = getattr(obj, slot)
                except AttributeError:
                    continue
                new = visit(value)
                if new is not value:
                    try:
                        setattr(obj, slot, new)
                    except (AttributeError, TypeError):
                        pass
        return obj

    visit(root)


class Model:
    """A probabilistic model: a directed acyclic graph of parameters, data and observations.

    Nodes are registered by creating ``Data``, ``RandomParameter``,
    ``ObservedParameter`` and ``DeterministicParameter`` objects inside a
    ``with Model() as model:`` block; leaving the block compiles the model
    (:meth:`compile`). Log densities are taken in the unconstrained space of
    every random parameter (with the log-Jacobian of its transform), and
    enumerated parameters are summed out.

    Attributes
    ----------
    params : dict[str, RandomParameter]
        The sampled random parameters.
    enumerated_params : dict[str, RandomParameter]
        Finite discrete parameters that are summed out instead of sampled.
    observed_params : dict[str, ObservedParameter]
        The likelihood terms.
    deterministic_params : dict[str, DeterministicParameter]
        Deterministic functions of other nodes.
    data : dict[str, Data]
        Mutable data inputs.
    graph : networkx.DiGraph
        The dependency graph; each node has a ``"type"`` attribute.
    n_chains : int or None
        The chain dimension while the model is batched (:meth:`batch`), else ``None``.
    """

    def __init__(self):
        self.data = {}
        self.params: dict[str, RandomParameter] = {}
        # Finite-support discrete parameters that are summed out of the joint
        # density instead of being sampled (see ``RandomParameter(enumerate=True)``).
        self.enumerated_params: dict[str, RandomParameter] = {}
        self.observed_params: dict[str, ObservedParameter] = {}
        self.deterministic_params: dict[str, DeterministicParameter] = {}
        self.graph = DiGraph()
        self._enumeration_groups = []
        self._enumeration_group_of: dict[str, object] = {}
        self._compiled = False
        self._affected_params: dict[str, list[RandomParameter]] = {}
        self._affected_observed: dict[str, list[ObservedParameter]] = {}
        self._all_params_list: list[RandomParameter] = []
        self._topo_order: list[str] = []
        # Batched (vectorized chains) evaluation; see BayesianDLL._batching.
        self.n_chains = None
        self._batch_mode = None
        self._batched_nodes: set[str] = set()
        self._alignment_cache = {}

    def __enter__(self):
        push_model(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pop_model(self)
        # Only compile a fully built model. When the block raised, compiling a
        # half-built graph could raise a second error that hides the first.
        if exc_type is None:
            self.compile()
        return False

    def compile(self):
        """Fix the topological order, the dependants of every parameter and the enumeration groups.

        Called when the ``with`` block ends and after :func:`condition` /
        :func:`marginalize`; call it again after adding nodes outside a block.
        """
        self._all_params_list = list(self.params.values())

        try:
            topo_order = list(nx.topological_sort(self.graph))
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            topo_order = list(self.graph.nodes)

        self._topo_order = topo_order

        self._affected_params = {}
        self._affected_observed = {}

        for name in self.params:
            visited = set()
            queue = deque([name])
            visited.add(name)

            while queue:
                curr = queue.popleft()
                for succ in self.graph.successors(curr):
                    if succ not in visited:
                        visited.add(succ)
                        queue.append(succ)

            self._affected_params[name] = [
                self.params[n] for n in visited if n in self.params
            ]
            self._affected_observed[name] = [
                self.observed_params[n] for n in visited if n in self.observed_params
            ]

        from ._enumeration import build_groups
        self._enumeration_groups = build_groups(self)
        self._enumeration_group_of = {
            term: group for group in self._enumeration_groups for term in group.terms
        }

        self._compiled = True

    def _markov_blanket(self, names):
        """Names of the log density terms (and graph nodes) that depend on ``names``.

        Returns ``(random_terms, observed_terms, nodes)``; ``random_terms``
        starts with ``names`` and may contain enumerated parameters.
        """
        names = [names] if isinstance(names, str) else list(names)
        nodes = set(names)
        random_terms = list(names)
        observed_terms = []
        queue = deque(names)
        while queue:
            current = queue.popleft()
            for successor in self.graph.successors(current):
                if successor in nodes:
                    continue
                if successor in self.deterministic_params:
                    nodes.add(successor)
                    queue.append(successor)
                elif successor in self.params or successor in self.enumerated_params:
                    nodes.add(successor)
                    random_terms.append(successor)
                elif successor in self.observed_params:
                    nodes.add(successor)
                    observed_terms.append(successor)
        return random_terms, observed_terms, nodes

    def _random_term(self, name):
        return self.params[name] if name in self.params else self.enumerated_params[name]

    def prepare_enumeration(self):
        """Check how every enumerated parameter is summed out (raises if it cannot be)."""
        if not self._compiled:
            self.compile()
        for group in self._enumeration_groups:
            group.prepare(self)
        return list(self._enumeration_groups)

    def _log_prob_of_terms(self, random_terms, observed_terms):
        """Sum the given prior and likelihood terms.

        Terms that belong to an enumeration group are replaced by the group's
        log marginal, which is added once per group.
        """
        if self.enumerated_params and not self._compiled:
            self.compile()
        if self.n_chains is not None:
            if self._batch_mode == "loop":
                return self._loop_over_chains(lambda: self._log_prob_of_terms(random_terms, observed_terms))
            return self._batched_log_prob_of_terms(random_terms, observed_terms)
        group_of = self._enumeration_group_of
        groups = []
        logp = 0.0
        for parameter in random_terms:
            group = group_of.get(parameter.name)
            if group is None:
                logp = logp + parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value)
            elif not any(group is g for g in groups):
                groups.append(group)
        for observed_parameter in observed_terms:
            group = group_of.get(observed_parameter.name)
            if group is None:
                logp = logp + observed_parameter.distribution.log_pdf(observed_parameter.observed_values).sum()
            elif not any(group is g for g in groups):
                groups.append(group)
        for group in groups:
            logp = logp + group.log_marginal(self)
        return logp

    def sample_enumerated(self):
        """Draw every enumerated parameter from its conditional posterior given the other values.

        The draws are set on the parameters. Returns ``{name: (value,
        probabilities)}``, where ``probabilities[..., k]`` is the posterior
        probability that a coordinate takes the ``k``-th support value.
        """
        if not self._compiled:
            self.compile()
        if self.n_chains is not None and self._batch_mode == "loop":
            draws = self._loop_over_chains(self.sample_enumerated)
            for name, (value, _) in draws.items():
                self.enumerated_params[name].set_constrained_value(value)
            return draws
        draws = {}
        with _batching.batched_evaluation(self.n_chains):
            for group in self._enumeration_groups:
                draws.update(group.sample(self))
        return draws

    def block_terms(self, names):
        """Return the log-density terms that depend on the parameters ``names``.

        These are the prior terms of the parameters themselves, of their random
        children and of the observed children (children reached through
        deterministic nodes included). All other terms of ``model_log_prob``
        are constant while only ``names`` change, so a Gibbs block can work
        with this Markov blanket instead of the full joint density.

        Returns ``(random_params, observed_params, nodes)``, where ``nodes``
        is the set of graph nodes needed to differentiate these terms with
        :meth:`joint_grad_log_prob`.

        When a term belongs to the factor group of an enumerated parameter,
        all terms of that group are included, because the summed-out group
        contributes a single, non-separable log marginal.
        """
        random_terms, observed_terms, nodes = self._markov_blanket(names)
        if self.enumerated_params:
            if not self._compiled:
                self.compile()
            for term in list(random_terms) + list(observed_terms):
                group = self._enumeration_group_of.get(term)
                if group is None:
                    continue
                for name in group.enumerated_names + group.random_terms:
                    if name not in random_terms:
                        random_terms.append(name)
                    nodes.add(name)
                for name in group.observed_terms:
                    if name not in observed_terms:
                        observed_terms.append(name)
                    nodes.add(name)
        return (
            [self._random_term(n) for n in random_terms],
            [self.observed_params[n] for n in observed_terms],
            nodes,
        )

    def block_log_prob(self, names):
        """Log density restricted to the terms that depend on ``names``.

        Differs from :meth:`model_log_prob` only by terms that are constant in
        the parameters ``names``.
        """
        random_terms, observed_terms, _ = self.block_terms(names)
        return self._log_prob_of_terms(random_terms, observed_terms)

    def model_log_prob(self):
        """Joint log density in the unconstrained space.

        Enumerated parameters are summed out, so this is the log density of
        the marginal model.
        """
        if not self._compiled:
            self.compile()
        random_terms = self._all_params_list + list(self.enumerated_params.values())
        return self._log_prob_of_terms(random_terms, list(self.observed_params.values()))

    def log_prob(self, name, theta):
        """Log density terms that change with parameter ``name``, at the unconstrained value ``theta``.

        The parameter is set to ``theta`` temporarily. The result is the sum of
        the prior terms (with Jacobians) of ``name`` and of the random
        parameters downstream of it and of the observed terms downstream of
        it, so it differs from :meth:`model_log_prob` by a constant in
        ``theta``. With enumerated parameters it is :meth:`block_log_prob`.
        """
        if not self._compiled:
            self.compile()

        if self.enumerated_params:
            with self.temporarily_set(name, theta):
                return self.block_log_prob(name)

        with self.temporarily_set(name, theta):
            logp = 0.0
            for param in self._affected_params[name]:
                logp += param.distribution._log_prob_unconstrained(param.unconstrained_value)
            for obs in self._affected_observed[name]:
                if obs.name in self.observed_params:
                    logp += obs.distribution.log_pdf(obs.observed_values).sum()

        return logp

    @contextmanager
    def temporarily_set(self, name, value):
        """Context manager: set the unconstrained value of parameter ``name`` and restore it afterwards."""
        try:
            old_value = self._random_term(name).unconstrained_value
            self._random_term(name).set_unconstrained_value(value)
            yield
        finally:
            self._random_term(name).set_unconstrained_value(old_value)
    
    @contextmanager
    def temporarily_set_many(self, values):
        """Temporarily set unconstrained values (enumerated parameters included)."""
        try:
            old_values = {}
            for name, value in values.items():
                parameter = self._random_term(name)
                old_values[name] = parameter.unconstrained_value
                parameter.set_unconstrained_value(value)
            yield
        finally:
            for name, old_value in old_values.items():
                self._random_term(name).set_unconstrained_value(old_value)

    def joint_grad_log_prob(self, param_names=None, nodes=None):
        """Gradient of the joint log density w.r.t. the unconstrained ``param_names``.

        ``nodes`` optionally restricts the backward pass to a set of graph
        nodes, e.g. the ``nodes`` returned by :meth:`block_terms`. Skipping the
        other nodes is exact for the parameters of that block, because no term
        outside their Markov blanket depends on them.

        Enumerated parameters are summed out: the gradient is the one of the
        marginal log density (see :mod:`BayesianDLL._enumeration`).
        """
        if not self._compiled:
            self.compile()

        if param_names is None:
            param_names = list(self.params.keys())

        if self.n_chains is not None:
            if self._batch_mode == "loop":
                return self._loop_over_chains(lambda: self.joint_grad_log_prob(param_names, nodes))
            with _batching.batched_evaluation(self.n_chains):
                return self._joint_grad(param_names, nodes)
        return self._joint_grad(param_names, nodes)

    def _joint_grad(self, param_names, nodes):
        if not self._enumeration_groups:
            return self._reverse_pass(param_names, nodes=nodes)

        grouped = self._enumeration_group_of
        result = self._reverse_pass(param_names, nodes=nodes, active_terms=lambda name: name not in grouped)
        for group in self._enumeration_groups:
            if nodes is not None and not any(term in nodes for term in group.terms):
                continue
            for name, grad in group.grad(self, param_names, nodes=nodes).items():
                result[name] = result[name] + grad
        return result

    def _reverse_pass(self, param_names, nodes=None, active_terms=None, term_weights=None):
        """Hand-written reverse pass over the model graph.

        ``active_terms`` (a container or a predicate on term names) selects the
        prior and likelihood terms that are differentiated; all terms by
        default. ``term_weights`` maps a term name to a tensor of the shape of
        its elementwise log density that multiplies that term elementwise
        (``None``: weight one), which yields the gradient of a weighted sum of
        the terms' elementwise log densities.
        """
        if self.n_chains is not None and self._batch_mode != "loop":
            return self._batched_reverse_pass(param_names, nodes=nodes, active_terms=active_terms, term_weights=term_weights)
        if active_terms is None:
            is_active = lambda name: True
        elif callable(active_terms):
            is_active = active_terms
        else:
            is_active = active_terms.__contains__
        term_weights = {} if term_weights is None else term_weights

        cotangents: dict[str, torch.Tensor] = {}

        def add_param_grads(distribution, value, weight):
            param_grads = distribution.log_pdf_param_grads(value)
            dependencies = _distribution_dependencies(distribution)
            for p_name, p_grad in param_grads.items():
                if p_name not in dependencies:
                    continue
                if p_name in self.params:
                    target_shape = self.params[p_name].constrained_value.shape
                elif p_name in self.deterministic_params:
                    target_shape = self.deterministic_params[p_name].constrained_value.shape
                else:
                    continue
                _add_grad(cotangents, p_name, _sum_to_shape(weight_gradient(p_grad, weight), target_shape))

        for node_name in reversed(self._topo_order):
            if nodes is not None and node_name not in nodes:
                continue
            if node_name in self.observed_params:
                if not is_active(node_name):
                    continue
                obs_param = self.observed_params[node_name]
                add_param_grads(obs_param.distribution, obs_param.observed_values, term_weights.get(node_name))

            elif node_name in self.deterministic_params:
                if node_name in cotangents:
                    det_param = self.deterministic_params[node_name]
                    bar_v = cotangents[node_name]
                    if getattr(det_param, "vjp_func", None) is not None:
                        # Vector-Jacobian product: no Jacobian is materialised.
                        input_values = det_param.input_values()
                        input_grads = det_param.vjp(bar_v, input_values)
                        seen = set()
                        for inp, value in zip(det_param.inputs, input_values):
                            inp_name = getattr(inp, "name", None)
                            if inp_name in seen or inp_name not in input_grads:
                                continue
                            seen.add(inp_name)
                            if inp_name in self.params or inp_name in self.deterministic_params:
                                _add_grad(cotangents, inp_name, _sum_to_shape(input_grads[inp_name], value.shape))
                        continue
                    local_derivatives = det_param.derivatives()
                    for inp in det_param.inputs:
                        inp_name = inp.name if hasattr(inp, "name") else None
                        if inp_name and (inp_name in self.params or inp_name in self.deterministic_params):
                            target_param = self.params[inp_name] if inp_name in self.params else self.deterministic_params[inp_name]
                            target_shape = target_param.constrained_value.shape
                            det_deriv = local_derivatives[inp_name]
                            if not isinstance(det_deriv, torch.Tensor):
                                det_deriv = torch.as_tensor(det_deriv, dtype=bar_v.dtype)

                            if (
                                det_deriv.ndim == 2
                                and det_deriv.shape[0] == bar_v.numel()
                                and det_deriv.shape[1] == target_param.constrained_value.numel()
                                and det_deriv.shape != bar_v.shape
                            ):
                                vjp = (bar_v.reshape(-1) @ det_deriv).reshape(target_shape)
                            else:
                                vjp = _sum_to_shape(bar_v * det_deriv, target_shape)

                            _add_grad(cotangents, inp_name, vjp)

            elif node_name in self.params or node_name in self.enumerated_params:
                if not is_active(node_name):
                    continue
                param = self._random_term(node_name)
                if getattr(param.distribution, "parameters", None):
                    add_param_grads(param.distribution, param.constrained_value, term_weights.get(node_name))

        result = {}
        for name in param_names:
            param = self.params[name]
            if is_active(name):
                prior_grad = param.distribution._log_prob_grad_unconstrained(param.unconstrained_value)
                prior_grad = weight_gradient(prior_grad, term_weights.get(name))
            else:
                prior_grad = torch.zeros_like(param.unconstrained_value)
            if name in cotangents:
                bar_p = cotangents[name]
                vjp = getattr(param.distribution.transform, "vjp", None)
                deriv = None if vjp is not None else param.distribution.transform.derivative(param.unconstrained_value)
                if vjp is not None:
                    likelihood_grad = vjp(param.unconstrained_value, bar_p)
                elif deriv.shape == param.unconstrained_value.shape:
                    likelihood_grad = bar_p * deriv
                else:
                    if deriv.ndim >= 2:
                        likelihood_grad = (deriv.transpose(-2, -1) @ bar_p.unsqueeze(-1)).squeeze(-1)
                    else:
                        likelihood_grad = bar_p @ deriv
                result[name] = prior_grad + likelihood_grad
            else:
                result[name] = prior_grad

        return result


    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------
    def _reference_tensor(self):
        for parameter in list(self.params.values()) + list(self.enumerated_params.values()):
            return parameter.unconstrained_value
        for data in self.data.values():
            return data.value
        for observed in self.observed_params.values():
            return torch.as_tensor(observed.observed_values)
        return None

    @property
    def device(self):
        """Device of the model's tensors (the device of its first parameter or data tensor)."""
        reference = self._reference_tensor()
        return reference.device if reference is not None else torch.device("cpu")

    @property
    def dtype(self):
        """Floating dtype of the model's parameters (the default dtype when there are none)."""
        for parameter in list(self.params.values()) + list(self.enumerated_params.values()):
            if parameter.unconstrained_value.is_floating_point():
                return parameter.unconstrained_value.dtype
        return torch.get_default_dtype()

    def to(self, device=None, dtype=None):
        """Move every tensor the model owns to ``device`` (and floating tensors to ``dtype``).

        The model is modified in place and returned. The objects of the model
        are walked generically, so this covers the values of ``Data``,
        parameters and observations, the constant tensors stored in
        distributions, transforms, deterministic nodes (including tensors
        captured by the closures of their functions), GP kernels and the
        supports of enumerated parameters.

        On a GPU, sample with ``sample(..., chain_method="vectorized")``
        (the default when the model lives on a CUDA/ROCm device): all chains
        then run in this process as one batched tensor computation. For AMD
        GPUs install the ROCm build of PyTorch; it exposes the GPU as device
        ``"cuda"``.

        Example::

            model.to("cuda")                      # or torch.device("cuda:0")
            result = model.sample(1000, 500, n_chains=64)
        """
        if device is None and dtype is None:
            return self
        device = self.device if device is None else torch.device(device)
        _move_tensors(self, device, dtype)
        self._clear_batch_caches()
        return self

    def _clear_batch_caches(self):
        self._alignment_cache = {}
        for parameter in self.deterministic_params.values():
            parameter._batched_cache = None

    # ------------------------------------------------------------------
    # Batched evaluation (vectorized chains)
    # ------------------------------------------------------------------
    def batch(self, n_chains, values=None, mode="auto", check=True):
        """Give every latent parameter a leading chain dimension of size ``n_chains``.

        The current (unbatched) values are replicated for every chain;
        ``values`` optionally maps parameter names to unconstrained values of
        shape ``(n_chains, *value_shape)``. Afterwards :meth:`model_log_prob`
        and :meth:`block_log_prob` return one value per chain (shape
        ``(n_chains,)``) and :meth:`joint_grad_log_prob` gradients of shape
        ``(n_chains, *value_shape)``.

        ``mode`` is ``"tensor"`` (one batched tensor computation),
        ``"loop"`` (the model is evaluated chain by chain; always correct but
        slower) or ``"auto"``: ``"tensor"`` unless a node of the model does
        not support batching. With ``check=True`` and ``mode="auto"`` the
        batched evaluation is compared once with chain-by-chain evaluation (of
        up to four chains) at randomly perturbed points, and the model falls
        back to ``"loop"`` (with a warning) when they differ, e.g. for a user
        ``DeterministicParameter`` whose function does not broadcast over a
        leading chain dimension. The reason of a fallback is stored in
        ``batch_fallback_reason``.

        The model is modified in place and returned; raises a
        ``RuntimeError`` if it is already batched.
        """
        if not self._compiled:
            self.compile()
        if self.n_chains is not None:
            raise RuntimeError("The model is already batched; call unbatch() first.")
        if mode not in ("auto", "tensor", "loop"):
            raise ValueError(f"mode must be 'auto', 'tensor' or 'loop', got {mode!r}.")
        n_chains = int(n_chains)
        if n_chains < 1:
            raise ValueError("n_chains must be at least 1.")
        for parameter in self.deterministic_params.values():
            try:
                parameter._ub_ndim = torch.as_tensor(parameter.constrained_value).ndim
            except Exception:
                parameter._ub_ndim = None
        batched = set(self.params) | set(self.enumerated_params)
        for name in self._topo_order:
            if name in self.deterministic_params and any(p in batched for p in self.graph.predecessors(name)):
                batched.add(name)
        self._batched_nodes = batched
        self._clear_batch_caches()
        for parameter in list(self.params.values()) + list(self.enumerated_params.values()):
            u = parameter.unconstrained_value
            c = parameter.constrained_value
            parameter.unconstrained_value = u.unsqueeze(0).expand(n_chains, *u.shape).clone()
            parameter.constrained_value = c.unsqueeze(0).expand(n_chains, *c.shape).clone()
        self.n_chains = n_chains
        reason = _non_batchable_reason(self)
        if mode == "loop" or (mode == "auto" and reason is not None):
            self._batch_mode = "loop"
        else:
            self._batch_mode = "tensor"
        self.batch_fallback_reason = reason if self._batch_mode == "loop" else None
        if values is not None:
            for name, value in values.items():
                self._random_term(name).set_unconstrained_value(value)
        if check and self._batch_mode == "tensor" and mode == "auto":
            self._check_batched_evaluation()
        return self

    def unbatch(self, chain=0):
        """Remove the chain dimension, keeping the values of chain ``chain``."""
        if self.n_chains is None:
            return self
        for parameter in list(self.params.values()) + list(self.enumerated_params.values()):
            parameter.unconstrained_value = parameter.unconstrained_value[chain].clone()
            parameter.constrained_value = parameter.constrained_value[chain].clone()
        self.n_chains = None
        self._batch_mode = None
        self._batched_nodes = set()
        self._clear_batch_caches()
        return self

    @contextmanager
    def batched(self, n_chains, values=None, mode="auto", check=True):
        """Context manager: :meth:`batch` the model and restore the unbatched values afterwards."""
        saved = {name: (p.unconstrained_value, p.constrained_value)
                 for name, p in {**self.params, **self.enumerated_params}.items()}
        self.batch(n_chains, values=values, mode=mode, check=check)
        try:
            yield self
        finally:
            self.n_chains = None
            self._batch_mode = None
            self._batched_nodes = set()
            self._clear_batch_caches()
            for name, (u, c) in saved.items():
                parameter = self._random_term(name)
                parameter.unconstrained_value = u
                parameter.constrained_value = c

    def _check_batched_evaluation(self, n_check=4):
        """Compare the batched evaluation with chain-by-chain evaluation; fall back to ``"loop"`` if they differ."""
        import warnings
        saved = {name: (p.unconstrained_value, p.constrained_value)
                 for name, p in {**self.params, **self.enumerated_params}.items()}
        continuous = [n for n, p in self.params.items() if p.distribution.transformed_state_space.is_continuous()]
        generator = torch.Generator().manual_seed(20261116)
        reason = None
        try:
            for name in continuous:
                u = saved[name][0]
                if u.is_floating_point():
                    noise = torch.randn(u.shape, generator=generator, dtype=u.dtype, device="cpu").to(u.device)
                    self.params[name].set_unconstrained_value(u + 0.3 * noise)
            chains = list(range(min(self.n_chains, n_check)))
            try:
                batched_lp = self.model_log_prob()
                batched_grads = self.joint_grad_log_prob(continuous) if continuous else {}
            except Exception as error:  # noqa: BLE001 - any failure means: not batchable
                reason = f"the batched evaluation raised {type(error).__name__}: {error}"
            else:
                mode = self._batch_mode
                self._batch_mode = "loop"
                try:
                    reference_lp = self._loop_over_chains(self.model_log_prob, chains=chains)
                    reference_grads = self._loop_over_chains(lambda: self.joint_grad_log_prob(continuous), chains=chains) if continuous else {}
                finally:
                    self._batch_mode = mode
                dtype = batched_lp.dtype if isinstance(batched_lp, torch.Tensor) else torch.float64
                rtol = 1e-6 if dtype == torch.float64 else 1e-3

                def same(a, b):
                    a = torch.as_tensor(a)
                    b = torch.as_tensor(b).to(device=a.device, dtype=a.dtype)
                    if a.shape != b.shape:
                        return False
                    scale = b.abs().max() if b.numel() and torch.isfinite(b).all() else 1.0
                    return bool(torch.isclose(a, b, rtol=rtol, atol=rtol * (1.0 + float(scale)), equal_nan=True).all())

                if not isinstance(batched_lp, torch.Tensor) or batched_lp.shape != (self.n_chains,):
                    reason = f"the batched log density has shape {tuple(torch.as_tensor(batched_lp).shape)} instead of ({self.n_chains},)"
                elif not same(batched_lp[chains], reference_lp):
                    reason = "the batched log density differs from the chain-by-chain evaluation"
                else:
                    for name in continuous:
                        if not same(batched_grads[name][chains], reference_grads[name]):
                            reason = f"the batched gradient of '{name}' differs from the chain-by-chain evaluation"
                            break
        finally:
            for name, (u, c) in saved.items():
                parameter = self._random_term(name)
                parameter.unconstrained_value = u
                parameter.constrained_value = c
        if reason is not None:
            self._batch_mode = "loop"
            self.batch_fallback_reason = reason
            warnings.warn(
                f"The model cannot be evaluated for all chains as one batched tensor computation ({reason}); "
                "the vectorized chains evaluate it chain by chain instead, which is slower. Nodes with "
                "user functions must broadcast over a leading chain dimension to be batched.",
                stacklevel=3,
            )

    def _loop_over_chains(self, function, chains=None):
        """Evaluate ``function`` unbatched once per chain and stack the results along a new first dimension."""
        n_chains = self.n_chains
        latents = {**self.params, **self.enumerated_params}
        saved = {name: (p.unconstrained_value, p.constrained_value) for name, p in latents.items()}
        chains = range(n_chains) if chains is None else chains
        results = []
        self.n_chains = None
        mode = self._batch_mode
        self._batch_mode = None
        try:
            for chain in chains:
                for name, parameter in latents.items():
                    u, c = saved[name]
                    parameter.unconstrained_value = u[chain]
                    parameter.constrained_value = c[chain]
                results.append(function())
        finally:
            self.n_chains = n_chains
            self._batch_mode = mode
            for name, parameter in latents.items():
                u, c = saved[name]
                parameter.unconstrained_value = u
                parameter.constrained_value = c
        return _stack_chain_results(results)

    def _is_batched_object(self, obj):
        name = getattr(obj, "name", None)
        if name is None or name not in self._batched_nodes:
            return False
        return (self.params.get(name) is obj or self.enumerated_params.get(name) is obj
                or self.deterministic_params.get(name) is obj)

    def _unbatched_ndim(self, obj, batched):
        from ._parameters import VariationalParameter
        if isinstance(obj, torch.Tensor):
            return obj.ndim
        if isinstance(obj, RandomParameter):
            return obj.constrained_value.ndim - (1 if batched else 0)
        if isinstance(obj, DeterministicParameter):
            if batched:
                return getattr(obj, "_ub_ndim", None)
            return torch.as_tensor(obj.constrained_value).ndim
        if isinstance(obj, VariationalParameter):
            return obj.value.ndim
        if isinstance(obj, (Data, ObservedParameter)):
            return torch.as_tensor(obj.constrained_value).ndim
        return None

    def _term_alignment(self, name):
        """``(pad, x_rank, z_rank, batched)`` for the log density term ``name`` (see BayesianDLL._batching)."""
        cached = self._alignment_cache.get(name)
        if cached is not None:
            return cached
        if name in self.observed_params:
            observed = self.observed_params[name]
            distribution = observed.distribution
            x_ndim = torch.as_tensor(observed.observed_values).ndim
            x_batched = False
        else:
            parameter = self._random_term(name)
            distribution = parameter.distribution
            x_ndim = parameter.constrained_value.ndim - 1
            x_batched = True
        event_shape = tuple(distribution.event_shape)
        event_ndim = len(event_shape)
        unconstrained_event_ndim = len(distribution.transform.unconstrained_shape(event_shape)) if event_ndim else 0
        batch_rank = max(x_ndim - event_ndim, 0)
        entries = []
        for argument, argument_event_ndim in _distribution_arguments(distribution):
            batched = self._is_batched_object(argument)
            ndim = self._unbatched_ndim(argument, batched)
            if ndim is None:
                continue
            batch_rank = max(batch_rank, ndim - argument_event_ndim)
            if batched:
                entries.append((argument, argument_event_ndim))
        pad = {id(argument): 1 + batch_rank + e for argument, e in entries}
        result = (pad, 1 + batch_rank + event_ndim, 1 + batch_rank + unconstrained_event_ndim, x_batched or bool(pad))
        self._alignment_cache[name] = result
        return result

    def _chain_zeros(self):
        reference = self._reference_tensor()
        dtype = reference.dtype if reference is not None and reference.is_floating_point() else torch.get_default_dtype()
        device = reference.device if reference is not None else None
        return torch.zeros(self.n_chains, dtype=dtype, device=device)

    def _chain_total(self, value, batched):
        """Sum an elementwise term to one value per chain (batched) or to a scalar."""
        if batched and isinstance(value, torch.Tensor) and value.ndim > 0:
            return value.reshape(value.shape[0], -1).sum(-1)
        return value.sum() if isinstance(value, torch.Tensor) else value

    def _prior_term(self, parameter):
        """Prior log density (with the Jacobian term) of a random parameter, per chain."""
        if self.n_chains is None:
            return parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value)
        pad, _, z_rank, _ = self._term_alignment(parameter.name)
        with _batching.batched_evaluation(self.n_chains, pad):
            return parameter.distribution._log_prob_unconstrained(pad_to(parameter.unconstrained_value, z_rank))

    def _elementwise_term(self, name):
        """Elementwise log density of the term ``name`` (without Jacobian terms).

        Batched terms have shape ``(n_chains, *log_density_shape)``.
        """
        if self.n_chains is None:
            if name in self.observed_params:
                observed = self.observed_params[name]
                return observed.distribution.log_pdf(observed.observed_values)
            parameter = self._random_term(name)
            return parameter.distribution.log_pdf(parameter.constrained_value)
        if self._batch_mode == "loop":
            return self._loop_over_chains(lambda: self._elementwise_term(name))
        pad, x_rank, _, _ = self._term_alignment(name)
        with _batching.batched_evaluation(self.n_chains, pad):
            if name in self.observed_params:
                observed = self.observed_params[name]
                return observed.distribution.log_pdf(observed.observed_values)
            parameter = self._random_term(name)
            return parameter.distribution.log_pdf(pad_to(parameter.constrained_value, x_rank))

    def _term_is_batched(self, name):
        return self.n_chains is not None and self._term_alignment(name)[3]

    def _batched_log_prob_of_terms(self, random_terms, observed_terms):
        group_of = self._enumeration_group_of
        groups = []
        logp = self._chain_zeros()
        with _batching.batched_evaluation(self.n_chains):
            for parameter in random_terms:
                group = group_of.get(parameter.name)
                if group is None:
                    logp = logp + self._prior_term(parameter)
                elif not any(group is g for g in groups):
                    groups.append(group)
            for observed_parameter in observed_terms:
                group = group_of.get(observed_parameter.name)
                if group is None:
                    value = self._elementwise_term(observed_parameter.name)
                    logp = logp + self._chain_total(value, self._term_is_batched(observed_parameter.name))
                elif not any(group is g for g in groups):
                    groups.append(group)
            for group in groups:
                logp = logp + group.log_marginal(self)
        return logp

    def prior_log_prob(self, names=None):
        """Sum of the prior terms (with Jacobians) of the random parameters ``names`` (all by default)."""
        names = list(self.params) if names is None else list(names)
        if self.n_chains is not None and self._batch_mode == "loop":
            return self._loop_over_chains(lambda: self.prior_log_prob(names))
        total = self._chain_zeros() if self.n_chains is not None else 0.0
        with _batching.batched_evaluation(self.n_chains):
            for name in names:
                total = total + self._prior_term(self.params[name])
        return total

    def _prior_gradient(self, names):
        """Gradient of the sum of all prior terms of the random parameters w.r.t. ``names``."""
        terms = set(self.params)
        if self.n_chains is not None and self._batch_mode == "loop":
            return self._loop_over_chains(lambda: self._reverse_pass(names, active_terms=terms))
        return self._reverse_pass(names, active_terms=terms)

    def _batched_reverse_pass(self, param_names, nodes=None, active_terms=None, term_weights=None):
        """Batched counterpart of :meth:`_reverse_pass` (gradients of shape ``(n_chains, *value_shape)``)."""
        if active_terms is None:
            is_active = lambda name: True
        elif callable(active_terms):
            is_active = active_terms
        else:
            is_active = active_terms.__contains__
        term_weights = {} if term_weights is None else term_weights
        n_chains = self.n_chains
        cotangents: dict[str, torch.Tensor] = {}
        deterministic_shapes = {}

        def target_of(name):
            if name in self.params:
                return self.params[name], self.params[name].constrained_value.shape
            if name in self.deterministic_params and name in self._batched_nodes:
                if name not in deterministic_shapes:
                    deterministic_shapes[name] = torch.as_tensor(self.deterministic_params[name].constrained_value).shape
                return self.deterministic_params[name], deterministic_shapes[name]
            return None, None

        def add_param_grads(term_name, distribution, value, weight):
            pad, x_rank, _, _ = self._term_alignment(term_name)
            if not pad:
                return
            with _batching.batched_evaluation(n_chains, pad):
                if term_name not in self.observed_params:
                    value = pad_to(value, x_rank)
                param_grads = distribution.log_pdf_param_grads(value)
            dependencies = _distribution_dependencies(distribution)
            for p_name, p_grad in param_grads.items():
                if p_name not in dependencies:
                    continue
                target, target_shape = target_of(p_name)
                if target is None or id(target) not in pad:
                    continue
                grad = weight_gradient(p_grad, weight)
                _add_grad(cotangents, p_name, sum_to_batched(grad, target_shape, batched_rank=pad[id(target)]))

        with _batching.batched_evaluation(n_chains):
            for node_name in reversed(self._topo_order):
                if nodes is not None and node_name not in nodes:
                    continue
                if node_name in self.observed_params:
                    if not is_active(node_name):
                        continue
                    observed = self.observed_params[node_name]
                    add_param_grads(node_name, observed.distribution, observed.observed_values, term_weights.get(node_name))
                elif node_name in self.deterministic_params:
                    if node_name not in cotangents:
                        continue
                    deterministic = self.deterministic_params[node_name]
                    input_values = deterministic.input_values()
                    input_grads = deterministic.batched_vjp(cotangents[node_name], input_values)
                    seen = set()
                    for inp, value in zip(deterministic.inputs, input_values):
                        inp_name = getattr(inp, "name", None)
                        if inp_name in seen or inp_name not in input_grads or input_grads[inp_name] is None:
                            continue
                        seen.add(inp_name)
                        if inp_name in self.params or (inp_name in self.deterministic_params and inp_name in self._batched_nodes):
                            _add_grad(cotangents, inp_name, sum_to_batched(input_grads[inp_name], value.shape))
                elif node_name in self.params or node_name in self.enumerated_params:
                    if not is_active(node_name):
                        continue
                    parameter = self._random_term(node_name)
                    if getattr(parameter.distribution, "parameters", None):
                        add_param_grads(node_name, parameter.distribution, parameter.constrained_value, term_weights.get(node_name))

            result = {}
            for name in param_names:
                parameter = self.params[name]
                distribution = parameter.distribution
                z = parameter.unconstrained_value
                pad, x_rank, z_rank, _ = self._term_alignment(name)
                with _batching.batched_evaluation(n_chains, pad):
                    z_aligned = pad_to(z, z_rank)
                    if is_active(name):
                        prior_grad = distribution._log_prob_grad_unconstrained(z_aligned)
                        prior_grad = weight_gradient(prior_grad, term_weights.get(name))
                        prior_grad = sum_to_batched(prior_grad, z.shape, batched_rank=z_rank)
                    else:
                        prior_grad = torch.zeros_like(z)
                    if name in cotangents:
                        bar_p = pad_to(cotangents[name], x_rank)
                        transform = distribution.transform
                        vjp = getattr(transform, "vjp", None)
                        if vjp is not None:
                            likelihood_grad = vjp(z_aligned, bar_p)
                        else:
                            derivative = transform.derivative(z_aligned)
                            if derivative.shape == z_aligned.shape or derivative.ndim <= z_aligned.ndim:
                                likelihood_grad = bar_p * derivative
                            else:
                                likelihood_grad = (derivative.transpose(-2, -1) @ bar_p.unsqueeze(-1)).squeeze(-1)
                        result[name] = prior_grad + sum_to_batched(likelihood_grad, z.shape, batched_rank=z_rank)
                    else:
                        result[name] = prior_grad
        return result


    def grad_log_prob(self, name, theta):
        """Gradient of the joint log density w.r.t. the unconstrained parameter ``name`` at ``theta``.

        The parameter is set to ``theta`` temporarily; returns a tensor of its
        unconstrained shape. Raises a ``RuntimeError`` if ``name`` is not a
        (sampled) random parameter.
        """
        if not self._compiled:
            self.compile()

        with self.temporarily_set(name, theta):
            if name not in self.params:
                raise RuntimeError("One is only able to compute the derivative with respect to a RandomParameter.")
            grads = self.joint_grad_log_prob([name])
            return grads[name]


    def sample(self, n_samples, warmup_length, n_chains=4, progress_bar=True, start_point_variance=1, blocks=None, seed=None, **sampler_params):
        """Sample the posterior; see :func:`BayesianDLL.sample` (e.g. ``chain_method="vectorized"``, ``device="cuda"``)."""
        from .Samplers import sample as _sample
        return _sample(n_samples, warmup_length, n_chains=n_chains, model=self, progress_bar=progress_bar, start_point_variance=start_point_variance, blocks=blocks, seed=seed, **sampler_params)

    def find_MAP(self, lr=1e-2, epochs=100, betas=(0.9, 0.999), callback_frequency=1, verbose=True):
        """Maximum a posteriori estimate of this model; see :func:`BayesianDLL.find_MAP`."""
        from .Variational import find_MAP as _find_MAP
        return _find_MAP(self, lr=lr, epochs=epochs, betas=betas, callback_frequency=callback_frequency, verbose=verbose)
    
    def posterior_predicative(self, trace, n_samples=None, samples_per_step=20, warmup_per_sample=20, progress_bar=True):
        """Posterior predictive draws for ``trace``; see :func:`BayesianDLL.posterior_predicative`."""
        from .Samplers import posterior_predicative as _pp
        return _pp(trace, n_samples=n_samples, samples_per_step=samples_per_step, warmup_per_sample=warmup_per_sample, model=self, progress_bar=progress_bar)

    def sample_prior_predicative(self, n_samples=20, samples_per_step=500, warmup_per_sample=100, progress_bar=True):
        """Prior predictive draws; see :func:`BayesianDLL.sample_prior_predicative`."""
        from .Samplers import sample_prior_predicative as _spp
        return _spp(n_samples, samples_per_step, warmup_per_sample, model=self, progress_bar=progress_bar)

    def condition(self, random_variable_names_and_values=None, **kwargs):
        """A conditioned copy of this model; see :func:`BayesianDLL.condition`."""
        return condition(self, random_variable_names_and_values, **kwargs)

    def marginalize(self, names):
        """A copy of this model with ``names`` summed out; see :func:`BayesianDLL.marginalize`."""
        return marginalize(self, names)


def condition(model: Model, random_variable_names_and_values: dict = None, **kwargs) -> Model:
    """Return a copy of ``model`` with some variables fixed to given values.

    Conditions are given as a dict (keys are names or the parameter objects)
    and/or as keyword arguments. A conditioned random (or enumerated)
    parameter becomes an ``ObservedParameter`` with that value; a
    conditioned observed parameter gets new observations; a conditioned
    ``Data`` node a new value. Values are reshaped (or broadcast) to the
    node's shape and must lie in its distribution's state space. The
    original model is not modified.

    Raises
    ------
    TypeError
        If ``model`` is not a :class:`Model` or the conditions are not a dict.
    ValueError
        For a deterministic node, a value that cannot be shaped like the node
        or one outside its state space.
    KeyError
        For an unknown name.
    """
    if not isinstance(model, Model):
        raise TypeError(f"model must be an instance of Model, got {type(model).__name__}.")

    conditions = {}
    if random_variable_names_and_values is not None:
        if not isinstance(random_variable_names_and_values, dict):
            raise TypeError(f"random_variable_names_and_values must be a dict, got {type(random_variable_names_and_values).__name__}.")
        conditions.update(random_variable_names_and_values)
    conditions.update(kwargs)

    # Validate everything on the original model first, so that nothing is
    # copied when a condition is invalid.
    data_values = {}
    observed_values = {}
    latent_values = {}
    for k, v in conditions.items():
        name = k.name if hasattr(k, "name") else str(k)

        if name in model.deterministic_params:
            raise ValueError(f"Cannot condition deterministic parameter '{name}'. Condition its input random variables instead.")

        if name in model.data:
            data_values[name] = v
            continue

        latent = name in model.params or name in model.enumerated_params
        if not latent and name not in model.observed_params:
            raise KeyError(f"Variable '{name}' not found in model parameters, observed parameters, or data.")

        if latent:
            param = model._random_term(name)
            dtype = param.constrained_value.dtype
            target_shape = param.constrained_value.shape
        else:
            param = model.observed_params[name]
            dtype = param.observed_values.dtype
            target_shape = param.observed_values.shape

        device = (param.constrained_value if latent else torch.as_tensor(param.observed_values)).device
        val_tensor = torch.as_tensor(v, dtype=dtype, device=device)
        if val_tensor.shape != target_shape:
            try:
                val_tensor = val_tensor.reshape(target_shape)
            except RuntimeError:
                try:
                    val_tensor = val_tensor.expand(target_shape).clone()
                except RuntimeError:
                    raise ValueError(f"Cannot shape conditioned value of shape {val_tensor.shape} to match parameter '{name}' shape {target_shape}.")

        if hasattr(param.distribution, "state_space") and hasattr(param.distribution.state_space, "contains"):
            contains = param.distribution.state_space.contains(val_tensor)
            if isinstance(contains, torch.Tensor):
                contains = contains.all().item()
            if not contains:
                raise ValueError(f"Conditioned value {val_tensor} for '{name}' is outside its distribution's state space.")

        if latent:
            latent_values[name] = val_tensor
        else:
            observed_values[name] = val_tensor

    # Each conditioned RandomParameter is replaced by a new ObservedParameter.
    # Registering the (still empty) replacement in the deepcopy memo makes every
    # reference to the random parameter in the copy (distributions of its
    # children, deterministic inputs, the model dictionaries) point to it.
    replacements = {}
    memo = {}
    for name in latent_values:
        shell = ObservedParameter.__new__(ObservedParameter)
        replacements[name] = shell
        memo[id(model._random_term(name))] = shell

    new_model = copy.deepcopy(model, memo)

    for name, val_tensor in latent_values.items():
        original = model._random_term(name)
        observed = ObservedParameter._from_random_parameter(
            original, val_tensor, memo=memo, instance=replacements[name],
        )
        new_model.params.pop(name, None)
        new_model.enumerated_params.pop(name, None)
        new_model.observed_params[name] = observed
        new_model.graph.nodes[name]["type"] = "observed"

    for name, val_tensor in observed_values.items():
        new_model.observed_params[name].observed_values = val_tensor

    for name, value in data_values.items():
        new_model.data[name].set_value(value)

    new_model.compile()
    return new_model


def marginalize(model: Model, names) -> Model:
    """Return a copy of ``model`` in which the parameters ``names`` are summed out.

    Equivalent to creating the parameters with ``RandomParameter(...,
    enumerate=True)``: the parameters must have a finite discrete support,
    they are no longer sampled, every log density and gradient of the new
    model is that of the marginal model, and :func:`~BayesianDLL.sample`
    adds exact posterior draws of them to the result.
    """
    from ._parameters import _check_enumerable
    if not isinstance(model, Model):
        raise TypeError(f"model must be an instance of Model, got {type(model).__name__}.")
    names = [names] if isinstance(names, str) or hasattr(names, "name") else list(names)
    names = [n.name if hasattr(n, "name") else str(n) for n in names]
    for name in names:
        if name in model.enumerated_params:
            continue
        if name not in model.params:
            raise KeyError(f"Random parameter '{name}' not found in the model.")
        _check_enumerable(name, model.params[name].distribution)

    new_model = copy.deepcopy(model)
    for name in names:
        if name in new_model.enumerated_params:
            continue
        parameter = new_model.params.pop(name)
        parameter.enumerated = True
        new_model.enumerated_params[name] = parameter
        new_model.graph.nodes[name]["enumerated"] = True
    new_model.compile()
    new_model.prepare_enumeration()
    return new_model
