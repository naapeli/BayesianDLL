import torch
from contextlib import contextmanager
import networkx as nx
from networkx import DiGraph
from collections import deque
import copy

from ._active_model._active_model import push_model, pop_model
from ._parameters import RandomParameter, ObservedParameter, DeterministicParameter
from ._enumeration import weight_gradient


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


class Model:
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
        self._topo_deterministic_params: list[DeterministicParameter] = []
        self._successors: dict[str, list[str]] = {}
        self._topo_order: list[str] = []

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
        self._all_params_list = list(self.params.values())
        self._successors = {n: list(self.graph.successors(n)) for n in self.graph.nodes}

        try:
            topo_order = list(nx.topological_sort(self.graph))
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            topo_order = list(self.graph.nodes)

        self._topo_order = topo_order
        self._topo_deterministic_params = [
            self.deterministic_params[n] for n in topo_order if n in self.deterministic_params
        ]

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
        draws = {}
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


    def grad_log_prob(self, name, theta):
        if not self._compiled:
            self.compile()

        with self.temporarily_set(name, theta):
            if name not in self.params:
                raise RuntimeError("One is only able to compute the derivative with respect to a RandomParameter.")
            grads = self.joint_grad_log_prob([name])
            return grads[name]


    def sample(self, n_samples, warmup_length, n_chains=4, progress_bar=True, start_point_variance=1, blocks=None, seed=None, **sampler_params):
        from .Samplers import sample as _sample
        return _sample(n_samples, warmup_length, n_chains=n_chains, model=self, progress_bar=progress_bar, start_point_variance=start_point_variance, blocks=blocks, seed=seed, **sampler_params)

    def find_MAP(self, lr=1e-2, epochs=100, betas=(0.9, 0.999), callback_frequency=1, verbose=True):
        from .Variational import find_MAP as _find_MAP
        return _find_MAP(self, lr=lr, epochs=epochs, betas=betas, callback_frequency=callback_frequency, verbose=verbose)
    
    def sample_posterior_predicative(self, n_samples=20, warmup_length=100, samples_per_step=500, warmup_per_sample=100, progress_bar=True):
        from .Samplers import sample_posterior_predicative as _spp
        return _spp(n_samples, warmup_length, samples_per_step, warmup_per_sample, model=self, progress_bar=progress_bar)

    def posterior_predicative(self, trace, n_samples=20, samples_per_step=500, warmup_per_sample=100, progress_bar=True):
        from .Samplers import posterior_predicative as _pp
        return _pp(trace, n_samples=n_samples, samples_per_step=samples_per_step, warmup_per_sample=warmup_per_sample, model=self, progress_bar=progress_bar)

    def sample_prior_predicative(self, n_samples=20, warmup_length=100, samples_per_step=500, warmup_per_sample=100, progress_bar=True):
        from .Samplers import sample_prior_predicative as _spp
        return _spp(n_samples, warmup_length, samples_per_step, warmup_per_sample, model=self, progress_bar=progress_bar)

    def condition(self, random_variable_names_and_values=None, **kwargs):
        return condition(self, random_variable_names_and_values, **kwargs)

    def marginalize(self, names):
        return marginalize(self, names)


class MeanFieldGuide(Model):
    pass


def condition(model: Model, random_variable_names_and_values: dict = None, **kwargs) -> Model:
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

        val_tensor = torch.as_tensor(v, dtype=dtype)
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
