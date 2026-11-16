import torch
from contextlib import contextmanager
import networkx as nx
from networkx import DiGraph
from collections import deque
import copy

from ._active_model._active_model import push_model, pop_model
from ._parameters import RandomParameter, ObservedParameter, DeterministicParameter


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


def _sum_log_prob_terms(random_terms, observed_terms):
    logp = 0.0
    for parameter in random_terms:
        logp = logp + parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value)
    for observed_parameter in observed_terms:
        logp = logp + observed_parameter.distribution.log_pdf(observed_parameter.observed_values).sum()
    return logp


class Model:
    def __init__(self):
        self.data = {}
        self.params: dict[str, RandomParameter] = {}
        self.observed_params: dict[str, ObservedParameter] = {}
        self.deterministic_params: dict[str, DeterministicParameter] = {}
        self.graph = DiGraph()
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

        self._compiled = True

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
                elif successor in self.params:
                    nodes.add(successor)
                    random_terms.append(successor)
                elif successor in self.observed_params:
                    nodes.add(successor)
                    observed_terms.append(successor)
        return (
            [self.params[n] for n in random_terms],
            [self.observed_params[n] for n in observed_terms],
            nodes,
        )

    def block_log_prob(self, names):
        """Log density restricted to the terms that depend on ``names``.

        Differs from :meth:`model_log_prob` only by terms that are constant in
        the parameters ``names``.
        """
        random_terms, observed_terms, _ = self.block_terms(names)
        return _sum_log_prob_terms(random_terms, observed_terms)

    def model_log_prob(self):
        if not self._compiled:
            self.compile()

        logp = 0.0
        # priors
        for parameter in self._all_params_list:
            diff = parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value)
            logp += diff

        # likelihood
        for observed_parameter in self.observed_params.values():
            logp += observed_parameter.distribution.log_pdf(observed_parameter.observed_values).sum()

        return logp

    def log_prob(self, name, theta):
        if not self._compiled:
            self.compile()

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
            old_value = self.params[name].unconstrained_value
            self.params[name].set_unconstrained_value(value)
            yield
        finally:
            self.params[name].set_unconstrained_value(old_value)
    
    @contextmanager
    def temporarily_set_many(self, values):
        try:
            old_values = {}
            for name, value in values.items():
                old_values[name] = self.params[name].unconstrained_value
                self.params[name].set_unconstrained_value(value)
            yield
        finally:
            for name, old_value in old_values.items():
                self.params[name].set_unconstrained_value(old_value)

    def joint_grad_log_prob(self, param_names=None, nodes=None):
        """Gradient of the joint log density w.r.t. the unconstrained ``param_names``.

        ``nodes`` optionally restricts the backward pass to a set of graph
        nodes, e.g. the ``nodes`` returned by :meth:`block_terms`. Skipping the
        other nodes is exact for the parameters of that block, because no term
        outside their Markov blanket depends on them.
        """
        if not self._compiled:
            self.compile()

        if param_names is None:
            param_names = list(self.params.keys())

        cotangents: dict[str, torch.Tensor] = {}

        for node_name in reversed(self._topo_order):
            if nodes is not None and node_name not in nodes:
                continue
            if node_name in self.observed_params:
                obs_param = self.observed_params[node_name]
                observed = obs_param.observed_values
                param_grads = obs_param.distribution.log_pdf_param_grads(observed)
                dependencies = _distribution_dependencies(obs_param.distribution)
                for p_name, p_grad in param_grads.items():
                    if p_name not in dependencies:
                        continue
                    if p_name in self.params:
                        target_shape = self.params[p_name].constrained_value.shape
                        grad_p = _sum_to_shape(p_grad, target_shape)
                        cotangents[p_name] = cotangents[p_name] + grad_p if p_name in cotangents else grad_p
                    elif p_name in self.deterministic_params:
                        target_shape = self.deterministic_params[p_name].constrained_value.shape
                        grad_p = _sum_to_shape(p_grad, target_shape)
                        cotangents[p_name] = cotangents[p_name] + grad_p if p_name in cotangents else grad_p

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

                            cotangents[inp_name] = cotangents[inp_name] + vjp if inp_name in cotangents else vjp

            elif node_name in self.params:
                param = self.params[node_name]
                if getattr(param.distribution, "parameters", None):
                    param_grads = param.distribution.log_pdf_param_grads(param.constrained_value)
                    dependencies = _distribution_dependencies(param.distribution)
                    for p_name, p_grad in param_grads.items():
                        if p_name not in dependencies:
                            continue
                        if p_name in self.params:
                            target_shape = self.params[p_name].constrained_value.shape
                            grad_p = _sum_to_shape(p_grad, target_shape)
                            cotangents[p_name] = cotangents[p_name] + grad_p if p_name in cotangents else grad_p
                        elif p_name in self.deterministic_params:
                            target_shape = self.deterministic_params[p_name].constrained_value.shape
                            grad_p = _sum_to_shape(p_grad, target_shape)
                            cotangents[p_name] = cotangents[p_name] + grad_p if p_name in cotangents else grad_p

        result = {}
        for name in param_names:
            param = self.params[name]
            prior_grad = param.distribution._log_prob_grad_unconstrained(param.unconstrained_value)
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

        if name not in model.params and name not in model.observed_params:
            raise KeyError(f"Variable '{name}' not found in model parameters, observed parameters, or data.")

        if name in model.params:
            param = model.params[name]
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

        if name in model.params:
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
        memo[id(model.params[name])] = shell

    new_model = copy.deepcopy(model, memo)

    for name, val_tensor in latent_values.items():
        original = model.params[name]
        observed = ObservedParameter._from_random_parameter(
            original, val_tensor, memo=memo, instance=replacements[name],
        )
        new_model.params.pop(name)
        new_model.observed_params[name] = observed
        new_model.graph.nodes[name]["type"] = "observed"

    for name, val_tensor in observed_values.items():
        new_model.observed_params[name].observed_values = val_tensor

    for name, value in data_values.items():
        new_model.data[name].set_value(value)

    new_model.compile()
    return new_model
