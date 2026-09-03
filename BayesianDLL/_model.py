import torch
from contextlib import contextmanager
import networkx as nx
from networkx import DiGraph
from collections import deque

from ._active_model import _active_model
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


class Model:
    def __init__(self):
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
        _active_model._active_model = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _active_model._active_model = None
        self.compile()

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

    def joint_grad_log_prob(self, param_names=None):
        if not self._compiled:
            self.compile()

        if param_names is None:
            param_names = list(self.params.keys())

        cotangents: dict[str, torch.Tensor] = {}

        for node_name in reversed(self._topo_order):
            if node_name in self.observed_params:
                obs_param = self.observed_params[node_name]
                observed = obs_param.observed_values
                param_grads = obs_param.distribution.log_pdf_param_grads(observed)
                for p_name, p_grad in param_grads.items():
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
                    for inp in det_param.inputs:
                        inp_name = inp.name if hasattr(inp, "name") else None
                        if inp_name and (inp_name in self.params or inp_name in self.deterministic_params):
                            target_param = self.params[inp_name] if inp_name in self.params else self.deterministic_params[inp_name]
                            target_shape = target_param.constrained_value.shape
                            det_deriv = det_param.derivative(inp_name)
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
                    for p_name, p_grad in param_grads.items():
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
                deriv = param.distribution.transform.derivative(param.unconstrained_value)
                if deriv.shape == param.unconstrained_value.shape:
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


    def sample(self, n_samples, warmup_length, n_chains=4, progress_bar=True, start_point_variance=1, blocks=None, **sampler_params):
        from .Samplers import sample as _sample
        return _sample(n_samples, warmup_length, n_chains=n_chains, model=self, progress_bar=progress_bar, start_point_variance=start_point_variance, blocks=blocks, **sampler_params)

    def find_MAP(self, lr=1e-2, epochs=100, betas=(0.9, 0.999), callback_frequency=1, verbose=True):
        from .Variational import find_MAP as _find_MAP
        return _find_MAP(self, lr=lr, epochs=epochs, betas=betas, callback_frequency=callback_frequency, verbose=verbose)
    
    def sample_posterior_predicative(self, n_samples=20, warmup_length=100, samples_per_step=500, warmup_per_sample=100, progress_bar=True):
        from .Samplers import sample_posterior_predicative as _spp
        return _spp(n_samples, warmup_length, samples_per_step, warmup_per_sample, model=self, progress_bar=progress_bar)

    def sample_prior_predicative(self, n_samples=20, warmup_length=100, samples_per_step=500, warmup_per_sample=100, progress_bar=True):
        from .Samplers import sample_prior_predicative as _spp
        return _spp(n_samples, warmup_length, samples_per_step, warmup_per_sample, model=self, progress_bar=progress_bar)

class MeanFieldGuide(Model):
    pass
