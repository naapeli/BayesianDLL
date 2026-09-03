import torch
from contextlib import contextmanager
import networkx as nx
from networkx import DiGraph
from collections import deque

from ._active_model import _active_model
from ._parameters import RandomParameter, ObservedParameter, DeterministicParameter

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

    def grad_log_prob(self, name, theta):
        if not self._compiled:
            self.compile()

        with self.temporarily_set(name, theta):
            if name not in self.params:
                raise RuntimeError("One is only able to compute the derivative with respect to a RandomParameter.")
            
            param = self.params[name]
            n_param = param.unconstrained_value.numel()
            # Prior gradient: shape same as unconstrained_value
            grad = param.distribution._log_prob_grad_unconstrained(param.unconstrained_value)
            # Flatten to (n_param,) for accumulation
            grad_flat = grad.reshape(-1)

            # chain_derivative: Jacobian of current node w.r.t. the original param
            # shape: (n_current, n_param)
            stack = [(name, torch.eye(n_param, dtype=param.unconstrained_value.dtype))]

            while stack:
                current_name, chain_derivative = stack.pop()
                # chain_derivative shape: (n_current, n_param)

                for successor_name in self.graph.successors(current_name):
                    if successor_name in self.params:
                        successor_param = self.params[successor_name]
                        local_grad = successor_param.distribution.log_pdf_param_grads(successor_param.constrained_value)[current_name]
                        # local_grad shape: (*successor_shape) w.r.t. current
                        # We need: d(log_p_successor)/d(param) = d(log_p)/d(current) @ d(current)/d(param)
                        grad_flat = grad_flat + local_grad.reshape(-1) @ chain_derivative

                    elif successor_name in self.observed_params:
                        successor_param = self.observed_params[successor_name]
                        observed = successor_param.observed_values
                        distribution_grad = successor_param.distribution.log_pdf_param_grads(observed)[current_name]
                        # distribution_grad has shape (*obs_batch, *current_event)
                        # chain_derivative has shape (n_current, n_param)
                        # We need sum over obs of: d(log_p)/d(current) @ d(current)/d(param)
                        # = distribution_grad.reshape(n_obs, n_current) @ chain_derivative
                        n_current = chain_derivative.shape[0]
                        dg_flat = distribution_grad.reshape(-1, n_current)  # (n_obs, n_current)
                        # Sum over observations: (n_obs, n_current) @ (n_current, n_param) -> (n_obs, n_param) -> sum -> (n_param,)
                        likelihood_grad = (dg_flat @ chain_derivative).sum(dim=0)  # (n_param,)
                        # Apply transform Jacobian: d(constrained)/d(unconstrained)
                        deriv = param.distribution.transform.derivative(param.unconstrained_value)
                        if deriv.shape == param.unconstrained_value.shape:
                            # Element-wise: Jacobian is diagonal
                            grad_flat = grad_flat + likelihood_grad * deriv.reshape(-1)
                        else:
                            # Full Jacobian (e.g. SoftMax)
                            grad_flat = grad_flat + likelihood_grad @ deriv.squeeze(0)
                        break

                    elif successor_name in self.deterministic_params:
                        deterministic_param = self.deterministic_params[successor_name]
                        deterministic_derivative = deterministic_param.derivative(current_name)
                        # deterministic_derivative: d(det_output)/d(current_name)
                        # shape conceptually (n_det_output, n_current) but stored as tensor
                        n_current = chain_derivative.shape[0]
                        n_det = deterministic_derivative.numel() // n_current if n_current > 0 else deterministic_derivative.numel()
                        det_jac = deterministic_derivative.reshape(n_det, n_current)  # (n_det, n_current)
                        new_chain = det_jac @ chain_derivative  # (n_det, n_param)
                        stack.append((successor_name, new_chain))

                    elif self.graph.nodes[successor_name].get("type") == "observed":
                        # This happens during prior predicative sampling when observed_params is temporarily cleared
                        pass

                    else:
                        raise RuntimeError(f"Node {successor_name} not in the compute graph")

        return grad_flat.reshape(param.unconstrained_value.shape)


    def sample(self, n_samples, warmup_length, n_chains=4, progress_bar=True, start_point_variance=1):
        from .Samplers import sample as _sample
        return _sample(n_samples, warmup_length, n_chains=n_chains, model=self, progress_bar=progress_bar, start_point_variance=start_point_variance)

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
