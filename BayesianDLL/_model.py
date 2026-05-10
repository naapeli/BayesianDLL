import torch
from contextlib import contextmanager
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

    def __enter__(self):
        _active_model._active_model = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _active_model._active_model = None

    def model_log_prob(self):
        logp = 0.0
        # priors
        for parameter in self.params.values():
            diff = parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value)
            logp += diff

        # likelihood
        for observed_parameter in self.observed_params.values():
            logp += observed_parameter.distribution.log_pdf(observed_parameter.observed_values).sum()

        return logp

    def log_prob(self, name, theta):
        with self.temporarily_set(name, theta):
            logp = 0
            node_queue = deque()
            node_queue.append(name)
            visited = {name}

            while node_queue:
                name = node_queue.popleft()
                if name in self.params:
                    parameter = self.params[name]
                    logp += parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value)
                elif name in self.observed_params:
                    observed_parameter = self.observed_params[name]
                    logp += observed_parameter.distribution.log_pdf(observed_parameter.observed_values).sum()
                elif name in self.deterministic_params:
                    pass
                elif self.graph.nodes[name].get("type") == "observed":
                    pass
                else:
                    raise RuntimeError(f"Node {name} not in the compute graph")
                
                for param in self.graph.successors(name):
                    if param not in visited:
                        visited.add(param)
                        node_queue.append(param)

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
        with self.temporarily_set(name, theta):
            if name not in self.params:
                raise RuntimeError("One is only able to compute the derivative with respect to a RandomParameter.")
            
            param = self.params[name]
            grad = param.distribution._log_prob_grad_unconstrained(param.unconstrained_value)

            stack = [(name, torch.eye(param.unconstrained_value.numel(), dtype=param.unconstrained_value.dtype))]

            while stack:
                current_name, chain_derivative = stack.pop()

                for successor_name in self.graph.successors(current_name):
                    if successor_name in self.params:
                        successor_param = self.params[successor_name]
                        local_grad = successor_param.distribution.log_pdf_param_grads(successor_param.constrained_value)[current_name]
                        grad = grad + local_grad @ chain_derivative

                    elif successor_name in self.observed_params:
                        successor_param = self.observed_params[successor_name]
                        if current_name == name:  # does not depend on a DeterministicParameter
                            distribution_grad = successor_param.distribution.log_pdf_param_grads(successor_param.observed_values)[current_name].sum(dim=0, keepdim=True)
                            transform_jacobian = param.distribution.transform.derivative(param.unconstrained_value)
                            grad = grad + distribution_grad @ transform_jacobian.squeeze(0)
                            break
                        else:  # depends on a DeterministicParameter
                            distribution_grad = successor_param.distribution.log_pdf_param_grads(successor_param.observed_values)[current_name]
                            transformed_grad = distribution_grad.T @ chain_derivative
                            transform_jacobian = param.distribution.transform.derivative(param.unconstrained_value)
                            grad = grad + transformed_grad @ transform_jacobian.squeeze(0)
                            break

                    elif successor_name in self.deterministic_params:
                        deterministic_param = self.deterministic_params[successor_name]
                        deterministic_derivative = deterministic_param.derivative(current_name)
                        stack.append((successor_name, deterministic_derivative @ chain_derivative.T))

                    elif self.graph.nodes[successor_name].get("type") == "observed":
                        # This happens during prior predicative sampling when observed_params is temporarily cleared
                        pass

                    else:
                        raise RuntimeError(f"Node {successor_name} not in the compute graph")

        return grad


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
