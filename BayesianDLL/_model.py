import torch
from contextlib import contextmanager
from networkx import DiGraph
from collections import deque

from ._active_model import _active_model

class Model:
    def __init__(self):
        self.params = {}
        self.observed_params = {}
        self.deterministic_params = {}
        self.graph = DiGraph()

    def __enter__(self):
        _active_model._active_model = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _active_model._active_model = None

    # def log_prob(self, name, theta):
    #     with self.temporarily_set(name, theta):
    #         logp = 0.0
    #         # priors
    #         for parameter in self.params.values():
    #             logp += parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value)

    #         # likelihood
    #         for observed_parameter in self.observed_params.values():
    #             logp += observed_parameter.distribution.log_pdf(observed_parameter.observed_values).sum()
        
    #     return logp

    def log_prob(self, name, theta):
        with self.temporarily_set(name, theta):
            logp = 0
            node_queue = deque()
            node_queue.append(name)
            visited = set(name)

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
                else:
                    raise RuntimeError(f"Node {parameter} not in the compute graph")
                
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

                    else:
                        raise RuntimeError(f"Node {successor_name} not in the compute graph")

        return grad
