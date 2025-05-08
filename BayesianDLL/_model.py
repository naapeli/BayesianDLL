import torch
from contextlib import contextmanager

from ._active_model import _active_model

class Model:
    def __init__(self):
        self.params = {}
        self.observed_params = {}
        self.deterministic_params = {}

    def __enter__(self):
        _active_model._active_model = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _active_model._active_model = None

    def log_prob(self, name, theta):
        with self.temporarily_set(name, theta):
            logp = 0.0
            # priors
            for parameter in self.params.values():
                logp += parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value)

            # likelihood
            for observed_parameter in self.observed_params.values():
                logp += observed_parameter.distribution.log_pdf(observed_parameter.observed_values).sum()
        
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
            grad = torch.zeros_like(theta)
            param = self.params[name]

            # handle the observed parameters
            for observed_parameter in self.observed_params.values():

                # if the observed depends on deterministic parameters
                for deterministic_parameter_name in observed_parameter.distribution.deterministic_parameters:
                    for _ in [input.name for input in self.deterministic_params[deterministic_parameter_name].inputs if input.name == name]:
                        grad += (self.deterministic_params[deterministic_parameter_name].derivative(name).T @ observed_parameter.distribution.log_pdf_param_grads(observed_parameter.observed_values)[deterministic_parameter_name]).T * param.distribution.transform.derivative(param.unconstrained_value)

                # otherwise if the observed depends straight up on the random variable
                if observed_parameter.distribution._depends_on_random_variable(name):
                    grad += observed_parameter.distribution.log_pdf_param_grads(observed_parameter.observed_values)[name].sum(dim=0) * param.distribution.transform.derivative(param.unconstrained_value)  # TODO: Make sure the dimensions work (.sum(dim=0) or .sum())

            grad += param.distribution._log_prob_grad_unconstrained(param.unconstrained_value)
        
        return grad
    
    def set_parameter_values(self, constrained_parameters: dict[str, torch.Tensor]):
        for name, value in constrained_parameters.items():
            self.params[name].set_constrained_value(value)
