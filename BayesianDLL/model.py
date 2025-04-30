import torch
from contextlib import contextmanager

from ._active_model import _active_model

class Model:
    def __init__(self):
        self.params = {}
        self.observed_params = {}

    def __enter__(self):
        _active_model._active_model = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _active_model._active_model = None

    def log_prob(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        with self.temporarily_set(theta):
            logp = 0.0
            # priors
            for parameter in self.params.values():
                logp += parameter.distribution._log_prob_unconstrained(parameter.unconstrained_value)

            # likelihood
            for parameter in self.observed_params.values():
                logp += parameter.distribution.log_pdf(parameter.observed_values).sum()

        return logp

    @contextmanager
    def temporarily_set(self, theta: dict[str, torch.Tensor]):
        old_values = {}
        try:
            # set new values to chosen parameters and store old ones
            for name, value in theta.items():
                old_values[name] = self.params[name].unconstrained_value
                self.params[name].set_unconstrained_value(value)
            yield
        finally:
            # Restore original values
            for name, value in old_values.items():
                self.params[name].set_unconstrained_value(value)

    # def grad_log_prob(self, theta_dict):
    #     theta = theta_dict or self.params
    #     grads = {name: torch.zeros_like(val) for name, val in theta.items()}
    #     for var in self.vars:
    #         if var.observed is None:
    #             val = theta[var.name]
    #             grads[var.name] += var.distribution.log_pdf_grad(val)
    #         else:
    #             param_grads = var.distribution.log_pdf_param_grads(var.observed)
    #             for pname, grad_val in param_grads.items():
    #                 if pname in grads:
    #                     grads[pname] += grad_val
    #     return grads
