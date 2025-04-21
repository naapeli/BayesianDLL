import torch

from . import _active_model

class Model:
    def __init__(self):
        self.vars = []
        self.params = {}
        self.observed_vars = {}

    def __enter__(self):
        global _active_model
        _active_model = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_model
        _active_model = None

    def log_prob(self, theta: dict[str, torch.Tensor]) -> torch.Tensor:
        logp = 0.0
        for var in self.vars:
            if var.observed is not None:
                logp += var.distribution.log_pdf(var.observed).sum()
            else:
                val = theta[var.name]
                logp += var.distribution.log_pdf(val)
        return logp

    def grad_log_prob(self, theta_dict):
        theta = theta_dict or self.params
        grads = {name: torch.zeros_like(val) for name, val in theta.items()}
        for var in self.vars:
            if var.observed is None:
                val = theta[var.name]
                grads[var.name] += var.distribution.log_pdf_grad(val)
            else:
                param_grads = var.distribution.log_pdf_param_grads(var.observed)
                for pname, grad_val in param_grads.items():
                    if pname in grads:
                        grads[pname] += grad_val
        return grads


class PosteriorWrapper:
    def __init__(self, model):
        self.model = model

    def log_pdf(self, theta_vec):
        keys = list(self.model.params.keys())
        theta_dict = {k: v for k, v in zip(keys, theta_vec)}
        return self.model.log_prob(theta_dict)

    def log_pdf_grad(self, theta_vec):
        keys = list(self.model.params.keys())
        theta_dict = {k: v for k, v in zip(keys, theta_vec)}
        grads_dict = self.model.grad_log_prob(theta_dict)
        return torch.stack([grads_dict[k] for k in keys])
