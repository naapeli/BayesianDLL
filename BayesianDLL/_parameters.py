import torch

from ._active_model import _active_model


class RandomParameter:
    def __init__(self, name, distribution, initial_value, sampler="auto", **sampler_params):
        if initial_value.ndim not in [0, 1]:
            raise ValueError("initial_value must be either 0 or 1 dimensional.")

        self.name = name
        self.distribution = distribution
        self.constrained_value = initial_value.reshape(1, -1)
        self.unconstrained_value = self.distribution.transform.forward(self.constrained_value)
        self.sampler = sampler
        self.sampler_params = sampler_params
        self.state_space = self.distribution.state_space
        self.transformed_state_space = self.distribution.transformed_state_space

        if _active_model._active_model is not None:
            _active_model._active_model.params[name] = self
        else:
            raise RuntimeError("One should select an active model before creating random variables.")

    def set_unconstrained_value(self, unconstrained_value):
        if not isinstance(unconstrained_value, torch.Tensor):
            raise TypeError("unconstrained_value should be a torch.Tensor.")
        if unconstrained_value.ndim != 2:
            raise ValueError("unconstrained_value.shape should be (n_samples, n_features).")

        self.unconstrained_value = unconstrained_value
        self.constrained_value = self.distribution.transform.inverse(unconstrained_value)
    
    # def set_constrained_value(self, constrained_value):
    #     if not isinstance(constrained_value, torch.Tensor):
    #         raise TypeError("constrained_value should be a torch.Tensor.")
    #     if constrained_value.ndim != 2:
    #         raise ValueError("constrained_value.shape should be (n_samples, n_features).")

    #     self.constrained_value = constrained_value
    #     self.unconstrained_value = self.distribution.transform.forward(constrained_value)

class ObservedParameter:
    def __init__(self, name, distribution, observed_values):
        self.name = name
        self.distribution = distribution
        self.observed_values = observed_values
        self.sampler = "metropolis"  # for predicative sampling
        self.sampler_params = {}

        if _active_model._active_model is not None:
            _active_model._active_model.observed_params[name] = self
        else:
            raise RuntimeError("One should select an active model before creating random variables.")

class DeterministicParameter:
    def __init__(self, name, forward_func, derivative_func, inputs):
        self.name = name
        self.forward_func = forward_func
        self.derivative_func = derivative_func
        self.inputs = inputs

        if _active_model._active_model is not None:
            _active_model._active_model.deterministic_params[name] = self
        else:
            raise RuntimeError("One should select an active model before creating random variables.")

    @property
    def constrained_value(self):
        inputs = [self._get_constrained_value(input) for input in self.inputs]
        return self.forward_func(*inputs)

    def derivative(self, name):
        inputs = [self._get_constrained_value(input) for input in self.inputs]
        local_derivative = self.derivative_func(*inputs)[name]
        return local_derivative
        # TODO: Currently chaining deterministic parameters is not possible as the derivative is not implemented

        # # Handle simple case: all inputs are leaf parameters
        # if all(not hasattr(input, "name") or input.name in _active_model._active_model.params for input in self.inputs):
        #     return local_derivative

        # # Chain rule: propagate through deterministic inputs
        # full_derivative = torch.zeros_like(local_derivative)
        # for i, input in enumerate(self.inputs):
        #     if hasattr(input, "name") and input.name in _active_model._active_model.deterministic_params:
        #         inner_derivative = _active_model._active_model.deterministic_params[input.name].derivative(name)
        #     elif hasattr(input, "name") and input.name in _active_model._active_model.params:
        #         # Identity for direct inputs
        #         inner_derivative = torch.eye(len(input.constrained_value))  # or appropriate shape
        #     else:
        #         # Constant tensor input
        #         inner_derivative = 0

        #     # Multiply partial derivative w.r.t. input[i] with its derivative
        #     full_derivative += local_derivative[i] @ inner_derivative

        # return full_derivative

    def _get_constrained_value(self, input):
        model = _active_model._active_model
        if isinstance(input, torch.Tensor):
            return input
        if hasattr(input, "name"):
            if input.name in model.params:
                return model.params[input.name].constrained_value
            elif input.name in model.deterministic_params:
                return model.deterministic_params[input.name].constrained_value
            raise KeyError(f"Parameter '{input.name}' not found in the active model.")
        raise TypeError(f"Parameter {input} has an unkown type.")
