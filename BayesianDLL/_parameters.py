import torch

from ._active_model import _active_model


class RandomParameter:
    def __init__(self, name, distribution, initial_value=None, shape=None, sampler="auto", **sampler_params):
        if initial_value is None:
            if shape is None:
                shape = distribution.shape
            elif isinstance(shape, int):
                shape = (shape,)
            
            if distribution.state_space.is_continuous():
                unconstrained = torch.zeros((1, *shape), dtype=torch.float64)
                initial_value = distribution.transform.inverse(unconstrained).squeeze(0)
            elif distribution.state_space.is_discrete():
                first_value = next(iter(distribution.state_space))
                initial_value = first_value * torch.ones(shape, dtype=torch.float64)

        if initial_value.ndim == 2 and initial_value.size(0) == 1:
            initial_value = initial_value.squeeze(0)

        if initial_value.ndim not in [0, 1]:
            raise ValueError(f"initial_value must be either 0 or 1 dimensional. Currently the shape is {initial_value.shape}.")

        self.name = name
        self.distribution = distribution
        self.constrained_value = initial_value.reshape(1, -1)
        self.unconstrained_value = self.distribution.transform.forward(self.constrained_value)
        self.sampler = sampler
        self.sampler_params = sampler_params
        self.state_space = self.distribution.state_space
        self.transformed_state_space = self.distribution.transformed_state_space

        if _active_model._active_model is not None:
            _active_model._active_model.params[name] = self  # TODO: remove this once nodes of the graph are objects and not strings
            graph = _active_model._active_model.graph
            graph.add_node(self.name, type="random")
            for parameter in self.distribution.parameters:
                if parameter in graph:
                    _active_model._active_model.graph.add_edge(parameter, self.name)
                else:
                    raise RuntimeError(f"{self.name} depends on {parameter}, which is not in the computation graph of the model.")
        else:
            raise RuntimeError("One should select an active model before creating random variables.")

    def set_unconstrained_value(self, unconstrained_value):
        if not isinstance(unconstrained_value, torch.Tensor):
            raise TypeError("unconstrained_value should be a torch.Tensor.")
        if unconstrained_value.ndim != 2:
            raise ValueError("unconstrained_value.shape should be (n_samples, n_features).")

        self.unconstrained_value = unconstrained_value
        self.constrained_value = self.distribution.transform.inverse(unconstrained_value)
    
    def set_constrained_value(self, constrained_value):
        if not isinstance(constrained_value, torch.Tensor):
            raise TypeError("constrained_value should be a torch.Tensor.")
        if constrained_value.ndim != 2:
            raise ValueError("constrained_value.shape should be (n_samples, n_features).")

        self.constrained_value = constrained_value
        self.unconstrained_value = self.distribution.transform.forward(constrained_value)

class ObservedParameter:
    def __init__(self, name, distribution, observed_values, sampler="auto", **sampler_params):
        self.name = name
        self.distribution = distribution
        self.observed_values = observed_values
        self.sampler = sampler  # for predicative sampling
        self.sampler_params = sampler_params

        if _active_model._active_model is not None:
            _active_model._active_model.observed_params[name] = self  # TODO: remove this once nodes of the graph are objects and not strings
            graph = _active_model._active_model.graph
            graph.add_node(self.name, type="observed")
            for parameter in self.distribution.parameters:
                if parameter in graph:
                    _active_model._active_model.graph.add_edge(parameter, self.name)
                else:
                    raise RuntimeError(f"{self.name} depends on {parameter}, which is not in the computation graph of the model.")
        else:
            raise RuntimeError("One should select an active model before creating random variables.")

class DeterministicParameter:
    def __init__(self, name, forward_func, derivative_func, inputs):
        self.name = name
        self.forward_func = forward_func
        self.derivative_func = derivative_func
        self.inputs = inputs
        self.owner_model = _active_model._active_model

        if _active_model._active_model is not None:
            _active_model._active_model.deterministic_params[name] = self  # TODO: remove this once nodes of the graph are objects and not strings
            graph = _active_model._active_model.graph
            graph.add_node(self.name, type="deterministic")
            for parameter in self.inputs:
                if parameter.name in graph:
                    _active_model._active_model.graph.add_edge(parameter.name, self.name)
                else:
                    raise RuntimeError(f"{self.name} depends on {parameter.name}, which is not in the computation graph of the model.")
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

    def _get_constrained_value(self, input):
        # model = _active_model._active_model
        if isinstance(input, torch.Tensor):
            return input
        if hasattr(input, "name"):
            if input.name in self.owner_model.params:
                return self.owner_model.params[input.name].constrained_value
            elif input.name in self.owner_model.deterministic_params:
                return self.owner_model.deterministic_params[input.name].constrained_value
            raise KeyError(f"Parameter '{input.name}' not found in the active model.")
        raise TypeError(f"Parameter {input} has an unkown type.")

class VariationalParameter:
    def __init__(self, name, value, min=-float("inf"), max=float("inf")):
        self.name = name
        self.value = torch.as_tensor(value, dtype=value.dtype if torch.is_tensor(value) else torch.float32).reshape(1, -1)
        self.min = min
        self.max = max

    def __repr__(self):
        return f"VariationalParameter {self.name}, value: {self.value}, limits: {(self.min, self.max)}"

    def set_new_value(self, value):
        value = value.reshape(self.value.shape)
        assert self.value.dtype == value.dtype
        self.value = torch.clamp(value, self.min, self.max)
