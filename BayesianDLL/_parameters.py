import torch

from ._active_model import _active_model
from ._plate import get_active_plates
from ._data import Data


class RandomParameter:
    def __init__(self, name, distribution, initial_value=None, shape=None, sampler="auto", **sampler_params):
        # Capture active plates at creation time
        self.plates = get_active_plates()

        # Determine the event shape from distribution or explicit shape
        if shape is not None:
            event_shape = (shape,) if isinstance(shape, int) else tuple(shape)
        else:
            event_shape = distribution.event_shape
            if event_shape == ():
                event_shape = (1,)  # minimum 1D for scalar distributions

        # Compute the full value shape: plate_shape + event_shape
        plate_shape = tuple(p.size for p in self.plates)
        full_shape = plate_shape + event_shape

        if initial_value is None:
            if distribution.state_space.is_continuous():
                unconstrained = torch.zeros(full_shape, dtype=torch.float64)
                initial_value = distribution.transform.inverse(unconstrained)
            elif distribution.state_space.is_discrete():
                first_value = next(iter(distribution.state_space))
                initial_value = first_value * torch.ones(full_shape, dtype=torch.float64)
        else:
            # Ensure initial_value has the right shape
            if initial_value.shape != full_shape:
                try:
                    initial_value = initial_value.reshape(full_shape)
                except RuntimeError:
                    initial_value = initial_value.expand(full_shape).clone()

        self.name = name
        self.distribution = distribution
        self.event_shape = event_shape
        self.constrained_value = initial_value
        self.unconstrained_value = self.distribution.transform.forward(self.constrained_value)
        self.sampler = sampler
        self.sampler_params = sampler_params
        self.state_space = self.distribution.state_space
        self.transformed_state_space = self.distribution.transformed_state_space

        if _active_model._active_model is not None:
            _active_model._active_model.params[name] = self  # TODO: remove this once nodes of the graph are objects and not strings
            _active_model._active_model._compiled = False
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
        self.unconstrained_value = unconstrained_value
        self.constrained_value = self.distribution.transform.inverse(unconstrained_value)
    
    def set_constrained_value(self, constrained_value):
        if not isinstance(constrained_value, torch.Tensor):
            raise TypeError("constrained_value should be a torch.Tensor.")
        self.constrained_value = constrained_value
        self.unconstrained_value = self.distribution.transform.forward(constrained_value)


class ObservedParameter:
    def __init__(self, name, distribution, observed_values, sampler="auto", **sampler_params):
        self.name = name
        self.distribution = distribution
        self.observed_values = observed_values
        self.sampler = sampler  # for predicative sampling
        self.sampler_params = sampler_params
        self.plates = get_active_plates()

        if _active_model._active_model is not None:
            _active_model._active_model.observed_params[name] = self  # TODO: remove this once nodes of the graph are objects and not strings
            _active_model._active_model._compiled = False
            graph = _active_model._active_model.graph
            graph.add_node(self.name, type="observed")
            if isinstance(self._observed_values, Data):
                if _active_model._active_model.data.get(self._observed_values.name) is not self._observed_values:
                    raise RuntimeError("Observed data must belong to the active model.")
                graph.add_edge(self._observed_values.name, self.name)
            for parameter in self.distribution.parameters:
                if parameter in graph:
                    _active_model._active_model.graph.add_edge(parameter, self.name)
                else:
                    raise RuntimeError(f"{self.name} depends on {parameter}, which is not in the computation graph of the model.")
        else:
            raise RuntimeError("One should select an active model before creating random variables.")


    @property
    def observed_values(self):
        if isinstance(self._observed_values, Data):
            return self._observed_values.value
        return self._observed_values

    @observed_values.setter
    def observed_values(self, values):
        self._observed_values = values

    @property
    def predictive_shape(self):
        """Shape of a generated value, with plate sizes resolved at runtime."""
        shape = list(self.observed_values.shape)
        for index, plate_info in enumerate(self.plates):
            if index < len(shape):
                shape[index] = plate_info.size
            else:
                shape.append(plate_info.size)
        return torch.Size(shape)


class DeterministicParameter:
    def __init__(self, name, forward_func, derivative_func, inputs):
        self.name = name
        self.forward_func = forward_func
        self.derivative_func = derivative_func
        self.inputs = inputs
        self.owner_model = _active_model._active_model
        self.plates = get_active_plates()

        if _active_model._active_model is not None:
            _active_model._active_model.deterministic_params[name] = self  # TODO: remove this once nodes of the graph are objects and not strings
            _active_model._active_model._compiled = False
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
        if isinstance(input, torch.Tensor):
            return input
        if hasattr(input, "name"):
            if input.name in self.owner_model.params:
                return self.owner_model.params[input.name].constrained_value
            elif input.name in self.owner_model.deterministic_params:
                return self.owner_model.deterministic_params[input.name].constrained_value
            elif input.name in self.owner_model.data:
                return self.owner_model.data[input.name].value
            raise KeyError(f"Parameter '{input.name}' not found in the active model.")
        raise TypeError(f"Parameter {input} has an unkown type.")


class VariationalParameter:
    def __init__(self, name, value, min=-float("inf"), max=float("inf")):
        self.name = name
        self.value = torch.as_tensor(value, dtype=value.dtype if torch.is_tensor(value) else torch.float32)
        if self.value.ndim == 0:
            self.value = self.value.unsqueeze(0)
        self.min = min
        self.max = max

    def __repr__(self):
        return f"VariationalParameter {self.name}, value: {self.value}, limits: {(self.min, self.max)}"

    def set_new_value(self, value):
        value = value.reshape(self.value.shape)
        assert self.value.dtype == value.dtype
        self.value = torch.clamp(value, self.min, self.max)
