from ._active_model import _active_model


class RandomParameter:
    def __init__(self, name, distribution, initial_value):
        self.name = name
        self.distribution = distribution
        self.constrained_value = initial_value
        self.unconstrained_value = self.distribution.transform.forward(initial_value)

        if _active_model._active_model is not None:
            _active_model._active_model.params[name] = self
        else:
            raise RuntimeError("One should select an active model before creating random variables.")

    def set_unconstrained_value(self, unconstrained_value):
        self.unconstrained_value = unconstrained_value
        self.constrained_value = self.distribution.transform.inverse(unconstrained_value)
    
    def set_constrained_value(self, constrained_value):
        self.constrained_value = constrained_value
        self.unconstrained_value = self.distribution.transform.forward(constrained_value)

class ObservedParameter:
    def __init__(self, name, distribution, observed_values):
        self.name = name
        self.distribution = distribution
        self.observed_values = observed_values

        if _active_model._active_model is not None:
            _active_model._active_model.observed_params[name] = self
        else:
            raise RuntimeError("One should select an active model before creating random variables.")

class DeterministicParameter:  # TransformedParameter
    def __init__(self, name):
        pass
