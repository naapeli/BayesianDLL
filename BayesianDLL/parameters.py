import torch
import math

from . import _active_model


class Parameter:
    def __init__(self, name, distribution, init=None, observed=None):
        self.name = name
        self.distribution = distribution
        self.observed = observed
        self.init_value = init

        if _active_model is not None:
            _active_model.vars.append(self)
            if observed is None:
                _active_model.params[name] = init
