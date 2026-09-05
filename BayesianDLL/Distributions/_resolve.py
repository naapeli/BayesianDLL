import torch
from .. import RandomParameter, DeterministicParameter, VariationalParameter, ObservedParameter
from .._data import Data


def resolve(parameter):
    if isinstance(parameter, torch.Tensor):
        return parameter
    elif isinstance(parameter, RandomParameter | DeterministicParameter | Data | ObservedParameter):
        return parameter.constrained_value
    elif isinstance(parameter, VariationalParameter):
         return parameter.value
    elif isinstance(parameter, int | float):
        return torch.as_tensor(parameter)
    elif isinstance(parameter, list | tuple):
            return torch.tensor(parameter)
    else:
        raise RuntimeError(f"Parameter {parameter} is not of type int, float, RandomParameter, DeterministicParameter or ObservedParameter.")
