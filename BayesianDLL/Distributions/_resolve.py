import torch
from .. import RandomParameter, DeterministicParameter, VariationalParameter, ObservedParameter
from .._data import Data
from .._batching import aligned


def resolve(parameter):
    """Current tensor value of a distribution parameter.

    Tensors are returned as they are, model nodes give their constrained
    value (inside a batched evaluation with the chain dimension aligned),
    variational parameters their value, and numbers, lists and tuples are
    converted to tensors.
    """
    if isinstance(parameter, torch.Tensor):
        return parameter
    elif isinstance(parameter, RandomParameter | DeterministicParameter):
        # Inside a batched term evaluation, batched nodes are aligned (see
        # BayesianDLL._batching).
        return aligned(parameter, parameter.constrained_value)
    elif isinstance(parameter, Data | ObservedParameter):
        return parameter.constrained_value
    elif isinstance(parameter, VariationalParameter):
         return parameter.value
    elif isinstance(parameter, int | float):
        return torch.as_tensor(parameter)
    elif isinstance(parameter, list | tuple):
            return torch.tensor(parameter)
    else:
        raise RuntimeError(f"Parameter {parameter} is not of type int, float, RandomParameter, DeterministicParameter or ObservedParameter.")
