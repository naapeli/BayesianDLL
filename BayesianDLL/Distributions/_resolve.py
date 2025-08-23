import torch
from .. import RandomParameter, DeterministicParameter


def resolve(parameter):
    if isinstance(parameter, torch.Tensor):
        return parameter
    elif isinstance(parameter, RandomParameter | DeterministicParameter):
        return parameter.constrained_value
    elif isinstance(parameter, int | float):
        return torch.as_tensor(parameter)
    elif isinstance(parameter, list | tuple):
            return torch.tensor(parameter)
    else:
        raise RuntimeError(f"Parameter {parameter} is not of type int, float, RandomParameter or DeterministicParameter.")
