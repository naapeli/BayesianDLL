import torch
from abc import ABC, abstractmethod


class StateSpace(ABC):
    @abstractmethod
    def contains(self, state) -> bool:
        pass

    @abstractmethod
    def __iter__(self):
        pass

class DiscreteRange(StateSpace):
    def __init__(self, low, high, values=None):
        self.values = torch.arange(low, high + 1) if values is None else torch.as_tensor(values)

    def contains(self, state):
        return state in self.values
    
    def __iter__(self):
        for i in range(len(self.values)):
            yield self.values[i].item()

class DiscretePositive(StateSpace):
    def contains(self, state):
        return state > 0 and state.item().is_integer()

    def __iter__(self):
        i = 1
        while True:
            yield i
            i += 1
