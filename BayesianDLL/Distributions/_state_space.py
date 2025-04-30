import torch
from abc import ABC, abstractmethod


class StateSpace(ABC):
    @abstractmethod
    def contains(self, state) -> bool:
        pass

    @abstractmethod
    def is_continuous(self):
        pass

    @abstractmethod
    def is_discrete(self):
        pass

class DiscreteSpace(StateSpace, ABC):
    @abstractmethod
    def __iter__(self):
        pass
    
    def is_continuous(self):
        return False

    def is_discrete(self):
        return True

class ContinuousSpace(StateSpace, ABC):
    def is_continuous(self):
        return True

    def is_discrete(self):
        return False

class DiscreteRange(DiscreteSpace):
    def __init__(self, low, high, values=None):
        self.values = torch.arange(low, high + 1) if values is None else torch.as_tensor(values)

    def contains(self, state):
        return state in self.values
    
    def __iter__(self):
        for i in range(len(self.values)):
            yield self.values[i].item()

class DiscretePositive(DiscreteSpace):
    def contains(self, state):
        return state > 0 and state.item().is_integer()

    def __iter__(self):
        i = 1
        while True:
            yield i
            i += 1

class ContinuousRange(ContinuousSpace):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def contains(self, state):
        return torch.all((self.low <= state) & (state <= self.high))

class ContinuousPositive(ContinuousSpace):
    def contains(self, state):
        return torch.all(state > 0)

class ContinuousReal (ContinuousSpace):
    def contains(self, state):
        return torch.all(torch.isfinite(state))
