import torch
from abc import ABC, abstractmethod
from ._resolve import resolve


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
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def contains(self, state):
        low = resolve(self.low)
        high = resolve(self.high)
        values = torch.arange(low, high + 1)
        return state in values
    
    def __iter__(self):
        low = resolve(self.low)
        high = resolve(self.high)
        values = torch.arange(low, high + 1)
        for i in range(len(values)):
            yield values[i].item()
    
    def __len__(self):
        low = resolve(self.low).item()
        high = resolve(self.high).item()
        return high - low + 1

    def __getitem__(self, index):
        low = resolve(self.low)
        high = resolve(self.high)
        values = torch.arange(low, high + 1)
        return values[index].item()
    
    @property
    def values(self):
        low = resolve(self.low)
        high = resolve(self.high)
        return torch.arange(low, high + 1)

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
