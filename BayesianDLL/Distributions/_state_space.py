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

class Union(StateSpace):
    def __init__(self, *spaces):
        self.state_spaces = spaces
        self._continuous = all(space.is_continuous() for space in spaces)

    def contains(self, x):
        return any(space.contains(x) for space in self.state_spaces)
    
    def is_continuous(self):
        return self._continuous
    
    def is_discrete(self):
        return not self._continuous

class DiscreteRange(DiscreteSpace):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def contains(self, state):
        low = resolve(self.low).item()
        high = resolve(self.high).item()
        values = torch.arange(low, high + 1)
        return state in values
    
    def __iter__(self):
        low = resolve(self.low).item()
        high = resolve(self.high).item()
        values = torch.arange(low, high + 1)
        for i in range(len(values)):
            yield values[i].item()
    
    def __len__(self):
        low = resolve(self.low).item()
        high = resolve(self.high).item()
        return high - low + 1

    def __getitem__(self, index):
        low = resolve(self.low).item()
        high = resolve(self.high).item()
        values = torch.arange(low, high + 1)
        return values[index].item()
    
    @property
    def values(self):
        low = resolve(self.low).item()
        high = resolve(self.high).item()
        return torch.arange(low, high + 1)

class DiscretePositive(DiscreteSpace):
    def contains(self, state):
        return state > 0 and (isinstance(state.item(), int) or state.item().is_integer())

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

class ContinuousReal(ContinuousSpace):
    def contains(self, state):
        return torch.all(torch.isfinite(state))

class ContinuousSimplex(ContinuousSpace):
    def contains(self, state):
        between_0_and_1 = torch.all((0 <= state) & (state <= 1))
        sum = torch.sum(state)
        sum_to_1 = torch.allclose(sum, torch.ones_like(sum))
        return between_0_and_1 & sum_to_1

class JointStateSpace(StateSpace):
    def __init__(self, spaces_dict, slices):
        self.spaces_dict = spaces_dict
        self.slices = slices
        self._is_continuous = all(s.is_continuous() for s in spaces_dict.values())

    def contains(self, theta_flat):
        for name, (start, end, shape) in self.slices.items():
            val = theta_flat[start:end].reshape(shape)
            if not self.spaces_dict[name].contains(val):
                return False
        return True

    def is_continuous(self):
        return self._is_continuous

    def is_discrete(self):
        return not self._is_continuous



class DiscreteNonNegative(DiscreteSpace):
    """The integers 0, 1, 2, ... (support of Poisson, NegativeBinomial, ...)."""

    def contains(self, state):
        state = torch.as_tensor(state)
        return bool(torch.all((state >= 0) & (state == torch.round(state))))

    def __iter__(self):
        i = 0
        while True:
            yield i
            i += 1


class ContinuousCorrCholesky(ContinuousSpace):
    """Lower-triangular Cholesky factors of correlation matrices (unit-norm rows, positive diagonal)."""

    def contains(self, state):
        state = torch.as_tensor(state)
        if state.ndim < 2 or state.shape[-1] != state.shape[-2]:
            return False
        tol = 1e3 * torch.finfo(state.dtype).eps if state.is_floating_point() else 0.0
        lower = bool(torch.all(torch.triu(state, diagonal=1).abs() <= tol))
        positive = bool(torch.all(torch.diagonal(state, dim1=-2, dim2=-1) > 0))
        unit_rows = bool(torch.allclose((state ** 2).sum(-1), torch.ones((), dtype=state.dtype, device=state.device)))
        return lower and positive and unit_rows
