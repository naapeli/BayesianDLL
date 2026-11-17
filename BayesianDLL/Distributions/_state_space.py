import torch
from abc import ABC, abstractmethod
from ._resolve import resolve


def _as_tensor(state):
    return state if isinstance(state, torch.Tensor) else torch.as_tensor(state)


def _bound_like(bound, state):
    """Resolve a (possibly random) bound to a tensor on the device of ``state``."""
    bound = resolve(bound)
    return bound.to(device=state.device)


def _is_integer_valued(state):
    if not (state.is_floating_point() or state.is_complex()):
        return torch.ones_like(state, dtype=torch.bool)
    return torch.isfinite(state) & (state == torch.round(state))


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

    def contains(self, x) -> bool:
        return any(bool(space.contains(x)) for space in self.state_spaces)
    
    def is_continuous(self):
        return self._continuous
    
    def is_discrete(self):
        return not self._continuous

class DiscreteRange(DiscreteSpace):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def contains(self, state) -> bool:
        """Return whether every element of ``state`` is an integer in ``[low, high]``."""
        state = _as_tensor(state)
        low = _bound_like(self.low, state)
        high = _bound_like(self.high, state)
        valid = (state >= low) & (state <= high) & _is_integer_valued(state)
        return bool(valid.all())
    
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
    def contains(self, state) -> bool:
        """Return whether every element of ``state`` is a positive integer."""
        state = _as_tensor(state)
        return bool(((state > 0) & _is_integer_valued(state)).all())

    def __iter__(self):
        i = 1
        while True:
            yield i
            i += 1

class ContinuousRange(ContinuousSpace):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def contains(self, state) -> bool:
        state = _as_tensor(state)
        low = _bound_like(self.low, state)
        high = _bound_like(self.high, state)
        return bool(torch.all((low <= state) & (state <= high)))

class ContinuousPositive(ContinuousSpace):
    def contains(self, state) -> bool:
        return bool(torch.all(_as_tensor(state) > 0))

class ContinuousReal(ContinuousSpace):
    def contains(self, state) -> bool:
        return bool(torch.all(torch.isfinite(_as_tensor(state))))

class ContinuousSimplex(ContinuousSpace):
    def contains(self, state) -> bool:
        state = _as_tensor(state)
        between_0_and_1 = bool(torch.all((0 <= state) & (state <= 1)))
        # Each simplex lies along the last dimension; leading dimensions are batches.
        total = torch.sum(state, dim=-1) if state.ndim > 0 else state
        sum_to_1 = torch.allclose(total, torch.ones_like(total))
        return between_0_and_1 and sum_to_1

class JointStateSpace(StateSpace):
    def __init__(self, spaces_dict, slices):
        self.spaces_dict = spaces_dict
        self.slices = slices
        self._is_continuous = all(s.is_continuous() for s in spaces_dict.values())

    def contains(self, theta_flat) -> bool:
        for name, (start, end, shape) in self.slices.items():
            val = theta_flat[start:end].reshape(shape)
            if not bool(self.spaces_dict[name].contains(val)):
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


class ContinuousPositiveDefinite(ContinuousSpace):
    """Symmetric positive-definite matrices (support of Wishart / InverseWishart)."""

    def contains(self, state):
        state = torch.as_tensor(state)
        if state.ndim < 2 or state.shape[-1] != state.shape[-2] or not state.is_floating_point():
            return False
        if not torch.allclose(state, state.transpose(-2, -1)):
            return False
        _, info = torch.linalg.cholesky_ex(state)
        return bool(torch.all(info == 0))


class ContinuousOrdered(ContinuousSpace):
    """Vectors that are strictly increasing along the last dimension.

    ``base`` (optional) is the support of each element, e.g.
    ``ContinuousPositive()`` for an ordered vector of positive values.
    Leading dimensions (e.g. the chain dimension) are batch dimensions.
    """

    def __init__(self, base=None):
        self.base = ContinuousReal() if base is None else base

    def contains(self, state) -> bool:
        state = _as_tensor(state)
        if state.ndim == 0 or not bool(torch.all(torch.isfinite(state))):
            return False
        increasing = bool(torch.all(state[..., 1:] > state[..., :-1]))
        return increasing and bool(self.base.contains(state))
