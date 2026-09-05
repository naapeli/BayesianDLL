import threading
from contextlib import contextmanager


# Thread-local storage for plate stack
_plate_context = threading.local()


def _get_plate_stack():
    if not hasattr(_plate_context, "stack"):
        _plate_context.stack = []
    return _plate_context.stack


def get_active_plates():
    """Returns a copy of the current plate stack (list of PlateInfo)."""
    return list(_get_plate_stack())


class PlateInfo:
    """Stores metadata about an active plate context."""
    __slots__ = ("name", "_size", "dim")

    def __init__(self, name, size, dim):
        self.name = name
        self._size = size
        self.dim = dim

    @property
    def size(self):
        if hasattr(self._size, "batch_shape"):
            try:
                return self._size.batch_shape[self.dim]
            except IndexError as error:
                raise ValueError(
                    f"Plate '{self.name}' uses dim {self.dim}, but data "
                    f"'{self._size.name}' has batch shape "
                    f"{tuple(self._size.batch_shape)}."
                ) from error
        return self._size

    def __repr__(self):
        return f"PlateInfo(name={self.name!r}, size={self.size}, dim={self.dim})"


class plate:
    """
    Context manager that declares a batch dimension as conditionally independent.

    Inspired by pyro.plate. When a RandomParameter or ObservedParameter is
    created inside a plate context, its value shape is expanded along the
    plate dimension. The plate tells the model that log-probabilities along
    this dimension should be summed (the data are conditionally independent).

    Usage::

        with plate("data", N):
            y = ObservedParameter("y", Normal(mu, sigma), y_data)

    The size can instead be read from mutable ``Data`` at runtime::

        with plate("data", x_data):
            y = ObservedParameter("y", Normal(mu, sigma), y_data)

    Calling ``x_data.set_value(...)`` then changes the plate size. Passing an
    integer keeps the plate size constant.

    Plates can be nested::

        with plate("batch", B):
            with plate("features", D):
                ...

    Args:
        name: Human-readable identifier for this plate.
        size: Number of independent repetitions along this dimension, or a
              ``Data`` object whose corresponding batch dimension determines
              the size at runtime.
        dim:  Which batch dimension this plate controls (negative, counting
              from the right).  When nesting plates without specifying dim,
              each nested plate automatically takes the next leftward
              dimension.
    """

    def __init__(self, name: str, size: int, dim: int | None = None):
        self.name = name
        self.size = size
        self._dim = dim

    def __enter__(self):
        stack = _get_plate_stack()
        if self._dim is not None:
            dim = self._dim
        else:
            # Auto-assign: rightmost unused dim = -(len(stack) + 1)
            dim = -(len(stack) + 1)
        self._info = PlateInfo(self.name, self.size, dim)
        stack.append(self._info)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stack = _get_plate_stack()
        stack.pop()
        return False
