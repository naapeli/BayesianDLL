import threading


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
        # ``plate`` is commonly called with the raw tensor used as data,
        # e.g. ``plate("observations", x)``.  A tensor has no ``batch_shape``
        # attribute, so returning it directly would make the model plot show
        # the complete tensor instead of the length of the plate.
        if hasattr(self._size, "shape"):
            try:
                return self._size.shape[self.dim]
            except IndexError as error:
                raise ValueError(
                    f"Plate '{self.name}' uses dim {self.dim}, but the "
                    f"plate size has shape {tuple(self._size.shape)}."
                ) from error
        return self._size

    def __repr__(self):
        return f"PlateInfo(name={self.name!r}, size={self.size}, dim={self.dim})"


class plate:
    """
    Context manager that declares a batch dimension as conditionally independent.

    Inspired by pyro.plate. When a RandomParameter is created inside plate
    contexts, its value shape is ``(*plate_sizes, *event_shape)``, with the
    sizes of the enclosing plates from the outermost to the innermost as the
    leading dimensions; an ObservedParameter's generated data
    (``predictive_shape``) take the plate sizes in its leading dimensions in
    the same order. The elementwise log-probabilities along these dimensions
    are summed (the values are conditionally independent).

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
        size: Number of independent repetitions along this dimension, a
              ``Data`` object whose batch dimension ``dim`` determines the
              size at runtime, or a tensor whose dimension ``dim`` gives the size.
        dim:  The dimension of a ``Data`` (its batch shape) or tensor ``size``
              that gives the plate size; ignored for an integer size. It
              does not change where the plate's dimension is placed in the
              value shape (see above). Default: ``-1`` for the outermost plate,
              ``-2`` for a plate nested in one other plate, and so on.
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
