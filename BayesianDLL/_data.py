import torch

from ._active_model._active_model import get_active_model


class Data:
    """Named, mutable model input that is held fixed during inference.

    Create inside a ``Model`` context and pass it directly to a distribution,
    an ``ObservedParameter``, or a deterministic parameter's ``inputs``.
    Call ``set_value(values)`` before inference to replace the data without
    rebuilding the model. ``event_ndim`` specifies how many trailing dimensions
    form one event; their sizes must be preserved when replacing values. Leading
    batch dimensions may change in number and size. The default, zero, treats
    each element as a scalar event. Dtype and device are preserved by converting
    the new values to match the original tensor.

    Parameters
    ----------
    name : str
        Unique, non-empty node name in the model.
    value : array-like
        The data (copied into a tensor).
    event_ndim : int, default 0
        Number of trailing event dimensions, between 0 and the rank of ``value``.

    Notes
    -----
    ``value`` (and ``constrained_value``) return the current tensor,
    ``event_shape`` its trailing ``event_ndim`` dimensions and ``batch_shape``
    the leading ones.
    """

    def __init__(self, name, value, event_ndim=0):
        model = get_active_model()
        if model is None:
            raise RuntimeError("One should select an active model before creating data.")
        if not isinstance(name, str) or not name:
            raise ValueError("Data name must be a non-empty string.")
        if name in model.graph:
            raise ValueError(f"A model node named '{name}' already exists.")
        self.name = name
        self._value = torch.as_tensor(value).detach().clone()
        if isinstance(event_ndim, bool) or not isinstance(event_ndim, int):
            raise TypeError("event_ndim must be an integer.")
        if not 0 <= event_ndim <= self._value.ndim:
            raise ValueError("event_ndim must be between zero and the data tensor rank.")
        self._event_shape = self._value.shape[-event_ndim:] if event_ndim else torch.Size()
        model.data[name] = self
        model.graph.add_node(name, type="data")
        model._compiled = False

    @property
    def value(self):
        return self._value

    @property
    def constrained_value(self):
        return self._value

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        event_ndim = len(self.event_shape)
        return self._value.shape[:-event_ndim] if event_ndim else self._value.shape

    def set_value(self, value):
        """Replace values, preserving event shape, dtype, and device."""
        value = torch.as_tensor(value, dtype=self._value.dtype, device=self._value.device)
        event_ndim = len(self.event_shape)
        event_shape = value.shape[-event_ndim:] if event_ndim else torch.Size()
        if value.ndim < event_ndim or event_shape != self.event_shape:
            raise ValueError(
                f"Data '{self.name}' requires event shape {tuple(self.event_shape)}, "
                f"got tensor shape {tuple(value.shape)}. "
                "Rebuild the model to change the event shape."
            )
        self._value = value.detach().clone()
