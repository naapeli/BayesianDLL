"""Thread-local stack of the models whose ``with Model():`` block is open.

The innermost open model is the *active* model: new parameters register
themselves with it. Each thread has its own stack, like the plate stack, so
models built concurrently in different threads do not interfere, and nested
``with Model()`` blocks restore the enclosing model on exit.

For backward compatibility the active model is still readable (and writable)
as the module attribute ``_active_model``::

    from BayesianDLL._active_model import _active_model
    _active_model._active_model  # innermost open model or None
"""

import sys
import threading
import types


_context = threading.local()


def _model_stack():
    if not hasattr(_context, "stack"):
        _context.stack = []
    return _context.stack


def get_active_model():
    """Return the innermost model whose context is open in this thread, or ``None``."""
    stack = _model_stack()
    return stack[-1] if stack else None


def push_model(model):
    """Make ``model`` the active model of this thread."""
    _model_stack().append(model)


def pop_model(model=None):
    """Remove ``model`` (default: the innermost model) from this thread's stack."""
    stack = _model_stack()
    if not stack:
        return None
    if model is None or stack[-1] is model:
        return stack.pop()
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] is model:
            return stack.pop(index)
    return None


class _ActiveModelModule(types.ModuleType):
    @property
    def _active_model(self):
        return get_active_model()

    @_active_model.setter
    def _active_model(self, model):
        # Legacy assignment API: ``None`` deactivates the innermost model and
        # any other value replaces it.
        stack = _model_stack()
        if model is None:
            if stack:
                stack.pop()
        elif stack:
            stack[-1] = model
        else:
            stack.append(model)


sys.modules[__name__].__class__ = _ActiveModelModule
