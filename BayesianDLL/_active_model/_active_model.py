"""Thread-local stack of the models whose ``with Model():`` block is open.

The innermost open model is the *active* model: new parameters register
themselves with it. Each thread has its own stack, like the plate stack, so
models built concurrently in different threads do not interfere, and nested
``with Model()`` blocks restore the enclosing model on exit.
"""

import threading


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

