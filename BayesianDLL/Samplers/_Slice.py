import math

import torch

from ._Metropolis import _CachedTarget


def _as_float(value):
    value = float(value)
    return value if math.isfinite(value) else -math.inf


class Slice(_CachedTarget):
    """Coordinate-wise univariate slice sampler (Neal, 2003).

    Every step updates each coordinate of the (unconstrained) state once, in
    random order, with the stepping-out and shrinkage procedure: a slice level
    ``log y = log f(x) - Exp(1)`` is drawn, an interval of size ``width`` is
    placed randomly around the coordinate and stepped out by ``width`` at most
    ``max_steps_out`` times in total until both ends lie outside the slice, and
    proposals drawn uniformly from the interval shrink it until one lands in
    the slice. Every update is accepted, so the sampler needs no gradient and
    no tuning of an acceptance rate.

    The ``max_steps_out`` extensions are split at random between the two
    ends (Neal's ``J = floor(m V)`` to the left, the rest to the right). The
    shrinkage gives up after 200 proposals and keeps the current value.

    Parameters
    ----------
    log_target : callable
        Maps the state tensor to its scalar log density; points outside
        ``state_space`` count as density zero.
    state_space : StateSpace
        A continuous state space (otherwise a ``RuntimeError`` is raised).
    width : float or Tensor, default 1.0
        Initial interval width, a scalar or one positive value per coordinate.
    max_steps_out : int, default 20
        Total number of stepping-out extensions per coordinate update.
    adapt_width : bool, default True
        Adapt the width of every coordinate during warmup to the running mean
        of ``2 |move|`` of the coordinate (the initial width counting as one
        observation), a running estimate of the scale of its conditional
        distribution.

    :meth:`step` returns ``(theta, mean_width, 1.0, False)``. After each call,
    ``last_stats`` holds ``n_evals`` (target evaluations of the step),
    ``accept_stat`` (always 1), ``step_size`` (mean width) and ``lp`` (log
    target at the new state).
    """

    def __init__(self, log_target, state_space, width=1.0, max_steps_out=20, adapt_width=True):
        if not state_space.is_continuous():
            raise RuntimeError("The slice sampler can only be used with continuous state spaces.")
        width_tensor = torch.as_tensor(width)
        if not bool((width_tensor > 0).all()):
            raise ValueError(f"width must be positive, got {width}.")
        if int(max_steps_out) < 0:
            raise ValueError(f"max_steps_out must be non-negative, got {max_steps_out}.")
        self.log_target = log_target
        self.state_space = state_space
        self.initial_width = width
        self.max_steps_out = int(max_steps_out)
        self.adapt_width = adapt_width
        self.max_shrinks = 200
        self.width = None
        self.m = 0
        self._n_tune = 0
        self.last_stats = {}
        self._init_cache()

    def _widths_like(self, flat):
        if self.width is None or self.width.numel() != flat.numel():
            width = torch.as_tensor(self.initial_width, dtype=flat.dtype, device=flat.device)
            self.width = (width.reshape(-1) * torch.ones_like(flat)).clone()
        return self.width.to(dtype=flat.dtype, device=flat.device)

    def _evaluate(self, theta):
        if not self.state_space.contains(theta):
            return -math.inf
        return _as_float(self.log_target(theta))

    def step(self, theta, warmup=False):
        shape = theta.shape
        current = _as_float(self._current_log_prob(theta))
        if current == -math.inf:
            raise RuntimeError("The slice sampler was started at a point with zero density.")
        widths = self._widths_like(theta.reshape(-1))
        state = theta.detach().clone().reshape(-1)
        n_evals = 0
        old_state = state.clone()

        def at(index, value):
            nonlocal n_evals
            n_evals += 1
            proposal = state.clone()
            proposal[index] = value
            return self._evaluate(proposal.reshape(shape))

        for index in torch.randperm(state.numel()).tolist():
            x0 = float(state[index])
            w = float(widths[index])
            log_y = current - float(torch.empty(()).exponential_())

            # Stepping out, with the total number of extensions bounded by
            # max_steps_out and split at random between both ends.
            left = x0 - w * float(torch.rand(()))
            right = left + w
            steps_left = int(float(torch.rand(())) * self.max_steps_out)
            steps_right = self.max_steps_out - steps_left
            while steps_left > 0 and at(index, left) > log_y:
                left -= w
                steps_left -= 1
            while steps_right > 0 and at(index, right) > log_y:
                right += w
                steps_right -= 1

            # Shrinkage.
            new_x, new_lp = x0, current
            for _ in range(self.max_shrinks):
                x1 = left + float(torch.rand(())) * (right - left)
                lp1 = at(index, x1)
                if lp1 > log_y:
                    new_x, new_lp = x1, lp1
                    break
                if x1 < x0:
                    left = x1
                else:
                    right = x1
            state[index] = new_x
            current = new_lp

        if warmup and self.adapt_width:
            # Running average of 2 |move| per coordinate, in which the
            # initial width counts as one observation.
            self._n_tune += 1
            moves = 2.0 * (state - old_state).abs()
            new_widths = widths + (moves - widths) / (self._n_tune + 1)
            self.width = torch.maximum(new_widths, torch.full_like(new_widths, 1e-8))

        self.m += 1
        new_theta = state.reshape(shape)
        log_prob = torch.tensor(current, dtype=theta.dtype, device=theta.device)
        self._store_cache(new_theta, log_prob)
        mean_width = float(self.width.mean())
        self.last_stats = {
            "n_evals": float(n_evals),
            "accept_stat": 1.0,
            "step_size": mean_width,
            "lp": current,
        }
        return new_theta, mean_width, 1.0, False

    def init_sampler(self):
        pass

    def reset(self):
        self.m = 0
        self._n_tune = 0
        self.width = None
        self.invalidate_cache()
