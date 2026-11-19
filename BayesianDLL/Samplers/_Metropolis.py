import math
from collections import deque

import torch


def _is_invalid(value):
    return bool(torch.isnan(value) or torch.isneginf(value))


class _CachedTarget:
    """Cache of the log target at the state a sampler last returned.

    The cache is keyed on the identity of the state tensor: when a caller passes
    the tensor that ``step`` returned, the target is not evaluated again. Any
    other tensor (or a call to :meth:`invalidate`) recomputes it, because the
    target may have changed (e.g. another Gibbs block moved other parameters).
    """

    def _init_cache(self):
        self._cached_theta = None
        self._cached_log_prob = None

    def invalidate_cache(self):
        self._init_cache()

    def _current_log_prob(self, theta):
        if self._cached_theta is not None and theta is self._cached_theta:
            return self._cached_log_prob
        return self.log_target(theta)

    def _store_cache(self, theta, log_prob):
        self._cached_theta = theta
        self._cached_log_prob = log_prob


class Metropolis(_CachedTarget):
    """Random-walk Metropolis sampler.

    Continuous states take isotropic Gaussian proposals
    ``theta + proposal_scale * N(0, I)``, accepted with probability
    ``min(1, p(theta') / p(theta))``; a proposal outside the state space has
    zero density and is rejected (acceptance probability 0, counted as a
    rejection by the acceptance statistics and the adaptation). The scale is adapted during warmup
    towards the acceptance rate ``target_accept_rate`` (0.234 by default;
    about 0.44 suits a single coordinate): after every warmup step it is
    multiplied by ``1 - adaptation_rate`` when the rolling acceptance rate
    (over the last 20 steps) is below the target and by
    ``1 + adaptation_rate`` when it is above. The names differ from NUTS's
    ``delta`` and ``kappa`` on purpose, so settings passed to
    :func:`~BayesianDLL.sample` for NUTS do not reach Metropolis blocks.
    Discrete states are updated one coordinate at a time: each step
    sweeps over the coordinates in random order and proposes a move of ±1 for
    each (rejected when it leaves the state space), so the acceptance rate
    does not collapse with the dimension; their proposals are not adapted.

    Parameters
    ----------
    log_target : callable
        Maps a state tensor to its scalar log density.
    state_space : StateSpace
        The space of the state; decides between the continuous and the
        discrete update.
    proposal_scale : float, default 1.0
        Initial standard deviation of the continuous proposal.
    adaptation_rate : float, default 0.05
        Relative change of ``proposal_scale`` per warmup step.
    target_accept_rate : float, default 0.234
        The acceptance rate the warmup adaptation aims for.

    :meth:`step` returns ``(theta, proposal_scale, rolling_acceptance_rate,
    False)`` and sets ``last_stats`` (``accept_stat``, the acceptance
    probability of this step or the mean over the coordinates of a discrete
    sweep; ``accepted``; ``step_size``, the proposal scale; ``lp``).
    """

    def __init__(self, log_target, state_space, proposal_scale=1.0, adaptation_rate=0.05, target_accept_rate=0.234):
        self.log_target = log_target
        self.state_space = state_space
        self.proposal_scale = proposal_scale
        self.adaptation_rate = adaptation_rate
        self.target_accept_rate = target_accept_rate
        self.m = 0
        self.accept_queue = deque([], maxlen=20)
        self.last_stats = {}
        self._init_cache()

    def step(self, theta, warmup=False):
        if self.state_space.is_discrete():
            theta, accept_stat, accepted = self._discrete_sweep(theta)
        elif self.state_space.is_continuous():
            theta, accept_stat, accepted = self._continuous_step(theta)
        else:
            raise ValueError("The state_space must either be discrete or continuous.")

        self.m += 1
        if warmup and self.state_space.is_continuous():
            self.adapt_proposal_scale()

        rolling = sum(self.accept_queue) / len(self.accept_queue)
        self.last_stats = {
            "accept_stat": float(accept_stat),
            "accepted": float(accepted),
            "step_size": float(self.proposal_scale),
            "lp": float(self._cached_log_prob) if self._cached_log_prob is not None else float("nan"),
        }
        return theta, self.proposal_scale, rolling, False

    def _accept(self, log_diff):
        """Return ``(accepted, acceptance probability)`` for a log density difference."""
        if _is_invalid(log_diff):
            return False, 0.0
        accept_prob = 1.0 if log_diff >= 0 else math.exp(float(log_diff))
        return bool(torch.rand(1) < accept_prob), accept_prob

    def _continuous_step(self, theta):
        current = self._current_log_prob(theta)
        proposal = self.get_proposal(theta)
        if proposal is None:
            # Outside the support: the proposal has zero density (log target
            # -inf), so it is rejected and counted as a rejection.
            accepted, accept_prob = False, 0.0
        else:
            proposal_log_prob = self.log_target(proposal)
            accepted, accept_prob = self._accept(proposal_log_prob - current)
        if accepted:
            theta, current = proposal, proposal_log_prob
        self.accept_queue.append(int(accepted))
        self._store_cache(theta, current)
        return theta, accept_prob, accepted

    def _discrete_sweep(self, theta):
        current = self._current_log_prob(theta)
        n_accepted = 0
        accept_probs = 0.0
        n = theta.numel()
        for index in torch.randperm(n).tolist():
            proposal = self._propose_coordinate(theta, index)
            if proposal is None:
                # Outside the support: the proposal has zero density.
                accepted, accept_prob = False, 0.0
            else:
                proposal_log_prob = self.log_target(proposal)
                accepted, accept_prob = self._accept(proposal_log_prob - current)
                if accepted:
                    theta, current = proposal, proposal_log_prob
            n_accepted += int(accepted)
            accept_probs += accept_prob
        self.accept_queue.append(n_accepted / max(n, 1))
        self._store_cache(theta, current)
        return theta, accept_probs / max(n, 1), n_accepted > 0

    def _propose_coordinate(self, theta, index):
        proposal = theta.clone()
        flat = proposal.view(-1)
        flat[index] = flat[index] + (1 if torch.rand(1) < 0.5 else -1)
        if self.state_space.contains(proposal):
            return proposal
        return None

    def get_proposal(self, theta):
        """Continuous Gaussian proposal, or ``None`` when it leaves the state space (a rejected proposal)."""
        theta_proposal = theta + self.proposal_scale * torch.randn_like(theta)
        if self.state_space.contains(theta_proposal):
            return theta_proposal
        return None

    def adapt_proposal_scale(self):
        acceptance_ratio = sum(self.accept_queue) / len(self.accept_queue)
        if acceptance_ratio < self.target_accept_rate:
            self.proposal_scale *= (1 - self.adaptation_rate)
        elif acceptance_ratio > self.target_accept_rate:
            self.proposal_scale *= (1 + self.adaptation_rate)

    def init_sampler(self):
        pass

    def reset(self):
        self.m = 0
        self.accept_queue.clear()
        self.invalidate_cache()


def _finite_supports(state_space, numel, slices=None):
    """Per-coordinate supports of a finite discrete state space.

    Returns a list of 1-D tensors (one per flat coordinate), or ``None`` when
    the space has no finite support (e.g. ``DiscretePositive``).
    """
    from ..Distributions._state_space import DiscreteRange, JointStateSpace
    from ..Distributions._resolve import resolve

    def range_support(space, count):
        low = resolve(space.low).reshape(-1)
        high = resolve(space.high).reshape(-1)
        if low.numel() == 1 and high.numel() == 1:
            values = torch.arange(int(low.item()), int(high.item()) + 1)
            return [values] * count
        low = low.expand(count) if low.numel() == 1 else low
        high = high.expand(count) if high.numel() == 1 else high
        if low.numel() != count or high.numel() != count:
            return None
        return [torch.arange(int(lo), int(hi) + 1) for lo, hi in zip(low.tolist(), high.tolist())]

    if isinstance(state_space, DiscreteRange):
        return range_support(state_space, numel)
    if isinstance(state_space, JointStateSpace):
        supports = [None] * numel
        for name, (start, end, _) in state_space.slices.items():
            space = state_space.spaces_dict[name]
            if not isinstance(space, DiscreteRange):
                return None
            part = range_support(space, end - start)
            if part is None:
                return None
            supports[start:end] = part
        return supports
    return None


def has_finite_support(state_space):
    """Whether :class:`FiniteGibbs` can sample from ``state_space``."""
    from ..Distributions._state_space import DiscreteRange, JointStateSpace
    if isinstance(state_space, DiscreteRange):
        return True
    if isinstance(state_space, JointStateSpace):
        return all(isinstance(space, DiscreteRange) for space in state_space.spaces_dict.values())
    return False


class FiniteGibbs(_CachedTarget):
    """Exact Gibbs sampler for discrete variables with finite support.

    Each step sweeps over the coordinates in random order. For every
    coordinate the log target is evaluated at all values of its support, and a
    new value is drawn exactly from the resulting conditional distribution.
    The current value's log target is known from the previous update, so a
    coordinate with ``K`` possible values costs ``K - 1`` evaluations.

    If ``elementwise_log_target`` is given, it must return a tensor of the
    shape of the state holding independent per-coordinate log probabilities
    (e.g. a factorised predictive distribution); when all coordinates share
    one support, they are then sampled jointly with one evaluation per
    support value (otherwise, or if the returned shape differs from the
    state's, the sweep is used).

    Parameters
    ----------
    log_target : callable
        Maps a state tensor to its scalar log density.
    state_space : DiscreteRange or JointStateSpace of DiscreteRange
        A discrete space with finite support per coordinate; anything else
        raises a ``RuntimeError``.
    elementwise_log_target : callable, optional
        Per-coordinate log probabilities, see above.

    :meth:`step` returns ``(theta, 1.0, 1.0, False)``: every Gibbs update is
    accepted.
    """

    def __init__(self, log_target, state_space, elementwise_log_target=None):
        if not has_finite_support(state_space):
            raise RuntimeError("FiniteGibbs requires a discrete state space with finite support (DiscreteRange).")
        self.log_target = log_target
        self.state_space = state_space
        self.elementwise_log_target = elementwise_log_target
        self.m = 0
        self.last_stats = {}
        self._init_cache()

    def step(self, theta, warmup=False):
        supports = _finite_supports(self.state_space, theta.numel())
        if supports is None:
            raise RuntimeError("Could not determine a finite support for every coordinate.")

        if self.elementwise_log_target is not None and self._shared_support(supports):
            theta = self._independent_update(theta, supports[0])
        else:
            theta = self._sweep(theta, supports)
        self.m += 1
        lp = self._cached_log_prob
        self.last_stats = {
            "accept_stat": 1.0,
            "step_size": 1.0,
            "lp": float(lp) if lp is not None else float("nan"),
        }
        return theta, 1.0, 1.0, False

    @staticmethod
    def _shared_support(supports):
        first = supports[0] if supports else None
        return first is not None and all(s is first or torch.equal(s, first) for s in supports)

    @staticmethod
    def _draw(log_probs):
        """Indices drawn from the normalised ``exp(log_probs)`` along the last dim, and which rows are possible."""
        log_probs = torch.nan_to_num(log_probs, nan=-math.inf)
        maximum = torch.amax(log_probs, dim=-1, keepdim=True)
        valid = torch.isfinite(maximum).squeeze(-1)
        weights = torch.exp(log_probs - torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum)))
        weights = torch.where(valid.unsqueeze(-1), weights, torch.ones_like(weights))
        return torch.multinomial(weights.reshape(-1, weights.shape[-1]), 1).reshape(weights.shape[:-1]), valid

    @staticmethod
    def _draw_index(log_probs):
        """Index drawn proportionally to ``exp(log_probs)`` (a list of floats), or ``None`` if all are impossible."""
        finite = [lp for lp in log_probs if math.isfinite(lp)]
        if not finite:
            return None
        maximum = max(finite)
        weights = [math.exp(lp - maximum) if math.isfinite(lp) else 0.0 for lp in log_probs]
        u = float(torch.rand(1)) * sum(weights)
        cumulative = 0.0
        last = 0
        for k, weight in enumerate(weights):
            if weight > 0.0:
                last = k
                cumulative += weight
                if u < cumulative:
                    return k
        return last

    def _sweep(self, theta, supports):
        current = float(self._current_log_prob(theta))
        theta = theta.clone()
        flat = theta.view(-1)
        for index in torch.randperm(theta.numel()).tolist():
            values = supports[index].tolist()
            old_value = flat[index].item()
            log_probs = []
            for value in values:
                if value == old_value:
                    log_probs.append(current)
                    continue
                flat[index] = value
                # Pass a copy: the target may keep references to its argument.
                log_probs.append(float(self.log_target(theta.clone())))
            k = self._draw_index(log_probs)
            if k is None:
                flat[index] = old_value
            else:
                flat[index] = values[k]
                current = log_probs[k]
        self._store_cache(theta, torch.tensor(current, dtype=theta.dtype, device=theta.device))
        return theta

    def _independent_update(self, theta, support):
        support = support.to(dtype=theta.dtype, device=theta.device)
        values = [self.elementwise_log_target(torch.full_like(theta, float(value))) for value in support]
        if any(value.shape != theta.shape for value in values):
            # Not one log probability per coordinate: fall back to the sweep.
            return self._sweep(theta, [support] * theta.numel())
        log_probs = torch.stack(values, dim=-1)
        choice, valid = self._draw(log_probs)
        new_theta = torch.where(valid, support[choice], theta)
        # The full target (which may contain terms beyond the elementwise
        # ones) at the new state, for ``lp`` and the cache.
        self._store_cache(new_theta, self.log_target(new_theta.clone()))
        return new_theta

    def init_sampler(self):
        pass

    def reset(self):
        self.m = 0
        self.invalidate_cache()
