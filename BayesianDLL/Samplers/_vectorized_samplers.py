"""MCMC transition kernels that move ``C`` chains at once.

Every sampler works on a state ``theta`` of shape ``(C, d)`` (one flat
unconstrained vector per chain) and on a log target returning ``(C,)`` and a
gradient returning ``(C, d)``. Accept/reject decisions, step sizes, metrics
and adaptation are per chain; the chains only share the iteration counter
(and therefore the warmup schedule). Chains that have finished their part of
an iteration (a NUTS trajectory that turned or diverged, a slice that has
been found) are masked: their state is no longer updated, but the batched
evaluations still run on them.
"""

import math

import torch

from .._batching import pad_to


_LOG_HALF = math.log(0.5)
_LOG_TWO = math.log(2.0)


def _where(mask, a, b):
    """``torch.where`` with a per-chain mask of shape ``(C,)``."""
    return torch.where(pad_to(mask, max(a.ndim, b.ndim, 1)) if a.ndim > 1 or b.ndim > 1 else mask, a, b)


def _finite_rows(x):
    return torch.isfinite(x).reshape(x.shape[0], -1).all(-1)


def _nan_to_neg_inf(lp):
    return torch.where(torch.isnan(lp), torch.full_like(lp, -math.inf), lp)


def _uniform(n, like):
    return torch.rand(n, dtype=like.dtype, device=like.device)


def _log_uniform(n, like):
    return torch.log(torch.rand(n, dtype=like.dtype, device=like.device).clamp(min=torch.finfo(like.dtype).tiny))


# ============================================================ state spaces
def chain_contains(state_space, theta, slices=None):
    """Per-chain membership ``(C,)`` of the flat states ``theta`` (shape ``(C, d)``)."""
    from ..Distributions._state_space import (
        ContinuousPositive, ContinuousRange, ContinuousReal, DiscreteNonNegative, DiscretePositive, DiscreteRange,
        JointStateSpace,
    )
    from ..Distributions._resolve import resolve

    def integer(x):
        return torch.isfinite(x) & (x == torch.round(x))

    def rows(mask):
        return mask.reshape(mask.shape[0], -1).all(-1)

    def bound(value, x):
        value = torch.as_tensor(resolve(value)).to(device=x.device, dtype=x.dtype)
        if value.numel() == 1:
            return value.reshape(())
        if value.numel() == x[0].numel():
            return value.reshape(-1)
        if value.numel() == x.numel():
            return value.reshape(x.shape)
        return None

    if isinstance(state_space, JointStateSpace):
        result = torch.ones(theta.shape[0], dtype=torch.bool, device=theta.device)
        for name, (start, end, _) in state_space.slices.items():
            result = result & chain_contains(state_space.spaces_dict[name], theta[:, start:end])
        return result
    if isinstance(state_space, ContinuousReal):
        return rows(torch.isfinite(theta))
    if isinstance(state_space, ContinuousPositive):
        return rows(theta > 0)
    if isinstance(state_space, DiscretePositive):
        return rows((theta > 0) & integer(theta))
    if isinstance(state_space, DiscreteNonNegative):
        return rows((theta >= 0) & integer(theta))
    if isinstance(state_space, (DiscreteRange, ContinuousRange)):
        low, high = bound(state_space.low, theta), bound(state_space.high, theta)
        if low is not None and high is not None:
            inside = (theta >= low) & (theta <= high)
            if isinstance(state_space, DiscreteRange):
                inside = inside & integer(theta)
            return rows(inside)
    return torch.tensor([bool(state_space.contains(t)) for t in theta], device=theta.device)


# ============================================================ Metropolis
class VectorizedMetropolis:
    """Random-walk Metropolis for ``C`` chains (per-chain proposal scales and adaptation).

    Continuous states take Gaussian proposals; discrete states are swept one
    coordinate at a time with ±1 proposals, as :class:`~BayesianDLL.Samplers.Metropolis`,
    whose settings (``proposal_scale``, ``adaptation_rate``,
    ``target_accept_rate``) it takes. Proposals outside the state space are
    rejected. :meth:`step` returns ``(theta, proposal_scale,
    rolling_acceptance_rate, diverging=False)`` with per-chain tensors.
    """

    def __init__(self, log_target, state_space, n_chains, proposal_scale=1.0, adaptation_rate=0.05, target_accept_rate=0.234):
        self.log_target = log_target
        self.state_space = state_space
        self.n_chains = n_chains
        self.initial_scale = proposal_scale
        self.adaptation_rate = adaptation_rate
        self.target_accept_rate = target_accept_rate
        self.proposal_scale = None
        self.accept_history = []
        self.last_stats = {}
        self._cached_theta = None
        self._cached_lp = None

    def _current(self, theta):
        if self._cached_theta is not None and theta is self._cached_theta:
            return self._cached_lp
        return _nan_to_neg_inf(self.log_target(theta))

    def step(self, theta, warmup=False):
        if self.proposal_scale is None:
            self.proposal_scale = torch.full((theta.shape[0],), float(self.initial_scale), dtype=theta.dtype, device=theta.device)
        current = self._current(theta)
        if self.state_space.is_discrete():
            theta, current, accept_stat, accepted = self._discrete_sweep(theta, current)
        else:
            theta, current, accept_stat, accepted = self._continuous_step(theta, current)
        self.accept_history = (self.accept_history + [accepted.to(theta.dtype)])[-20:]
        rolling = torch.stack(self.accept_history).mean(0)
        if warmup and self.state_space.is_continuous():
            factor = torch.where(rolling < self.target_accept_rate, 1 - self.adaptation_rate,
                                 torch.where(rolling > self.target_accept_rate, 1 + self.adaptation_rate, 1.0))
            self.proposal_scale = self.proposal_scale * factor
        self._cached_theta, self._cached_lp = theta, current
        self.last_stats = {
            "accept_stat": accept_stat,
            "accepted": (accepted > 0).to(theta.dtype),
            "step_size": self.proposal_scale.clone(),
            "lp": current,
        }
        return theta, self.proposal_scale, rolling, torch.zeros_like(rolling, dtype=torch.bool)

    def _accept(self, log_diff, valid):
        log_diff = torch.where(valid, _nan_to_neg_inf(log_diff), torch.full_like(log_diff, -math.inf))
        accept_prob = torch.exp(log_diff.clamp(max=0.0))
        accepted = valid & (_log_uniform(log_diff.shape[0], log_diff) < log_diff)
        return accepted, accept_prob

    def _continuous_step(self, theta, current):
        proposal = theta + self.proposal_scale.unsqueeze(-1) * torch.randn_like(theta)
        valid = chain_contains(self.state_space, proposal)
        proposal_lp = _nan_to_neg_inf(self.log_target(proposal))
        accepted, accept_prob = self._accept(proposal_lp - current, valid)
        theta = _where(accepted, proposal, theta)
        current = torch.where(accepted, proposal_lp, current)
        return theta, current, accept_prob, accepted.to(theta.dtype)

    def _discrete_sweep(self, theta, current):
        n_chains, dim = theta.shape
        n_accepted = torch.zeros(n_chains, dtype=theta.dtype, device=theta.device)
        accept_probs = torch.zeros_like(n_accepted)
        for index in torch.randperm(dim).tolist():
            direction = torch.where(_uniform(n_chains, theta) < 0.5, 1.0, -1.0).to(theta.dtype)
            proposal = theta.clone()
            proposal[:, index] = proposal[:, index] + direction
            valid = chain_contains(self.state_space, proposal)
            proposal_lp = _nan_to_neg_inf(self.log_target(proposal))
            accepted, accept_prob = self._accept(proposal_lp - current, valid)
            theta = _where(accepted, proposal, theta)
            current = torch.where(accepted, proposal_lp, current)
            n_accepted = n_accepted + accepted.to(theta.dtype)
            accept_probs = accept_probs + accept_prob
        return theta, current, accept_probs / max(dim, 1), n_accepted / max(dim, 1)

    def invalidate_cache(self):
        self._cached_theta = None


# ============================================================ finite Gibbs
class VectorizedFiniteGibbs:
    """Exact Gibbs sweeps for finite discrete states, all chains at once.

    For every coordinate (in random order) the log target is evaluated at
    every support value for all chains with one batched call per value (``K``
    calls for ``K`` values), and each chain draws its new value from its own
    conditional distribution. The supports must be the same for every chain.
    """

    def __init__(self, log_target, state_space, n_chains):
        from ._Metropolis import has_finite_support
        if not has_finite_support(state_space):
            raise RuntimeError("FiniteGibbs requires a discrete state space with finite support (DiscreteRange).")
        self.log_target = log_target
        self.state_space = state_space
        self.n_chains = n_chains
        self.last_stats = {}
        self._supports = None

    def _coordinate_supports(self, theta):
        from ._Metropolis import _finite_supports
        if self._supports is None:
            try:
                supports = _finite_supports(self.state_space, theta.shape[1])
            except (RuntimeError, ValueError) as error:
                raise NotImplementedError(
                    "chain_method='vectorized' needs the support of a Gibbs-sampled parameter to be the same for "
                    "every chain (its bounds must not be random)."
                ) from error
            if supports is None:
                raise RuntimeError("Could not determine a finite support for every coordinate.")
            self._supports = [s.to(dtype=theta.dtype, device=theta.device) for s in supports]
        return self._supports

    def step(self, theta, warmup=False):
        supports = self._coordinate_supports(theta)
        theta = theta.clone()
        n_chains = theta.shape[0]
        current = None
        for index in torch.randperm(theta.shape[1]).tolist():
            values = supports[index]
            log_probs = []
            for value in values:
                proposal = theta.clone()
                proposal[:, index] = value
                log_probs.append(_nan_to_neg_inf(self.log_target(proposal)))
            log_probs = torch.stack(log_probs, dim=-1)  # (C, K)
            maximum = torch.amax(log_probs, dim=-1, keepdim=True)
            possible = torch.isfinite(maximum).squeeze(-1)
            weights = torch.exp(log_probs - torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum)))
            weights = torch.where(possible.unsqueeze(-1), weights, torch.ones_like(weights))
            choice = torch.multinomial(weights, 1).squeeze(-1)
            theta[:, index] = torch.where(possible, values[choice], theta[:, index])
            chosen_lp = log_probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            current = chosen_lp if current is None else torch.where(possible, chosen_lp, current)
        if current is None:
            current = _nan_to_neg_inf(self.log_target(theta))
        ones = torch.ones(n_chains, dtype=theta.dtype, device=theta.device)
        self.last_stats = {"accept_stat": ones, "step_size": ones, "lp": current}
        return theta, ones, ones, torch.zeros(n_chains, dtype=torch.bool, device=theta.device)

    def invalidate_cache(self):
        pass


# ============================================================ slice
class VectorizedSlice:
    """Coordinate-wise slice sampling (stepping out and shrinkage) for ``C`` chains.

    Every chain has its own slice level, interval and (adapted) widths; the
    stepping-out and shrinkage loops run until every chain is done, with
    finished chains masked. The settings (``width``, ``max_steps_out``,
    ``adapt_width``) are those of :class:`~BayesianDLL.Samplers.Slice`.
    """

    def __init__(self, log_target, state_space, n_chains, width=1.0, max_steps_out=20, adapt_width=True):
        if not state_space.is_continuous():
            raise RuntimeError("The slice sampler can only be used with continuous state spaces.")
        width_tensor = torch.as_tensor(width)
        if not bool((width_tensor > 0).all()):
            raise ValueError(f"width must be positive, got {width}.")
        self.log_target = log_target
        self.state_space = state_space
        self.n_chains = n_chains
        self.initial_width = width
        self.max_steps_out = int(max_steps_out)
        self.adapt_width = adapt_width
        self.max_shrinks = 200
        self.width = None
        self._n_tune = 0
        self.last_stats = {}
        self._cached_theta = None
        self._cached_lp = None

    def _evaluate(self, theta):
        lp = _nan_to_neg_inf(self.log_target(theta))
        return torch.where(chain_contains(self.state_space, theta), lp, torch.full_like(lp, -math.inf))

    def step(self, theta, warmup=False):
        n_chains, dim = theta.shape
        if self.width is None:
            width = torch.as_tensor(self.initial_width, dtype=theta.dtype, device=theta.device).reshape(-1)
            self.width = (width * torch.ones(n_chains, dim, dtype=theta.dtype, device=theta.device)).clone()
        if self._cached_theta is not None and theta is self._cached_theta:
            current = self._cached_lp
        else:
            current = self._evaluate(theta)
        if bool((current == -math.inf).any()):
            raise RuntimeError("The slice sampler was started at a point with zero density.")
        state = theta.clone()
        old_state = state.clone()
        n_evals = torch.zeros(n_chains, dtype=theta.dtype, device=theta.device)

        def at(index, values, mask):
            nonlocal n_evals
            n_evals = n_evals + mask.to(theta.dtype)
            proposal = state.clone()
            proposal[:, index] = values
            return self._evaluate(proposal)

        for index in torch.randperm(dim).tolist():
            x0 = state[:, index]
            w = self.width[:, index]
            log_y = current - torch.empty_like(current).exponential_()
            left = x0 - w * _uniform(n_chains, theta)
            right = left + w
            steps_left = torch.floor(_uniform(n_chains, theta) * self.max_steps_out)
            steps_right = self.max_steps_out - steps_left
            for bound, steps, sign in ((left, steps_left, -1.0), (right, steps_right, 1.0)):
                active = steps > 0
                while bool(active.any()):
                    lp = at(index, bound, active)
                    extend = active & (lp > log_y)
                    bound = torch.where(extend, bound + sign * w, bound)
                    steps = steps - extend.to(steps.dtype)
                    active = extend & (steps > 0)
                if sign < 0:
                    left = bound
                else:
                    right = bound
            done = torch.zeros(n_chains, dtype=torch.bool, device=theta.device)
            new_x, new_lp = x0.clone(), current.clone()
            for _ in range(self.max_shrinks):
                x1 = left + _uniform(n_chains, theta) * (right - left)
                lp1 = at(index, x1, ~done)
                accept = ~done & (lp1 > log_y)
                new_x = torch.where(accept, x1, new_x)
                new_lp = torch.where(accept, lp1, new_lp)
                done = done | accept
                if bool(done.all()):
                    break
                left = torch.where(~done & (x1 < x0), x1, left)
                right = torch.where(~done & (x1 >= x0), x1, right)
            state[:, index] = new_x
            current = new_lp

        if warmup and self.adapt_width:
            self._n_tune += 1
            moves = 2.0 * (state - old_state).abs()
            self.width = torch.clamp(self.width + (moves - self.width) / (self._n_tune + 1), min=1e-8)
        self._cached_theta, self._cached_lp = state, current
        mean_width = self.width.mean(-1)
        ones = torch.ones(n_chains, dtype=theta.dtype, device=theta.device)
        self.last_stats = {"n_evals": n_evals, "accept_stat": ones, "step_size": mean_width, "lp": current}
        return state, mean_width, ones, torch.zeros(n_chains, dtype=torch.bool, device=theta.device)

    def invalidate_cache(self):
        self._cached_theta = None


# ============================================================ NUTS
class VectorizedNUTS:
    """Multinomial No-U-Turn sampler running ``C`` chains in lock step.

    Each iteration doubles every chain's trajectory in a random direction
    until it makes a U-turn, diverges or reaches ``max_depth`` doublings.
    A doubling builds a subtree of ``2**j`` leapfrog steps for all chains at
    once; the recursion of the original algorithm is unrolled: the U-turn
    criterion of every sub-subtree is checked against checkpoints of its
    first state, and the subtree's proposal is drawn by progressive
    (multinomial) sampling: each leaf has the weight ``W = exp(H_0 - H)``
    and replaces the subtree's proposal with probability
    ``W_leaf / W_subtree``. The subtree replaces the current proposal with
    probability ``min(1, W_subtree / W_tree)`` (biased progressive sampling);
    a subtree that diverges or contains a U-turn is discarded. A chain whose
    trajectory terminated is masked: it stops updating while the other
    chains continue.

    This is the multinomial NUTS of Betancourt (2017) and current Stan; the
    single-chain :class:`~BayesianDLL.Samplers.NUTS` used by
    ``chain_method="parallel"`` is the slice-sampling variant of Hoffman and
    Gelman (2014). Both leave the posterior invariant.

    Step sizes (dual averaging), the inverse metric (``"dense"`` or
    ``"diag"``, Stan's windowed adaptation with per-chain Welford
    estimates) and all statistics are per chain. The statistics match those
    of :class:`~BayesianDLL.Samplers.NUTS`: ``step_size``, ``accept_stat``,
    ``tree_depth``, ``n_leapfrog``, ``diverging``, ``energy`` and ``lp``,
    each of shape ``(C,)``. ``log_target`` and ``gradient`` map ``(C, d)`` to
    ``(C,)`` and ``(C, d)``; ``n_chains`` is ``C``. The other settings and
    their defaults are those of :class:`~BayesianDLL.Samplers.NUTS`, e.g. the
    target acceptance statistic ``delta`` (default 0.8). :meth:`step`
    returns ``(theta, step_size, accept_stat, diverging)`` with per-chain
    tensors, ``step_size`` being the one for the next iteration.
    """

    def __init__(self, log_target, gradient, n_chains, delta=0.8, gamma=0.05, step_size_bar=1, max_depth=10, t0=10,
                 kappa=0.75, H_bar=0, min_step_size=1e-4, max_step_size=100, max_delta=1000.0,
                 adapt_mass_matrix=True, metric="dense"):
        from ._NUTS import _adaptation_windows
        if metric not in ("diag", "dense"):
            raise ValueError(f"metric must be 'diag' or 'dense', got {metric!r}.")
        self._windows = _adaptation_windows
        self.log_target = log_target
        self.gradient = gradient
        self.n_chains = n_chains
        self.delta = delta
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.max_depth = max_depth
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.max_delta = max_delta
        self.adapt_mass_matrix = adapt_mass_matrix
        self.metric = metric
        self._H_bar0 = float(H_bar)
        self._step_size_bar0 = float(step_size_bar)
        self.step_size = None
        self.inv_metric = None
        self._inv_metric_chol = None
        self.warmup_length = None
        self._warmup_iteration = 0
        self._window_start, self._window_ends = None, []
        self._reset_welford()
        self._cached_theta = None
        self.last_stats = {}
        # Whether the step size has been adapted since the last non-warmup step.
        self._adapting = False
        # Optional ``theta -> (log_target, gradient)`` that shares work between the two.
        self.value_and_gradient = None

    # ------------------------------------------------------------ metric
    def set_warmup_length(self, warmup_length):
        self.warmup_length = warmup_length
        self._warmup_iteration = 0
        if self.adapt_mass_matrix:
            self._window_start, self._window_ends = self._windows(warmup_length)
        else:
            self._window_start, self._window_ends = None, []

    def _reset_welford(self):
        self._welford_n = 0
        self._welford_mean = None
        self._welford_m2 = None

    def _welford_update(self, theta):
        if self._welford_mean is None:
            self._welford_mean = torch.zeros_like(theta)
            if self.metric == "dense":
                self._welford_m2 = torch.zeros(*theta.shape, theta.shape[-1], dtype=theta.dtype, device=theta.device)
            else:
                self._welford_m2 = torch.zeros_like(theta)
        self._welford_n += 1
        delta = theta - self._welford_mean
        self._welford_mean = self._welford_mean + delta / self._welford_n
        after = theta - self._welford_mean
        if self.metric == "dense":
            self._welford_m2 = self._welford_m2 + delta.unsqueeze(-1) * after.unsqueeze(-2)
        else:
            self._welford_m2 = self._welford_m2 + delta * after

    def _velocity(self, r):
        if self.inv_metric is None:
            return r
        if self.metric == "dense":
            return (self.inv_metric @ r.unsqueeze(-1)).squeeze(-1)
        return self.inv_metric * r

    def _kinetic(self, r):
        return 0.5 * (r * self._velocity(r)).sum(-1)

    def _sample_momentum(self, theta):
        z = torch.randn_like(theta)
        if self.inv_metric is None:
            return z
        if self.metric == "dense":
            if self._inv_metric_chol is None:
                self._inv_metric_chol = torch.linalg.cholesky(self.inv_metric)
            # With M^-1 = L L^T, r = L^-T z has covariance M.
            return torch.linalg.solve_triangular(self._inv_metric_chol.transpose(-2, -1), z.unsqueeze(-1), upper=True).squeeze(-1)
        return z / torch.sqrt(self.inv_metric)

    def _end_adaptation_window(self, theta):
        n = self._welford_n
        if n >= 2:
            covariance = self._welford_m2 / (n - 1)
            if self.metric == "dense":
                identity = torch.eye(covariance.shape[-1], dtype=covariance.dtype, device=covariance.device)
                inv_metric = (n / (n + 5.0)) * covariance + 1e-3 * (5.0 / (n + 5.0)) * identity
                inv_metric = 0.5 * (inv_metric + inv_metric.transpose(-2, -1))
                chol, info = torch.linalg.cholesky_ex(inv_metric)
                ok = info == 0
                previous = self.inv_metric if self.inv_metric is not None else identity.expand_as(inv_metric)
                previous_chol = self._inv_metric_chol if self._inv_metric_chol is not None else identity.expand_as(inv_metric)
                self.inv_metric = torch.where(ok[:, None, None], inv_metric, previous)
                self._inv_metric_chol = torch.where(ok[:, None, None], chol, previous_chol)
            else:
                self.inv_metric = (n / (n + 5.0)) * covariance + 1e-3 * (5.0 / (n + 5.0))
        self._reset_welford()
        self._restart_dual_averaging(theta)

    def _restart_dual_averaging(self, theta):
        self.step_size = self.find_reasonable_step_size(theta, self._grad, self._lp)
        self.mu = torch.log(10 * self.step_size)
        self.H_bar = torch.full_like(self.step_size, self._H_bar0)
        self.step_size_bar = torch.full_like(self.step_size, self._step_size_bar0)
        self.m = 1

    # ------------------------------------------------------------ dynamics
    def _evaluate(self, theta):
        if self.value_and_gradient is not None:
            lp, grad = self.value_and_gradient(theta)
        else:
            lp, grad = self.log_target(theta), self.gradient(theta)
        return _nan_to_neg_inf(lp), grad

    def leapfrog(self, theta, r, grad, step):
        """One leapfrog step per chain with signed step sizes ``step`` (shape ``(C,)``)."""
        step = step.unsqueeze(-1)
        r_half = r + 0.5 * step * grad
        theta_new = theta + step * self._velocity(r_half)
        lp_new, grad_new = self._evaluate(theta_new)
        r_new = r_half + 0.5 * step * grad_new
        return theta_new, r_new, grad_new, lp_new

    def find_reasonable_step_size(self, theta, grad, lp):
        step = torch.ones(theta.shape[0], dtype=theta.dtype, device=theta.device)
        r0 = self._sample_momentum(theta)
        joint0 = lp - self._kinetic(r0)

        def trial(step):
            _, r1, g1, lp1 = self.leapfrog(theta, r0, grad, step)
            ok = torch.isfinite(lp1) & _finite_rows(g1)
            return lp1 - self._kinetic(r1) - joint0, ok

        log_accept, ok = trial(step)
        for _ in range(64):
            bad = ~ok & (step >= self.min_step_size)
            if not bool(bad.any()):
                break
            step = torch.where(bad, 0.5 * step, step)
            new_accept, new_ok = trial(step)
            log_accept = torch.where(bad, new_accept, log_accept)
            ok = torch.where(bad, new_ok, ok)
        step = torch.where(ok, step, torch.full_like(step, self.min_step_size))
        active = ok & torch.isfinite(log_accept)
        step = torch.where(ok & ~torch.isfinite(log_accept), torch.full_like(step, self.min_step_size), step)
        direction = torch.where(log_accept > _LOG_HALF, 1.0, -1.0).to(step.dtype)
        for _ in range(100):
            going = active & (direction * log_accept > -direction * _LOG_TWO)
            if not bool(going.any()):
                break
            new_step = torch.where(going, step * torch.pow(2.0, direction), step)
            outside = going & ((new_step < self.min_step_size) | (new_step > self.max_step_size))
            step = new_step
            active = going & ~outside
            new_accept, new_ok = trial(step)
            failed = active & ~new_ok
            step = torch.where(failed, 0.5 * step, step)
            active = active & new_ok
            log_accept = torch.where(active, new_accept, log_accept)
        return step.clamp(self.min_step_size, self.max_step_size)

    def _uturn_free(self, theta_minus, theta_plus, r_minus, r_plus):
        delta = theta_plus - theta_minus
        return ((delta * self._velocity(r_minus)).sum(-1) >= 0) & ((delta * self._velocity(r_plus)).sum(-1) >= 0)

    # ------------------------------------------------------------ transition
    def step(self, theta, warmup=False):
        if self.step_size is None:
            self._lp, self._grad = self._evaluate(theta)
            self._cached_theta = theta
            self._restart_dual_averaging(theta)
        elif theta is not self._cached_theta:
            self._lp, self._grad = self._evaluate(theta)
            self._cached_theta = theta

        if not warmup and self._adapting:
            # First draw after warmup: switch to the averaged step size before
            # building the trajectory, so every post-warmup draw uses it.
            self.step_size = self.step_size_bar
            self._adapting = False
        n_chains = theta.shape[0]
        device, dtype = theta.device, theta.dtype
        step_size = self.step_size
        lp0, grad0 = self._lp, self._grad
        r0 = self._sample_momentum(theta)
        H0 = -lp0 + self._kinetic(r0)

        # Ends of the trajectory and the current proposal.
        minus = [theta, r0, grad0]
        plus = [theta, r0, grad0]
        proposal = [theta, lp0, grad0, r0]
        log_w_tree = torch.zeros(n_chains, dtype=dtype, device=device)
        active = torch.ones(n_chains, dtype=torch.bool, device=device)
        depth = torch.zeros(n_chains, dtype=dtype, device=device)
        n_leapfrog = torch.zeros(n_chains, dtype=dtype, device=device)
        sum_alpha = torch.zeros(n_chains, dtype=dtype, device=device)
        diverging = torch.zeros(n_chains, dtype=torch.bool, device=device)
        neg_inf = torch.full((n_chains,), -math.inf, dtype=dtype, device=device)

        for j in range(self.max_depth):
            if not bool(active.any()):
                break
            depth = depth + active.to(dtype)
            forward = _uniform(n_chains, theta) < 0.5
            direction = torch.where(forward, 1.0, -1.0).to(dtype)
            start = [_where(forward, p, m) for p, m in zip(plus, minus)]
            cursor = list(start)
            sub_active = active.clone()
            sub_log_w = neg_inf.clone()
            sub_proposal = [start[0], lp0, start[2], start[1]]
            checkpoints = {}
            for n in range(2 ** j):
                if not bool(sub_active.any()):
                    break
                theta_n, r_n, grad_n, lp_n = self.leapfrog(cursor[0], cursor[1], cursor[2], direction * step_size)
                H = -lp_n + self._kinetic(r_n)
                finite = torch.isfinite(H) & _finite_rows(grad_n) & _finite_rows(theta_n)
                energy_error = torch.where(finite, H - H0, torch.full_like(H, math.inf))
                divergent = sub_active & (~finite | (energy_error > self.max_delta))
                diverging = diverging | divergent
                valid = sub_active & ~divergent
                log_w = torch.where(finite, -energy_error, neg_inf)
                alpha = torch.where(finite, torch.exp((-energy_error).clamp(max=0.0)), torch.zeros_like(H))
                sum_alpha = sum_alpha + torch.where(sub_active, alpha, torch.zeros_like(alpha))
                n_leapfrog = n_leapfrog + sub_active.to(dtype)
                # Progressive (multinomial) sampling within the subtree.
                new_sub_log_w = torch.logaddexp(sub_log_w, log_w)
                take = valid & (_log_uniform(n_chains, theta) < log_w - new_sub_log_w)
                leaf = [theta_n, lp_n, grad_n, r_n]
                sub_proposal = [_where(take, a, b) for a, b in zip(leaf, sub_proposal)]
                sub_log_w = torch.where(valid, new_sub_log_w, sub_log_w)
                cursor = [_where(valid, a, b) for a, b in zip([theta_n, r_n, grad_n], cursor)]
                # U-turns of every sub-subtree that ends with this leaf.
                level = 1
                while level <= j:
                    size = 2 ** level
                    if n % size == 0:
                        checkpoints[level] = (theta_n, r_n)
                    level += 1
                level = 1
                while level <= j:
                    size = 2 ** level
                    if (n + 1) % size == 0:
                        first_theta, first_r = checkpoints[level]
                        # first leaf is the minus end when moving forward
                        theta_minus = _where(forward, first_theta, theta_n)
                        theta_plus = _where(forward, theta_n, first_theta)
                        r_minus = _where(forward, first_r, r_n)
                        r_plus = _where(forward, r_n, first_r)
                        valid = valid & self._uturn_free(theta_minus, theta_plus, r_minus, r_plus)
                    level += 1
                sub_active = valid
            completed = active & sub_active
            # Extend the trajectory with the subtree's far end.
            plus = [_where(active & forward, c, p) for c, p in zip(cursor, plus)]
            minus = [_where(active & ~forward, c, m) for c, m in zip(cursor, minus)]
            # Biased progressive sampling between the old tree and the subtree.
            take = completed & (_log_uniform(n_chains, theta) < sub_log_w - log_w_tree)
            proposal = [_where(take, a, b) for a, b in zip(sub_proposal, proposal)]
            log_w_tree = torch.where(completed, torch.logaddexp(log_w_tree, sub_log_w), log_w_tree)
            active = completed & self._uturn_free(minus[0], plus[0], minus[1], plus[1])

        new_theta, new_lp, new_grad, new_r = proposal
        accept_stat = (sum_alpha / n_leapfrog.clamp(min=1.0)).clamp(0.0, 1.0)
        energy = -new_lp + self._kinetic(new_r)

        if warmup:
            eta = 1.0 / (self.m + self.t0)
            self.H_bar = (1 - eta) * self.H_bar + eta * (self.delta - accept_stat)
            log_step = self.mu - math.sqrt(self.m) / self.gamma * self.H_bar
            self.step_size = torch.exp(log_step).clamp(self.min_step_size, self.max_step_size)
            weight = self.m ** -self.kappa
            self.step_size_bar = torch.exp((1 - weight) * torch.log(self.step_size_bar) + weight * torch.log(self.step_size))
            self._adapting = True
        self.m += 1

        self._lp, self._grad = new_lp, new_grad
        self._cached_theta = new_theta

        if warmup and self._window_start is not None:
            iteration = self._warmup_iteration
            self._warmup_iteration += 1
            if self._window_ends and self._window_start <= iteration <= self._window_ends[-1]:
                self._welford_update(new_theta)
                if iteration in self._window_ends:
                    self._end_adaptation_window(new_theta)

        self.last_stats = {
            "step_size": step_size,
            "accept_stat": accept_stat,
            "tree_depth": depth,
            "n_leapfrog": n_leapfrog,
            "diverging": diverging.to(dtype),
            "energy": energy,
            "lp": new_lp,
        }
        return new_theta, self.step_size, accept_stat, diverging

    def invalidate_cache(self):
        self._cached_theta = None
