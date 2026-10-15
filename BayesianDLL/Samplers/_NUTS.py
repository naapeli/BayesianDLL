import torch
import math
from collections import namedtuple


Tree = namedtuple("Tree", ["theta_minus", "r_minus", "grad_minus", "theta_plus", "r_plus", "grad_plus", "theta_prime", "grad_prime", "log_prob_prime", "n_prime", "s_prime", "alpha_prime", "n_prime_alpha", "diverging"])

def _dot(a, b):
    """Dot product that works for any-shape tensors (flattened)."""
    return (a * b).sum()

def _adaptation_windows(warmup_length, init_buffer=75, term_buffer=50, base_window=25):
    """Return the iterations that end a metric-adaptation window.

    Follows Stan's windowed adaptation: a fast initial buffer that only adapts
    the step size, a sequence of doubling slow windows in which the diagonal
    metric is estimated, and a terminal buffer where only the step size is
    adapted to the final metric. Returns ``(start, ends)``, where ``start`` is
    the first iteration of the first slow window and ``ends`` contains the
    last iteration of each slow window.
    """
    if warmup_length < 20:
        return None, []
    if init_buffer + base_window + term_buffer > warmup_length:
        init_buffer = int(0.15 * warmup_length)
        term_buffer = int(0.1 * warmup_length)
        base_window = warmup_length - init_buffer - term_buffer
    slow_end = warmup_length - term_buffer
    ends = []
    window = base_window
    start = init_buffer
    while start < slow_end:
        end = start + window
        # Stretch the last window to the terminal buffer instead of leaving a
        # window that is too short to give a useful variance estimate.
        if end + 2 * window > slow_end:
            end = slow_end
        ends.append(end - 1)
        start = end
        window *= 2
    return init_buffer, ends


class NUTS:
    def __init__(self, log_target, gradient, inverse_transformation, delta=0.6, gamma=0.05, step_size_bar=1, max_depth=10, t0=10, kappa=0.75, H_bar=0, min_step_size=1e-4, max_step_size=100, max_delta=1000.0, adapt_mass_matrix=True):
        self.log_target = log_target
        self.gradient = gradient
        self.inverse_transformation = inverse_transformation
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.step_size_bar = step_size_bar
        self.H_bar = H_bar
        self.delta = delta
        self.max_depth = max_depth
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.max_delta = max_delta
        self.adapt_mass_matrix = adapt_mass_matrix

        # Diagonal inverse metric (posterior variance estimate). ``None`` means
        # the identity, which is used until the first adaptation window ends.
        self.inv_metric = None
        self.warmup_length = None
        self._warmup_iteration = 0
        self._window_start = None
        self._window_ends = []
        self._reset_welford()

        # save in case of resetting
        self._H_bar = H_bar
        self._step_size_bar = step_size_bar

    def reset(self):
        if hasattr(self, "step_size"): del self.step_size  # force step to reset things
        self.H_bar = self._H_bar
        self.step_size_bar = self._step_size_bar
        self.inv_metric = None
        self._warmup_iteration = 0
        self._reset_welford()

    def set_warmup_length(self, warmup_length):
        """Enable windowed diagonal mass-matrix adaptation for ``warmup_length`` warmup steps."""
        self.warmup_length = warmup_length
        self._warmup_iteration = 0
        if self.adapt_mass_matrix:
            self._window_start, self._window_ends = _adaptation_windows(warmup_length)
        else:
            self._window_start, self._window_ends = None, []

    def _reset_welford(self):
        self._welford_n = 0
        self._welford_mean = None
        self._welford_m2 = None

    def _welford_update(self, theta):
        theta = theta.detach().reshape(-1)
        if self._welford_mean is None:
            self._welford_mean = torch.zeros_like(theta)
            self._welford_m2 = torch.zeros_like(theta)
        self._welford_n += 1
        delta = theta - self._welford_mean
        self._welford_mean = self._welford_mean + delta / self._welford_n
        self._welford_m2 = self._welford_m2 + delta * (theta - self._welford_mean)

    def _metric_like(self, theta):
        if self.inv_metric is None:
            return torch.ones_like(theta)
        return self.inv_metric.reshape(theta.shape).to(dtype=theta.dtype)

    def _sample_momentum(self, theta):
        return torch.randn_like(theta) / torch.sqrt(self._metric_like(theta))

    def _kinetic(self, r):
        return 0.5 * _dot(r * self._metric_like(r), r)

    def _end_adaptation_window(self, theta):
        n = self._welford_n
        if n >= 2:
            variance = self._welford_m2 / (n - 1)
            # Regularize towards a small multiple of the identity (as in Stan).
            self.inv_metric = (n / (n + 5.0)) * variance + 1e-3 * (5.0 / (n + 5.0))
        self._reset_welford()
        # Restart dual averaging for the new metric.
        self.step_size = self.find_reasonable_step_size(theta, self.gradient_cache, self.log_prob_cache)
        self.mu = math.log(10 * self.step_size)
        self.H_bar = self._H_bar
        self.step_size_bar = self._step_size_bar
        self.m = 1

    def leapfrog(self, theta, r, grad, step_size):
        r_prime = r + 0.5 * step_size * grad
        theta_prime = theta + step_size * self._metric_like(r_prime) * r_prime
        log_prob_prime = self.log_target(theta_prime)
        grad_prime = self.gradient(theta_prime)
        r_prime = r_prime + 0.5 * step_size * grad_prime
        return theta_prime, r_prime, grad_prime, log_prob_prime

    def find_reasonable_step_size(self, theta_init, grad_init, log_prob_init):
        step_size = 1.0
        r0 = self._sample_momentum(theta_init)
        _, r_prime, grad_prime, log_prob_prime = self.leapfrog(theta_init, r0, grad_init, step_size)
        while torch.isinf(log_prob_prime) or torch.isnan(log_prob_prime) or torch.isinf(grad_prime).any() or torch.isnan(grad_prime).any():
            step_size *= 0.5
            if step_size < self.min_step_size:
                return self.min_step_size
            _, r_prime, grad_prime, log_prob_prime = self.leapfrog(theta_init, r0, grad_init, step_size)
        log_accept_prob = log_prob_prime - log_prob_init - (self._kinetic(r_prime) - self._kinetic(r0))
        if torch.isnan(log_accept_prob) or torch.isinf(log_accept_prob):
            return self.min_step_size
        a = 1 if log_accept_prob > math.log(0.5) else -1
        count = 0
        while a * log_accept_prob > -a * math.log(2) and count < 100:
            step_size *= 2 ** a
            if step_size < self.min_step_size or step_size > self.max_step_size:
                break
            _, r_prime, grad_prime, log_prob_prime = self.leapfrog(theta_init, r0, grad_init, step_size)
            if torch.isnan(log_prob_prime) or torch.isinf(log_prob_prime) or torch.isnan(grad_prime).any() or torch.isinf(grad_prime).any():
                step_size *= 0.5
                break
            log_accept_prob = log_prob_prime - log_prob_init - (self._kinetic(r_prime) - self._kinetic(r0))
            count += 1
        
        step_size = min(max(step_size, self.min_step_size), self.max_step_size)
        return step_size

    def build_tree(self, theta, r, grad, log_u, v, j, step_size, joint0):
        if j == 0:
            theta_prime, r_prime, grad_prime, log_prob_prime = self.leapfrog(theta, r, grad, v * step_size)
            
            non_finite = (
                torch.isnan(log_prob_prime)
                or torch.isinf(log_prob_prime)
                or torch.isnan(grad_prime).any()
                or torch.isinf(grad_prime).any()
                or torch.isnan(r_prime).any()
                or torch.isinf(r_prime).any()
            )
            if non_finite:
                diverging = True
                return Tree(theta_prime, r_prime, grad_prime, theta_prime, r_prime, grad_prime, theta_prime, grad_prime, log_prob_prime, 0, 0, 0.0, 1.0, diverging)

            log_joint_prime = log_prob_prime - self._kinetic(r_prime)
            energy_diff = joint0 - log_joint_prime
            diverging = bool(
                energy_diff > self.max_delta
                or torch.isnan(log_joint_prime)
                or torch.isinf(log_joint_prime)
            )

            if diverging:
                return Tree(theta_prime, r_prime, grad_prime, theta_prime, r_prime, grad_prime, theta_prime, grad_prime, log_prob_prime, 0, 0, 0.0, 1.0, diverging)

            n_prime = 1 if log_u < log_joint_prime else 0
            s_prime = 1
            log_alpha = log_joint_prime - joint0
            alpha_prime = 1.0 if log_alpha >= 0 else float(torch.exp(log_alpha).clamp(max=1.0))
            return Tree(theta_prime, r_prime, grad_prime, theta_prime, r_prime, grad_prime, theta_prime, grad_prime, log_prob_prime, n_prime, s_prime, alpha_prime, 1.0, diverging)
        else:
            tree = self.build_tree(theta, r, grad, log_u, v, j - 1, step_size, joint0)
            theta_minus, r_minus, grad_minus = tree.theta_minus, tree.r_minus, tree.grad_minus
            theta_plus, r_plus, grad_plus = tree.theta_plus, tree.r_plus, tree.grad_plus
            theta_prime, grad_prime, log_prob_prime = tree.theta_prime, tree.grad_prime, tree.log_prob_prime
            n_prime, s_prime, alpha_prime, n_prime_alpha = tree.n_prime, tree.s_prime, tree.alpha_prime, tree.n_prime_alpha

            if tree.s_prime == 1:
                if v == -1:
                    tree_prime = self.build_tree(theta_minus, r_minus, grad_minus, log_u, v, j - 1, step_size, joint0)
                    theta_minus, r_minus, grad_minus = tree_prime.theta_minus, tree_prime.r_minus, tree_prime.grad_minus
                else:
                    tree_prime = self.build_tree(theta_plus, r_plus, grad_plus, log_u, v, j - 1, step_size, joint0)
                    theta_plus, r_plus, grad_plus = tree_prime.theta_plus, tree_prime.r_plus, tree_prime.grad_plus

                if tree_prime.s_prime == 1 and (tree.n_prime + tree_prime.n_prime) > 0:
                    if torch.rand(1) < tree_prime.n_prime / (tree.n_prime + tree_prime.n_prime):
                        theta_prime, grad_prime, log_prob_prime = tree_prime.theta_prime, tree_prime.grad_prime, tree_prime.log_prob_prime

                n_prime = tree.n_prime + tree_prime.n_prime
                s_prime = tree.s_prime * tree_prime.s_prime * self._uturn(theta_minus, theta_plus, r_minus, r_plus)
                alpha_prime = tree.alpha_prime + tree_prime.alpha_prime
                n_prime_alpha = tree.n_prime_alpha + tree_prime.n_prime_alpha
                diverging = tree.diverging or tree_prime.diverging
            else:
                diverging = tree.diverging
            return Tree(theta_minus, r_minus, grad_minus, theta_plus, r_plus, grad_plus, theta_prime, grad_prime, log_prob_prime, n_prime, s_prime, alpha_prime, n_prime_alpha, diverging)

    def _uturn(self, theta_minus, theta_plus, r_minus, r_plus):
        delta_theta = theta_plus - theta_minus
        metric = self._metric_like(delta_theta)
        return _dot(delta_theta, metric * r_minus) >= 0 and _dot(delta_theta, metric * r_plus) >= 0

    def step(self, theta, warmup=False):
        if not hasattr(self, "step_size"):
            log_prob = self.log_target(theta)
            gradient = self.gradient(theta)
            self.step_size = self.find_reasonable_step_size(theta, gradient, log_prob)
            self.mu = math.log(10 * self.step_size)
            self.gradient_cache = gradient
            self.log_prob_cache = log_prob
            self.m = 1

        r0 = self._sample_momentum(theta)
        joint = self.log_prob_cache - self._kinetic(r0)
        log_u = joint + torch.log(torch.rand(1))

        theta_minus = theta
        theta_plus = theta
        r_minus = r0
        r_plus = r0
        grad_minus = self.gradient_cache
        grad_plus = self.gradient_cache
        j, n, s = 0, 1, 1

        new_theta = theta
        new_log_prob = self.log_prob_cache
        new_gradient = self.gradient_cache

        alpha_sum_total = 0.0
        n_alpha_sum_total = 0.0
        diverging = False

        while s == 1 and j <= self.max_depth:
            v = 1 if torch.rand(1) < 0.5 else -1
            if v == -1:
                tree = self.build_tree(theta_minus, r_minus, grad_minus, log_u, v, j, self.step_size, joint)
                theta_minus, r_minus, grad_minus = tree.theta_minus, tree.r_minus, tree.grad_minus
            else:
                tree = self.build_tree(theta_plus, r_plus, grad_plus, log_u, v, j, self.step_size, joint)
                theta_plus, r_plus, grad_plus = tree.theta_plus, tree.r_plus, tree.grad_plus
            
            alpha_sum_total += float(tree.alpha_prime)
            n_alpha_sum_total += float(tree.n_prime_alpha)
            
            if tree.s_prime == 1 and n > 0:
                _tmp = min(1.0, tree.n_prime / n)
                if torch.rand(1) < _tmp:
                    new_theta = tree.theta_prime
                    new_log_prob = tree.log_prob_prime
                    new_gradient = tree.grad_prime
            n += tree.n_prime
            s = tree.s_prime * self._uturn(theta_minus, theta_plus, r_minus, r_plus)
            j += 1
            diverging = diverging or tree.diverging
        
        mean_alpha = (alpha_sum_total / n_alpha_sum_total) if n_alpha_sum_total > 0 else 0.0
        mean_alpha = min(max(mean_alpha, 0.0), 1.0)

        if warmup:
            eta = 1 / (self.m + self.t0)
            self.H_bar = (1 - eta) * self.H_bar + eta * (self.delta - mean_alpha)
            self.step_size = math.exp(self.mu - math.sqrt(self.m) / self.gamma * self.H_bar)
            self.step_size = min(max(self.step_size, self.min_step_size), self.max_step_size)
            eta = self.m ** -self.kappa
            self.step_size_bar = math.exp((1 - eta) * math.log(self.step_size_bar) + eta * math.log(self.step_size))
        else:
            self.step_size = self.step_size_bar

        # Only update cache if new state is finite and valid
        if not torch.isnan(new_log_prob) and not torch.isinf(new_log_prob) and not torch.isnan(new_gradient).any():
            self.gradient_cache = new_gradient
            self.log_prob_cache = new_log_prob
        self.m += 1

        if warmup and self._window_start is not None:
            iteration = self._warmup_iteration
            self._warmup_iteration += 1
            if self._window_ends and self._window_start <= iteration <= self._window_ends[-1]:
                self._welford_update(new_theta)
                if iteration in self._window_ends:
                    self._end_adaptation_window(new_theta)

        return new_theta, self.step_size, mean_alpha, diverging

    def init_sampler(self):
        pass
