import torch
import math
from collections import namedtuple


Tree = namedtuple("Tree", ["theta_minus", "r_minus", "grad_minus", "theta_plus", "r_plus", "grad_plus", "theta_prime", "grad_prime", "log_prob_prime", "n_prime", "s_prime", "alpha_prime", "n_prime_alpha", "diverging"])

def _dot(a, b):
    """Dot product that works for any-shape tensors (flattened)."""
    return (a * b).sum()

class NUTS:
    def __init__(self, log_target, gradient, inverse_transformation, delta=0.6, gamma=0.05, step_size_bar=1, max_depth=10, t0=10, kappa=0.75, H_bar=0, min_step_size=1e-4, max_step_size=100, max_delta=1000.0):
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

        # save in case of resetting
        self._H_bar = H_bar
        self._step_size_bar = step_size_bar

    def reset(self):
        if hasattr(self, "step_size"): del self.step_size  # force step to reset things
        self.H_bar = self._H_bar
        self.step_size_bar = self._step_size_bar

    def leapfrog(self, theta, r, grad, step_size):
        r_prime = r + 0.5 * step_size * grad
        theta_prime = theta + step_size * r_prime
        log_prob_prime = self.log_target(theta_prime)
        grad_prime = self.gradient(theta_prime)
        r_prime = r_prime + 0.5 * step_size * grad_prime
        return theta_prime, r_prime, grad_prime, log_prob_prime

    def find_reasonable_step_size(self, theta_init, grad_init, log_prob_init):
        step_size = 1.0
        r0 = torch.randn_like(theta_init)
        _, r_prime, grad_prime, log_prob_prime = self.leapfrog(theta_init, r0, grad_init, step_size)
        while torch.isinf(log_prob_prime) or torch.isnan(log_prob_prime) or torch.isinf(grad_prime).any() or torch.isnan(grad_prime).any():
            step_size *= 0.5
            if step_size < self.min_step_size:
                return self.min_step_size
            _, r_prime, grad_prime, log_prob_prime = self.leapfrog(theta_init, r0, grad_init, step_size)
        log_accept_prob = log_prob_prime - log_prob_init - 0.5 * (_dot(r_prime, r_prime) - _dot(r0, r0))
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
            log_accept_prob = log_prob_prime - log_prob_init - 0.5 * (_dot(r_prime, r_prime) - _dot(r0, r0))
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

            log_joint_prime = log_prob_prime - 0.5 * _dot(r_prime, r_prime)
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
        return _dot(delta_theta, r_minus) >= 0 and _dot(delta_theta, r_plus) >= 0

    def step(self, theta, warmup=False):
        if not hasattr(self, "step_size"):
            log_prob = self.log_target(theta)
            gradient = self.gradient(theta)
            self.step_size = self.find_reasonable_step_size(theta, gradient, log_prob)
            self.mu = math.log(10 * self.step_size)
            self.gradient_cache = gradient
            self.log_prob_cache = log_prob
            self.m = 1

        r0 = torch.randn_like(theta)
        joint = self.log_prob_cache - 0.5 * _dot(r0, r0)
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

        return new_theta, self.step_size, mean_alpha, diverging

    def init_sampler(self):
        pass
