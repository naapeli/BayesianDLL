import torch
import math
from tqdm import tqdm
from collections import namedtuple


Tree = namedtuple("Tree", ["theta_minus", "r_minus", "grad_minus", "theta_plus", "r_plus", "grad_plus", "theta_prime", "grad_prime", "log_prob_prime", "n_prime", "s_prime", "alpha_prime", "n_prime_alpha"])

class NUTS:
    # def __init__(self, distribution):
    def __init__(self, log_target, gradient, inverse_transformation):
        # self.log_target = distribution._log_prob_unconstrained
        # self.gradient = distribution._log_prob_grad_unconstrained
        # self.inverse_transformation = distribution.transform.inverse
        self.log_target = log_target
        self.gradient = gradient
        self.inverse_transformation = inverse_transformation
        self.gamma = 0.05
        self.t0 = 10
        self.kappa = 0.75
        self.step_size_bar = 1
        self.H_bar = 0
        self.delta = 0.6
        self.max_depth = 10

    def leapfrog(self, theta, r, grad, step_size):
        r_prime = r +  0.5 * step_size * grad
        theta_prime = theta + step_size * r_prime
        log_prob_prime, grad_prime = self.log_target(theta_prime), self.gradient(theta_prime)
        r_prime = r_prime + 0.5 * step_size * grad_prime
        return theta_prime, r_prime, grad_prime, log_prob_prime

    def find_reasonable_step_size(self, theta_init, grad_init, log_prob_init):
        step_size = 1
        r0 = torch.randn_like(theta_init)
        _, r_prime, grad_prime, log_prob_prime = self.leapfrog(theta_init, r0, grad_init, step_size)
        k = 1
        while torch.isinf(log_prob_prime) or torch.isinf(grad_prime).any():
            k *= 0.5
            _, r_prime, _, log_prob_prime = self.leapfrog(theta_init, r0, grad_init, step_size * k)
        step_size = 0.5 * k * step_size
        log_accept_prob = log_prob_prime - log_prob_init - 0.5 * (r_prime @ r_prime - r0 @ r0)
        a = 1 if log_accept_prob > math.log(0.5) else -1
        while a * log_accept_prob > -a * math.log(2):
            step_size = step_size * (2 ** a)
            _, r_prime, _, log_prob_prime = self.leapfrog(theta_init, r0, grad_init, step_size)
            log_accept_prob = log_prob_prime - log_prob_init - 0.5 * (r_prime @ r_prime - r0 @ r0)
        return step_size

    def build_tree(self, theta, r, grad, log_u, v, j, step_size, joint0):
        if j == 0:
            theta_prime, r_prime, grad_prime, log_prob_prime = self.leapfrog(theta, r, grad, v * step_size)
            log_joint_prime = log_prob_prime - 0.5 * r_prime @ r_prime
            # n_prime = int(log_u < log_joint_prime)
            # s_prime = int((log_u - 1000) < log_joint_prime)
            n_prime = 1 if log_u < log_joint_prime else 0
            s_prime = 1 if log_u < (log_joint_prime + 1000) else 0
            return Tree(theta_prime, r_prime, grad_prime, theta_prime, r_prime, grad_prime, theta_prime, grad_prime, log_prob_prime, n_prime, s_prime, min(1, torch.exp(log_joint_prime - joint0)), 1)
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
                if torch.rand(1) < tree_prime.n_prime / max(tree.n_prime + tree_prime.n_prime, 1):
                    theta_prime, grad_prime, log_prob_prime = tree_prime.theta_prime, tree_prime.grad_prime, tree_prime.log_prob_prime
                n_prime = tree.n_prime + tree_prime.n_prime
                s_prime = tree.s_prime * tree_prime.s_prime * self._uturn(theta_minus, theta_plus, r_minus, r_plus)
                alpha_prime = tree.alpha_prime + tree_prime.alpha_prime
                n_prime_alpha = tree.n_prime_alpha + tree_prime.n_prime_alpha
            return Tree(theta_minus, r_minus, grad_minus, theta_plus, r_plus, grad_plus, theta_prime, grad_prime, log_prob_prime, n_prime, s_prime, alpha_prime, n_prime_alpha)

    def _uturn(self, theta_minus, theta_plus, r_minus, r_plus):
        delta_theta = theta_plus - theta_minus
        return delta_theta @ r_minus >= 0 and delta_theta @ r_plus >= 0

    def sample(self, M, theta_init, M_adapt=10, delta=0.6):
        if theta_init.ndim != 1:
            raise ValueError("theta_init should be a one dimensional tensor.")
        
        D = len(theta_init)
        samples = torch.empty((M + M_adapt + 1, D), dtype=theta_init.dtype)
        lnprob = torch.empty((M + M_adapt + 1,), dtype=theta_init.dtype)

        log_prob, gradient = self.log_target(theta_init), self.gradient(theta_init)
        samples[0] = theta_init
        lnprob[0] = log_prob

        step_size = self.find_reasonable_step_size(theta_init, gradient, log_prob)

        mu = math.log(10 * step_size)
        step_size_bar = 1
        H_bar = 0

        progress_bar = tqdm(range(1, M + M_adapt + 1))
        for m in progress_bar:
            if m < M_adapt: progress_bar.set_description(f"Warmup")
            else: progress_bar.set_description(f"Sample")
            
            r0 = torch.randn(D, dtype=theta_init.dtype)
            joint = log_prob - 0.5 * r0 @ r0
            log_u = joint + torch.log(torch.rand(1))
            
            samples[m] = samples[m - 1]
            lnprob[m] = lnprob[m - 1]
            theta_minus = samples[m - 1]
            theta_plus = samples[m - 1]
            r_minus = r0
            r_plus = r0
            grad_minus = gradient
            grad_plus = gradient
            j, n, s = 0, 1, 1

            while s == 1 and j <= self.max_depth:
                v = 1 if torch.rand(1) < 0.5 else -1
                if v == -1:
                    tree = self.build_tree(theta_minus, r_minus, grad_minus, log_u, v, j, step_size, joint)
                    theta_minus, r_minus, grad_minus = tree.theta_minus, tree.r_minus, tree.grad_minus
                else:
                    tree = self.build_tree(theta_plus, r_plus, grad_plus, log_u, v, j, step_size, joint)
                    theta_plus, r_plus, grad_plus = tree.theta_plus, tree.r_plus, tree.grad_plus
                
                _tmp = min(1, tree.n_prime / n)
                if tree.s_prime == 1 and torch.rand(1) < _tmp:
                    samples[m] = tree.theta_prime
                    lnprob[m] = tree.log_prob_prime
                    log_prob = tree.log_prob_prime
                    gradient = tree.grad_prime
                n += tree.n_prime
                s = tree.s_prime * self._uturn(theta_minus, theta_plus, r_minus, r_plus)
                j += 1
            
            if m <= M_adapt:
                eta = 1 / (m + self.t0)
                H_bar = (1 - eta) * H_bar + eta * (delta - tree.alpha_prime / tree.n_prime_alpha)
                step_size = math.exp(mu - math.sqrt(m) / self.gamma * H_bar)
                eta = m ** -self.kappa
                step_size_bar = math.exp((1 - eta) * math.log(step_size_bar) + eta * math.log(step_size))
            else:
                step_size = step_size_bar
        samples = self.inverse_transformation(samples[M_adapt:])
        lnprob = lnprob[M_adapt:]
        return samples, lnprob, step_size

    def step(self, theta, warmup=False):
        if theta.ndim != 1:
            raise ValueError("theta should be a one-dimensional tensor.")
        
        D = len(theta)
        if not hasattr(self, "step_size"):
            log_prob = self.log_target(theta)
            gradient = self.gradient(theta)
            self.step_size = self.find_reasonable_step_size(theta, gradient, log_prob)
            self.mu = math.log(10 * self.step_size)
            self.gradient_cache = gradient
            self.log_prob_cache = log_prob
            self.m = 1

        r0 = torch.randn(D, dtype=theta.dtype)
        joint = self.log_prob_cache - 0.5 * r0 @ r0
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

        while s == 1 and j <= self.max_depth:
            v = 1 if torch.rand(1) < 0.5 else -1
            if v == -1:
                tree = self.build_tree(theta_minus, r_minus, grad_minus, log_u, v, j, self.step_size, joint)
                theta_minus, r_minus, grad_minus = tree.theta_minus, tree.r_minus, tree.grad_minus
            else:
                tree = self.build_tree(theta_plus, r_plus, grad_plus, log_u, v, j, self.step_size, joint)
                theta_plus, r_plus, grad_plus = tree.theta_plus, tree.r_plus, tree.grad_plus
            
            _tmp = min(1, tree.n_prime / n)
            if tree.s_prime == 1 and torch.rand(1) < _tmp:
                new_theta = tree.theta_prime
                new_log_prob = tree.log_prob_prime
                new_gradient = tree.grad_prime
            n += tree.n_prime
            s = tree.s_prime * self._uturn(theta_minus, theta_plus, r_minus, r_plus)
            j += 1
        
        if warmup:
            eta = 1 / (self.m + self.t0)
            self.H_bar = (1 - eta) * self.H_bar + eta * (self.delta - tree.alpha_prime / tree.n_prime_alpha)
            self.step_size = math.exp(self.mu - math.sqrt(self.m) / self.gamma * self.H_bar)
            eta = self.m ** -self.kappa
            self.step_size_bar = math.exp((1 - eta) * math.log(self.step_size_bar) + eta * math.log(self.step_size))
        else:
            self.step_size = self.step_size_bar

        self.gradient_cache = new_gradient
        self.log_prob_cache = new_log_prob
        self.m += 1

        return new_theta

    def init_sampler(self):
        pass
