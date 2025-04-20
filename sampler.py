import torch
import math
from collections import namedtuple


Tree = namedtuple("Tree", ["theta_minus", "r_minus", "theta_plus", "r_plus", "theta_prime", "n_prime", "s_prime", "alpha_prime", "n_prime_alpha"])


class NUTS:
    def __init__(self, distribution, step_size=0.1, max_depth=10, target_accept=0.65):
        self.log_target = distribution.log_pdf
        self.gradient = distribution.log_pdf_grad
        self.step_size = step_size
        self.max_depth = max_depth
        self.target_accept = target_accept
        self.mu = math.log(10 * step_size)
        self.H = 0.0
        self.gamma = 0.05
        self.t0 = 10
        self.kappa = 0.75

    def leapfrog(self, theta, r, step_size):
        r += 0.5 * step_size * self.gradient(theta)
        theta += step_size * r
        r += 0.5 * step_size * self.gradient(theta)
        return theta, r

    def find_reasonable_step_size(self, theta_init):
        r = torch.randn_like(theta_init)
        log_p = self.log_target(theta_init)
        log_joint = log_p - 0.5 * r.dot(r)

        step_size = 1.0
        theta, r = self.leapfrog(theta_init, r, step_size)
        log_p_new = self.log_target(theta)
        log_joint_new = log_p_new - 0.5 * r.dot(r)
        a = 1.0 if math.exp(log_joint_new - log_joint) > 0.5 else -1.0

        while a * math.exp(log_joint_new - log_joint) > 2**(-a):
            step_size *= 2.0**a
            theta, r = self.leapfrog(theta_init, r, step_size)
            log_p_new = self.log_target(theta)
            log_joint_new = log_p_new - 0.5 * r.dot(r)
        return step_size

    def build_tree(self, theta, r, log_u, v, j, step_size, joint0):
        if j == 0:
            theta_prime, r_prime = self.leapfrog(theta, r, v * step_size)
            log_p_prime = self.log_target(theta_prime)
            log_joint_prime = log_p_prime - 0.5 * r_prime @ r_prime
            n_prime = 1 if log_u <= log_joint_prime else 0
            s_prime = 1 if log_u < (log_joint_prime + 1000.0) else 0
            return Tree(theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime, min(1, torch.exp(log_joint_prime - joint0)), 1)
        else:
            tree = self.build_tree(theta, r, log_u, v, j - 1, step_size, joint0)
            theta_minus, r_minus, theta_plus, r_plus = tree.theta_minus, tree.r_minus, tree.theta_plus, tree.r_plus
            if tree.s_prime == 1:
                if v == -1:
                    tree_prime = self.build_tree(tree.theta_minus, tree.r_minus, log_u, v, j - 1, step_size, joint0)
                    theta_minus, r_minus = tree_prime.theta_minus, tree_prime.r_minus
                else:
                    tree_prime = self.build_tree(tree.theta_plus, tree.r_plus, log_u, v, j - 1, step_size, joint0)
                    theta_plus, r_plus = tree_prime.theta_plus, tree_prime.r_plus
                n_prime = tree_prime.n_prime + tree.n_prime
                alpha = tree.alpha_prime + tree_prime.alpha_prime
                n_prime_alpha = tree.n_prime_alpha + tree_prime.n_prime_alpha
                accept_prob = min(1.0, tree_prime.n_prime / n_prime) if n_prime != 0 else 1
                if torch.rand(1) < accept_prob:
                    proposal = tree_prime.theta_prime
                else:
                    proposal = tree.theta_prime
                s_prime = tree_prime.s_prime * self._uturn(theta_plus, theta_minus, r_plus, r_minus)
                return Tree(theta_minus, r_minus, theta_plus, r_plus, proposal, n_prime, s_prime, alpha, n_prime_alpha)
            else:
                return tree

    def _uturn(self, theta_plus, theta_minus, r_plus, r_minus):
        delta_theta = theta_plus - theta_minus
        return delta_theta @ r_minus >= 0 and delta_theta @ r_plus >= 0

    def sample(self, num_samples, theta_init, M_adapt=10):
        theta = theta_init.clone()
        samples = []
        step_size = self.find_reasonable_step_size(theta_init)

        for m in range(1, num_samples + M_adapt + 1):
            r = torch.randn_like(theta)
            log_p = self.log_target(theta)
            joint = log_p - 0.5 * r @ r
            random = torch.rand(1)
            log_u = joint + torch.log(random)
            # log_u = torch.log(random * torch.exp(joint))

            theta_minus = theta.clone()
            theta_plus = theta.clone()
            r_minus = r.clone()
            r_plus = r.clone()
            j = 0
            n = 1
            s = 1

            while s == 1 and j < self.max_depth:
                v = 1 if torch.rand(1) < 0.5 else -1
                if v == -1:
                    tree = self.build_tree(theta_minus, r_minus, log_u, v, j, step_size, joint)
                    theta_minus, r_minus = tree.theta_minus, tree.r_minus
                else:
                    tree = self.build_tree(theta_plus, r_plus, log_u, v, j, step_size, joint)
                    theta_plus, r_plus = tree.theta_plus, tree.r_plus

                if tree.s_prime == 1:
                    accept_prob = min(1, tree.n_prime / n)
                    if torch.rand(1) < accept_prob:
                        theta = tree.theta_prime

                n += tree.n_prime
                s = tree.s_prime * self._uturn(theta_plus, theta_minus, r_plus, r_minus)
                j += 1

            eta = 1 / (m + self.t0)
            self.H = (1 - eta) * self.H + eta * (self.target_accept - tree.alpha_prime / tree.n_prime_alpha)
            if m < M_adapt:
                log_step_size = self.mu - math.sqrt(m + 1) / self.gamma * self.H
                step_size = math.exp(log_step_size)
            else:
                step_size = math.exp(self.mu - math.sqrt(self.t0) / self.gamma * self.H)

            samples.append(theta.clone())
        return torch.stack(samples)[M_adapt:]
