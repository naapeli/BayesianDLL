import torch
from tqdm import tqdm
from collections import deque


class Metropolis:
    def __init__(self, log_target, state_space, proposal_variance=1, adapt_rate=0.1, acceptance_low=0.2, acceptance_high=0.5):
        self.log_target = log_target
        self.state_space = state_space
        self.proposal_variance = proposal_variance
        self.adapt_rate = adapt_rate
        self.acceptance_low = acceptance_low
        self.acceptance_high = acceptance_high
        self.m = 0
        self.accept_queue = deque([], maxlen=20)
    
    def step(self, theta, warmup=False):
        theta_proposal = self.get_proposal(theta)
        acceptance_ratio = min(torch.ones(1), torch.exp(self.log_target(theta_proposal) - self.log_target(theta)))
        if torch.rand(1) < acceptance_ratio:
            theta = theta_proposal
            self.accept_queue.append(1)
        else:
            self.accept_queue.append(0)
        self.m += 1

        if warmup:
            if self.state_space.is_continuous(): self.adapt_proposal_variance()
        return theta, self.proposal_variance, sum(self.accept_queue) / len(self.accept_queue)
    
    def get_proposal(self, theta):
        if self.state_space.is_discrete():
            theta_proposal = theta + 2 * torch.randint(0, 2, size=(1,)) - 1
        elif self.state_space.is_continuous():
            theta_proposal = theta + self.proposal_variance * torch.randn_like(theta)
        else:
            raise ValueError("The state_space must either be discrete or continuous.")
        
        if self.state_space.contains(theta_proposal): return theta_proposal
        return theta
    
    def adapt_proposal_variance(self):
        acceptance_ratio = sum(self.accept_queue) / len(self.accept_queue)
        if acceptance_ratio < self.acceptance_low:
            self.proposal_variance *= (1 - self.adapt_rate)
        elif acceptance_ratio > self.acceptance_high:
            self.proposal_variance *= (1 + self.adapt_rate)
        
    def init_sampler(self):
        pass
