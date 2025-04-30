import torch
from tqdm import tqdm



class Metropolis:
    def __init__(self, log_target, state_space):
        self.log_target = log_target
        self.state_space = state_space
        self.proposal_variance = 1
        self.adapt_rate = 0.1
        self.acceptance_low = 0.2
        self.acceptance_high = 0.5
        self.m = 0
        self.n_accepted = 0

    def sample(self, M, theta_init, M_adapt=10):
        D = len(theta_init)
        samples = torch.empty((M + M_adapt + 1, D), dtype=theta_init.dtype)
        theta = theta_init

        progress_bar = tqdm(range(1, M + M_adapt + 1))
        for _ in progress_bar:
            if self.m < M_adapt: progress_bar.set_description(f"Warmup")
            else: progress_bar.set_description(f"Sample")

            theta = self.step(theta, self.m < M_adapt)
            samples[self.m] = theta

        return samples[M_adapt:]
    
    def step(self, theta, warmup=False):
        theta_proposal = self.get_proposal(theta)
        acceptance_ratio = min(1, torch.exp(self.log_target(theta_proposal) - self.log_target(theta)))
        if torch.rand(1) < acceptance_ratio:
            theta = theta_proposal
            self.n_accepted += 1
        self.m += 1

        if warmup:
            if self.state_space.is_continuous(): self.adapt_proposal_variance()
        return theta
    
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
        acceptance_ratio = self.n_accepted / self.m
        if acceptance_ratio < self.acceptance_low:
            self.proposal_variance *= (1 - self.adapt_rate)
        elif acceptance_ratio > self.acceptance_high:
            self.proposal_variance *= (1 + self.adapt_rate)
        
    def init_sampler(self):
        pass
