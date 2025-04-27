import torch
from tqdm import tqdm



class Metropolis:
    def __init__(self, log_target, state_space):
        self.log_target = log_target
        self.state_space = state_space

    def sample(self, M, theta_init, M_adapt=10):
        D = len(theta_init)
        samples = torch.empty((M + M_adapt, D), dtype=theta_init.dtype)
        theta = theta_init

        progress_bar = tqdm(range(1, M + M_adapt))
        for m in progress_bar:
            if m < M_adapt: progress_bar.set_description(f"Warmup")
            else: progress_bar.set_description(f"Sample")

            theta_proposal = theta + 2 * torch.randint(0, 2, size=(1,)) - 1  # either move left or right
            theta_proposal = theta_proposal if self.state_space.contains(theta_proposal) else theta

            p_current = self.log_target(theta)
            p_proposal = self.log_target(theta_proposal)

            acceptance_ratio = min(1, torch.exp(p_proposal - p_current))

            if torch.rand(1) < acceptance_ratio:
                theta = theta_proposal

            samples[m] = theta

        return samples[M_adapt:]
