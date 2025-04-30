import torch
from tqdm import tqdm

from . import NUTS, Metropolis
from ..parameters import RandomParameter


def sample(model, n_samples, warmup_length):
    steps = {}
    for name, parameter in model.params.items():
        steps[name] = _decide_step(model, parameter)
    
    samples = {name: [] for name in model.params.keys()}
    
    progress_bar = tqdm(range(1, n_samples + warmup_length + 1))
    for m in progress_bar:
        if m < warmup_length: progress_bar.set_description(f"Warmup")
        else: progress_bar.set_description(f"Sample")

        for name, sampler in steps.items():
            theta = model.params[name].unconstrained_value
            new_theta = sampler.step(theta, m < warmup_length)
            model.params[name].set_unconstrained_value(new_theta)
            samples[name].append(model.params[name].constrained_value)

    samples = {name: torch.tensor(_samples[warmup_length:]) for name, _samples in samples.items()}
    return samples

def _decide_step(model, parameter: RandomParameter):
    def _log_prob(theta):
        return model.log_prob({parameter.name: theta})

    state_space = parameter.distribution.transformed_state_space

    if state_space.is_continuous():
        sampler = Metropolis(_log_prob, state_space)  # TODO: change to NUTS once derivative is implemented
    elif state_space.is_discrete():
        sampler = Metropolis(_log_prob, state_space)
    else:
        raise RuntimeError("A distribution must be either discrete or continuous.")
    
    sampler.init_sampler()
    return sampler
