"""Sequential Monte Carlo with the particles as the chain dimension of a batched model.

The particles of :func:`~BayesianDLL.sample_smc` are independent given the
inverse temperature, so the model is batched with one "chain" per particle
(:meth:`~BayesianDLL.Model.batch`): the reweighting evaluates the prior and
the likelihood of all particles with one call, and the mutation kernels are
the vectorized samplers, which move all particles at once.
"""

import math

import torch
from tqdm import tqdm

from ._vectorized_samplers import (
    VectorizedFiniteGibbs, VectorizedMetropolis, VectorizedNUTS, VectorizedSlice, _log_uniform, _where,
)
from ._Metropolis import has_finite_support
from ._result import SamplingResult, SampleStats


_NUMERICAL_ERRORS = (ValueError, RuntimeError, OverflowError, ZeroDivisionError)


def _tempered(prior, likelihood, beta):
    ok = torch.isfinite(prior) & torch.isfinite(likelihood)
    return torch.where(ok, prior + beta * torch.where(ok, likelihood, torch.zeros_like(likelihood)), torch.full_like(prior, -math.inf))


class _BatchedTarget:
    """Prior and likelihood of all particles (shape ``(N, dim)``) of a batched model."""

    def __init__(self, model, target):
        self.model = model
        self.target = target
        self.n = model.n_chains

    def set(self, particles):
        for name, (start, end, shape) in self.target.slices.items():
            self.model.params[name].set_unconstrained_value(particles[:, start:end].reshape(self.n, *shape))

    def _guarded(self, evaluate, failure):
        try:
            return evaluate()
        except _NUMERICAL_ERRORS:
            def per_particle():
                try:
                    return evaluate()
                except _NUMERICAL_ERRORS:
                    return failure()
            mode = self.model._batch_mode
            self.model._batch_mode = "loop"
            try:
                return self.model._loop_over_chains(per_particle)
            finally:
                self.model._batch_mode = mode

    def log_prior_and_likelihood(self, particles):
        """``(log prior, log likelihood)``, each of shape ``(N,)``; ``-inf`` where the model fails numerically."""
        self.set(particles)
        model = self.model

        def evaluate():
            prior = model.prior_log_prob(self.target.names)
            likelihood = model._log_prob_of_terms(list(model.enumerated_params.values()), list(model.observed_params.values()))
            return (torch.as_tensor(prior, dtype=particles.dtype, device=particles.device).expand(self.n),
                    torch.as_tensor(likelihood, dtype=particles.dtype, device=particles.device).expand(self.n))

        def failure():
            inf = torch.tensor(-math.inf, dtype=particles.dtype, device=particles.device)
            return inf, inf

        prior, likelihood = self._guarded(evaluate, failure)
        nan_to_inf = lambda x: torch.where(torch.isnan(x), torch.full_like(x, -math.inf), x)
        return nan_to_inf(prior), nan_to_inf(likelihood)

    def tempered_gradient(self, particles, beta):
        """Gradient of ``log prior + beta * log likelihood`` w.r.t. the continuous coordinates, ``(N, d_c)``."""
        self.set(particles)
        names = self.target.continuous_names
        model = self.model

        def evaluate():
            joint = model.joint_grad_log_prob(names)
            prior = model._prior_gradient(names)
            return torch.cat([(prior[n] + beta * (joint[n] - prior[n])).reshape(self.n, -1) for n in names], dim=1)

        def failure():
            return torch.full((len(self.target.continuous_index),), math.nan, dtype=particles.dtype, device=particles.device)

        return self._guarded(evaluate, failure)


class _BatchedMutation:
    """MCMC moves of all particles at once at a fixed inverse temperature."""

    def __init__(self, target, batched_target, kernel, n_steps):
        self.target = target
        self.batched = batched_target
        self.kernel = kernel
        self.n_steps = n_steps
        self.beta = 0.0
        self.n = batched_target.n
        self.dim_c = len(target.continuous_index)
        self.scale = 2.38 / math.sqrt(max(self.dim_c, 1))
        self._full = None
        self._discrete = None
        self._step_size = None
        space = target.discrete_state_space()
        if space is not None:
            if has_finite_support(space):
                self._discrete = VectorizedFiniteGibbs(self._discrete_log_target, space, self.n)
            else:
                self._discrete = VectorizedMetropolis(self._discrete_log_target, space, self.n)

    def _log_target_at(self, flat):
        return _tempered(*self.batched.log_prior_and_likelihood(flat), self.beta)

    def _with(self, index, values):
        flat = self._full.clone()
        flat[:, index] = values.to(flat.dtype)
        return flat

    def _continuous_log_target(self, theta):
        return self._log_target_at(self._with(self.target.continuous_index, theta))

    def _continuous_gradient(self, theta):
        return self.batched.tempered_gradient(self._with(self.target.continuous_index, theta), self.beta)

    def _discrete_log_target(self, theta):
        return self._log_target_at(self._with(self.target.discrete_index, theta))

    def _prepare_continuous(self, continuous):
        n = continuous.shape[0]
        centred = continuous - continuous.mean(dim=0, keepdim=True)
        covariance = centred.T @ centred / max(n - 1, 1)
        variance = covariance.diagonal().clamp(min=1e-10)
        identity = torch.eye(self.dim_c, dtype=continuous.dtype, device=continuous.device)
        covariance = covariance + 1e-8 * variance.mean() * identity
        cholesky, info = torch.linalg.cholesky_ex(covariance)
        if int(info) != 0:
            cholesky = torch.diag(variance.sqrt())
        self._cholesky = cholesky
        if self.kernel == "nuts":
            self._sampler = VectorizedNUTS(self._continuous_log_target, self._continuous_gradient, self.n, delta=0.8, max_depth=8, metric="diag")
            self._sampler.inv_metric = variance.expand(self.n, self.dim_c).clone()
        elif self.kernel == "slice":
            self._sampler = VectorizedSlice(self._continuous_log_target, _real_space(), self.n, adapt_width=False)
            self._sampler.width = (2.0 * variance.sqrt()).expand(self.n, self.dim_c).clone()

    def mutate(self, particles, log_prior, log_likelihood, beta):
        self.beta = beta
        dtype, device = particles.dtype, particles.device
        acceptance = torch.zeros(self.n, dtype=dtype, device=device)
        particles = particles.clone()
        index = self.target.continuous_index.to(device)
        discrete_index = self.target.discrete_index.to(device)
        if self.dim_c > 0:
            self._prepare_continuous(particles[:, index])
        for _ in range(self.n_steps):
            if self.dim_c > 0:
                if self.kernel == "metropolis":
                    noise = torch.randn((self.n, self.dim_c), dtype=dtype, device=device)
                    proposals = particles.clone()
                    proposals[:, index] = particles[:, index] + self.scale * noise @ self._cholesky.T
                    prior, likelihood = self.batched.log_prior_and_likelihood(proposals)
                    new = _tempered(prior, likelihood, beta)
                    old = _tempered(log_prior, log_likelihood, beta)
                    accepted = torch.isfinite(new) & (_log_uniform(self.n, particles) < new - old)
                    particles = _where(accepted, proposals, particles)
                    log_prior = torch.where(accepted, prior, log_prior)
                    log_likelihood = torch.where(accepted, likelihood, log_likelihood)
                    rate = float(accepted.to(dtype).mean())
                    self.scale = min(max(self.scale * math.exp(rate - 0.234), 1e-4), 1e2)
                    acceptance = acceptance + accepted.to(dtype)
                else:
                    self._full = particles
                    self._sampler.invalidate_cache()
                    theta = particles[:, index].clone()
                    if self.kernel == "nuts":
                        self._set_pooled_step_size(theta)
                    new_theta, _, accept_stat, _ = self._sampler.step(theta, warmup=False)
                    if self.kernel == "nuts":
                        # The particles are exchangeable at a fixed temperature, so
                        # one step size is adapted from their mean acceptance.
                        mean_accept = float(torch.as_tensor(accept_stat).mean())
                        self._step_size = min(max(self._step_size * math.exp(2.0 * (mean_accept - 0.8)), 1e-4), 1e2)
                    flat = self._with(index, new_theta)
                    prior, likelihood = self.batched.log_prior_and_likelihood(flat)
                    keep = torch.isfinite(_tempered(prior, likelihood, beta))
                    particles = _where(keep, flat, particles)
                    log_prior = torch.where(keep, prior, log_prior)
                    log_likelihood = torch.where(keep, likelihood, log_likelihood)
                    acceptance = acceptance + torch.as_tensor(accept_stat, dtype=dtype, device=device)
            if self._discrete is not None:
                self._full = particles
                self._discrete.invalidate_cache()
                new_theta, *_ = self._discrete.step(particles[:, discrete_index].clone())
                particles = self._with(discrete_index, new_theta)
                log_prior, log_likelihood = self.batched.log_prior_and_likelihood(particles)
        if self.dim_c > 0 and self.kernel != "metropolis":
            self.scale = self._step_size if self.kernel == "nuts" else float(self._sampler.width.mean())
        if self.dim_c == 0:
            acceptance.fill_(float(self.n_steps))
        return particles, log_prior, log_likelihood, acceptance / max(self.n_steps, 1)


def _set_pooled_step_size(self, theta):
    """Run the next NUTS step with the population's step size (initially the median heuristic one)."""
    sampler = self._sampler
    if self._step_size is None:
        log_prob, gradient = sampler._evaluate(theta)
        self._step_size = float(sampler.find_reasonable_step_size(theta, gradient, log_prob).median())
    step = torch.full((self.n,), self._step_size, dtype=theta.dtype, device=theta.device)
    sampler.step_size = step
    sampler.step_size_bar = step
    sampler.mu = torch.log(10 * step)
    sampler.H_bar = torch.zeros_like(step)
    sampler.m = 1


_BatchedMutation._set_pooled_step_size = _set_pooled_step_size


def _real_space():
    from ..Distributions._state_space import ContinuousReal
    return ContinuousReal()


def run_smc_batched(model, n_particles, kernel, n_mutation_steps, target_ess, progress_bar):
    from ._SMC import _TemperedTarget, _draw_prior, _next_beta, _ess, _systematic_resample
    target = _TemperedTarget(model)
    reference = next(iter(model.params.values())).unconstrained_value
    dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
    device = reference.device
    particles = _draw_prior(model, target, n_particles, dtype, device)
    values = {name: particles[:, start:end].reshape(n_particles, *shape).to(model.params[name].unconstrained_value.dtype)
              for name, (start, end, shape) in target.slices.items()}

    with model.batched(n_particles, values=values):
        batched = _BatchedTarget(model, target)
        log_prior, log_likelihood = batched.log_prior_and_likelihood(particles)
        mutation = _BatchedMutation(target, batched, kernel, n_mutation_steps)
        beta = 0.0
        log_evidence = 0.0
        betas = [0.0]
        stage_ess = []
        stage_acceptance = []
        acceptance = torch.ones(n_particles, dtype=dtype, device=device)

        bar = tqdm(total=1.0, desc="SMC", disable=not progress_bar, bar_format=r"{desc}: {percentage:3.0f}% | {bar} | {elapsed} {postfix}")
        try:
            while beta < 1.0:
                safe_log_likelihood = torch.nan_to_num(log_likelihood, nan=-math.inf)
                new_beta = _next_beta(beta, safe_log_likelihood, target_ess * n_particles)
                log_weights = (new_beta - beta) * safe_log_likelihood
                if not bool(torch.isfinite(log_weights).any()):
                    raise RuntimeError("Every particle has zero likelihood; SMC cannot proceed.")
                log_evidence += float(torch.logsumexp(log_weights, dim=0)) - math.log(n_particles)
                stage_ess.append(_ess(log_weights))
                indices = _systematic_resample(torch.softmax(log_weights, dim=0))
                particles, log_prior, log_likelihood = particles[indices], log_prior[indices], log_likelihood[indices]
                bar.update(new_beta - beta)
                beta = new_beta
                betas.append(beta)
                particles, log_prior, log_likelihood, acceptance = mutation.mutate(particles, log_prior, log_likelihood, beta)
                stage_acceptance.append(float(acceptance.mean()))
                bar.set_postfix({"stage": len(betas) - 1, "beta": f"{beta:.4f}", "acc": f"{stage_acceptance[-1]:.3f}"}, refresh=True)
        finally:
            bar.close()

        batched.set(particles)
        trace = {name: p.constrained_value.clone().unsqueeze(0) for name, p in model.params.items()}
        deterministic_trace = {}
        for name, p in model.deterministic_params.items():
            value = torch.as_tensor(p.constrained_value)
            if name not in model._batched_nodes:
                value = value.unsqueeze(0).expand(n_particles, *value.shape)
            deterministic_trace[name] = value.clone().unsqueeze(0)
        enumerated_probs = {}
        if model.enumerated_params:
            for name, (value, probabilities) in model.sample_enumerated().items():
                trace[name] = value.clone().unsqueeze(0)
                enumerated_probs[name] = probabilities.clone().unsqueeze(0)

    log_posterior = log_prior + log_likelihood
    stats = SampleStats([{
        "params": list(target.names),
        "sampler": "smc",
        "stats": {
            "lp": log_posterior.detach().to(torch.float64).reshape(1, n_particles),
            "accept_stat": acceptance.detach().to(torch.float64).reshape(1, n_particles),
        },
    }])
    return SamplingResult(
        trace, [0], [[stage_acceptance[-1] if stage_acceptance else 1.0]], [[mutation.scale]],
        deterministic_trace=deterministic_trace, sample_stats=stats, enumerated_probs=enumerated_probs,
        log_marginal_likelihood=log_evidence, beta_schedule=betas, stage_ess=stage_ess,
        stage_acceptance=stage_acceptance, kernel=kernel,
    )
