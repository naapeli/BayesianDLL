"""Sequential Monte Carlo with adaptive tempering."""

import contextlib
import math

import torch
from tqdm import tqdm

from ._NUTS import NUTS
from ._Metropolis import Metropolis, FiniteGibbs, has_finite_support
from ._Slice import Slice
from ._result import SamplingResult, SampleStats
from .._active_model._active_model import get_active_model
from ..Distributions._state_space import ContinuousReal, JointStateSpace


_KERNELS = ("metropolis", "nuts", "slice")
_DEFAULT_MUTATION_STEPS = {"metropolis": 10, "nuts": 2, "slice": 2}
_NUMERICAL_ERRORS = (ValueError, RuntimeError, OverflowError, ZeroDivisionError)


def _to_float(value):
    value = float(value)
    return value if not math.isnan(value) else -math.inf


class _TemperedTarget:
    """The model's prior and likelihood as functions of a flat unconstrained vector.

    The prior consists of the prior terms of the sampled random parameters;
    the likelihood of everything else, i.e. the observed terms and the factor
    groups of enumerated (summed out) parameters. The tempered target at
    inverse temperature ``beta`` is ``log prior + beta * log likelihood``,
    which is the prior at ``beta = 0`` and the posterior at ``beta = 1``.
    """

    def __init__(self, model):
        self.model = model
        for name in model.params:
            if name in model._enumeration_group_of:
                raise NotImplementedError(
                    f"sample_smc does not support the random parameter '{name}', whose prior term depends on an "
                    "enumerated parameter: its prior cannot be separated from the likelihood."
                )
        self.names = list(model.params)
        self.slices = {}
        offset = 0
        for name in self.names:
            value = model.params[name].unconstrained_value
            self.slices[name] = (offset, offset + value.numel(), value.shape)
            offset += value.numel()
        self.dim = offset

        self.continuous_names = [n for n in self.names if model.params[n].distribution.transformed_state_space.is_continuous()]
        self.discrete_names = [n for n in self.names if n not in self.continuous_names]
        self.continuous_index = self._indices(self.continuous_names)
        self.discrete_index = self._indices(self.discrete_names)
        self._prior_terms = set(self.names)
        self._evaluated = False

    def _indices(self, names):
        indices = [torch.arange(*self.slices[n][:2]) for n in names]
        return torch.cat(indices) if indices else torch.zeros(0, dtype=torch.long)

    def discrete_state_space(self):
        if not self.discrete_names:
            return None
        spaces = {n: self.model.params[n].distribution.transformed_state_space for n in self.discrete_names}
        if len(spaces) == 1:
            return next(iter(spaces.values()))
        slices, offset = {}, 0
        for name in self.discrete_names:
            start, end, shape = self.slices[name]
            slices[name] = (offset, offset + end - start, shape)
            offset += end - start
        return JointStateSpace(spaces, slices)

    def pack(self):
        values = [self.model.params[n].unconstrained_value.reshape(-1) for n in self.names]
        return torch.cat(values) if values else torch.zeros(0)

    def set(self, flat):
        for name, (start, end, shape) in self.slices.items():
            self.model.params[name].set_unconstrained_value(flat[start:end].reshape(shape))

    def _guarded(self, function, failure):
        if not self._evaluated:
            value = function()
            self._evaluated = True
            return value
        try:
            return function()
        except _NUMERICAL_ERRORS:
            return failure

    def log_prior_and_likelihood(self, flat):
        """``(log prior, log likelihood)`` as floats; ``-inf`` where the model fails numerically."""
        self.set(flat)
        model = self.model

        def evaluate():
            prior = sum((model.params[n].distribution._log_prob_unconstrained(model.params[n].unconstrained_value) for n in self.names), 0.0)
            likelihood = model._log_prob_of_terms(list(model.enumerated_params.values()), list(model.observed_params.values()))
            return _to_float(prior), _to_float(likelihood)

        return self._guarded(evaluate, (-math.inf, -math.inf))

    def tempered_gradient(self, flat, beta):
        """Gradient of ``log prior + beta * log likelihood`` w.r.t. the continuous coordinates of ``flat``."""
        self.set(flat)
        names = self.continuous_names
        model = self.model

        def evaluate():
            joint = model.joint_grad_log_prob(names)
            prior = model._reverse_pass(names, active_terms=self._prior_terms)
            return torch.cat([(prior[n] + beta * (joint[n] - prior[n])).reshape(-1) for n in names])

        return self._guarded(evaluate, torch.full((len(self.continuous_index),), float("nan"), dtype=flat.dtype, device=flat.device))


def _draw(distribution, shape, n):
    """``n`` prior draws of a parameter with value shape ``shape``, shape ``(n, *shape)``."""
    try:
        single = distribution.sample(1)
    except NotImplementedError as error:
        raise NotImplementedError(f"sample_smc draws the initial particles from the prior, but {type(distribution).__name__} cannot be sampled.") from error
    per_draw = max(single[0].numel(), 1)
    total = math.prod(shape)
    if total % per_draw:
        raise RuntimeError(f"Cannot shape draws of {type(distribution).__name__} with shape {tuple(single.shape[1:])} to the parameter shape {tuple(shape)}.")
    repeats = total // per_draw
    if n == 1 and repeats == 1:
        return single.reshape(1, *shape)
    return distribution.sample(n * repeats).reshape(n, *shape)


def _draw_prior(model, target, n_particles, dtype, device):
    """Particles drawn from the prior, as flat unconstrained vectors of shape ``(n_particles, dim)``."""
    random_terms = {**model.params, **model.enumerated_params}
    order = [name for name in model._topo_order if name in random_terms]
    stochastic_nodes = set(random_terms) | set(model.deterministic_params)
    independent = {}
    for name in order:
        parameter = random_terms[name]
        if not (set(getattr(parameter.distribution, "parameters", ())) & stochastic_nodes):
            shape = parameter.constrained_value.shape
            independent[name] = _draw(parameter.distribution, shape, n_particles).to(dtype=parameter.constrained_value.dtype, device=device)

    particles = torch.empty((n_particles, target.dim), dtype=dtype, device=device)
    for i in range(n_particles):
        for name in order:
            parameter = random_terms[name]
            if name in independent:
                value = independent[name][i]
            else:
                value = _draw(parameter.distribution, parameter.constrained_value.shape, 1)[0]
                value = value.to(dtype=parameter.constrained_value.dtype, device=device)
            parameter.set_constrained_value(value)
        particles[i] = target.pack().to(dtype=dtype, device=device)
    return particles


def _ess(log_weights):
    weights = torch.softmax(log_weights, dim=0)
    return float(1.0 / weights.square().sum())


def _next_beta(beta, log_likelihood, target_ess, iterations=60):
    """Largest inverse temperature in ``(beta, 1]`` whose incremental weights have ESS >= ``target_ess``."""
    remaining = 1.0 - beta
    if _ess(remaining * log_likelihood) >= target_ess:
        return 1.0
    low, high = 0.0, remaining
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if _ess(middle * log_likelihood) >= target_ess:
            low = middle
        else:
            high = middle
    increment = low if low > 0 else high
    return min(beta + increment, 1.0)


def _systematic_resample(weights):
    """Indices of a systematic resample of the normalised ``weights``."""
    n = weights.numel()
    positions = (torch.rand((), dtype=weights.dtype, device=weights.device) + torch.arange(n, dtype=weights.dtype, device=weights.device)) / n
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions).clamp(max=n - 1)


class _Mutation:
    """MCMC kernels that move the particles at a fixed inverse temperature."""

    def __init__(self, target, kernel, n_steps):
        self.target = target
        self.kernel = kernel
        self.n_steps = n_steps
        self.beta = 0.0
        self.dim_c = len(target.continuous_index)
        self.scale = 2.38 / math.sqrt(max(self.dim_c, 1))
        self._nuts = None
        self._slice = None
        self._discrete = None
        space = target.discrete_state_space()
        if space is not None:
            sampler_class = FiniteGibbs if has_finite_support(space) else Metropolis
            self._discrete = sampler_class(self._discrete_log_target, space)
        self._full = None

    def tempered(self, prior, likelihood):
        if prior == -math.inf or likelihood == -math.inf:
            return -math.inf
        return prior + self.beta * likelihood

    def _log_target_at(self, flat):
        return self.tempered(*self.target.log_prior_and_likelihood(flat))

    def _with_continuous(self, theta):
        flat = self._full.clone()
        flat[self.target.continuous_index] = theta.reshape(-1)
        return flat

    def _continuous_log_target(self, theta):
        return torch.tensor(self._log_target_at(self._with_continuous(theta)), dtype=theta.dtype, device=theta.device)

    def _continuous_gradient(self, theta):
        return self.target.tempered_gradient(self._with_continuous(theta), self.beta)

    def _discrete_log_target(self, theta):
        flat = self._full.clone()
        flat[self.target.discrete_index] = theta.reshape(-1).to(flat.dtype)
        return torch.tensor(self._log_target_at(flat), dtype=flat.dtype, device=flat.device)

    def mutate(self, particles, log_prior, log_likelihood, beta):
        """Move every particle ``n_steps`` times; returns the new particles, terms and per-particle acceptance."""
        self.beta = beta
        n = particles.shape[0]
        acceptance = torch.zeros(n, dtype=particles.dtype, device=particles.device)
        particles = particles.clone()
        log_prior = log_prior.clone()
        log_likelihood = log_likelihood.clone()
        if self.dim_c > 0:
            continuous = particles[:, self.target.continuous_index]
            self._prepare_continuous(continuous)
        for _ in range(self.n_steps):
            if self.dim_c > 0:
                if self.kernel == "metropolis":
                    accepted = self._metropolis_step(particles, log_prior, log_likelihood)
                else:
                    accepted = self._gradient_or_slice_step(particles, log_prior, log_likelihood)
                acceptance += accepted
            if self._discrete is not None:
                self._discrete_step(particles, log_prior, log_likelihood)
        if self.dim_c == 0:
            acceptance.fill_(float(self.n_steps))
        return particles, log_prior, log_likelihood, acceptance / max(self.n_steps, 1)

    def _prepare_continuous(self, continuous):
        n = continuous.shape[0]
        centred = continuous - continuous.mean(dim=0, keepdim=True)
        covariance = centred.T @ centred / max(n - 1, 1)
        variance = covariance.diagonal().clamp(min=1e-10)
        identity = torch.eye(self.dim_c, dtype=continuous.dtype, device=continuous.device)
        covariance = covariance + 1e-8 * variance.mean() * identity
        cholesky, info = torch.linalg.cholesky_ex(covariance)
        if int(info) != 0:
            # Fall back to the diagonal; the NUTS metric and its factor must agree.
            covariance = torch.diag(variance)
            cholesky = torch.diag(variance.sqrt())
        self._cholesky = cholesky
        if self.kernel == "nuts":
            if self._nuts is None:
                self._nuts = NUTS(self._continuous_log_target, self._continuous_gradient, lambda x: x, delta=0.8, max_depth=8, metric="dense")
            self._nuts.reset()
            self._nuts.inv_metric = covariance
            self._nuts._inv_metric_chol = cholesky
        elif self.kernel == "slice":
            if self._slice is None:
                self._slice = Slice(self._continuous_log_target, ContinuousReal(), adapt_width=False)
            self._slice.reset()
            self._slice.width = 2.0 * variance.sqrt()

    def _metropolis_step(self, particles, log_prior, log_likelihood):
        index = self.target.continuous_index
        noise = torch.randn((particles.shape[0], self.dim_c), dtype=particles.dtype, device=particles.device)
        proposals = particles[:, index] + self.scale * noise @ self._cholesky.T
        log_u = torch.log(torch.rand(particles.shape[0], dtype=particles.dtype, device=particles.device)).tolist()
        accepted = torch.zeros(particles.shape[0], dtype=particles.dtype, device=particles.device)
        for i in range(particles.shape[0]):
            flat = particles[i].clone()
            flat[index] = proposals[i]
            prior, likelihood = self.target.log_prior_and_likelihood(flat)
            new = self.tempered(prior, likelihood)
            old = self.tempered(float(log_prior[i]), float(log_likelihood[i]))
            if new != -math.inf and log_u[i] < new - old:
                particles[i] = flat
                log_prior[i], log_likelihood[i] = prior, likelihood
                accepted[i] = 1.0
        # Adapt the proposal scale between steps towards an acceptance rate of 0.234.
        rate = float(accepted.mean())
        self.scale = min(max(self.scale * math.exp(rate - 0.234), 1e-4), 1e2)
        return accepted

    def _gradient_or_slice_step(self, particles, log_prior, log_likelihood):
        index = self.target.continuous_index
        sampler = self._nuts if self.kernel == "nuts" else self._slice
        accepted = torch.zeros(particles.shape[0], dtype=particles.dtype, device=particles.device)
        for i in range(particles.shape[0]):
            self._full = particles[i].clone()
            sampler.invalidate_cache()
            theta = particles[i, index].clone()
            # NUTS adapts its step size across the particles of a stage; each
            # transition still leaves the tempered target invariant.
            new_theta, _, accept_stat, _ = sampler.step(theta, warmup=self.kernel == "nuts")
            flat = self._with_continuous(new_theta)
            prior, likelihood = self.target.log_prior_and_likelihood(flat)
            if self.tempered(prior, likelihood) != -math.inf:
                particles[i] = flat
                log_prior[i], log_likelihood[i] = prior, likelihood
            accepted[i] = float(accept_stat)
        if self.kernel == "nuts":
            self.scale = float(self._nuts.step_size)
        else:
            self.scale = float(self._slice.width.mean())
        return accepted

    def _discrete_step(self, particles, log_prior, log_likelihood):
        index = self.target.discrete_index
        for i in range(particles.shape[0]):
            self._full = particles[i].clone()
            self._discrete.invalidate_cache()
            theta = particles[i, index].clone()
            new_theta, *_ = self._discrete.step(theta)
            flat = self._full.clone()
            flat[index] = new_theta.reshape(-1).to(flat.dtype)
            prior, likelihood = self.target.log_prior_and_likelihood(flat)
            particles[i] = flat
            log_prior[i], log_likelihood[i] = prior, likelihood


def sample_smc(model=None, n_particles=1000, kernel="metropolis", n_mutation_steps=None, target_ess=0.5, seed=None, progress_bar=True, vectorized=True):
    """Sequential Monte Carlo sampler with adaptive tempering.

    The particles are drawn from the prior and moved through the tempered
    posteriors ``p(theta) p(y | theta)^beta`` with ``0 = beta_0 < beta_1 <
    ... < beta_K = 1``. At every stage

    1. the next inverse temperature is the largest ``beta_{k+1} <= 1``
       (found by bisection) for which the effective sample size
       ``(sum w)^2 / sum w^2`` of the incremental weights
       ``w = exp((beta_{k+1} - beta_k) log p(y | theta))`` is at least
       ``target_ess * n_particles``;
    2. the particles are resampled (systematic resampling) and
    3. moved with ``n_mutation_steps`` MCMC steps that leave the new tempered
       posterior invariant.

    ``kernel`` selects the MCMC kernel for the continuous parameters:
    ``"metropolis"`` (random walk with the particle covariance as proposal
    covariance, initially scaled by ``2.38^2 / d``; after every step the
    scale ``2.38 / sqrt(d)`` is multiplied by ``exp(rate - 0.234)``, i.e.
    adapted towards an acceptance rate of 0.234), ``"nuts"`` (tempered log density and gradient
    ``grad log p(theta) + beta grad log p(y | theta)``, with the particle
    covariance as dense metric and a maximal tree depth of 8) or
    ``"slice"`` (coordinate-wise slice sampling with widths of twice the
    particle standard deviations). The NUTS step size is adapted across the
    particles of a stage towards a mean acceptance statistic of 0.8: by dual
    averaging with ``vectorized=False``, and with ``vectorized=True`` as one
    pooled step size multiplied by ``exp(2 (mean accept_stat - 0.8))`` after
    every step. Discrete parameters are updated with the exact finite Gibbs
    sampler when their support is finite and with discrete Metropolis
    otherwise, whatever ``kernel`` is. Enumerated parameters are summed out
    of the likelihood, and posterior draws of them are added to the result.

    The particles explore all modes the prior covers, so SMC can sample
    multimodal posteriors that trap a single MCMC chain. The marginal
    likelihood is estimated as the product over the stages of the mean
    incremental weights, ``log p(y) ~= sum_k log mean_i w_i^(k)``. For a fixed
    temperature schedule this estimate of ``p(y)`` is unbiased; with the
    adaptive schedule it is consistent but slightly biased.

    Parameters
    ----------
    model : Model, optional
        The model; the active model (inside ``with model:``) by default.
    n_particles : int, default 1000
        Number of particles (at least 2).
    kernel : {"metropolis", "nuts", "slice"}, default "metropolis"
        Mutation kernel for the continuous parameters, see above.
    n_mutation_steps : int, optional
        MCMC steps per stage; 10 for ``"metropolis"`` and 2 for ``"nuts"`` and
        ``"slice"`` by default.
    target_ess : float, default 0.5
        Fraction of ``n_particles``, in ``(0, 1)``, that the ESS of the
        incremental weights must keep at each stage; larger values give more,
        smaller temperature steps.
    seed : int, optional
        Makes the run reproducible without changing torch's global random
        state; with ``seed=None`` the global generator is used.
    progress_bar : bool, default True
        Show a progress bar over the inverse temperature.
    vectorized : bool, default True
        Use the particles as the chain dimension of the batched model (see
        :meth:`~BayesianDLL.Model.batch`): the weights of all particles are
        computed with one evaluation and the mutation kernels move all
        particles at once, also on a GPU. Models with nodes that cannot be
        batched are evaluated particle by particle inside the same scheme.
        ``False`` uses the particle-by-particle implementation.

    Returns
    -------
    SamplingResult
        One chain of ``n_particles`` draws (trace tensors of shape
        ``(1, n_particles, *event_shape)``), with ``lp`` and ``accept_stat``
        sample statistics. It also holds ``log_marginal_likelihood`` (the
        estimate of ``log p(y)``), ``beta_schedule`` (the inverse
        temperatures, starting with 0), ``stage_ess`` (the effective sample
        size of every stage's incremental weights), ``stage_acceptance`` (the
        mean acceptance statistic of every stage's mutation) and ``kernel``.

    Raises
    ------
    NotImplementedError
        If the prior of a random parameter depends on an enumerated
        parameter, or a prior cannot be sampled.
    """
    model = get_active_model() if model is None else model
    model._capture_training_data()
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}, got {kernel!r}.")
    if not 0.0 < target_ess < 1.0:
        raise ValueError(f"target_ess must be in (0, 1), got {target_ess}.")
    if int(n_particles) < 2:
        raise ValueError(f"n_particles must be at least 2, got {n_particles}.")
    n_particles = int(n_particles)
    n_mutation_steps = _DEFAULT_MUTATION_STEPS[kernel] if n_mutation_steps is None else int(n_mutation_steps)
    if n_mutation_steps < 1:
        raise ValueError(f"n_mutation_steps must be at least 1, got {n_mutation_steps}.")

    model.prepare_enumeration()
    if not model.params:
        raise ValueError("The model has no random parameters to sample.")

    device = model.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    rng = torch.random.fork_rng(devices=devices) if seed is not None else contextlib.nullcontext()
    saved = {name: p.unconstrained_value for name, p in {**model.params, **model.enumerated_params}.items()}
    try:
        with rng:
            if seed is not None:
                torch.manual_seed(int(seed))
            if vectorized:
                from ._SMC_batched import run_smc_batched
                return run_smc_batched(model, n_particles, kernel, n_mutation_steps, target_ess, progress_bar)
            return _run_smc(model, n_particles, kernel, n_mutation_steps, target_ess, progress_bar)
    finally:
        for name, value in saved.items():
            model._random_term(name).set_unconstrained_value(value)


def _run_smc(model, n_particles, kernel, n_mutation_steps, target_ess, progress_bar):
    target = _TemperedTarget(model)
    reference = next(iter(model.params.values())).unconstrained_value
    dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
    device = reference.device

    particles = _draw_prior(model, target, n_particles, dtype, device)
    log_prior = torch.empty(n_particles, dtype=dtype, device=device)
    log_likelihood = torch.empty(n_particles, dtype=dtype, device=device)
    for i in range(n_particles):
        log_prior[i], log_likelihood[i] = target.log_prior_and_likelihood(particles[i])

    mutation = _Mutation(target, kernel, n_mutation_steps)
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

    return _build_result(model, target, particles, log_prior + log_likelihood, acceptance, mutation, kernel, log_evidence, betas, stage_ess, stage_acceptance)


def _build_result(model, target, particles, log_posterior, acceptance, mutation, kernel, log_evidence, betas, stage_ess, stage_acceptance):
    n = particles.shape[0]
    trace = {
        name: torch.empty((1, n, *p.constrained_value.shape), dtype=p.constrained_value.dtype, device=particles.device)
        for name, p in model.params.items()
    }
    deterministic_trace = {
        name: torch.empty((1, n, *p.constrained_value.shape), dtype=p.constrained_value.dtype, device=particles.device)
        for name, p in model.deterministic_params.items()
    }
    enumerated, enumerated_probs = {}, {}
    for i in range(n):
        target.set(particles[i])
        for name, (value, probabilities) in model.sample_enumerated().items():
            if name not in enumerated:
                enumerated[name] = torch.empty((1, n, *value.shape), dtype=value.dtype, device=value.device)
                enumerated_probs[name] = torch.empty((1, n, *probabilities.shape), dtype=probabilities.dtype, device=probabilities.device)
            enumerated[name][0, i] = value
            enumerated_probs[name][0, i] = probabilities
        for name, parameter in model.params.items():
            trace[name][0, i] = parameter.constrained_value
        for name, parameter in model.deterministic_params.items():
            deterministic_trace[name][0, i] = parameter.constrained_value
    trace.update(enumerated)

    stats = SampleStats([{
        "params": list(target.names),
        "sampler": "smc",
        "stats": {
            "lp": log_posterior.detach().to(torch.float64).reshape(1, n),
            "accept_stat": acceptance.detach().to(torch.float64).reshape(1, n),
        },
    }])
    return SamplingResult(
        trace, [0], [[stage_acceptance[-1] if stage_acceptance else 1.0]], [[mutation.scale]],
        deterministic_trace=deterministic_trace, sample_stats=stats, enumerated_probs=enumerated_probs,
        log_marginal_likelihood=log_evidence, beta_schedule=betas, stage_ess=stage_ess,
        stage_acceptance=stage_acceptance, kernel=kernel,
    )
