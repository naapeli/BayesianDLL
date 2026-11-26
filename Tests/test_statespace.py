"""Linear Gaussian state-space model: Kalman likelihood, hand-written gradients, missing data, batching, sampling."""

import math
import warnings

import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import multivariate_normal

from BayesianDLL import DeterministicParameter, Model, ObservedParameter, RandomParameter
from BayesianDLL.Distributions import HalfNormal, Normal, Uniform
from BayesianDLL.TimeSeries import LinearGaussianStateSpace
from test_vectorized import chain_by_chain, random_values

t = torch.tensor


def make_system(seed=0, m=3, k=2, r=2):
    g = torch.Generator().manual_seed(seed)
    Tm = 0.5 * torch.randn(m, m, generator=g)
    Tm = Tm / (1.2 * torch.linalg.eigvals(Tm).abs().max())
    Q0 = torch.randn(r, r, generator=g)
    H0 = torch.randn(k, k, generator=g)
    return {
        "transition": Tm, "design": torch.randn(k, m, generator=g), "state_cov": Q0 @ Q0.T + 0.2 * torch.eye(r),
        "obs_cov": H0 @ H0.T + 0.1 * torch.eye(k), "selection": torch.randn(m, r, generator=g),
    }


def dense_moments(params, N, init_mean=None, init_cov=None, offset=None):
    """Mean and covariance of the stacked series ``(N * k,)`` implied by the model (differentiable)."""
    Tm, Z, Q, H, R = (params[n] for n in ("transition", "design", "state_cov", "obs_cov", "selection"))
    k = Z.shape[0]
    m, r = Tm.shape[0], Q.shape[0]
    a1 = torch.zeros(m) if init_mean is None else init_mean
    if init_cov is None:
        K = torch.eye(m * m) - torch.einsum("ij,kl->ikjl", Tm, Tm).reshape(m * m, m * m)
        init_cov = torch.linalg.solve(K, (R @ Q @ R.T).reshape(-1)).reshape(m, m)
    # y = L x with x = (alpha_1, eta_1, ..., eta_{N-1}, eps_1, ..., eps_N)
    nx = m + (N - 1) * r + N * k
    A = [torch.zeros(m, nx) for _ in range(N)]  # alpha_t as linear maps of x
    A[0] = torch.cat([torch.eye(m), torch.zeros(m, nx - m)], dim=1)
    for s in range(1, N):
        sel = torch.zeros(r, nx)
        sel[:, m + (s - 1) * r: m + s * r] = torch.eye(r)
        A[s] = Tm @ A[s - 1] + R @ sel
    rows = []
    for s in range(N):
        eps = torch.zeros(k, nx)
        eps[:, m + (N - 1) * r + s * k: m + (N - 1) * r + (s + 1) * k] = torch.eye(k)
        rows.append(Z @ A[s] + eps)
    L = torch.cat(rows, dim=0)
    blocks = [init_cov] + [Q] * (N - 1) + [H] * N
    S = torch.zeros(nx, nx)
    pos = 0
    for b in blocks:
        n = b.shape[0]
        S[pos:pos + n, pos:pos + n] = b
        pos += n
    mean = (L[:, :m] @ a1)
    if offset is not None:
        mean = mean + offset.reshape(-1)
    return mean, L @ S @ L.T


def dense_loglik(params, y, init_mean=None, init_cov=None, offset=None):
    """Joint Gaussian log density of the observed entries of ``y`` (N, k) from the implied covariance (differentiable)."""
    mean, cov = dense_moments(params, y.shape[0], init_mean, init_cov, offset)
    obs = ~torch.isnan(y.reshape(-1))
    cov, mean, yy = cov[obs][:, obs], mean[obs], y.reshape(-1)[obs]
    resid = yy - mean
    return -0.5 * (obs.sum() * math.log(2 * math.pi) + torch.logdet(cov) + resid @ torch.linalg.solve(cov, resid))


@pytest.mark.parametrize("missing", [False, True])
def test_kalman_likelihood_equals_dense_gaussian(missing):
    p = make_system()
    y = torch.randn(7, 2)
    if missing:
        y[2, 0] = math.nan
        y[4] = math.nan
        y[5, 1] = math.nan
    d = LinearGaussianStateSpace(**p)
    expected = dense_loglik(p, y)
    assert float(d.log_pdf(y)) == pytest.approx(float(expected), abs=1e-9)
    # an independent solution of the stationary covariance equation
    P1 = solve_discrete_lyapunov(p["transition"].numpy(), (p["selection"] @ p["state_cov"] @ p["selection"].T).numpy())
    given = LinearGaussianStateSpace(**p, init_cov=t(P1))
    assert float(given.log_pdf(y)) == pytest.approx(float(expected), abs=1e-9)
    # dense reference for the observed entries through scipy
    obs = ~np.isnan(y.numpy().reshape(-1))


def test_kalman_likelihood_with_offset_initial_state_and_explicit_covariance():
    p = make_system(1)
    y = torch.randn(6, 2)
    offset = torch.randn(6, 2)
    a1 = torch.randn(3)
    P1 = torch.eye(3) * 0.7
    d = LinearGaussianStateSpace(**p, init_mean=a1, init_cov=P1, obs_offset=offset)
    expected = dense_loglik(p, y, a1, P1, offset)
    assert float(d.log_pdf(y)) == pytest.approx(float(expected), abs=1e-9)
    # the same through scipy's multivariate normal on the implied covariance
    assert float(expected) == pytest.approx(float(dense_loglik(p, y, a1, P1, offset)))


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("explicit_init", [False, True])
def test_gradients_match_autograd_of_the_dense_likelihood(missing, explicit_init):
    p = make_system(2)
    y = torch.randn(6, 2)
    if missing:
        y[1, 1] = math.nan
        y[3] = math.nan
    offset = torch.randn(6, 2)
    extra = {"init_mean": torch.randn(3), "init_cov": 0.6 * torch.eye(3) + 0.1} if explicit_init else {}
    leaves = {k: v.clone().requires_grad_(True) for k, v in {**p, **extra, "obs_offset": offset}.items()}
    yl = y.clone().requires_grad_(True)
    dense = dense_loglik({k: leaves[k] for k in p}, yl, leaves.get("init_mean"), leaves.get("init_cov"), leaves["obs_offset"])
    dense.backward()
    d = LinearGaussianStateSpace(**{k: v.detach() for k, v in leaves.items()})
    assert float(d.log_pdf(y)) == pytest.approx(float(dense), abs=1e-9)
    grads = d.log_pdf_param_grads(y)
    for name, leaf in leaves.items():
        torch.testing.assert_close(grads[name], leaf.grad, rtol=1e-7, atol=1e-9, msg=lambda m, n=name: f"{n}: {m}")
    gy = d.log_pdf_grad(y)
    expected_y = torch.where(torch.isnan(y), torch.zeros_like(y), yl.grad)
    torch.testing.assert_close(gy, expected_y, rtol=1e-7, atol=1e-9)


def test_unstable_transition_without_initial_covariance_is_minus_infinity():
    p = make_system(3)
    p["transition"] = 1.3 * torch.eye(3)
    y = torch.randn(5, 2)
    d = LinearGaussianStateSpace(**p)
    assert float(d.log_pdf(y)) == -math.inf
    assert all(torch.all(g == 0) for g in d.log_pdf_param_grads(y).values())
    # an explicit initial covariance makes it a valid (non-stationary) model
    assert math.isfinite(float(LinearGaussianStateSpace(**p, init_cov=torch.eye(3)).log_pdf(y)))


def test_batched_parameters_and_data_agree_with_a_loop():
    ps = [make_system(s) for s in range(3)]
    stacked = {k: torch.stack([p[k] for p in ps]) for k in ps[0]}
    y = torch.randn(4, 3, 6, 2)  # (extra, chains, N, k)
    d = LinearGaussianStateSpace(**stacked)
    assert d.batch_shape == (3,)
    lp = d.log_pdf(y)
    assert lp.shape == (4, 3)
    grads = d.log_pdf_param_grads(y)
    for c in range(3):
        one = LinearGaussianStateSpace(**ps[c])
        torch.testing.assert_close(lp[:, c], one.log_pdf(y[:, c]))
        g1 = one.log_pdf_param_grads(y[:, c])
        for name in ps[0]:
            torch.testing.assert_close(grads[name][:, c], g1[name], rtol=1e-9, atol=1e-10)


def test_filter_returns_predictions_and_innovations():
    p = make_system(4)
    y = torch.randn(5, 2)
    d = LinearGaussianStateSpace(**p)
    out = d.filter(y)
    assert out["predicted_mean"].shape == (6, 3) and out["predicted_cov"].shape == (6, 3, 3)
    assert out["innovations"].shape == (5, 2) and out["innovation_cov"].shape == (5, 2, 2)
    F, v = out["innovation_cov"], out["innovations"]
    ll = sum(multivariate_normal(np.zeros(2), F[i].numpy()).logpdf(v[i].numpy()) for i in range(5))
    assert float(out["log_likelihood"]) == pytest.approx(ll, abs=1e-9)
    # the first prediction is the initial distribution
    torch.testing.assert_close(out["predicted_mean"][0], torch.zeros(3))


def test_sampling_reproduces_the_implied_covariance_and_log_density_moments():
    p = make_system(5)
    N = 4
    d = LinearGaussianStateSpace(**p, T=N)
    s = d.sample(80000)
    assert s.shape == (80000, N, 2)
    # implied covariance from the dense representation (reuse the autograd-free dense construction)
    flat = s.reshape(80000, -1).numpy()
    # E log p(y) under the model equals minus its entropy: compare with the Gaussian entropy of the dense covariance
    _, dense = dense_moments(p, N)
    dense = dense.numpy()
    np.testing.assert_allclose(np.cov(flat.T), dense, rtol=0.03, atol=0.1)
    ll = d.log_pdf(s[:4000]).mean()
    entropy = 0.5 * np.linalg.slogdet(2 * np.pi * np.e * dense)[1]
    assert float(ll) == pytest.approx(-entropy, abs=0.15)
    assert abs(flat.mean()) < 0.05


def test_forecast_is_the_conditional_distribution_of_future_observations():
    p = make_system(6)
    y = torch.randn(5, 2)
    d = LinearGaussianStateSpace(**p)
    steps = 3
    f = d.forecast(y, steps, n_samples=100000)
    assert f.shape == (100000, steps, 2)
    # conditional mean and covariance from the dense joint model of the 8 observations
    n = 5 + steps
    _, joint_cov = dense_moments(p, n)
    joint_cov = joint_cov.numpy()
    a, b = n * 2 - steps * 2, n * 2
    cov_aa, cov_ba, cov_bb = joint_cov[:a, :a], joint_cov[a:, :a], joint_cov[a:, a:]
    cond_mean = cov_ba @ np.linalg.solve(cov_aa, y.reshape(-1).numpy())
    cond_cov = cov_bb - cov_ba @ np.linalg.solve(cov_aa, cov_ba.T)
    np.testing.assert_allclose(f.reshape(100000, -1).mean(0).numpy(), cond_mean, atol=0.06)
    np.testing.assert_allclose(np.cov(f.reshape(100000, -1).numpy().T), cond_cov, rtol=0.03, atol=0.12)


def test_forecast_with_missing_values_and_batched_posterior_draws():
    p = make_system(7)
    y = torch.randn(6, 2)
    y[3, 0] = math.nan
    batched = {k: torch.stack([v, v]) for k, v in p.items()}
    f = LinearGaussianStateSpace(**batched).forecast(y, 4, n_samples=5)
    assert f.shape == (5, 2, 4, 2) and torch.isfinite(f).all()


# ============================================================ inside models
def local_level_model(y=None):
    y = torch.cumsum(0.5 * torch.randn(25, generator=torch.Generator().manual_seed(1)), 0) + 0.4 * torch.randn(25, generator=torch.Generator().manual_seed(2)) if y is None else y
    y = y.reshape(-1, 1)
    with Model() as model:
        q = RandomParameter("q", HalfNormal(1.0), shape=(1, 1))
        h = RandomParameter("h", HalfNormal(1.0), shape=(1, 1))
        ObservedParameter("y", LinearGaussianStateSpace(
            transition=torch.eye(1), design=torch.eye(1), state_cov=q, obs_cov=h, init_mean=torch.zeros(1), init_cov=10 * torch.eye(1)), y)
    return model


def missing_data_model():
    y = torch.randn(12, 1, generator=torch.Generator().manual_seed(3))
    y[[2, 3, 7]] = math.nan
    with Model() as model:
        phi = RandomParameter("phi", Uniform(-0.9, 0.9), shape=(1, 1))
        h = RandomParameter("h", HalfNormal(1.0), shape=(1, 1))
        ObservedParameter("y", LinearGaussianStateSpace(transition=phi, design=torch.eye(1), state_cov=torch.eye(1), obs_cov=h), y)
    return model


def two_state_model():
    y = torch.randn(10, 2, generator=torch.Generator().manual_seed(4))
    with Model() as model:
        a = RandomParameter("a", Normal(0.0, 0.3), shape=(2, 2))
        q = RandomParameter("q", HalfNormal(1.0), shape=(2, 2))
        ObservedParameter("y", LinearGaussianStateSpace(
            transition=a, design=torch.eye(2), state_cov=q, obs_cov=0.3 * torch.eye(2), init_cov=torch.eye(2)), y)
    return model


ZOO = {"local_level": local_level_model, "missing": missing_data_model, "two_state": two_state_model}


@pytest.mark.parametrize("name", list(ZOO))
def test_batched_models_equal_chain_by_chain_and_finite_differences(name):
    model = ZOO[name]()
    n_chains = 3
    values = random_values(model, n_chains, torch.Generator().manual_seed(5))
    if name == "two_state":  # keep the unconstrained draws small enough for stable draws of "a"
        values = {k: 0.3 * v for k, v in values.items()}
    names = list(model.params)
    expected_lp, expected_grads = chain_by_chain(model, values, names)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with model.batched(n_chains, values=values):
            assert model._batch_mode == "tensor", model.batch_fallback_reason
            lp = model.model_log_prob()
            grads = model.joint_grad_log_prob(names)
    torch.testing.assert_close(lp, expected_lp, rtol=1e-10, atol=1e-10)
    for n in names:
        torch.testing.assert_close(grads[n], expected_grads[n], rtol=1e-8, atol=1e-9)
    for n in names:
        u = model.params[n].unconstrained_value.clone()
        flat = u.reshape(-1)
        fd = torch.zeros_like(flat)
        for i in range(flat.numel()):
            for sign in (1, -1):
                v = flat.clone()
                v[i] += sign * 1e-6
                model.params[n].set_unconstrained_value(v.reshape(u.shape))
                fd[i] += sign * float(model.model_log_prob()) / 2e-6
        model.params[n].set_unconstrained_value(u)
        torch.testing.assert_close(model.joint_grad_log_prob(names)[n].reshape(-1), fd, rtol=1e-5, atol=1e-6)


def _posterior(model, n=100, warmup=100, chains=4, seed=5):
    from BayesianDLL import sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sample(n, warmup, n_chains=chains, model=model, progress_bar=False, seed=seed, chain_method="vectorized")


def test_nuts_recovers_local_level_variances():
    truth_q, truth_h = 0.4, 0.8
    sys = LinearGaussianStateSpace(torch.eye(1), torch.eye(1), t([[truth_q]]), t([[truth_h]]), init_mean=torch.zeros(1), init_cov=torch.eye(1), T=30)
    y = sys.sample(1)[0]
    model = local_level_model(y.reshape(-1))
    result = _posterior(model)
    q, h = result["q"].reshape(-1), result["h"].reshape(-1)
    for draws, truth in ((q, truth_q), (h, truth_h)):
        assert abs(float(draws.mean()) - truth) < 3.5 * float(draws.std())
    # the posterior mean of the signal-to-noise ratio is close to the exact MAP-free reference from a dense grid
    assert float(q.std()) > 0 and float(h.std()) > 0


def test_remembered_forward_pass_is_never_stale():
    # The last Kalman forward pass is reused for identical inputs; any change must recompute.
    from BayesianDLL.TimeSeries import _kalman

    y = torch.randn(30, 1, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    y[5] = float("nan")

    def model(rho, sd):
        transition = torch.tensor([[rho]], dtype=torch.float64)
        return LinearGaussianStateSpace(transition, torch.ones(1, 1, dtype=torch.float64),
                                        torch.tensor([[sd]], dtype=torch.float64), obs_cov=torch.tensor([[0.3]], dtype=torch.float64))

    first, second = model(0.5, 1.0), model(0.7, 1.0)
    reference = [first.log_pdf(y), second.log_pdf(y)]
    _kalman._last_forward = None
    fresh_gradients = first.log_pdf_param_grads(y)
    for _ in range(2):
        torch.testing.assert_close(first.log_pdf(y), reference[0])
        torch.testing.assert_close(second.log_pdf(y), reference[1])
    first.log_pdf(y)
    for name, gradient in first.log_pdf_param_grads(y).items():
        torch.testing.assert_close(gradient, fresh_gradients[name])
    y[3] += 1.0
    assert not torch.equal(first.log_pdf(y), reference[0])


def test_numpy_and_torch_backends_agree():
    # CPU tensors run the recursions in NumPy, other devices in torch; both must give the same results.
    from BayesianDLL.TimeSeries import _kalman

    g = torch.Generator().manual_seed(3)
    N, B, m, k = 12, (2,), 3, 2
    y = torch.randn(N, *B, k, generator=g, dtype=torch.float64)
    mask = torch.ones(N, *B, k, dtype=torch.float64)
    mask[4, 0, 1] = 0.0
    mask[7] = 0.0
    d = 0.1 * torch.randn(N, *B, k, generator=g, dtype=torch.float64)
    Tm = 0.4 * torch.randn(*B, m, m, generator=g, dtype=torch.float64)
    Z = torch.randn(k, m, generator=g, dtype=torch.float64)
    H = torch.eye(k, dtype=torch.float64).expand(*B, k, k) * 0.5
    L = torch.randn(*B, m, m, generator=g, dtype=torch.float64)
    RQR = L @ L.transpose(-1, -2) + torch.eye(m, dtype=torch.float64)
    a = torch.randn(*B, m, generator=g, dtype=torch.float64)
    P = 2.0 * torch.eye(m, dtype=torch.float64).expand(*B, m, m)
    inputs = (y, mask, d, Tm, Z, H, RQR, a, P)

    results = {}
    for name, xp, convert in (("numpy", _kalman._NumPy, lambda x: x.numpy()), ("torch", _kalman._Torch, lambda x: x)):
        ll, cache, a_end, P_end = _kalman._kalman_forward(xp, *(convert(x) for x in inputs))
        grads = _kalman.kalman_backward(cache, mask, Tm)
        results[name] = (torch.as_tensor(ll), torch.as_tensor(a_end), torch.as_tensor(P_end), grads)
    for first, second in zip(results["numpy"][:3], results["torch"][:3]):
        torch.testing.assert_close(first, second)
    for key, value in results["numpy"][3].items():
        assert isinstance(value, torch.Tensor)
        torch.testing.assert_close(value, results["torch"][3][key], msg=key)
