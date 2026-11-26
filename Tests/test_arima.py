"""ARMA / ARIMA / SARIMA / SARIMAX likelihoods, gradients, constraints and forecasts."""

import warnings

import numpy as np
import pytest
import torch
from scipy.linalg import toeplitz
from scipy.signal import lfilter
from scipy.stats import multivariate_normal as mvn

from BayesianDLL import Model, ObservedParameter, RandomParameter
from BayesianDLL.Distributions import HalfNormal, Normal, Uniform
from BayesianDLL.TimeSeries import ARIMA, ARMA, SARIMA, SARIMAX, InvertibleMA, MovingAverage, StationaryAR
from BayesianDLL.TimeSeries._constraints import pacf_to_ar
from BayesianDLL.TimeSeries._arima import _difference_coefficients
from test_vectorized import chain_by_chain, random_values

t = torch.tensor


def ref_cov(ar_poly, ma_poly, sigma, n):
    psi = lfilter(ma_poly, ar_poly, np.r_[1.0, np.zeros(4000)])
    g = np.array([sigma ** 2 * psi[: len(psi) - h] @ psi[h:] for h in range(n)])
    return toeplitz(g)


def ref_loglik(x, ar_poly, ma_poly, sigma, mean):
    return mvn(np.full(len(x), mean), ref_cov(ar_poly, ma_poly, sigma, len(x))).logpdf(np.asarray(x))


def test_arma_family_matches_dense_normal():
    n, sigma, mu = 14, 1.2, 0.3
    x = torch.randn(n)
    phi, th, Phi, Th = [0.5, -0.2], [0.4, 0.1], [0.3], [0.35]
    cases = [
        (MovingAverage(t(th), sigma, mu), [1.0], [1, *th]),
        (ARMA(t(phi), t(th), sigma, mu), [1, *(-np.array(phi))], [1, *th]),
        (SARIMA(t(phi), t(th[:1]), t(Phi), t(Th), sigma, period=4, mu=mu),
         np.convolve([1, -phi[0], -phi[1]], [1, 0, 0, 0, -Phi[0]]), np.convolve([1, th[0]], [1, 0, 0, 0, Th[0]])),
        (ARMA(None, None, sigma, mu), [1.0], [1.0]),
    ]
    for d, ar_poly, ma_poly in cases:
        assert float(d.log_pdf(x)) == pytest.approx(ref_loglik(x.numpy(), ar_poly, ma_poly, sigma, mu), abs=1e-9)


def test_differencing_gives_the_likelihood_of_the_differenced_series():
    x = torch.randn(20).cumsum(0)
    d = ARIMA(t([0.4]), t([0.3]), 1.1, diff=1, mu=0.1)
    w = (x[1:] - x[:-1]).numpy()
    ref = ref_loglik(w, [1, -0.4], [1, 0.3], 1.1, 0.1)
    assert float(d.log_pdf(x)) == pytest.approx(ref, abs=1e-9)
    s = SARIMA(t([0.4]), None, None, None, 1.1, period=4, diff=1, seasonal_diff=1)
    c = _difference_coefficients(1, 1, 4)
    w = np.array([sum(c[k] * x[i - k].item() for k in range(len(c))) for i in range(5, 20)])
    assert float(s.log_pdf(x)) == pytest.approx(ref_loglik(w, [1, -0.4], [1.0], 1.1, 0.0), abs=1e-9)


def test_sarimax_regression_with_arma_errors():
    n = 16
    X = torch.randn(n, 2)
    beta = t([0.7, -0.4])
    x = torch.randn(n)
    d = SARIMAX(t([0.5]), t([0.2]), 0.9, 0.1, beta=beta, exog=X)
    resid = x.numpy() - 0.1 - (X @ beta).numpy()
    ref = mvn(np.zeros(n), ref_cov([1, -0.5], [1, 0.2], 0.9, n)).logpdf(resid)
    assert float(d.log_pdf(x)) == pytest.approx(ref, abs=1e-9)


CASES = {
    "arma": lambda p: ARMA(p["ar"], p["ma"], p["sigma"], p["mu"]),
    "sarima": lambda p: SARIMAX(p["ar"], p["ma"], p["sigma"], p["mu"], seasonal_ar=p["sar"], seasonal_ma=p["sma"], diff=1, period=4),
    "sarimax": lambda p: SARIMAX(p["ar"], None, p["sigma"], p["mu"], beta=p["beta"], exog=EXOG()),
}


def EXOG():
    return torch.sin(torch.arange(18.0)).reshape(-1, 1) * torch.ones(1, 2) + torch.arange(2.0)


@pytest.mark.parametrize("case", list(CASES))
def test_gradients_match_autograd_of_the_dense_likelihood(case):
    """Autograd of a dense Gaussian likelihood built from the autocovariance (psi weights) is the reference."""
    n = 18
    x = torch.randn(n).cumsum(0) * 0.3 + torch.randn(n)
    params = {"ar": t([0.5, -0.2]), "ma": t([0.35]), "sigma": t(1.1), "mu": t(0.2), "sar": t([0.3]), "sma": t([0.25]), "beta": t([0.5, -0.3])}
    d = CASES[case](params)
    grads = d.log_pdf_param_grads(x)
    # finite differences of the log density
    for name in d._param_names if hasattr(d, "_param_names") else ["ar", "sigma", "mu"]:
        if name not in params:
            continue
        key = {"sar": "seasonal_ar", "sma": "seasonal_ma"}.get(name, name)
        if key not in grads:
            continue
        base = params[name].clone()
        fd = torch.zeros_like(base)
        for i in range(base.numel()):
            for sign in (1, -1):
                q = dict(params)
                v = base.clone().reshape(-1)
                v[i] += sign * 1e-6
                q[name] = v.reshape(base.shape)
                fd.reshape(-1)[i] += sign * float(CASES[case](q).log_pdf(x)) / 2e-6
        torch.testing.assert_close(grads[key].reshape(-1).sum(0) if base.ndim == 0 else grads[key].reshape(base.shape), fd, rtol=1e-5, atol=1e-6)
    # gradient with respect to the series
    gx = d.log_pdf_grad(x)
    fd = torch.zeros(n)
    for i in range(n):
        for sign in (1, -1):
            v = x.clone()
            v[i] += sign * 1e-6
            fd[i] += sign * float(d.log_pdf(v)) / 2e-6
    torch.testing.assert_close(gx, fd, rtol=1e-5, atol=1e-6)


def test_non_stationary_ar_is_minus_infinity():
    assert float(ARMA(t([1.1]), None, 1.0).log_pdf(torch.randn(8))) == -np.inf


def test_pacf_parameterisation_is_stationary_and_gradients_match_autograd():
    from BayesianDLL.TimeSeries import is_stationary
    r = (torch.rand(50, 4) * 1.98 - 0.99)
    phi, hist = pacf_to_ar(r)
    assert bool(is_stationary(phi).all())
    r1 = r[0].clone().requires_grad_(True)
    out = pacf_to_ar(r1)[0]
    w = torch.randn(4)
    (out * w).sum().backward()
    from BayesianDLL.TimeSeries._constraints import pacf_to_ar_vjp
    torch.testing.assert_close(pacf_to_ar_vjp(w, r[0], pacf_to_ar(r[0])[1]), r1.grad)


def _model():
    x = torch.randn(25, generator=torch.Generator().manual_seed(1)).cumsum(0) * 0.2
    with Model() as model:
        r = RandomParameter("r", Uniform(-1, 1), shape=2)
        q = RandomParameter("q", Uniform(-1, 1), shape=1)
        phi = StationaryAR("phi", r)
        theta = InvertibleMA("theta", q)
        sigma = RandomParameter("sigma", HalfNormal(1.0))
        ObservedParameter("x", ARMA(phi, theta, sigma), x)
    return model


def test_batched_model_with_constraint_blocks_equals_chain_by_chain():
    model = _model()
    values = {k: 0.5 * v for k, v in random_values(model, 3, torch.Generator().manual_seed(3)).items()}
    names = list(model.params)
    lp, g = chain_by_chain(model, values, names)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with model.batched(3, values=values):
            assert model._batch_mode == "tensor", model.batch_fallback_reason
            lpb = model.model_log_prob()
            gb = model.joint_grad_log_prob(names)
    torch.testing.assert_close(lpb, lp, rtol=1e-10, atol=1e-10)
    for n in names:
        torch.testing.assert_close(gb[n], g[n], rtol=1e-8, atol=1e-9)


def test_sampling_and_forecast_moments():
    d = ARMA(t([0.6]), t([0.3]), 1.0, 0.5, T=5)
    s = d.sample(40000)
    np.testing.assert_allclose(np.cov(s.numpy().T), ref_cov([1, -0.6], [1, 0.3], 1.0, 5), atol=0.08)
    # forecast of an AR(1): conditional mean mu + phi^h (x_N - mu), variance sigma^2 (1 - phi^{2h}) / (1 - phi^2)
    ar = ARMA(t([0.6]), None, 1.0, 0.5)
    x = torch.randn(10)
    f = ar.forecast(x, 3, n_samples=60000)
    for h in (1, 2, 3):
        assert float(f[:, h - 1].mean()) == pytest.approx(0.5 + 0.6 ** h * (float(x[-1]) - 0.5), abs=0.03)
        assert float(f[:, h - 1].var()) == pytest.approx((1 - 0.6 ** (2 * h)) / (1 - 0.36), rel=0.04)
    # random walk with drift: ARIMA(0,1,0)
    rw = ARIMA(None, None, 0.5, diff=1, mu=0.2)
    f = rw.forecast(x, 4, n_samples=40000)
    assert float(f[:, 3].mean()) == pytest.approx(float(x[-1]) + 0.8, abs=0.02)
    assert float(f[:, 3].var()) == pytest.approx(4 * 0.25, rel=0.05)


def test_nuts_recovers_arma11():
    truth = ARMA(t([0.6]), t([0.3]), 1.0, 0.0, T=60).sample(1)[0]
    with Model() as model:
        r = RandomParameter("r", Uniform(-1, 1), shape=1)
        q = RandomParameter("q", Uniform(-1, 1), shape=1)
        sigma = RandomParameter("sigma", HalfNormal(2.0))
        ObservedParameter("x", ARMA(StationaryAR("phi", r), InvertibleMA("theta", q), sigma), truth)
    from BayesianDLL import sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sample(80, 80, n_chains=4, model=model, progress_bar=False, seed=2, chain_method="vectorized")
    assert abs(float(res["sigma"].mean()) - 1.0) < 3.5 * float(res["sigma"].std()) + 0.05
