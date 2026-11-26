"""SARIMAX likelihood on the state-space model (Harvey's representation)."""
import math
import torch

from ..Distributions._distributions import Distribution, _as_float
from ..Distributions._resolve import resolve
from ..Distributions._transforms import IdentityTransform
from ..Distributions._state_space import ContinuousReal
from ._statespace import StateSpace


# ------------------------------------------------------------ lag polynomials
def _ar_polynomial(phi, Phi, s):
    """Coefficients ``c_1..c_{p+sP}`` of ``1 - sum c_k L^k = (1 - sum phi_i L^i)(1 - sum Phi_j L^{sj})``."""
    p, P = phi.shape[-1], Phi.shape[-1]
    batch = torch.broadcast_shapes(phi.shape[:-1], Phi.shape[:-1])
    c = torch.zeros(*batch, p + s * P, dtype=phi.dtype, device=phi.device)
    c[..., :p] += phi
    for j in range(P):
        c[..., s * (j + 1) - 1] += Phi[..., j]
        c[..., s * (j + 1): s * (j + 1) + p] -= Phi[..., j:j + 1] * phi
    return c


def _ar_polynomial_vjp(g, phi, Phi, s):
    p, P = phi.shape[-1], Phi.shape[-1]
    g_phi = g[..., :p].clone()
    g_Phi = []
    for j in range(P):
        block = g[..., s * (j + 1): s * (j + 1) + p]
        g_phi = g_phi - Phi[..., j:j + 1] * block
        g_Phi.append(g[..., s * (j + 1) - 1] - (phi * block).sum(-1))
    return g_phi, (torch.stack(g_Phi, dim=-1) if P else Phi.new_zeros(*g.shape[:-1], 0))


def _ma_polynomial(theta, Theta, s):
    """Coefficients ``e_1..e_{q+sQ}`` of ``1 + sum e_k L^k = (1 + sum theta_i L^i)(1 + sum Theta_j L^{sj})``."""
    q, Q = theta.shape[-1], Theta.shape[-1]
    batch = torch.broadcast_shapes(theta.shape[:-1], Theta.shape[:-1])
    e = torch.zeros(*batch, q + s * Q, dtype=theta.dtype, device=theta.device)
    e[..., :q] += theta
    for j in range(Q):
        e[..., s * (j + 1) - 1] += Theta[..., j]
        e[..., s * (j + 1): s * (j + 1) + q] += Theta[..., j:j + 1] * theta
    return e


def _ma_polynomial_vjp(g, theta, Theta, s):
    q, Q = theta.shape[-1], Theta.shape[-1]
    g_theta = g[..., :q].clone()
    g_Theta = []
    for j in range(Q):
        block = g[..., s * (j + 1): s * (j + 1) + q]
        g_theta = g_theta + Theta[..., j:j + 1] * block
        g_Theta.append(g[..., s * (j + 1) - 1] + (theta * block).sum(-1))
    return g_theta, (torch.stack(g_Theta, dim=-1) if Q else Theta.new_zeros(*g.shape[:-1], 0))


def _difference_coefficients(d, D, s):
    """Coefficients ``c_0 = 1, c_1, ...`` of ``(1 - L)^d (1 - L^s)^D``."""
    c = [1.0]
    for _ in range(d):
        c = [a - b for a, b in zip(c + [0.0], [0.0] + c)]
    for _ in range(D):
        shifted = [0.0] * s + c
        c = [a - b for a, b in zip(c + [0.0] * s, shifted)]
    return c


_difference_matrices = {}


def _difference_matrix(N, coefficients, dtype, device):
    """Matrix ``(N - n, N)`` applying the differencing filter."""
    n = len(coefficients) - 1
    key = (N, tuple(coefficients), dtype, device)
    if key not in _difference_matrices:
        D = torch.zeros(N - n, N, dtype=dtype, device=device)
        for k, c in enumerate(coefficients):
            if c != 0:
                D[torch.arange(N - n), torch.arange(N - n) + n - k] = c
        if len(_difference_matrices) > 64:
            _difference_matrices.clear()
        _difference_matrices[key] = D
    return _difference_matrices[key]


def _integrate(w, history, coefficients, keep_history=False):
    """Undo the differencing: ``y_t = w_t - sum_{k>=1} c_k y_{t-k}`` started from the raw ``history`` values (last dim)."""
    n = len(coefficients) - 1
    if n == 0:
        return w
    y = list(history.unbind(-1))
    for t in range(w.shape[-1]):
        nxt = w[..., t]
        for k in range(1, n + 1):
            nxt = nxt - coefficients[k] * y[-k]
        y.append(nxt)
    return torch.stack(y if keep_history else y[n:], dim=-1)


class SARIMAX(Distribution):
    r"""Seasonal ARIMA model with exogenous regressors (SARIMAX) as the distribution of a whole series.

    The series :math:`y_t` is differenced, :math:`w_t = (1 - L)^d (1 - L^s)^D y_t`, and the differenced
    series follows

    .. math::

        w_t = \delta + \tilde x_t^\top \beta + u_t, \qquad
        \phi(L)\,\Phi(L^s)\, u_t = \theta(L)\,\Theta(L^s)\, \sigma \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)

    with :math:`\phi(L) = 1 - \sum_i \phi_i L^i`, :math:`\Phi(L^s) = 1 - \sum_j \Phi_j L^{sj}`,
    :math:`\theta(L) = 1 + \sum_i \theta_i L^i`, :math:`\Theta(L^s) = 1 + \sum_j \Theta_j L^{sj}` (the usual
    Box-Jenkins sign convention for the MA part), and :math:`\tilde x_t` the regressors differenced like the series.

    **Mean and drift.** Without differencing (``difference = seasonal_difference = 0``) the mean of the series as
    written is ``mu`` (:math:`\delta = \mu`). With differencing the series has no stationary mean and the data
    cannot identify a level, so ``mu`` is not allowed; the constant of the *differenced* series is ``drift``
    (:math:`\delta` = ``drift``, the slope of a linear trend when ``difference = 1``), which in turn needs
    differencing. This follows R's ``forecast::Arima`` (``include.mean`` only without differencing,
    ``include.drift`` with it) and statsmodels (the constant of a differenced model is its drift). Simulated
    differenced series start from zero.

    **Special cases.** Leave arguments out to get the classical models (``y`` is the series, ``n_steps`` its length)::

        SARIMAX(ma=theta, sigma=sigma, mu=mu)                          # MA(q)
        SARIMAX(ar=phi, ma=theta, sigma=sigma, mu=mu)                  # ARMA(p, q)
        SARIMAX(ar=phi, ma=theta, sigma=sigma, difference=1, drift=c)  # ARIMA(p, 1, q) with drift
        SARIMAX(ar=phi, ma=theta, sigma=sigma, seasonal_ar=Phi, seasonal_ma=Theta, period=12,
                difference=1, seasonal_difference=1)                   # SARIMA(p,1,q)(P,1,Q)_12
        SARIMAX(ar=phi, sigma=sigma, exog=X, beta=beta)                # regression with AR errors

    **Likelihood.** :math:`u_t` is written as a state-space model in Harvey's form: the state has dimension
    :math:`m = \max(p_{tot}, q_{tot} + 1)` (:math:`p_{tot} = p + sP`, :math:`q_{tot} = q + sQ`), the transition matrix has
    the AR coefficients of the product polynomial :math:`\phi(L)\Phi(L^s)` in its first column and ones on the
    superdiagonal, the selection vector is :math:`(1, e_1, \dots, e_{q_{tot}}, 0, \dots)` with :math:`e` the
    coefficients of the product MA polynomial, and the observation picks the first state. The state starts from the
    stationary distribution, so the likelihood is the exact Gaussian likelihood of :math:`w_{d+sD+1}, \dots, w_N`,
    evaluated with the Kalman filter of :class:`StateSpace` and its hand-written gradients. The
    likelihood of the raw series is the likelihood of the differenced series, i.e. it conditions on the first
    :math:`d + sD` observations (differencing is a unit-determinant linear map, so no Jacobian arises).

    **Constraints.** The AR polynomial must be *stationary* (all roots outside the unit circle); the log density is
    ``-inf`` otherwise. The MA polynomial does not need to be invertible for the likelihood, but is not identified
    without that (the MA roots can be reflected into the unit circle without changing the density); use
    :class:`StationaryAR` / :class:`InvertibleMA` to parameterise both through partial autocorrelations in
    :math:`(-1, 1)`, which enforces the constraints (apply them to the non-seasonal and seasonal parts separately; the
    products of stationary polynomials are stationary). Missing values (``nan``) are allowed; with differencing a
    missing value makes every differenced value that contains it missing, which loses a little information.

    **Sampling.** ``sample`` simulates a series of length ``n_steps`` whose first :math:`d + sD` values equal ``mu`` and whose
    differenced series follows the model (the stationary process started from its stationary distribution).
    :meth:`forecast` simulates the future of an observed series exactly (given the observed history).

    Parameters
    ----------
    ar : torch.Tensor or Parameter, optional
        AR coefficients :math:`\phi_1, \dots, \phi_p`, shape ``(..., p)``.
    ma : torch.Tensor or Parameter, optional
        MA coefficients :math:`\theta_1, \dots, \theta_q`, shape ``(..., q)``.
    sigma : float or torch.Tensor or Parameter
        Innovation standard deviation :math:`\sigma > 0`.
    mu : float or torch.Tensor or Parameter, optional
        Mean :math:`\mu` of the series (default 0); not allowed with differencing (use ``drift``).
    seasonal_ar, seasonal_ma : torch.Tensor or Parameter, optional
        Seasonal coefficients :math:`\Phi_1, \dots, \Phi_P` and :math:`\Theta_1, \dots, \Theta_Q`, shape ``(..., P)`` / ``(..., Q)``.
    beta : torch.Tensor or Parameter, optional
        Regression coefficients of shape ``(..., r)`` (needs ``exog``).
    exog : torch.Tensor or Data, optional
        Regressors of shape ``(N, r)`` (a constant tensor or ``Data``; no gradient is taken with respect to it).
    drift : float or torch.Tensor or Parameter, optional
        Constant :math:`\delta` of the differenced series; needs ``difference > 0`` or ``seasonal_difference > 0``.
    difference : int, default 0
        Order :math:`d` of ordinary differencing.
    seasonal_difference : int, default 0
        Order :math:`D` of seasonal differencing.
    period : int, optional
        Seasonal period :math:`s`; needed for seasonal terms and seasonal differencing.
    n_steps : int, optional
        Series length. Needed for sampling and to give a latent series its shape.
    """

    _param_event_ndims = {"ar": 1, "ma": 1, "seasonal_ar": 1, "seasonal_ma": 1, "beta": 1, "exog": 2}

    def __init__(self, ar=None, ma=None, sigma=1.0, mu=None, *, seasonal_ar=None, seasonal_ma=None, period=None,
                 difference=0, seasonal_difference=0, drift=None, beta=None, exog=None, n_steps=None):
        super().__init__(IdentityTransform(), ContinuousReal(), ContinuousReal())
        if (seasonal_ar is not None or seasonal_ma is not None or seasonal_difference) and (period is None or period < 2):
            raise ValueError("Seasonal terms and seasonal differencing need a period >= 2.")
        if mu is not None and (difference or seasonal_difference):
            raise ValueError("A differenced series has no identifiable mean: use drift (the constant of the differenced series) instead of mu.")
        if mu is None:
            mu = 0.0
        if drift is not None and not (difference or seasonal_difference):
            raise ValueError("drift is the constant of the differenced series: it needs difference > 0 or seasonal_difference > 0.")
        if (beta is None) != (exog is None):
            raise ValueError("Give both beta and exog, or neither.")
        if exog is not None:
            from .._parameters import RandomParameter, DeterministicParameter
            if isinstance(exog, (RandomParameter, DeterministicParameter)):
                raise ValueError("exog must be constant data (a tensor or Data), not a latent node.")
        self.ar, self.ma, self.sigma, self.mu, self.drift = ar, ma, sigma, mu, drift
        self.seasonal_ar, self.seasonal_ma = seasonal_ar, seasonal_ma
        self.beta, self.exog = beta, exog
        self.difference, self.seasonal_difference = int(difference), int(seasonal_difference)
        self.period = None if period is None else int(period)
        self.n_steps = None if n_steps is None else int(n_steps)
        self.difference_coefficients = _difference_coefficients(self.difference, self.seasonal_difference, self.period or 1)
        for p in (ar, ma, sigma, mu, drift, seasonal_ar, seasonal_ma, beta, exog):
            if p is not None:
                self.add_dependency(p)

    # ------------------------------------------------------------ bookkeeping
    @property
    def n_differences(self):
        return len(self.difference_coefficients) - 1

    @property
    def event_shape(self):
        return () if self.n_steps is None else (self.n_steps,)

    def _length(self):
        if self.n_steps is None:
            raise ValueError(f"{type(self).__name__}: give the series length n_steps to sample or to use it as a latent parameter.")
        return self.n_steps

    @property
    def batch_shape(self):
        shapes = []
        for name, node in (("ar", self.ar), ("ma", self.ma), ("seasonal_ar", self.seasonal_ar), ("seasonal_ma", self.seasonal_ma),
                           ("beta", self.beta), ("exog", self.exog), ("sigma", self.sigma), ("mu", self.mu), ("drift", self.drift)):
            if node is not None:
                shape = resolve(node).shape
                shapes.append(tuple(shape[: len(shape) - self._param_event_ndims.get(name, 0)]))
        return torch.broadcast_shapes(*shapes)

    def _values(self):
        names = ("ar", "ma", "seasonal_ar", "seasonal_ma", "beta", "exog", "sigma", "mu", "drift")
        nodes = [getattr(self, n) for n in names]
        present = [n for n, v in zip(names, nodes) if v is not None]
        tensors = _as_float(*[getattr(self, n) for n in present]) if len(present) > 1 else [_as_float(getattr(self, present[0]))]
        values = dict(zip(present, tensors))
        ref = values["sigma"]
        empty = torch.zeros(0, dtype=ref.dtype, device=ref.device)
        for n in ("ar", "ma", "seasonal_ar", "seasonal_ma"):
            values.setdefault(n, empty)
        return values

    def _constant(self, values, like):
        """The constant of the differenced series, shape ``(..., 1)``: ``mu`` or, with differencing, ``drift``."""
        node = values["mu"] if self.n_differences == 0 else values.get("drift")
        if node is None:
            node = torch.zeros_like(values["mu"])
        return node.unsqueeze(-1) + torch.zeros(1, dtype=like.dtype, device=like.device)

    def _polynomials(self, values):
        s = self.period or 1
        c = _ar_polynomial(values["ar"], values["seasonal_ar"], s)
        e = _ma_polynomial(values["ma"], values["seasonal_ma"], s)
        return c, e

    def _system(self, values, c, e, offset, N):
        """The state-space model of the (mean-adjusted) differenced series."""
        sigma = values["sigma"]
        p_tot, q_tot = c.shape[-1], e.shape[-1]
        m = max(p_tot, q_tot + 1)
        batch = torch.broadcast_shapes(c.shape[:-1], e.shape[:-1], sigma.shape, offset.shape[:-1])
        dtype, device = sigma.dtype, sigma.device
        Tm = torch.zeros(*batch, m, m, dtype=dtype, device=device)
        Tm[..., :p_tot, 0] = c.expand(*batch, p_tot)
        if m > 1:
            idx = torch.arange(m - 1, device=device)
            Tm[..., idx, idx + 1] = 1.0
        R = torch.zeros(*batch, m, 1, dtype=dtype, device=device)
        R[..., 0, 0] = 1.0
        R[..., 1: 1 + q_tot, 0] = e.expand(*batch, q_tot)
        Z = torch.zeros(1, m, dtype=dtype, device=device)
        Z[0, 0] = 1.0
        Q = (sigma ** 2).expand(batch)[..., None, None]
        return StateSpace(Tm, Z, Q, selection=R, observation_offset=offset.unsqueeze(-1), n_steps=N), m

    def _prepare(self, y, values):
        y = torch.as_tensor(y).to(values["sigma"].dtype)
        N = y.shape[-1]
        if N <= self.n_differences:
            raise ValueError("The series is too short for the differencing.")
        Dm = _difference_matrix(N, self.difference_coefficients, y.dtype, y.device) if self.n_differences else None
        w = y @ Dm.T if Dm is not None else y
        offset = self._constant(values, y)
        Xd = None
        if self.exog is not None:
            X = values["exog"]
            Xd = Dm @ X if Dm is not None else X
            offset = offset + (Xd @ values["beta"].unsqueeze(-1)).squeeze(-1)
        offset = offset.expand(*offset.shape[:-1], w.shape[-1]) if offset.shape[-1] == 1 else offset
        return y, w, Dm, Xd, offset

    # ------------------------------------------------------------ density
    def log_pdf(self, y):
        values = self._values()
        c, e = self._polynomials(values)
        y, w, Dm, Xd, offset = self._prepare(y, values)
        system, _ = self._system(values, c, e, offset, None)
        return system.log_pdf(w.unsqueeze(-1))

    def _gradients(self, y):
        values = self._values()
        c, e = self._polynomials(values)
        y, w, Dm, Xd, offset = self._prepare(y, values)
        system, m = self._system(values, c, e, offset, None)
        grads, g, ok, s = system._gradients(w.unsqueeze(-1))
        p_tot, q_tot = c.shape[-1], e.shape[-1]
        sp = self.period or 1
        g_c = grads["transition"][..., :p_tot, 0]
        g_e = grads["selection"][..., 1: 1 + q_tot, 0]
        sigma = values["sigma"]
        out = {}

        def add(name, node, value):
            if node is None:
                return
            key = self.resolve_name(name, node)
            out[key] = out[key] + value if key in out else value

        g_phi, g_Phi = _ar_polynomial_vjp(g_c, values["ar"], values["seasonal_ar"], sp)
        g_theta, g_Theta = _ma_polynomial_vjp(g_e, values["ma"], values["seasonal_ma"], sp)
        add("ar", self.ar, g_phi)
        add("seasonal_ar", self.seasonal_ar, g_Phi)
        add("ma", self.ma, g_theta)
        add("seasonal_ma", self.seasonal_ma, g_Theta)
        add("sigma", self.sigma, 2 * sigma * grads["state_cov"][..., 0, 0])
        g_off = grads["observation_offset"][..., 0]
        if self.n_differences == 0:
            add("mu", self.mu, g_off.sum(-1))
        else:
            add("drift", self.drift, g_off.sum(-1))
        if self.beta is not None:
            add("beta", self.beta, (Xd.transpose(-1, -2) @ g_off.unsqueeze(-1)).squeeze(-1))
        return out, g, ok, Dm

    def log_pdf_param_grads(self, y):
        return self._gradients(y)[0]

    def log_pdf_grad(self, y):
        y = torch.as_tensor(y)
        _, g, ok, Dm = self._gradients(y)
        gw = g["y"].movedim(0, -2)[..., 0]
        gw = torch.where(ok.unsqueeze(-1), gw, torch.zeros_like(gw))
        grad = gw @ Dm if Dm is not None else gw
        return grad.sum_to_size(y.shape) if grad.shape != y.shape else grad

    # ------------------------------------------------------------ simulation
    def _simulation_system(self, N_w):
        values = self._values()
        c, e = self._polynomials(values)
        offset = self._constant(values, values["sigma"])
        return values, c, e, offset

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        """Simulate series ``(n_samples, *batch_shape, n_steps)``; the first ``difference + seasonal_difference * period`` values equal ``mu``."""
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        N = self._length()
        n = self.n_differences
        Nw = N - n
        if Nw < 1:
            raise ValueError("n_steps is too short for the differencing.")
        values, c, e, offset = self._simulation_system(Nw)
        if self.exog is not None:
            X = values["exog"]
            Dm = _difference_matrix(N, self.difference_coefficients, X.dtype, X.device) if n else None
            Xd = Dm @ X if Dm is not None else X
            offset = offset + (Xd @ values["beta"].unsqueeze(-1)).squeeze(-1)
        offset = offset.expand(*offset.shape[:-1], Nw)
        system, _ = self._system(values, c, e, offset, Nw)
        w = system.sample(n_samples)[..., 0]
        if n == 0:
            return w
        history = torch.zeros(*w.shape[:-1], n, dtype=w.dtype, device=w.device)
        return _integrate(w, history, self.difference_coefficients, keep_history=True) + values["mu"].unsqueeze(-1)

    def forecast(self, y, steps, n_samples=1, future_exog=None):
        """Simulate the continuation of the observed series ``y`` (shape ``(..., N)``) by ``steps`` steps.

        Exact given the observed history: the state after the last observation has the Kalman-filtered predictive
        distribution; the differencing is undone with the last raw observations. Returns
        ``(n_samples, *batch_shape, steps)``; give parameters with leading batch dimensions (posterior draws) to forecast
        for every draw.

        Parameters
        ----------
        y : torch.Tensor
            Observed series (``nan`` entries are allowed).
        steps : int
            Forecast horizon.
        n_samples : int
            Simulated futures per batch element.
        future_exog : torch.Tensor, optional
            Regressors of the forecast period, shape ``(steps, r)`` (required with ``exog``).
        """
        values = self._values()
        c, e = self._polynomials(values)
        y = torch.as_tensor(y).to(values["sigma"].dtype)
        N = y.shape[-1]
        n = self.n_differences
        if (self.exog is None) != (future_exog is None):
            raise ValueError("Give future_exog exactly when the model has exog.")
        _, w, Dm, Xd, offset = self._prepare(y, values)
        future_offset = self._constant(values, y) + torch.zeros(steps, dtype=y.dtype, device=y.device)
        if self.exog is not None:
            X_all = torch.cat([values["exog"], torch.as_tensor(future_exog, dtype=y.dtype)], dim=-2)
            D_all = _difference_matrix(N + steps, self.difference_coefficients, y.dtype, y.device) if n else None
            Xd_all = D_all @ X_all if D_all is not None else X_all
            future_offset = future_offset + (Xd_all[..., -steps:, :] @ values["beta"].unsqueeze(-1)).squeeze(-1)
        system, _ = self._system(values, c, e, offset, None)
        wf = system.forecast(w.unsqueeze(-1), steps, n_samples, future_offset=future_offset.unsqueeze(-1))[..., 0]
        if n == 0:
            return wf
        history = y[..., -n:].expand(*wf.shape[:-1], n)
        return _integrate(wf, history, self.difference_coefficients)
