"""Gaussian random walk and autoregressive distributions over a whole series."""
import math
import torch

from ..Distributions._distributions import Distribution, _as_float, _sample_shape
from ..Distributions._resolve import resolve
from ..Distributions._transforms import IdentityTransform
from ..Distributions._state_space import ContinuousReal
from ._lyapunov import _companion, _solve_lyapunov

_LOG_2PI = math.log(2 * math.pi)


def _pad_batch(*values):
    """Broadcast the batch shapes of tensors (their shapes without the last dimension handled by the caller)."""
    return torch.broadcast_shapes(*[v.shape for v in values])


class _SeriesDistribution(Distribution):
    """Common bookkeeping of the distributions over a series of length ``n_steps`` (the last dimension)."""

    def __init__(self, n_steps):
        super().__init__(IdentityTransform(), ContinuousReal(), ContinuousReal())
        self.n_steps = None if n_steps is None else int(n_steps)

    @property
    def event_shape(self):
        return () if self.n_steps is None else (self.n_steps,)

    def _length(self):
        if self.n_steps is None:
            raise ValueError(f"{type(self).__name__}: give the series length n_steps to sample or to use it as a latent parameter.")
        return self.n_steps


# ========================= Gaussian random walk =========================
class GaussianRandomWalk(_SeriesDistribution):
    r"""Gaussian random walk with drift over a series :math:`x_1, \dots, x_T` (the last dimension of the value).

    .. math::

        x_1 \sim \mathcal{N}(m_0, s_0^2), \qquad x_t \mid x_{t-1} \sim \mathcal{N}(x_{t-1} + \mu, \sigma^2), \quad t = 2, \dots, T

    The density is the exact joint density of the series (a proper distribution): the first
    value has its own normal distribution, so the walk is not conditioned on a starting value.
    By default :math:`s_0 = \sigma`, i.e. the walk starts at ``init_mu`` at time 0 with the
    drift applied from the second step on; give ``init_sigma`` for a wider or tighter start.
    All parameters are scalars per series (they broadcast over leading batch dimensions);
    use it as a prior on a latent series, ``RandomParameter("x", GaussianRandomWalk(sigma, n_steps=100))``,
    or as the likelihood of an observed series.

    Parameters
    ----------
    sigma : float or torch.Tensor or Parameter
        Innovation standard deviation :math:`\sigma > 0`.
    mu : float or torch.Tensor or Parameter, default 0.0
        Drift per step.
    init_mu : float or torch.Tensor or Parameter, default 0.0
        Mean :math:`m_0` of the first value.
    init_sigma : float or torch.Tensor or Parameter, optional
        Standard deviation :math:`s_0` of the first value (default: ``sigma``).
    n_steps : int, optional
        Series length. Needed for sampling and to give a latent series its shape.
    """

    def __init__(self, sigma, mu=0.0, init_mu=0.0, init_sigma=None, n_steps=None):
        super().__init__(n_steps)
        self.sigma = sigma
        self.mu = mu
        self.init_mu = init_mu
        self.init_sigma = init_sigma
        for p in (sigma, mu, init_mu, init_sigma):
            if p is not None:
                self.add_dependency(p)

    def _params(self):
        names = [self.sigma, self.mu, self.init_mu] + ([self.init_sigma] if self.init_sigma is not None else [])
        values = _as_float(*names)
        sigma, mu, init_mu = values[:3]
        return sigma, mu, init_mu, (values[3] if self.init_sigma is not None else sigma)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(*[resolve(p).shape for p in (self.sigma, self.mu, self.init_mu, self.init_sigma) if p is not None])

    def _parts(self, x):
        sigma, mu, init_mu, s0 = self._params()
        u = lambda v: v.unsqueeze(-1)
        d = x[..., 1:] - x[..., :-1] - u(mu)
        r0 = x[..., 0] - init_mu
        return sigma, mu, init_mu, s0, d, r0

    def log_pdf(self, x):
        x = torch.as_tensor(x)
        sigma, _, _, s0, d, r0 = self._parts(x)
        T = x.shape[-1]
        return (-0.5 * T * _LOG_2PI - torch.log(s0) - 0.5 * (r0 / s0) ** 2
                - (T - 1) * torch.log(sigma) - 0.5 * (d ** 2).sum(-1) / sigma ** 2)

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        sigma, _, _, s0, d, r0 = self._parts(x)
        scaled = d / sigma.unsqueeze(-1) ** 2
        grad = torch.zeros(torch.broadcast_shapes(x.shape, scaled.shape[:-1] + x.shape[-1:]), dtype=scaled.dtype, device=scaled.device)
        grad[..., 1:] -= scaled
        grad[..., :-1] += scaled
        grad[..., 0] -= r0 / s0 ** 2
        return grad.sum_to_size(x.shape)

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        sigma, _, _, s0, d, r0 = self._parts(x)
        T = x.shape[-1]
        grad_sigma = -(T - 1) / sigma + (d ** 2).sum(-1) / sigma ** 3
        grad_s0 = -1 / s0 + r0 ** 2 / s0 ** 3
        grads = {}

        def add(parameter, key, value):
            name = self.resolve_name(key, parameter)
            grads[name] = grads[name] + value if name in grads else value

        add(self.sigma, "sigma", grad_sigma + (grad_s0 if self.init_sigma is None else 0))
        add(self.mu, "mu", d.sum(-1) / sigma ** 2)
        add(self.init_mu, "init_mu", r0 / s0 ** 2)
        if self.init_sigma is not None:
            add(self.init_sigma, "init_sigma", grad_s0)
        return grads

    def mean(self):
        sigma, mu, init_mu, _ = self._params()
        t = torch.arange(self._length(), dtype=sigma.dtype, device=sigma.device)
        return init_mu.unsqueeze(-1) + mu.unsqueeze(-1) * t + torch.zeros_like(sigma).unsqueeze(-1)

    def variance(self):
        sigma, _, _, s0 = self._params()
        t = torch.arange(self._length(), dtype=sigma.dtype, device=sigma.device)
        return s0.unsqueeze(-1) ** 2 + sigma.unsqueeze(-1) ** 2 * t

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        T = self._length()
        sigma, mu, init_mu, s0 = self._params()
        shape = _sample_shape(n_samples, self.batch_shape, (T,))
        eps = torch.randn(shape, dtype=sigma.dtype, device=sigma.device)
        scale = torch.cat([s0.unsqueeze(-1).expand(*shape[:-1], 1), sigma.unsqueeze(-1).expand(*shape[:-1], T - 1)], dim=-1)
        shift = torch.cat([init_mu.unsqueeze(-1).expand(*shape[:-1], 1), mu.unsqueeze(-1).expand(*shape[:-1], T - 1)], dim=-1)
        return torch.cumsum(shift + scale * eps, dim=-1)


# ========================= autoregressive process =========================
def _is_stationary(phi, tol=1e-9):
    """Whether the AR polynomial :math:`1 - \\sum_j \\phi_j z^j` has all roots outside the unit circle (``(...,)`` boolean)."""
    radius = torch.linalg.eigvals(_companion(phi)).abs().amax(-1)
    return radius < 1 - tol


class AR(_SeriesDistribution):
    r"""Autoregressive process AR(:math:`p`) over a series :math:`x_1, \dots, x_T` (the last dimension of the value).

    .. math::

        x_t - \mu = \sum_{j=1}^p \phi_j (x_{t-j} - \mu) + \sigma \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1)

    with :math:`\mu` the process mean. Two likelihoods are available:

    * ``init="stationary"`` (default): the exact joint density of the series. The first
      :math:`p` values are drawn from the stationary distribution
      :math:`\mathcal{N}(\mu \mathbf{1}, \sigma^2 \Gamma)` (``\Gamma`` solves the discrete Lyapunov equation
      of the companion matrix), the rest follow the recursion. This needs a *stationary*
      process: all roots of :math:`1 - \sum_j \phi_j z^j` outside the unit circle, which
      the density enforces by returning ``-inf`` outside the stationary region (see
      :class:`~BayesianDLL.Deterministic.StationaryAR` for a parameterisation that stays inside).
      It is a proper distribution, so it can be a prior for a latent series.
    * ``init="conditional"``: the conditional density of :math:`x_{p+1}, \dots, x_T` given the
      first :math:`p` values, which are left unmodelled. It needs no stationarity but is improper
      as a prior for the first :math:`p` values; use it as a likelihood for observed series
      (it is what least squares maximises). The gradient with respect to the first :math:`p` values is still the conditional one.

    Sampling always starts from the stationary distribution (a stationary process is required).
    Parameters broadcast over leading batch dimensions.

    Parameters
    ----------
    coefficients : torch.Tensor or Parameter
        AR coefficients :math:`\phi_1, \dots, \phi_p` of shape ``(..., p)``.
    sigma : float or torch.Tensor or Parameter
        Innovation standard deviation :math:`\sigma > 0`.
    mu : float or torch.Tensor or Parameter, default 0.0
        Process mean :math:`\mu`.
    n_steps : int, optional
        Series length. Needed for sampling and to give a latent series its shape.
    init : {"stationary", "conditional"}
        Treatment of the first :math:`p` values, see above.
    """

    _param_event_ndims = {"coefficients": 1}

    def __init__(self, coefficients, sigma, mu=0.0, n_steps=None, init="stationary"):
        if init not in ("stationary", "conditional"):
            raise ValueError("init must be 'stationary' or 'conditional'.")
        super().__init__(n_steps)
        self.coefficients = coefficients
        self.sigma = sigma
        self.mu = mu
        self.init = init
        for p in (coefficients, sigma, mu):
            self.add_dependency(p)

    @property
    def order(self):
        return resolve(self.coefficients).shape[-1]

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.coefficients).shape[:-1], resolve(self.sigma).shape, resolve(self.mu).shape)

    def _stationary_part(self, phi, sigma):
        """``(Gamma, ok)``: stationary covariance of the state (unit innovation variance times sigma^2) and its validity."""
        ok = _is_stationary(phi)
        T = _companion(torch.where(ok.unsqueeze(-1), phi, torch.zeros_like(phi)))
        Q = torch.zeros_like(T)
        Q[..., 0, 0] = 1.0
        G1 = _solve_lyapunov(T, Q)
        G1 = 0.5 * (G1 + G1.transpose(-1, -2))
        return T, G1 * (sigma ** 2)[..., None, None], ok

    def _evaluate(self, x, need_grads):
        p = self.order
        x = torch.as_tensor(x)
        phi, sigma, mu = _as_float(self.coefficients, self.sigma, self.mu)
        n = x.shape[-1]
        if n < p + (self.init == "conditional"):
            raise ValueError(f"A series of length {n} is too short for an AR({p}) with init='{self.init}'.")
        z = x - mu.unsqueeze(-1)
        windows = z.unfold(-1, p + 1, 1)  # (..., n - p, p + 1): z_{t-p}, ..., z_t
        lagged = windows[..., :p].flip(-1)  # z_{t-1}, ..., z_{t-p}
        e = windows[..., p] - (lagged * phi.unsqueeze(-2)).sum(-1)  # (..., n - p)
        m = n - p
        s2 = sigma ** 2
        lp = -0.5 * m * _LOG_2PI - m * torch.log(sigma) - 0.5 * (e ** 2).sum(-1) / s2
        out = {"z": z, "e": e, "lagged": lagged, "phi": phi, "sigma": sigma, "mu": mu, "lp": lp, "n": n, "m": m, "ok": None}
        if self.init == "stationary" and n >= p:
            T, Gam, ok = self._stationary_part(phi, sigma)
            z0 = z[..., :p]
            Gam = torch.where(ok[..., None, None], Gam, torch.eye(p, dtype=Gam.dtype, device=Gam.device))
            chol, info = torch.linalg.cholesky_ex(Gam)
            ok = ok & (info == 0)
            chol = torch.where(ok[..., None, None], chol, torch.eye(p, dtype=Gam.dtype, device=Gam.device))
            lead = torch.broadcast_shapes(z0.shape[:-1], chol.shape[:-2])
            sol = torch.cholesky_solve(z0.expand(*lead, p).unsqueeze(-1).contiguous(), chol.expand(*lead, p, p).contiguous())
            quad = (z0.unsqueeze(-1) * sol).sum((-1, -2))
            logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
            lp = lp - 0.5 * p * _LOG_2PI - 0.5 * logdet - 0.5 * quad
            lp = torch.where(ok, lp, torch.full_like(lp, -math.inf))
            out.update(T=T, Gam=Gam, chol=chol, ok=ok, sol=sol.squeeze(-1), lp=lp)
        return out

    def log_pdf(self, x):
        return self._evaluate(x, False)["lp"]

    def _grads(self, x):
        d = self._evaluate(x, True)
        p, n, m = self.order, d["n"], d["m"]
        z, e, lagged, phi, sigma, mu, ok = d["z"], d["e"], d["lagged"], d["phi"], d["sigma"], d["mu"], d["ok"]
        s2 = sigma ** 2
        # conditional part
        g_phi = (e.unsqueeze(-1) * lagged).sum(-2) / s2.unsqueeze(-1)
        g_sigma = -m / sigma + (e ** 2).sum(-1) / sigma ** 3
        g_mu = e.sum(-1) * (1 - phi.sum(-1)) / s2
        E = torch.cat([torch.zeros(*e.shape[:-1], p, dtype=e.dtype, device=e.device), e], dim=-1)
        g_z = -E
        for j in range(1, p + 1):
            g_z[..., : n - j] += phi[..., j - 1].unsqueeze(-1) * E[..., j:]
        g_z = g_z / s2.unsqueeze(-1)
        if ok is not None:
            T, Gam, chol, sol = d["T"], d["Gam"], d["chol"], d["sol"]
            A = torch.cholesky_inverse(chol)
            z0 = z[..., :p]
            u = sol  # Gamma^{-1} z0
            g_z0 = -u
            g_mu = g_mu + u.sum(-1)
            G = 0.5 * (u.unsqueeze(-1) * u.unsqueeze(-2) - A)
            Lam = _solve_lyapunov(T, G, transpose=True)
            g_T = 2 * Lam @ T @ Gam
            g_phi = g_phi + g_T[..., 0, :]
            g_sigma = g_sigma + 2 * sigma * Lam[..., 0, 0]
            pad = torch.cat([g_z0, torch.zeros(*g_z0.shape[:-1], n - p, dtype=g_z0.dtype, device=g_z0.device)], dim=-1)
            g_z = g_z + pad
            valid = ok
            g_phi = torch.where(valid.unsqueeze(-1), g_phi, torch.zeros_like(g_phi))
            g_sigma = torch.where(valid, g_sigma, torch.zeros_like(g_sigma))
            g_mu = torch.where(valid, g_mu, torch.zeros_like(g_mu))
            g_z = torch.where(valid.unsqueeze(-1), g_z, torch.zeros_like(g_z))
        return g_z, g_phi, g_sigma, g_mu

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        return self._grads(x)[0].sum_to_size(x.shape)

    def log_pdf_param_grads(self, x):
        _, g_phi, g_sigma, g_mu = self._grads(x)
        grads = {}
        for parameter, key, value in ((self.coefficients, "coefficients", g_phi), (self.sigma, "sigma", g_sigma), (self.mu, "mu", g_mu)):
            name = self.resolve_name(key, parameter)
            grads[name] = grads[name] + value if name in grads else value
        return grads

    def _stationary_moments(self):
        phi, sigma = _as_float(self.coefficients, self.sigma)
        _, Gam, ok = self._stationary_part(phi, sigma)
        return Gam, ok

    def mean(self):
        mu = _as_float(self.mu)
        return mu.expand(self.batch_shape).unsqueeze(-1).expand(*self.batch_shape, self._length()).clone()

    def variance(self):
        """Marginal variance (constant over time); needs a stationary process."""
        Gam, ok = self._stationary_moments()
        var = torch.where(ok, Gam[..., 0, 0], torch.full_like(Gam[..., 0, 0], math.nan))
        return var.expand(self.batch_shape).unsqueeze(-1).expand(*self.batch_shape, self._length()).clone()

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        n = self._length()
        p = self.order
        phi, sigma, mu = _as_float(self.coefficients, self.sigma, self.mu)
        batch = self.batch_shape
        _, Gam, ok = self._stationary_part(phi, sigma)
        if not bool(ok.all()):
            raise ValueError("AR sampling needs stationary coefficients.")
        chol = torch.linalg.cholesky(Gam).expand(*batch, p, p)
        eps0 = torch.randn(n_samples, *batch, p, 1, dtype=phi.dtype, device=phi.device)
        z = [(chol @ eps0).squeeze(-1)[..., i] for i in range(p)]
        z = z[:n]
        phi = phi.expand(*batch, p)
        sig = sigma.expand(batch)
        for t in range(p, n):
            nxt = sum(phi[..., j] * z[t - 1 - j] for j in range(p)) + sig * torch.randn(n_samples, *batch, dtype=phi.dtype, device=phi.device)
            z.append(nxt)
        return torch.stack(z, dim=-1) + mu.expand(batch).unsqueeze(-1)
