"""Ordinal, censored, hurdle and overdispersed count distributions."""
import math
import torch
import torch.nn.functional as F

from ._distributions import (
    Distribution, Multinomial, _as_float, _device, _discrete_grad_error, _expand, _float_dtype,
    _in_support, _is_integer, _restrict, _sample_beta, _sample_shape, _zero_outside, _standard_gamma,
)
from ._transforms import IdentityTransform
from ._state_space import DiscreteRange
from ._resolve import resolve


# ========================= Ordered logistic =========================
class OrderedLogistic(Distribution):
    r"""Ordered logistic (proportional odds) distribution for ordinal outcomes.

    An ordinal outcome :math:`y \in \{0, 1, \dots, K\}` is generated from a latent logistic
    variable with location :math:`\eta` that falls between consecutive cutpoints
    :math:`c_1 < \dots < c_K` (:math:`c_0 = -\infty`, :math:`c_{K+1} = \infty`):

    .. math::

        P(y = k \mid \eta, c) = \sigma(c_{k+1} - \eta) - \sigma(c_k - \eta)

    where :math:`\sigma` is the logistic function. The cutpoints are usually given an
    :class:`Ordered` prior, which keeps them sorted. Categories are coded ``0, ..., K``.
    The distribution does not check that the cutpoints are increasing.

    Parameters
    ----------
    eta : float or torch.Tensor or Parameter
        Linear predictor (location of the latent logistic variable).
    cutpoints : torch.Tensor or Parameter
        Increasing cutpoints of shape ``(..., K)``; there are ``K + 1`` categories.
    """

    _param_event_ndims = {"cutpoints": 1}

    def __init__(self, eta, cutpoints):
        n_cut = resolve(cutpoints).shape[-1]
        super().__init__(IdentityTransform(), DiscreteRange(0, n_cut), DiscreteRange(0, n_cut))
        self.eta = eta
        self.cutpoints = cutpoints
        self.add_dependency(eta)
        self.add_dependency(cutpoints)

    @property
    def n_categories(self):
        return resolve(self.cutpoints).shape[-1] + 1

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.eta).shape, resolve(self.cutpoints).shape[:-1])

    def _support(self, y):
        y = torch.as_tensor(y)
        inside = (y >= 0) & (y <= self.n_categories - 1) & _is_integer(y)
        return _in_support(y, inside)

    def _edges(self, y):
        """``(a, b, idx, K)``: upper and lower cutpoint of ``y`` minus ``eta``, the category index and the number of cutpoints."""
        eta, cuts = _as_float(self.eta, self.cutpoints)
        K = cuts.shape[-1]
        shape = torch.broadcast_shapes(y.shape, eta.shape, cuts.shape[:-1])
        cuts = cuts.expand(*shape, K)
        inf = torch.full((*shape, 1), math.inf, dtype=cuts.dtype, device=cuts.device)
        padded = torch.cat([-inf, cuts, inf], dim=-1)
        idx = y.long().expand(shape).unsqueeze(-1)
        return padded.gather(-1, idx + 1).squeeze(-1) - eta, padded.gather(-1, idx).squeeze(-1) - eta, idx.squeeze(-1), K

    @staticmethod
    def _log_prob(a, b):
        # log(sigma(a) - sigma(b)) = log sigma(a) + log sigma(-b) + log(1 - exp(b - a)) for a > b
        return F.logsigmoid(a) + F.logsigmoid(-b) + torch.log(-torch.expm1(b - a))

    def log_pdf(self, y):
        inside, y = self._support(y)
        a, b, _, _ = self._edges(y)
        return _restrict(self._log_prob(a, b), inside)

    def log_pdf_grad(self, y):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, y):
        inside, y = self._support(y)
        a, b, idx, K = self._edges(y)
        lp = self._log_prob(a, b)
        # d sigma(a) = sigma(a) sigma(-a); the quotients with P are formed in log space
        up = torch.exp(F.logsigmoid(a) + F.logsigmoid(-a) - lp)
        down = torch.exp(F.logsigmoid(b) + F.logsigmoid(-b) - lp)
        grad_eta = _zero_outside(down - up, inside)
        ar = torch.arange(K, device=idx.device)
        grad_c = up.unsqueeze(-1) * (ar == idx.unsqueeze(-1)) - down.unsqueeze(-1) * (ar == idx.unsqueeze(-1) - 1)
        grad_c = _zero_outside(grad_c, inside.unsqueeze(-1))
        return {self.resolve_name("eta", self.eta): grad_eta, self.resolve_name("cutpoints", self.cutpoints): grad_c}

    def probs(self):
        """Category probabilities of shape ``(*batch_shape, K + 1)``."""
        eta, cuts = _as_float(self.eta, self.cutpoints)
        shape = torch.broadcast_shapes(eta.shape, cuts.shape[:-1])
        cdf = torch.sigmoid(cuts.expand(*shape, cuts.shape[-1]) - eta.expand(shape).unsqueeze(-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf, torch.ones_like(cdf[..., :1])], dim=-1)
        return cdf[..., 1:] - cdf[..., :-1]

    def mean(self):
        p = self.probs()
        return (p * torch.arange(p.shape[-1], dtype=p.dtype, device=p.device)).sum(-1)

    def variance(self):
        p = self.probs()
        k = torch.arange(p.shape[-1], dtype=p.dtype, device=p.device)
        return (p * k ** 2).sum(-1) - self.mean() ** 2

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        eta, cuts = _as_float(self.eta, self.cutpoints)
        shape = _sample_shape(n_samples, self.batch_shape)
        info = torch.finfo(eta.dtype)
        u = torch.rand(shape, dtype=eta.dtype, device=eta.device).clamp(min=info.tiny, max=1 - info.eps)
        latent = eta + torch.log(u) - torch.log1p(-u)
        cuts = cuts.expand(*self.batch_shape, cuts.shape[-1])
        return (latent.unsqueeze(-1) > cuts).sum(-1).to(eta.dtype)


# ========================= Censored =========================
class Censored(Distribution):
    r"""Distribution of observations that are only known to lie below or above a limit.

    An observation equal to the lower limit :math:`a` means "the value is at most :math:`a`",
    one equal to the upper limit :math:`b` means "the value is at least :math:`b`". With the
    base density :math:`f` and distribution function :math:`F`:

    .. math::

        p(x) = \begin{cases}
            F(a) & x \le a \quad\text{(left-censored)} \\
            1 - F(b) & x \ge b \quad\text{(right-censored)} \\
            f(x) & \text{otherwise.}
        \end{cases}

    Use it as the distribution of an observed node whose censored entries are recorded at the
    limit. Interval censoring is not covered. The base needs ``cdf`` (and, for gradients with
    respect to its parameters, ``cdf_param_grads``): Normal, LogNormal, HalfNormal,
    Exponential, Gamma, Beta, StudentT, Cauchy and Laplace have them. Sampling draws from the
    base and clips to the limits.

    Parameters
    ----------
    base : Distribution
        Univariate continuous distribution of the uncensored value.
    low : float or torch.Tensor or Parameter, optional
        Lower limit :math:`a`. ``None``: no left-censoring.
    high : float or torch.Tensor or Parameter, optional
        Upper limit :math:`b`. ``None``: no right-censoring.
    """

    def __init__(self, base, low=None, high=None):
        if base.event_shape != ():
            raise ValueError("Censored needs a univariate base distribution.")
        if low is None and high is None:
            raise ValueError("Give at least one of low / high.")
        super().__init__(base.transform, base.state_space, base.transformed_state_space)
        self.base = base
        self.low = low
        self.high = high
        self._add_dependencies_of(base)
        for bound in (low, high):
            if bound is not None:
                self.add_dependency(bound)

    @property
    def batch_shape(self):
        shapes = [tuple(self.base.batch_shape)] + [resolve(b).shape for b in (self.low, self.high) if b is not None]
        return torch.broadcast_shapes(*shapes)

    def _limits(self, x):
        low = None if self.low is None else _as_float(self.low).to(x.dtype)
        high = None if self.high is None else _as_float(self.high).to(x.dtype)
        return low, high

    def _masks(self, x, low, high):
        left = torch.zeros_like(x, dtype=torch.bool) if low is None else x <= low
        right = torch.zeros_like(x, dtype=torch.bool) if high is None else x >= high
        return left, right

    def _tails(self, x):
        low, high = self._limits(x)
        left, right = self._masks(x, low, high)
        log_f_low = torch.log(self.base.cdf(low)) if low is not None else None
        log_s_high = torch.log(self.base.sf(high)) if high is not None else None
        return low, high, left, right, log_f_low, log_s_high

    def log_pdf(self, x):
        x = torch.as_tensor(x)
        _, _, left, right, log_f_low, log_s_high = self._tails(x)
        out = self.base.log_pdf(x)
        if log_s_high is not None:
            out = torch.where(right, log_s_high, out)
        if log_f_low is not None:
            out = torch.where(left, log_f_low, out)
        return out

    def log_pdf_grad(self, x):
        x = torch.as_tensor(x)
        low, high = self._limits(x)
        left, right = self._masks(x, low, high)
        return torch.where(left | right, torch.zeros_like(x), self.base.log_pdf_grad(x))

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        low, high, left, right, log_f_low, log_s_high = self._tails(x)
        has_params = hasattr(self.base, "cdf_param_grads")
        if not has_params and (self.base.parameters or self.base.variational_parameters):
            raise NotImplementedError(f"Censored({type(self.base).__name__}) parameter gradients need {type(self.base).__name__}.cdf_param_grads.")
        grads = dict(self.base.log_pdf_param_grads(x)) if has_params else {}
        zero = torch.zeros_like(x)
        if low is not None:
            if has_params:
                f_low = torch.exp(log_f_low)
                for name, g in self.base.cdf_param_grads(low).items():
                    grads[name] = torch.where(left, g / f_low, grads.get(name, zero))
            name = self.resolve_name("low", self.low)
            grads[name] = grads.get(name, 0) + torch.where(left, torch.exp(self.base.log_pdf(low) - log_f_low) + zero, zero)
        if high is not None:
            if has_params:
                s_high = torch.exp(log_s_high)
                for name, g in self.base.cdf_param_grads(high).items():
                    grads[name] = torch.where(right, -g / s_high, grads.get(name, zero))
            name = self.resolve_name("high", self.high)
            grads[name] = grads.get(name, 0) + torch.where(right, -torch.exp(self.base.log_pdf(high) - log_s_high) + zero, zero)
        return grads

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        draws = self.base.sample(n_samples).expand(_sample_shape(n_samples, self.batch_shape))
        low, high = self._limits(draws)
        if low is not None:
            draws = torch.maximum(draws, low)
        if high is not None:
            draws = torch.minimum(draws, high)
        return draws


# ========================= Hurdle =========================
class Hurdle(Distribution):
    r"""Hurdle distribution for counts: a zero-versus-positive process and a zero-truncated count.

    A zero occurs with probability :math:`\pi`; given a positive outcome the count follows
    the base distribution truncated to :math:`\{1, 2, \dots\}`:

    .. math::

        P(X = 0) = \pi, \qquad P(X = k) = (1 - \pi) \frac{P_{\text{base}}(k)}{1 - P_{\text{base}}(0)} \quad (k \ge 1)

    In contrast to :class:`ZeroInflated`, where zeros come from two sources (the gate and the
    base) and :math:`\pi` is only the extra zero probability, here every zero comes from the
    hurdle and :math:`\pi` is the total zero probability. The two processes are separate, so
    :math:`\pi` can be modelled independently of the count parameters. Sampling draws the
    positive part by rejection from the base.

    Parameters
    ----------
    base : Distribution
        Discrete base distribution on the non-negative integers (e.g. :class:`Poisson`, :class:`NegativeBinomial`).
    zero_prob : float or torch.Tensor or Parameter
        Total probability of a zero :math:`\pi \in (0, 1)`.
    """

    def __init__(self, base, zero_prob):
        super().__init__(base.transform, base.state_space, base.transformed_state_space)
        self.base = base
        self.zero_prob = zero_prob
        self._add_dependencies_of(base)
        self.add_dependency(zero_prob)

    @property
    def event_shape(self):
        return self.base.event_shape

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(tuple(self.base.batch_shape), resolve(self.zero_prob).shape)

    def _parts(self, x):
        pi = _as_float(self.zero_prob)
        zero = torch.zeros_like(x)
        base_lp = self.base.log_pdf(x)
        lp0 = self.base.log_pdf(zero)
        log1m_f0 = torch.log(-torch.expm1(lp0))
        return pi, zero, base_lp, lp0, log1m_f0, x == 0

    def log_pdf(self, x):
        x = torch.as_tensor(x)
        pi, _, base_lp, _, log1m_f0, is_zero = self._parts(x)
        return torch.where(is_zero, torch.log(pi) + torch.zeros_like(base_lp), torch.log1p(-pi) + base_lp - log1m_f0)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        x = torch.as_tensor(x)
        pi, zero, base_lp, lp0, log1m_f0, is_zero = self._parts(x)
        positive = ~is_zero & torch.isfinite(base_lp)
        grad_pi = torch.where(is_zero, 1 / pi + zero, -1 / (1 - pi) + zero)
        grad_pi = _zero_outside(grad_pi, is_zero | positive)
        # d/dtheta [log p(x) - log(1 - p(0))] = d log p(x) + p(0) / (1 - p(0)) d log p(0)
        ratio = torch.exp(lp0 - log1m_f0)
        grads = {}
        g_x = self.base.log_pdf_param_grads(x)
        g_0 = self.base.log_pdf_param_grads(zero)
        for name in g_x:
            gx, g0 = g_x[name], g_0[name]
            w, r = positive, ratio
            if gx.ndim > w.ndim:
                pad = (1,) * (gx.ndim - w.ndim)
                w, r = w.reshape(*w.shape, *pad), r.reshape(*r.shape, *pad)
            grads[name] = torch.where(w, gx + r * g0, torch.zeros_like(gx))
        name = self.resolve_name("zero_prob", self.zero_prob)
        grads[name] = grads.get(name, 0) + grad_pi
        return grads

    def _moments(self):
        pi = _as_float(self.zero_prob)
        mu, var = self.base.mean(), self.base.variance()
        f0 = self.base.pdf(torch.zeros((), dtype=mu.dtype, device=mu.device))
        scale = (1 - pi) / (1 - f0)
        return scale * mu, scale * (var + mu ** 2)

    def mean(self):
        return self._moments()[0]

    def variance(self):
        m, m2 = self._moments()
        return m2 - m ** 2

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        pi = _as_float(self.zero_prob)
        shape = _sample_shape(n_samples, self.batch_shape)
        positive = self.base.sample(n_samples).expand(shape).clone()
        for _ in range(10000):
            bad = positive == 0
            if not bool(bad.any()):
                break
            positive = torch.where(bad, self.base.sample(n_samples).expand(shape), positive)
        else:
            raise RuntimeError("Hurdle: rejection sampling of the zero-truncated base did not terminate (P_base(0) too close to 1).")
        u = torch.rand(shape, dtype=pi.dtype, device=positive.device)
        return torch.where(u < pi, torch.zeros_like(positive), positive)


# ========================= Beta-binomial / Dirichlet-multinomial =========================
class BetaBinomial(Distribution):
    r"""Beta-binomial distribution: a binomial count whose success probability is :math:`\mathrm{Beta}(\alpha, \beta)`.

    The probability mass function on :math:`\{0, 1, \dots, n\}` is:

    .. math::

        P(X = k \mid n, \alpha, \beta) = \binom{n}{k} \frac{\mathrm{B}(k + \alpha, n - k + \beta)}{\mathrm{B}(\alpha, \beta)}

    It is an overdispersed binomial: the variance is
    :math:`n \bar p (1 - \bar p) (\alpha + \beta + n) / (\alpha + \beta + 1)` with :math:`\bar p = \alpha / (\alpha + \beta)`.

    Parameters
    ----------
    n : int or torch.Tensor or Parameter
        Number of trials :math:`n \ge 0`.
    alpha : float or torch.Tensor or Parameter
        First shape parameter :math:`\alpha > 0`.
    beta : float or torch.Tensor or Parameter
        Second shape parameter :math:`\beta > 0`.
    """

    def __init__(self, n, alpha, beta):
        super().__init__(IdentityTransform(), DiscreteRange(0, n), DiscreteRange(0, n))
        self.n = n
        self.alpha = alpha
        self.beta = beta
        for p in (n, alpha, beta):
            self.add_dependency(p)

    @property
    def batch_shape(self):
        return torch.broadcast_shapes(resolve(self.n).shape, resolve(self.alpha).shape, resolve(self.beta).shape)

    def _count(self, *like):
        n = resolve(self.n)
        return n.to(dtype=_float_dtype(*like, resolve(self.alpha)), device=_device(*like, n))

    def _support(self, x):
        x = torch.as_tensor(x)
        return _in_support(x, (x >= 0) & (x <= self._count(x)) & _is_integer(x))

    def log_pdf(self, x):
        inside, x = self._support(x)
        a, b = _as_float(self.alpha, self.beta)
        n = self._count(x)
        lg = torch.lgamma
        log_p = (lg(n + 1) - lg(x + 1) - lg(n - x + 1) + lg(x + a) + lg(n - x + b) - lg(n + a + b)
                 - lg(a) - lg(b) + lg(a + b))
        return _restrict(log_p, inside)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        inside, x = self._support(x)
        a, b = _as_float(self.alpha, self.beta)
        n = self._count(x)
        dg = torch.digamma
        common = dg(a + b) - dg(n + a + b)
        grad_a = dg(x + a) - dg(a) + common
        grad_b = dg(n - x + b) - dg(b) + common
        return {self.resolve_name("alpha", self.alpha): _zero_outside(grad_a, inside),
                self.resolve_name("beta", self.beta): _zero_outside(grad_b, inside)}

    def mean(self):
        a, b = _as_float(self.alpha, self.beta)
        return _expand(self._count(a) * a / (a + b), self.batch_shape)

    def variance(self):
        a, b = _as_float(self.alpha, self.beta)
        n, s = self._count(a), a + b
        return _expand(n * a * b * (s + n) / (s ** 2 * (s + 1)), self.batch_shape)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        a, b = _as_float(self.alpha, self.beta)
        shape = _sample_shape(n_samples, self.batch_shape)
        p = _sample_beta(a, b, shape).clamp(0, 1)
        n = self._count(p).expand(shape).contiguous()
        return torch.binomial(n, p.contiguous())


class DirichletMultinomial(Distribution):
    r"""Dirichlet-multinomial distribution: a multinomial whose probabilities are :math:`\mathrm{Dirichlet}(\boldsymbol\alpha)`.

    For counts :math:`\mathbf{k}` with :math:`\sum_j k_j = n` and :math:`\alpha_0 = \sum_j \alpha_j`:

    .. math::

        P(\mathbf{k} \mid n, \boldsymbol\alpha) = \frac{n!\,\Gamma(\alpha_0)}{\Gamma(n + \alpha_0)}
        \prod_{j=1}^K \frac{\Gamma(k_j + \alpha_j)}{k_j!\,\Gamma(\alpha_j)}

    It is an overdispersed multinomial (the counts are positively dependent through the
    shared random probabilities).

    Parameters
    ----------
    total_count : int or torch.Tensor or Parameter
        Number of trials :math:`n`.
    concentration : torch.Tensor or Parameter
        Positive concentration vector :math:`\boldsymbol\alpha` of shape ``(..., K)``.
    """

    _param_event_ndims = {"concentration": 1}

    def __init__(self, total_count, concentration):
        super().__init__(IdentityTransform(), DiscreteRange(0, total_count), DiscreteRange(0, total_count))
        self.total_count = total_count
        self.concentration = concentration
        self.add_dependency(total_count)
        self.add_dependency(concentration)

    @property
    def event_shape(self):
        return resolve(self.concentration).shape[-1:]

    @property
    def batch_shape(self):
        n = resolve(self.total_count)
        return torch.broadcast_shapes(resolve(self.concentration).shape[:-1], tuple(n.shape))

    def _support(self, x):
        x = torch.as_tensor(x)
        n = torch.as_tensor(resolve(self.total_count), device=x.device)
        inside = ((x >= 0) & _is_integer(x)).all(-1) & (x.sum(-1) == n)
        return inside, torch.where(inside.unsqueeze(-1), x, torch.zeros_like(x))

    def log_pdf(self, x):
        inside, x = self._support(x)
        alpha = _as_float(self.concentration)
        a0 = alpha.sum(-1)
        n = x.sum(-1)
        lg = torch.lgamma
        log_p = lg(n + 1) + lg(a0) - lg(n + a0) + (lg(x + alpha) - lg(x + 1) - lg(alpha)).sum(-1)
        return _restrict(log_p, inside)

    def log_pdf_grad(self, x):
        raise _discrete_grad_error()

    def log_pdf_param_grads(self, x):
        inside, x = self._support(x)
        alpha = _as_float(self.concentration)
        a0 = alpha.sum(-1, keepdim=True)
        n = x.sum(-1, keepdim=True)
        dg = torch.digamma
        grad = dg(a0) - dg(n + a0) + dg(x + alpha) - dg(alpha)
        return {self.resolve_name("concentration", self.concentration): _zero_outside(grad, inside.unsqueeze(-1))}

    def _n(self, like):
        n = _as_float(self.total_count).to(like.dtype)
        return n.unsqueeze(-1) if n.ndim else n

    def mean(self):
        alpha = _as_float(self.concentration)
        return self._n(alpha) * alpha / alpha.sum(-1, keepdim=True)

    def variance(self):
        alpha = _as_float(self.concentration)
        a0 = alpha.sum(-1, keepdim=True)
        p = alpha / a0
        n = self._n(alpha)
        return n * p * (1 - p) * (n + a0) / (1 + a0)

    def covariance(self):
        """``n (diag(p) - p p^T) (n + a0) / (1 + a0)``, shape ``(*batch_shape, K, K)``."""
        alpha = _as_float(self.concentration)
        a0 = alpha.sum(-1)[..., None, None]
        p = alpha / alpha.sum(-1, keepdim=True)
        n = self._n(alpha).unsqueeze(-1)
        return n * (torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)) * (n + a0) / (1 + a0)

    def sample(self, n_samples=1, _reparametrization_trick_grad=False):
        if _reparametrization_trick_grad:
            raise self._no_reparam()
        alpha = _as_float(self.concentration)
        shape = (n_samples, *self.batch_shape, alpha.shape[-1])
        g = _standard_gamma(alpha.expand(shape).contiguous())
        p = g / g.sum(-1, keepdim=True)
        n = resolve(self.total_count).to(p.dtype)
        # one multinomial draw per sampled probability vector (the sample dimension acts as a batch dimension)
        return Multinomial(n, probs=p).sample(1)[0]
