"""Exact expected values (at finite density) for sprinklings of flat intervals.

Central result (derivation in the README, sec. Goal 2): let x be the top
vertex of a flat causal interval of volume Lambda (in units of 1/rho).
For any function h of n(y, x), by the properties of the Poisson process,

    E[ sum_{y ≺ x} h(n(y, x)) ] = W[H](Lambda),   H(z) = E[h(Poisson(z))],

    W[H](Lambda) = (Omega_{d-2} / (d c_d)) int_0^Lambda H(z) K_d(Lambda / z) dz,

where c_d tau^d is the volume of an interval of proper time tau,
Omega_{d-2} the area of the sphere S^{d-2} and K_d(u) = int_0^{ln(u)/d} sinh^{d-2}.
Explicitly:

    d = 2:  W[H] = int_0^Lambda H(z) ln(Lambda / z) dz
    d = 4:  W[H] = 3 int_0^Lambda H(z) [sqrt(Lambda/z) - sqrt(z/Lambda) - ln(Lambda/z)] dz

(obtained by parametrising y = x + tau (-cosh chi, sinh chi n) and integrating
the hyperbolic angle chi up to the past edge of the interval). The same
distribution of volumes describes the points x of a diamond according to the
volume of I(p, x), which gives the expected abundances of the whole diamond:

    E[N_k] = (Omega_{d-2} / (d c_d)) int_0^Lambda K_d(Lambda / lam) W[p_k](lam) d lam,

with p_k(z) = z^k e^{-z} / k!.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln

from .action import coefficients

# Above this z, e^{-z} z^k is negligible for the k we use.
_Z_CUT = 200.0


def poisson_pmf(k: int, z):
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore"):
        logp = np.where(z > 0, k * np.log(np.where(z > 0, z, 1)) - z - gammaln(k + 1), -np.inf)
    return np.where((z == 0) & (k == 0), 1.0, np.exp(logp))


def layer_mean(z, dim: int, eps: float = 1.0):
    """H(z) = E[eps * f_d(n, eps)] with n ~ Poisson(z) = eps * sum_k C_k p_k(eps z)."""
    return eps * sum(c * poisson_pmf(k, eps * np.asarray(z, float)) for k, c in enumerate(coefficients(dim).layers))


def _kernel(dim: int, u):
    """Omega_{d-2}/(d c_d) * K_d(u), u = Lambda / z >= 1."""
    u = np.asarray(u, dtype=float)
    if dim == 2:
        return np.log(u)
    if dim == 4:
        s = np.sqrt(u)
        return 3.0 * (s - 1.0 / s - np.log(u))
    raise ValueError("kernel implemented only for d = 2 and d = 4")


def past_weight(H, dim: int, lam: float, decay: float = 1.0) -> float:
    """W[H](lam) for an interval of volume ``lam`` (units of 1/rho).

    Integrated in u = sqrt(z) to smooth the integrable singularity at z = 0.
    H must decay as e^{-decay * z} (cut off at z = 200 / decay).
    """
    upper = np.sqrt(min(lam, _Z_CUT / decay))

    def integrand(u):
        z = u * u
        return H(z) * _kernel(dim, lam / z) * 2 * u if z > 0 else 0.0

    val, _ = quad(integrand, 0.0, upper, limit=400, epsabs=1e-13, epsrel=1e-10)
    return float(val)


def expected_bd_bracket_at_top(dim: int, lam: float, eps: float = 1.0) -> float:
    """E[ l_eps^2 (B_eps 1)(x) ] / alpha_d at the top vertex of a flat interval
    of volume ``lam`` (units of 1/rho). Tends to 0 as eps*lam -> inf."""
    c = coefficients(dim)
    return c.beta + past_weight(lambda z: layer_mean(z, dim, eps), dim, lam, decay=eps)


def expected_bd_on_constant_at_top(dim: int, density: float, volume: float, eps: float = 1.0) -> float:
    """E[(B_eps 1)(x)] at the top vertex of a flat interval."""
    c = coefficients(dim)
    lsq = (eps * density) ** (-2.0 / dim)
    return c.alpha * expected_bd_bracket_at_top(dim, density * volume, eps) / lsq


def _diamond_pair_integral(H, dim: int, lam: float, decay: float = 1.0) -> float:
    """E[ sum_{pairs y ≺ x} H(n(y,x)) ] in a diamond of volume ``lam``."""

    def outer(u):
        v = u * u
        if v == 0:
            return 0.0
        return _kernel(dim, lam / v) * past_weight(H, dim, v, decay) * 2 * u

    val, _ = quad(outer, 0.0, np.sqrt(lam), limit=400, epsabs=1e-10, epsrel=1e-9)
    return float(val)


def expected_abundance(dim: int, lam: float, k: int) -> float:
    """E[N_k] for a Poisson sprinkling of a flat diamond with rho V = lam."""
    return _diamond_pair_integral(lambda z: poisson_pmf(k, z), dim, lam)


def _cone_position_integral(mu, lam: float):
    """I(mu) = ∫ (c - a - k/a)^2 da / a over a- <= a <= a+ (roots of
    a^2 - c a + k = 0), with c = 2 sqrt(lam), k = 2 mu: rapidity range of
    the pairs of volume mu in a 2D truncated cone of volume lam (rho = 1).
    Antiderivative: (c^2 + 2k) ln a + a^2/2 - k^2/(2a^2) - 2ca + 2ck/a."""
    mu = np.asarray(mu, dtype=float)
    c = 2.0 * np.sqrt(lam)
    k = 2.0 * mu
    disc = np.sqrt(np.maximum(c * c - 4 * k, 0.0))
    ap = (c + disc) / 2
    am = np.where(k > 0, 2 * k / np.where(c + disc > 0, c + disc, 1.0), 1.0)  # = (c - disc)/2 without cancellation

    def F(a):
        return a * a / 2 - k * k / (2 * a * a) - 2 * c * a + 2 * c * k / a

    return np.where(k > 0, (c * c + 2 * k) * np.log(ap / am) + F(ap) - F(am), np.inf)


def cone_pair_integral_2d(H, lam: float, decay: float = 1.0) -> float:
    """E[ sum_{pairs y ≺ x} H(rho V(y, x)) ] in the truncated past light cone
    J^-(q) ∩ {t >= 0} of 2D Minkowski with rho V = lam (``TruncatedPastCone``).

    Derivation (item 1a, see README): in light-cone coordinates
    u = t + x, v = t - x the region is {u, v <= h, u + v >= 0} (V = h^2) and a
    pair with sides Δu = a, Δv = b has both endpoints inside if and only if
    the bottom vertex lies in a triangle of physical area (2h - a - b)^2 / 4.
    With pair measure da db / 2 and rho = 1:
        E[sum h] = (1/8) ∫∫_{a + b <= 2h} (2h - a - b)^2 H(ab/2) da db
                 = (1/4) ∫_0^{lam/2} H(mu) I(mu) dmu      (b = 2 mu / a),
    with I(mu) from ``_cone_position_integral``. At fixed mu and lam -> ∞,
    I(mu) = 4 lam ln(lam / mu) + O(lam): the same logarithmic rapidity range
    as in the diamond, and therefore the same coefficient
    E[sum h] = lam ln(lam) ∫ H + O(lam)."""
    upper = min(lam / 2, _Z_CUT / decay)
    edges = np.unique(np.concatenate([[0.0], np.geomspace(1e-8, upper, 60)]))
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        val, _ = quad(lambda mu: H(mu) * float(_cone_position_integral(mu, lam)), lo, hi,
                      limit=200, epsabs=0, epsrel=1e-11)
        tot += val
    return tot / 4


def expected_abundance_cone(lam: float, k: int, eps: float = 1.0) -> float:
    """E[Ñ_k] in the 2D truncated cone with rho V = lam (Ñ_k = N_k for eps = 1):
    H(mu) = P_k(eps mu) (Poisson thinning)."""
    return cone_pair_integral_2d(lambda mu: float(poisson_pmf(k, eps * mu)), lam, decay=eps)


def cone_pair_integral(H, dim: int, lam: float, decay: float = 1.0) -> float:
    """E[ sum_{pairs y ≺ x} H(rho V(y, x)) ] in the truncated past cone
    J^-(q) ∩ {t >= 0} of Minkowski (``TruncatedPastCone``), d = 2 or 4, with
    rho V = lam (rho = 1).

    Derivation (item 1c, see README): the past of x is already in J^-(q), so
    y = x - tau (cosh chi, sinh chi n) lies in the cone if and only if
    tau cosh chi <= t_x. With d^d y = tau^(d-1) dtau sinh^(d-2)chi dchi dOmega,
        E[sum H] = Omega_{d-2} ∫_cone dV_x ∫_0^{t_x} tau^(d-1) S_d(arccosh(t_x/tau)) H(c_d tau^d) dtau,
    S_d(X) = ∫_0^X sinh^(d-2), and dV_x = V_{d-1}(1) (h - t)^(d-1) dt. It only
    depends on t_x: a double integral. In 2D it agrees with ``cone_pair_integral_2d``."""
    from .regions import TruncatedPastCone, unit_ball_volume, unit_sphere_area

    h = (lam / TruncatedPastCone(dim, 1.0).volume) ** (1.0 / dim)
    c = 2 * 0.5**dim * unit_ball_volume(dim - 1) / dim  # c_d: volume of the interval of unit proper time
    omega = unit_sphere_area(dim - 2) if dim > 2 else 2.0
    tau_cut = (_Z_CUT / decay / c) ** (1.0 / dim)

    def S(X):
        return X if dim == 2 else np.sinh(2 * X) / 4 - X / 2

    def inner(t):
        def f(tau):
            X = np.arccosh(max(t / tau, 1.0))
            return tau ** (dim - 1) * S(X) * H(c * tau**dim)

        val, _ = quad(f, 0.0, min(t, tau_cut), limit=200, epsabs=0, epsrel=1e-10)
        return val

    def outer(t):
        return unit_ball_volume(dim - 1) * (h - t) ** (dim - 1) * inner(t)

    # inner(t) grows as t^(d-2) for t >> tau_cut: the integral is smooth except near t = 0
    edges = np.unique(np.concatenate([[0.0], np.geomspace(min(1e-3, h / 10), h, 30)]))
    tot = sum(quad(outer, a, b, limit=200, epsabs=0, epsrel=1e-10)[0] for a, b in zip(edges[:-1], edges[1:]))
    return float(omega * tot)


def expected_abundance_cone_d(dim: int, lam: float, k: int, eps: float = 1.0) -> float:
    """E[Ñ_k] in the d-dimensional truncated cone with rho V = lam."""
    return cone_pair_integral(lambda mu: float(poisson_pmf(k, eps * mu)), dim, lam, decay=eps)


def expected_bd_action_diamond(dim: int, lam: float, eps: float = 1.0) -> float:
    """E[S_eps] for a Poisson sprinkling of a flat diamond with rho V = lam."""
    c = coefficients(dim)
    weighted = _diamond_pair_integral(lambda z: layer_mean(z, dim, eps), dim, lam, decay=eps)
    return -c.action_scale * eps ** (2.0 / dim) * c.alpha * (c.beta * lam + weighted)


# ---------------------------------------------------------------------------
# de Sitter
# ---------------------------------------------------------------------------


def _gauss_panels(lo: float, hi: float, n_panels: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of composite Gauss-Legendre quadrature on [lo, hi]."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(lo, hi, n_panels + 1)
    half = np.diff(edges)[:, None] / 2
    mid = (edges[:-1] + edges[1:])[:, None] / 2
    return (mid + half * x).ravel(), (half * w).ravel()


def expected_bd_bracket_de_sitter(
    dim: int, hubble: float, density: float, h_tau_px: float, eps: float = 1.0, resolution: int = 1
) -> float:
    """E[ l_eps^2 (B_eps 1)(x) ] / alpha_d in de Sitter, when the sprinkled
    past of x is the interval I(p, x) with proper time tau(p, x).

    Mecke formula for the Poisson process:

        E[bracket] = beta + int_{I(p,x)} rho dV_y  H(rho V(y, x))

    with H = ``layer_mean``. By maximal symmetry it only depends on tau(p, x),
    so it is evaluated with p, x on the axis: x = (-1/H, 0), p = (-e^{H tau}/H, 0).

    Coordinates: eta (conformal time of y) and w = H tau(y, x). At fixed eta,
    r^2 = eta^2 + eta_x^2 - 2 eta eta_x cosh(w), and
    dV = a(eta)^d Omega_{d-2} r^{d-3} eta eta_x sinh(w) dw d eta.
    In w only the layer w <= w_cut with eps rho V(w_cut) = 200 is integrated,
    next to the past light cone of x (outside it, H is negligible). To absorb
    the integrable singularity r^{d-3} at r = 0 (d = 2) one integrates in
    s = sqrt(w0 - w), with w0 = H tau on the axis (r = 0).
    Composite Gauss-Legendre quadrature; ``resolution`` multiplies the number
    of panels (to check convergence).
    """
    from .regions import DeSitterIntervalVolume, de_sitter_scale_factor, unit_sphere_area

    c = coefficients(dim)
    a = de_sitter_scale_factor(hubble)
    eta_x = -1.0 / hubble
    eta_p = -np.exp(hubble * h_tau_px) / hubble
    vol = DeSitterIntervalVolume(hubble, dim, hubble * h_tau_px * 1.0001 + 1e-9)
    omega = unit_sphere_area(dim - 2)

    grid = np.linspace(0, hubble * h_tau_px, 20001)
    z_vals = eps * density * vol.from_h_tau(grid)
    w_cut = grid[-1] if z_vals[-1] <= _Z_CUT else grid[np.searchsorted(z_vals, _Z_CUT)]

    # Outer variable w0 = ln(eta / eta_x) = H tau on the axis, in [0, W].
    # The integrand has corners where w0 = w_cut (layer reaching x),
    # at eta_mid (r_max switches from eta_x - eta to eta - eta_p) and where
    # w_lo = w_cut (the layer no longer fits over the past boundary): they
    # are used as segment edges.
    W = hubble * h_tau_px
    eta_mid = (eta_x + eta_p) / 2

    def w_lo_of(e):
        r_max = min(eta_x - e, e - eta_p)
        return np.arccosh(max((e * e + eta_x * eta_x - r_max * r_max) / (2 * e * eta_x), 1.0))

    breaks = {0.0, W, np.log(eta_mid / eta_x)}
    if w_cut < W:
        breaks.add(w_cut)
        if w_lo_of(eta_p * (1 - 1e-12)) > w_cut:
            e_b = brentq(lambda e: w_lo_of(e) - w_cut, eta_p * (1 - 1e-12), eta_mid)
            breaks.add(np.log(e_b / eta_x))
    breaks = np.array(sorted(b for b in breaks if 0 <= b <= W))
    nodes, weights = [], []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        if hi - lo > 1e-14:
            x_, w_ = _gauss_panels(lo, hi, 20 * resolution, 16)
            nodes.append(x_)
            weights.append(w_)
    w0_nodes = np.concatenate(nodes)
    eta = (eta_x * np.exp(w0_nodes))[:, None]
    w_eta = (np.concatenate(weights) * np.abs(eta[:, 0]))[:, None]  # d eta = |eta| d w0

    prod = eta * eta_x
    w0 = np.arccosh((eta**2 + eta_x**2) / (2 * prod))  # w on the axis (r = 0)
    r_max = np.minimum(eta_x - eta, eta - eta_p)
    w_lo = np.arccosh(np.maximum((eta**2 + eta_x**2 - r_max**2) / (2 * prod), 1.0))
    w_hi = np.minimum(w0, w_cut)
    valid = w_hi > w_lo

    # s = sqrt(w0 - w): s in [sqrt(w0 - w_hi), sqrt(w0 - w_lo)]
    s_lo = np.sqrt(np.maximum(w0 - w_hi, 0))
    s_hi = np.sqrt(np.maximum(w0 - w_lo, 0))
    u, wu = _gauss_panels(0.0, 1.0, 8 * resolution, 16)
    s = s_lo + (s_hi - s_lo) * u[None, :]
    ds = (s_hi - s_lo) * wu[None, :]
    w = w0 - s * s
    r2 = np.maximum(eta**2 + eta_x**2 - 2 * prod * np.cosh(w), 0)
    r = np.sqrt(r2)
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = omega * r ** (dim - 3) * prod * np.sinh(w) * 2 * s  # dV / (a^d d eta ds)
    jac = np.where(np.isfinite(jac), jac, 0.0)
    integrand = jac * layer_mean(density * vol.from_h_tau(w), dim, eps)
    inner = np.sum(integrand * ds, axis=1, keepdims=True)
    total = np.sum(np.where(valid, a(eta) ** dim * density * inner * w_eta, 0.0))
    return c.beta + float(total)


def expected_bd_on_constant_de_sitter(
    dim: int, hubble: float, density: float, h_tau_px: float, eps: float = 1.0, resolution: int = 1
) -> float:
    """E[(B_eps 1)(x)] in de Sitter (see ``expected_bd_bracket_de_sitter``).
    In the limit rho -> infinity it tends to -R/2 = -d(d-1)H^2/2."""
    c = coefficients(dim)
    lsq = (eps * density) ** (-2.0 / dim)
    return c.alpha * expected_bd_bracket_de_sitter(dim, hubble, density, h_tau_px, eps, resolution) / lsq
