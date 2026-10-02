"""Valores esperados exactos (a densidad finita) para sprinklings de intervalos planos.

Resultado central (derivación en el README, sec. Goal 2): sea x el vértice
superior de un intervalo causal plano de volumen Lambda (en unidades de 1/rho).
Para cualquier función h de n(y, x), por las propiedades del proceso de
Poisson,

    E[ sum_{y ≺ x} h(n(y, x)) ] = W[H](Lambda),   H(z) = E[h(Poisson(z))],

    W[H](Lambda) = (Omega_{d-2} / (d c_d)) int_0^Lambda H(z) K_d(Lambda / z) dz,

donde c_d tau^d es el volumen de un intervalo de tiempo propio tau,
Omega_{d-2} el área de la esfera S^{d-2} y K_d(u) = int_0^{ln(u)/d} sinh^{d-2}.
Explícitamente:

    d = 2:  W[H] = int_0^Lambda H(z) ln(Lambda / z) dz
    d = 4:  W[H] = 3 int_0^Lambda H(z) [sqrt(Lambda/z) - sqrt(z/Lambda) - ln(Lambda/z)] dz

(se obtiene parametrizando y = x + tau (-cosh chi, sinh chi n) e integrando
el ángulo hiperbólico chi hasta el borde pasado del intervalo). La misma
distribución de volúmenes describe los puntos x de un diamante según el
volumen de I(p, x), lo que da las abundancias esperadas del diamante entero:

    E[N_k] = (Omega_{d-2} / (d c_d)) int_0^Lambda K_d(Lambda / lam) W[p_k](lam) d lam,

con p_k(z) = z^k e^{-z} / k!.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln

from .action import coefficients

# Por encima de este z, e^{-z} z^k es despreciable para las k que usamos.
_Z_CUT = 200.0


def poisson_pmf(k: int, z):
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore"):
        logp = np.where(z > 0, k * np.log(np.where(z > 0, z, 1)) - z - gammaln(k + 1), -np.inf)
    return np.where((z == 0) & (k == 0), 1.0, np.exp(logp))


def layer_mean(z, dim: int, eps: float = 1.0):
    """H(z) = E[eps * f_d(n, eps)] con n ~ Poisson(z) = eps * sum_k C_k p_k(eps z)."""
    return eps * sum(c * poisson_pmf(k, eps * np.asarray(z, float)) for k, c in enumerate(coefficients(dim).layers))


def _kernel(dim: int, u):
    """Omega_{d-2}/(d c_d) * K_d(u), u = Lambda / z >= 1."""
    u = np.asarray(u, dtype=float)
    if dim == 2:
        return np.log(u)
    if dim == 4:
        s = np.sqrt(u)
        return 3.0 * (s - 1.0 / s - np.log(u))
    raise ValueError("núcleo implementado sólo para d = 2 y d = 4")


def past_weight(H, dim: int, lam: float, decay: float = 1.0) -> float:
    """W[H](lam) para un intervalo de volumen ``lam`` (unidades de 1/rho).

    Se integra en u = sqrt(z) para suavizar la singularidad integrable en z = 0.
    H debe decaer como e^{-decay * z} (se corta en z = 200 / decay).
    """
    upper = np.sqrt(min(lam, _Z_CUT / decay))

    def integrand(u):
        z = u * u
        return H(z) * _kernel(dim, lam / z) * 2 * u if z > 0 else 0.0

    val, _ = quad(integrand, 0.0, upper, limit=400, epsabs=1e-13, epsrel=1e-10)
    return float(val)


def expected_bd_bracket_at_top(dim: int, lam: float, eps: float = 1.0) -> float:
    """E[ l_eps^2 (B_eps 1)(x) ] / alpha_d en el vértice superior de un intervalo
    plano de volumen ``lam`` (unidades de 1/rho). Tiende a 0 cuando eps*lam -> inf."""
    c = coefficients(dim)
    return c.beta + past_weight(lambda z: layer_mean(z, dim, eps), dim, lam, decay=eps)


def expected_bd_on_constant_at_top(dim: int, density: float, volume: float, eps: float = 1.0) -> float:
    """E[(B_eps 1)(x)] en el vértice superior de un intervalo plano."""
    c = coefficients(dim)
    lsq = (eps * density) ** (-2.0 / dim)
    return c.alpha * expected_bd_bracket_at_top(dim, density * volume, eps) / lsq


def _diamond_pair_integral(H, dim: int, lam: float, decay: float = 1.0) -> float:
    """E[ sum_{pares y ≺ x} H(n(y,x)) ] en un diamante de volumen ``lam``."""

    def outer(u):
        v = u * u
        if v == 0:
            return 0.0
        return _kernel(dim, lam / v) * past_weight(H, dim, v, decay) * 2 * u

    val, _ = quad(outer, 0.0, np.sqrt(lam), limit=400, epsabs=1e-10, epsrel=1e-9)
    return float(val)


def expected_abundance(dim: int, lam: float, k: int) -> float:
    """E[N_k] para un sprinkling de Poisson de un diamante plano con rho V = lam."""
    return _diamond_pair_integral(lambda z: poisson_pmf(k, z), dim, lam)


def _cone_position_integral(mu, lam: float):
    """I(mu) = ∫ (c - a - k/a)^2 da / a sobre a- <= a <= a+ (raíces de
    a^2 - c a + k = 0), con c = 2 sqrt(lam), k = 2 mu: rango de rapidez
    de los pares de volumen mu en un cono truncado 2D de volumen lam (rho = 1).
    Primitiva: (c^2 + 2k) ln a + a^2/2 - k^2/(2a^2) - 2ca + 2ck/a."""
    mu = np.asarray(mu, dtype=float)
    c = 2.0 * np.sqrt(lam)
    k = 2.0 * mu
    disc = np.sqrt(np.maximum(c * c - 4 * k, 0.0))
    ap = (c + disc) / 2
    am = np.where(k > 0, 2 * k / np.where(c + disc > 0, c + disc, 1.0), 1.0)  # = (c - disc)/2 sin cancelación

    def F(a):
        return a * a / 2 - k * k / (2 * a * a) - 2 * c * a + 2 * c * k / a

    return np.where(k > 0, (c * c + 2 * k) * np.log(ap / am) + F(ap) - F(am), np.inf)


def cone_pair_integral_2d(H, lam: float, decay: float = 1.0) -> float:
    """E[ sum_{pares y ≺ x} H(rho V(y, x)) ] en el cono de luz pasado truncado
    J^-(q) ∩ {t >= 0} de Minkowski 2D con rho V = lam (``TruncatedPastCone``).

    Derivación (punto 1a, ver README): en coordenadas de cono de luz
    u = t + x, v = t - x la región es {u, v <= h, u + v >= 0} (V = h^2) y un
    par con lados Δu = a, Δv = b tiene sus extremos dentro si y sólo si
    el vértice inferior está en un triángulo de área física (2h - a - b)^2 / 4.
    Con medida de pares da db / 2 y rho = 1:
        E[sum h] = (1/8) ∫∫_{a + b <= 2h} (2h - a - b)^2 H(ab/2) da db
                 = (1/4) ∫_0^{lam/2} H(mu) I(mu) dmu      (b = 2 mu / a),
    con I(mu) de ``_cone_position_integral``. A mu fijo y lam -> ∞,
    I(mu) = 4 lam ln(lam / mu) + O(lam): el mismo rango de rapidez
    logarítmico que en el diamante, y por tanto el mismo coeficiente
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
    """E[Ñ_k] en el cono truncado 2D con rho V = lam (Ñ_k = N_k con eps = 1):
    H(mu) = P_k(eps mu) (adelgazamiento de Poisson)."""
    return cone_pair_integral_2d(lambda mu: float(poisson_pmf(k, eps * mu)), lam, decay=eps)


def expected_bd_action_diamond(dim: int, lam: float, eps: float = 1.0) -> float:
    """E[S_eps] para un sprinkling de Poisson de un diamante plano con rho V = lam."""
    c = coefficients(dim)
    weighted = _diamond_pair_integral(lambda z: layer_mean(z, dim, eps), dim, lam, decay=eps)
    return -c.action_scale * eps ** (2.0 / dim) * c.alpha * (c.beta * lam + weighted)


# ---------------------------------------------------------------------------
# de Sitter
# ---------------------------------------------------------------------------


def _gauss_panels(lo: float, hi: float, n_panels: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodos y pesos de Gauss-Legendre compuesta en [lo, hi]."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(lo, hi, n_panels + 1)
    half = np.diff(edges)[:, None] / 2
    mid = (edges[:-1] + edges[1:])[:, None] / 2
    return (mid + half * x).ravel(), (half * w).ravel()


def expected_bd_bracket_de_sitter(
    dim: int, hubble: float, density: float, h_tau_px: float, eps: float = 1.0, resolution: int = 1
) -> float:
    """E[ l_eps^2 (B_eps 1)(x) ] / alpha_d en de Sitter, cuando el pasado
    sprinkleado de x es el intervalo I(p, x) con tiempo propio tau(p, x).

    Fórmula de Mecke para el proceso de Poisson:

        E[bracket] = beta + int_{I(p,x)} rho dV_y  H(rho V(y, x))

    con H = ``layer_mean``. Por simetría máxima sólo depende de tau(p, x), así
    que se evalúa con p, x sobre el eje: x = (-1/H, 0), p = (-e^{H tau}/H, 0).

    Coordenadas: eta (tiempo conforme de y) y w = H tau(y, x). A eta fijo,
    r^2 = eta^2 + eta_x^2 - 2 eta eta_x cosh(w), y
    dV = a(eta)^d Omega_{d-2} r^{d-3} eta eta_x sinh(w) dw d eta.
    En w sólo se integra la capa w <= w_cut con eps rho V(w_cut) = 200, junto
    al cono de luz pasado de x (fuera, H es despreciable). Para absorber la
    singularidad integrable r^{d-3} en r = 0 (d = 2) se integra en
    s = sqrt(w0 - w), con w0 = H tau sobre el eje (r = 0).
    Cuadratura de Gauss-Legendre compuesta; ``resolution`` multiplica el
    número de paneles (para comprobar la convergencia).
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

    # Variable exterior w0 = ln(eta / eta_x) = H tau sobre el eje, en [0, W].
    # El integrando tiene esquinas donde w0 = w_cut (capa que alcanza x),
    # en eta_mid (r_max pasa de eta_x - eta a eta - eta_p) y donde
    # w_lo = w_cut (la capa deja de caber sobre la frontera pasada): se
    # usan como bordes de segmento.
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
    w0 = np.arccosh((eta**2 + eta_x**2) / (2 * prod))  # w sobre el eje (r = 0)
    r_max = np.minimum(eta_x - eta, eta - eta_p)
    w_lo = np.arccosh(np.maximum((eta**2 + eta_x**2 - r_max**2) / (2 * prod), 1.0))
    w_hi = np.minimum(w0, w_cut)
    valid = w_hi > w_lo

    # s = sqrt(w0 - w): s en [sqrt(w0 - w_hi), sqrt(w0 - w_lo)]
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
    """E[(B_eps 1)(x)] en de Sitter (ver ``expected_bd_bracket_de_sitter``).
    En el límite rho -> infinito tiende a -R/2 = -d(d-1)H^2/2."""
    c = coefficients(dim)
    lsq = (eps * density) ** (-2.0 / dim)
    return c.alpha * expected_bd_bracket_de_sitter(dim, hubble, density, h_tau_px, eps, resolution) / lsq
