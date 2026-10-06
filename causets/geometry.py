"""Geometrías para el cálculo de fluctuaciones: todo lo que el formalismo de
``fluctuations.py`` necesita saber de la variedad M (la región sprinkleada).

Una geometría expone, en coordenadas en las que la estructura causal es la
de Minkowski (cartesianas en Minkowski, conformes en métricas a(eta)^2 eta):

  dim, volume                         volumen físico de M
  coordinate_volume                   volumen coordenado de M (= volume en
                                      Minkowski)
  sample_M(n, rng)                    n puntos de M distribuidos según dV físico
  contains(z)                         z en M
  weight(z)                           densidad física por unidad de volumen
                                      coordenado (a^d; 1 en Minkowski)
  precedes(a, b)                      a ≺ b (fila a fila)
  interval_volume(y, x)               volumen físico de I(y, x) (0 si y no ≺ x)
  overlap_volume(y1, x1, y2, x2)      |I(y1,x1) ∩ I(y2,x2)| si hay forma
                                      cerrada; si no, None
  sample_in_intervals(y, x, k, rng)   k[m] puntos en I(y[m], x[m]) (dV físico)

La relación causal de ``causal.py`` sirve para todas (métricas conformemente
planas). Los volúmenes de intervalos son exactos; la intersección de dos
intervalos sólo tiene forma cerrada en 2D (coordenadas de cono de luz, donde
los intervalos son rectángulos); en Minkowski 4D se calcula con una
cuadratura determinista (``overlap.py``).
"""

from __future__ import annotations

import numpy as np

from .regions import (
    CausalDiamond,
    ConformalDiamond,
    ConformalPastCone,
    TruncatedPastCone,
    DeSitterIntervalVolume,
    de_sitter_invariant,
    de_sitter_scale_factor,
    diamond_volume,
    power_law_scale_factor,
)
from .sprinkling import sample_uniform


def minkowski_precedes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a ≺ b estrictamente (fila a fila, con broadcasting)."""
    dt = b[..., 0] - a[..., 0]
    dx2 = np.sum((b[..., 1:] - a[..., 1:]) ** 2, axis=-1)
    return (dt > 0) & (dt * dt >= dx2)


def proper_time(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Tiempo propio de Minkowski entre y y x (0 si no hay separación temporal)."""
    dt = x[..., 0] - y[..., 0]
    dx2 = np.sum((x[..., 1:] - y[..., 1:]) ** 2, axis=-1)
    return np.sqrt(np.maximum(dt * dt - dx2, 0.0)) * (dt > 0)


def _boost_from_rest(points: np.ndarray, delta: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Aplica a ``points`` (en el sistema en reposo, eje temporal) el boost que
    lleva (tau, 0) a ``delta`` (vector temporal futuro). Fila a fila."""
    gamma = delta[:, 0] / tau
    u = delta[:, 1:] / delta[:, :1]  # velocidad
    u2 = np.sum(u * u, axis=1)
    t_p, x_p = points[:, 0], points[:, 1:]
    ux = np.sum(u * x_p, axis=1)
    t = gamma * (t_p + ux)
    with np.errstate(invalid="ignore", divide="ignore"):
        coef = np.where(u2 > 0, (gamma - 1) * ux / u2, 0.0) + gamma * t_p
    x = x_p + coef[:, None] * u
    return np.column_stack([t, x])


def _sample_unit_diamond(dim: int, n: int, rng) -> np.ndarray:
    """n puntos uniformes en el diamante con vértices (0, 0) y (1, 0)."""
    pts = sample_uniform(CausalDiamond(dim, 0.5), n, rng)
    pts[:, 0] += 0.5
    return rng.permutation(pts)


class _ConformallyFlatGeometry:
    dim: int
    region: object

    @property
    def volume(self) -> float:
        return self.region.volume

    def sample_M(self, n: int, rng) -> np.ndarray:
        return sample_uniform(self.region, n, rng)

    def contains(self, z: np.ndarray) -> np.ndarray:
        return self.region.contains(z)

    def precedes(self, a, b):
        return minkowski_precedes(a, b)

    def overlap_volume(self, y1, x1, y2, x2):
        return None

    def _sample_coordinate_uniform(self, y, x, k, rng):
        """Puntos uniformes (medida coordenada) en los intervalos coordenados."""
        k = np.asarray(k, dtype=int)
        total = int(k.sum())
        if total == 0:
            return np.empty((0, self.dim)), np.empty(0, dtype=int)
        owner = np.repeat(np.arange(len(k)), k)
        unit = _sample_unit_diamond(self.dim, total, rng)
        delta = x[owner] - y[owner]
        tau = proper_time(y[owner], x[owner])
        pts = _boost_from_rest(unit * tau[:, None], delta, tau) + y[owner]
        return pts, owner

    def _sample_conformal_weighted(self, y, x, k, rng):
        """Puntos según dV físico en métricas a(eta)^2 eta con a creciente:
        rechazo con peso (a / a_max)^d dentro de cada intervalo (a_max está en
        el vértice superior). Requiere ``self.a``."""
        k = np.asarray(k, dtype=int)
        out_pts, out_owner = [], []
        need = k.copy()
        while need.sum() > 0:
            pts, owner = self._sample_coordinate_uniform(y, x, 2 * need + 2 * (need > 0), rng)
            accept = rng.random(len(pts)) < (self.a(pts[:, 0]) / self.a(x[owner, 0])) ** self.dim
            pts, owner = pts[accept], owner[accept]
            # quedarse con como mucho need[m] puntos por intervalo
            order = np.argsort(owner, kind="stable")
            pts, owner = pts[order], owner[order]
            rank = np.arange(len(owner)) - np.searchsorted(owner, owner)
            keep = rank < need[owner]
            out_pts.append(pts[keep])
            out_owner.append(owner[keep])
            need -= np.bincount(owner[keep], minlength=len(k))
        pts, owner = np.concatenate(out_pts), np.concatenate(out_owner)
        order = np.argsort(owner, kind="stable")
        return pts[order], owner[order]


class _MinkowskiGeometry(_ConformallyFlatGeometry):
    """Región de Minkowski causalmente convexa (``region``), d = 2, 3, 4."""

    overlap_nodes: int = 32  # nodos de la cuadratura 4D en t y en z

    @property
    def coordinate_volume(self) -> float:
        return self.volume

    def weight(self, z):
        return np.ones(len(np.atleast_2d(z)))

    def interval_volume(self, y, x):
        return diamond_volume(self.dim, proper_time(y, x) / 2)

    def overlap_volume(self, y1, x1, y2, x2):
        """2D: forma cerrada. 4D: cuadratura determinista (``overlap.py``)."""
        if self.dim == 4:
            from .overlap import interval_intersection_volume_4d

            n = self.overlap_nodes
            return interval_intersection_volume_4d(y1, x1, y2, x2, n, n)
        if self.dim != 2:
            return None
        # coordenadas de cono de luz u = t + x, v = t - x;  dV = du dv / 2
        def uv(p):
            return p[..., 0] + p[..., 1], p[..., 0] - p[..., 1]

        (yu1, yv1), (xu1, xv1), (yu2, yv2), (xu2, xv2) = uv(y1), uv(x1), uv(y2), uv(x2)
        du = np.minimum(xu1, xu2) - np.maximum(yu1, yu2)
        dv = np.minimum(xv1, xv2) - np.maximum(yv1, yv2)
        return np.where((du > 0) & (dv > 0), du * dv / 2, 0.0)

    def sample_in_intervals(self, y, x, k, rng):
        return self._sample_coordinate_uniform(y, x, k, rng)


class MinkowskiDiamondGeometry(_MinkowskiGeometry):
    """Diamante causal de Minkowski (``CausalDiamond``), d = 2, 3, 4."""

    def __init__(self, dim: int, half_height: float = 1.0, overlap_nodes: int = 32):
        self.dim = dim
        self.region = CausalDiamond(dim, half_height)
        self.overlap_nodes = overlap_nodes


class MinkowskiConeGeometry(_MinkowskiGeometry):
    """Cono de luz pasado truncado de Minkowski (``TruncatedPastCone``):
    J^-(q) ∩ {t >= 0}, d = 2, 3, 4. Al ser causalmente convexo, los
    intervalos de pares de la región están contenidos en ella y todo lo
    demás es igual que en el diamante."""

    def __init__(self, dim: int, height: float = 1.0, overlap_nodes: int = 32):
        self.dim = dim
        self.region = TruncatedPastCone(dim, height)
        self.overlap_nodes = overlap_nodes


class DeSitterGeometry(_ConformallyFlatGeometry):
    """Diamante en coordenadas conformes del parche plano de de Sitter,
    a(eta) = -1/(H eta). Volúmenes de intervalos exactos por simetría máxima
    (``DeSitterIntervalVolume``); intersecciones sin forma cerrada."""

    def __init__(self, dim: int, eta_bottom: float, eta_top: float, hubble: float = 1.0):
        if not eta_bottom < eta_top < 0:
            raise ValueError("se requiere eta_bottom < eta_top < 0")
        self.dim, self.hubble = dim, hubble
        self.a = de_sitter_scale_factor(hubble)
        self.region = ConformalDiamond(dim, (eta_top - eta_bottom) / 2, (eta_top + eta_bottom) / 2, self.a)
        self._vol = DeSitterIntervalVolume(hubble, dim, hubble * np.log(eta_bottom / eta_top) * 1.001 + 1e-9)

    @property
    def coordinate_volume(self) -> float:
        return diamond_volume(self.dim, self.region.half_height)

    def weight(self, z):
        # fuera del parche (eta >= 0) el punto no está en M: peso 0
        eta = np.atleast_2d(z)[:, 0]
        inside = eta < 0
        return np.where(inside, self.a(np.where(inside, eta, -1.0)) ** self.dim, 0.0)

    def interval_volume(self, y, x):
        """0 si y no ≺ x o si algún punto está fuera del parche (eta >= 0) o
        el par excede el tiempo propio máximo dentro de M (puntos propuestos
        fuera de M, que el integrador descarta)."""
        related = minkowski_precedes(y, x) & (np.atleast_2d(y)[:, 0] < 0) & (np.atleast_2d(x)[:, 0] < 0)
        Z = np.where(related, de_sitter_invariant(y, x), 1.0)
        h_tau = np.arccosh(np.maximum(Z, 1.0))
        related &= h_tau <= self._vol._grid[-1]
        return np.where(related, self._vol.from_h_tau(np.where(related, h_tau, 0.0)), 0.0)

    def sample_in_intervals(self, y, x, k, rng):
        return self._sample_conformal_weighted(y, x, k, rng)


def power_law_rectangle_volume(s0, du, dv, p: int) -> np.ndarray:
    """∫∫ ((u + v)/2)^p du dv / 2 sobre el rectángulo [u0, u0 + du] x
    [v0, v0 + dv], con s0 = u0 + v0 >= 0 y p entero >= 0: volumen físico de
    un intervalo (o de la intersección de dos) en el FLRW 2D a(eta) = eta^(p/2).

    Con t = u - u0, r = v - v0 y el binomio de Newton,
        ∫∫ (s0 + t + r)^p = sum_k C(p, k) s0^(p-k) ∫_0^du ∫_0^dv (t + r)^k,
        ∫∫ (t + r)^k = sum_{j=1}^{k+1} C(k+2, j) du^j dv^(k+2-j) / ((k+1)(k+2)):
    todos los términos son >= 0, así que no hay cancelaciones (exacto también
    para intervalos casi nulos)."""
    from math import comb

    s0, du, dv = (np.asarray(v, dtype=float) for v in (s0, du, dv))
    total = np.zeros(np.broadcast(s0, du, dv).shape)
    for k in range(p + 1):
        inner = sum(comb(k + 2, j) * du**j * dv ** (k + 2 - j) for j in range(1, k + 2)) / ((k + 1) * (k + 2))
        total = total + comb(p, k) * s0 ** (p - k) * inner
    return total / 2.0 ** (p + 1)


def _lens_volume(r1, r2, d):
    """Volumen de la intersección de dos bolas de R^3 de radios r1, r2 cuyos
    centros distan d (con d <= r1 + r2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lens = np.pi * (r1 + r2 - d) ** 2 * (d * d + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2) / (12 * d)
    small = 4 * np.pi / 3 * np.minimum(r1, r2) ** 3
    return np.where(d <= np.abs(r1 - r2), small, np.where(d > 0, lens, small))


def conformal_interval_volume_4d(y, x, a, nodes: int = 8) -> np.ndarray:
    """Volumen físico ∫ a(eta)^4 d^4x de I(y, x) en una métrica a(eta)^2 eta
    (Minkowski 4D en coordenadas conformes), fila a fila; 0 si y no ≺ x.

    A eta fijo, la sección de I(y, x) es la intersección de las bolas de
    radios r1 = eta - eta_y y r2 = eta_x - eta centradas en x_y y x_x
    (``_lens_volume``). Se integra en eta con Gauss-Legendre en los tres
    tramos separados por los puntos donde una bola contiene a la otra,
    eta = (eta_y + eta_x ∓ d)/2. En cada tramo la sección es un polinomio en
    eta (grado <= 4), así que con a^4 polinómico de grado <= 2 nodes - 5 el
    resultado es exacto."""
    y, x = np.atleast_2d(y), np.atleast_2d(x)
    ey, ex = y[:, 0], x[:, 0]
    d = np.linalg.norm(x[:, 1:] - y[:, 1:], axis=1)
    related = ex - ey > d
    gx, gw = np.polynomial.legendre.leggauss(nodes)
    mid_lo, mid_hi = (ey + ex - d) / 2, (ey + ex + d) / 2
    total = np.zeros(len(y))
    for lo, hi in ((ey, mid_lo), (mid_lo, mid_hi), (mid_hi, ex)):
        half = (hi - lo) / 2
        eta = (lo + hi)[:, None] / 2 + half[:, None] * gx[None, :]
        sec = _lens_volume(eta - ey[:, None], ex[:, None] - eta, d[:, None])
        total += half * np.sum(gw * a(eta) ** 4 * sec, axis=1)
    return np.where(related, total, 0.0)


class FLRWConeGeometry(_ConformallyFlatGeometry):
    """Cono de luz pasado hasta el big bang en un FLRW plano con
    a(eta) = eta^alpha (``ConformalPastCone``), d = 2 o 4.

    2D (2 alpha entero): en coordenadas de cono de luz u = eta + x,
    v = eta - x los intervalos y sus intersecciones son rectángulos y sus
    volúmenes físicos tienen forma cerrada (``power_law_rectangle_volume``).
    4D: volúmenes de intervalos por ``conformal_interval_volume_4d`` (exacto
    para alpha entero <= 2); intersecciones sin forma cerrada (sólo se usan
    las medias de abundancias). Con alpha = 0 se recupera Minkowski."""

    local_proposal = True  # escala de la propuesta según a(eta)^d local (``fluctuations._local``)

    def __init__(self, alpha: float, height: float = 1.0, dim: int = 2):
        if dim == 2 and (abs(2 * alpha - round(2 * alpha)) > 1e-12 or alpha < 0):
            raise ValueError("en 2D se requiere 2 alpha entero >= 0")
        if dim == 4 and (alpha not in (0, 1, 2)):
            raise ValueError("en 4D se requiere alpha = 0, 1 o 2 (volúmenes exactos)")
        if dim not in (2, 4):
            raise ValueError("dim debe ser 2 o 4")
        self.dim, self.alpha, self._p = dim, alpha, int(round(2 * alpha))
        self.a = power_law_scale_factor(alpha)
        self.region = ConformalPastCone(dim, height, self.a)

    @property
    def coordinate_volume(self) -> float:
        return TruncatedPastCone(self.dim, self.region.height).volume

    def sample_M(self, n: int, rng) -> np.ndarray:
        """n puntos según dV físico, sin rechazo: la densidad marginal de eta es
        ∝ eta^(d alpha) (h - eta)^(d-1), es decir eta = h · Beta(d alpha + 1, d),
        y a eta fijo el punto es uniforme en la bola de radio h - eta. (El
        rechazo genérico de ``sample_uniform`` acepta ~V/(caja·a_max^d), del
        orden de 1e-4 con alpha = 2 en 4D: inviable para millones de puntos.)
        En 2D se mantiene el rechazo genérico, con el que se calcularon los
        resultados del punto 1b (reproducibilidad bit a bit)."""
        if self.dim == 2:
            return super().sample_M(n, rng)
        h, d = self.region.height, self.dim
        eta = h * rng.beta(d * self.alpha + 1, d, size=n)
        u = rng.normal(size=(n, d - 1))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        r = (h - eta) * rng.random(n) ** (1.0 / (d - 1))
        pts = np.column_stack([eta, r[:, None] * u])
        return pts[np.argsort(eta, kind="stable")]

    def weight(self, z):
        return self.a(np.atleast_2d(z)[:, 0]) ** self.dim

    @staticmethod
    def _uv(p):
        return p[..., 0] + p[..., 1], p[..., 0] - p[..., 1]

    def interval_volume(self, y, x):
        if self.dim == 4:
            y = np.atleast_2d(y)
            return np.where(y[:, 0] >= 0, conformal_interval_volume_4d(y, x, self.a), 0.0)
        (yu, yv), (xu, xv) = self._uv(np.atleast_2d(y)), self._uv(np.atleast_2d(x))
        du, dv = xu - yu, xv - yv
        ok = (du > 0) & (dv > 0) & (yu + yv >= 0)
        z = np.zeros_like(du)
        return np.where(ok, power_law_rectangle_volume(np.where(ok, yu + yv, 0.0), np.where(ok, du, z), np.where(ok, dv, z), self._p), 0.0)

    def overlap_volume(self, y1, x1, y2, x2):
        if self.dim != 2:
            return None
        (yu1, yv1), (xu1, xv1), (yu2, yv2), (xu2, xv2) = (self._uv(np.atleast_2d(p)) for p in (y1, x1, y2, x2))
        u0, v0 = np.maximum(yu1, yu2), np.maximum(yv1, yv2)
        du = np.minimum(xu1, xu2) - u0
        dv = np.minimum(xv1, xv2) - v0
        ok = (du > 0) & (dv > 0) & (u0 + v0 >= 0)
        z = np.zeros_like(du)
        return np.where(ok, power_law_rectangle_volume(np.where(ok, u0 + v0, 0.0), np.where(ok, du, z), np.where(ok, dv, z), self._p), 0.0)

    def sample_in_intervals(self, y, x, k, rng):
        return self._sample_conformal_weighted(y, x, k, rng)
