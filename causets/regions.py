"""Spacetime regions in which the sprinkling is done.

Convention: coordinates ``(t, x_1, ..., x_{d-1})`` with c = 1; column 0 is
always the time (conformal time eta in conformally flat spacetimes).

Interface common to every region (used by ``sprinkling.sample_uniform``):
  - ``dim``, ``volume`` (physical volume, with the measure sqrt(-g))
  - ``contains(coords)``, ``bounding_box()``
  - ``weight(coords)`` in [0, 1]: relative density of physical volume with
    respect to the coordinate measure (1 in Minkowski; (a / a_max)^d in
    conformally flat metrics g = a^2 eta).
  - ``top_vertex``: the future vertex of the diamond (point at which the BD
    operator is evaluated when its whole past is sprinkled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def unit_ball_volume(n: int) -> float:
    """Volume of the unit ball in R^n: pi^(n/2) / Gamma(n/2 + 1)."""
    return float(np.pi ** (n / 2) / gamma(n / 2 + 1))


def unit_sphere_area(n: int) -> float:
    """Area of the unit sphere S^n (in R^(n+1)): 2 pi^((n+1)/2) / Gamma((n+1)/2)."""
    return float(2 * np.pi ** ((n + 1) / 2) / gamma((n + 1) / 2))


def diamond_volume(dim: int, half_height: float) -> float:
    """Volume of a flat causal diamond of half-height h (two cones)."""
    return 2.0 * half_height**dim * unit_ball_volume(dim - 1) / dim


@dataclass(frozen=True)
class CausalDiamond:
    """Causal diamond J+(p) ∩ J-(q) in d-dimensional Minkowski.

    The vertices are p = (-h, 0, ..., 0) and q = (+h, 0, ..., 0), with
    h = ``half_height``. A point is inside if |t| + |x| <= h.

    Volume: two cones of height h with a (d-1)-dimensional ball of radius h
    as base, V = 2 * (h / d) * V_{d-1}(1) * h^(d-1).
    """

    dim: int
    half_height: float = 1.0

    def __post_init__(self):
        if self.dim < 2:
            raise ValueError("dim must be >= 2")
        if self.half_height <= 0:
            raise ValueError("half_height must be > 0")

    @property
    def volume(self) -> float:
        return diamond_volume(self.dim, self.half_height)

    @property
    def top_vertex(self) -> np.ndarray:
        q = np.zeros(self.dim)
        q[0] = self.half_height
        return q

    def contains(self, coords: np.ndarray) -> np.ndarray:
        coords = np.atleast_2d(coords)
        t = coords[:, 0]
        r = np.linalg.norm(coords[:, 1:], axis=1)
        return np.abs(t) + r <= self.half_height

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        h = self.half_height
        return np.full(self.dim, -h), np.full(self.dim, h)

    def weight(self, coords: np.ndarray) -> np.ndarray:
        return np.ones(len(np.atleast_2d(coords)))


@dataclass(frozen=True)
class TruncatedPastCone:
    """Causal past of an observer cut off by an initial surface:
    J^-(q) ∩ {t >= 0} in d-dimensional Minkowski, with q = (h, 0, ..., 0) and
    h = ``height``. A point is inside if t >= 0 and t + |x| <= h.

    It is the flat analogue of the region used by Everpresent Λ (the causal
    past of the observer down to the initial surface). It is causally convex:
    the interval between two points of the region is contained in it.

    Volume: a cone of height h with a (d-1)-dimensional ball of radius h as
    base, V = (h / d) * V_{d-1}(1) * h^(d-1)  (V = h^2 in 2D).
    """

    dim: int
    height: float = 1.0

    def __post_init__(self):
        if self.dim < 2:
            raise ValueError("dim must be >= 2")
        if self.height <= 0:
            raise ValueError("height must be > 0")

    @property
    def volume(self) -> float:
        return self.height**self.dim * unit_ball_volume(self.dim - 1) / self.dim

    @property
    def top_vertex(self) -> np.ndarray:
        q = np.zeros(self.dim)
        q[0] = self.height
        return q

    def contains(self, coords: np.ndarray) -> np.ndarray:
        coords = np.atleast_2d(coords)
        t = coords[:, 0]
        r = np.linalg.norm(coords[:, 1:], axis=1)
        return (t >= 0) & (t + r <= self.height)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        h = self.height
        lo, hi = np.full(self.dim, -h), np.full(self.dim, h)
        lo[0] = 0.0
        return lo, hi

    def weight(self, coords: np.ndarray) -> np.ndarray:
        return np.ones(len(np.atleast_2d(coords)))


@dataclass(frozen=True)
class ConformalPastCone:
    """Past light cone of an observer down to the surface eta = 0, in
    conformal coordinates of a metric g = a(eta)^2 eta_Minkowski:
    {eta >= 0, eta + |x| <= h}. With a(0) = 0 it is the causal past of the
    observer down to the big bang of a flat FLRW (the Everpresent Λ
    region). The causal structure is that of Minkowski; the measure is
    dV = a(eta)^d d eta d^(d-1)x. ``scale_factor`` must be increasing."""

    dim: int
    height: float
    scale_factor: Callable[[np.ndarray], np.ndarray]

    @property
    def volume(self) -> float:
        d, h = self.dim, self.height
        vb = unit_ball_volume(d - 1)
        val, _ = quad(lambda eta: self.scale_factor(np.array(eta)) ** d * vb * (h - eta) ** (d - 1),
                      0.0, h, epsabs=0, epsrel=1e-12, limit=200)
        return float(val)

    @property
    def top_vertex(self) -> np.ndarray:
        q = np.zeros(self.dim)
        q[0] = self.height
        return q

    def contains(self, coords: np.ndarray) -> np.ndarray:
        return TruncatedPastCone(self.dim, self.height).contains(coords)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return TruncatedPastCone(self.dim, self.height).bounding_box()

    def weight(self, coords: np.ndarray) -> np.ndarray:
        eta = np.maximum(np.atleast_2d(coords)[:, 0], 0.0)
        return (self.scale_factor(eta) / self.scale_factor(np.array(self.height))) ** self.dim


def power_law_scale_factor(alpha: float) -> Callable[[np.ndarray], np.ndarray]:
    """a(eta) = eta^alpha (eta >= 0): flat FLRW with big bang at eta = 0.
    In 4D, alpha = 1 is radiation and alpha = 2 matter (a ∝ t^(1/2), t^(2/3))."""

    def a(eta):
        return np.maximum(np.asarray(eta, dtype=float), 0.0) ** alpha

    return a


@dataclass(frozen=True)
class ConformalDiamond:
    """Diamond in conformal coordinates of a metric g = a(eta)^2 eta_Minkowski.

    In coordinates (eta, x) the diamond is |eta - eta_c| + |x| <= h, and since
    the metric is conformal to Minkowski, its causal structure is that of
    Minkowski (the diamond is still J+(p) ∩ J-(q)). What changes is the
    volume measure: dV = a(eta)^d d eta d^(d-1)x.

    ``scale_factor`` must be positive and monotonic over the time range of the
    diamond (so that the maximum is at an endpoint).
    """

    dim: int
    half_height: float
    center_time: float
    scale_factor: Callable[[np.ndarray], np.ndarray]

    @property
    def _a_max(self) -> float:
        h, c = self.half_height, self.center_time
        return float(max(self.scale_factor(np.array(c - h)), self.scale_factor(np.array(c + h))))

    @property
    def volume(self) -> float:
        d, h, c = self.dim, self.half_height, self.center_time
        vb = unit_ball_volume(d - 1)

        def integrand(eta):
            return self.scale_factor(np.array(eta)) ** d * vb * (h - abs(eta - c)) ** (d - 1)

        val, _ = quad(integrand, c - h, c + h, points=[c], epsabs=0, epsrel=1e-11, limit=200)
        return float(val)

    @property
    def top_vertex(self) -> np.ndarray:
        q = np.zeros(self.dim)
        q[0] = self.center_time + self.half_height
        return q

    def contains(self, coords: np.ndarray) -> np.ndarray:
        coords = np.atleast_2d(coords)
        t = coords[:, 0] - self.center_time
        r = np.linalg.norm(coords[:, 1:], axis=1)
        return np.abs(t) + r <= self.half_height

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        h, c = self.half_height, self.center_time
        lo = np.full(self.dim, -h)
        hi = np.full(self.dim, h)
        lo[0], hi[0] = c - h, c + h
        return lo, hi

    def weight(self, coords: np.ndarray) -> np.ndarray:
        eta = np.atleast_2d(coords)[:, 0]
        return (self.scale_factor(eta) / self._a_max) ** self.dim


def de_sitter_scale_factor(hubble: float) -> Callable[[np.ndarray], np.ndarray]:
    """a(eta) = -1 / (H eta), eta < 0: flat (cosmological) patch of de Sitter.

    ds^2 = a(eta)^2 (-d eta^2 + dx^2), scalar curvature R = d(d-1) H^2
    (R = 2H^2 in 2D, R = 12H^2 in 4D).
    """

    def a(eta):
        eta = np.asarray(eta, dtype=float)
        if np.any(eta >= 0):
            raise ValueError("the flat patch of de Sitter requires eta < 0")
        return -1.0 / (hubble * eta)

    return a


class DeSitterIntervalVolume:
    """Physical volume of a de Sitter causal interval as a function of its
    proper time tau (or, equivalently, of the invariant Z = cosh(H tau)).

    De Sitter is maximally symmetric, so the volume of I(y, x) only depends
    on tau(y, x). It is computed for the equivalent pair on the axis,
    x' = (-1/H, 0), y' = (-e^{H tau}/H, 0), whose interval is a
    ``ConformalDiamond``, tabulated in H tau and interpolated in V^(1/d)
    (almost linear in tau for small intervals, where V ~ c_d tau^d).
    """

    def __init__(self, hubble: float, dim: int, max_h_tau: float, n_grid: int = 600):
        self.hubble, self.dim = hubble, dim
        a = de_sitter_scale_factor(hubble)
        self._grid = np.linspace(0.0, max_h_tau, n_grid)
        vols = [
            ConformalDiamond(dim, (np.exp(g) - 1) / (2 * hubble), -(np.exp(g) + 1) / (2 * hubble), a).volume
            if g > 0
            else 0.0
            for g in self._grid
        ]
        self._root = np.array(vols) ** (1.0 / dim)

    def from_h_tau(self, h_tau) -> np.ndarray:
        h_tau = np.asarray(h_tau, dtype=float)
        if np.any(h_tau > self._grid[-1] * (1 + 1e-12)):
            raise ValueError("H tau outside the tabulated range")
        return np.interp(h_tau, self._grid, self._root) ** self.dim

    def from_invariant(self, Z) -> np.ndarray:
        """V(Z) with Z = cosh(H tau) >= 1."""
        Z = np.maximum(np.asarray(Z, dtype=float), 1.0)
        return self.from_h_tau(np.arccosh(Z))


def de_sitter_invariant(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Z(y, x) = (eta_y^2 + eta_x^2 - |dx|^2) / (2 eta_y eta_x) in the flat patch.
    For timelike separated y, x, Z = cosh(H tau(y, x)) > 1."""
    y, x = np.atleast_2d(y), np.atleast_2d(x)
    dx2 = np.sum((y[:, 1:] - x[:, 1:]) ** 2, axis=1)
    return (y[:, 0] ** 2 + x[:, 0] ** 2 - dx2) / (2 * y[:, 0] * x[:, 0])


def de_sitter_interval_volume(past: np.ndarray, x: np.ndarray, hubble: float, dim: int) -> np.ndarray:
    """Physical volume of I(y, x) in de Sitter for each pair of rows (y, x)
    (``x`` may be a single point). Returns 0 if y does not precede x."""
    past = np.atleast_2d(past)
    x = np.broadcast_to(np.atleast_2d(x), past.shape)
    Z = de_sitter_invariant(past, x)
    related = (Z > 1) & (past[:, 0] < x[:, 0])
    h_tau = np.arccosh(np.where(related, Z, 1.0))
    vol = DeSitterIntervalVolume(hubble, dim, h_tau.max() * 1.001 + 1e-9)
    return np.where(related, vol.from_h_tau(h_tau), 0.0)
