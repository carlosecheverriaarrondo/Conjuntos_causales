"""Regiones del espacio-tiempo donde se hace el sprinkling.

Convención: coordenadas ``(t, x_1, ..., x_{d-1})`` con c = 1; la columna 0 es
siempre el tiempo (tiempo conforme eta en espaciotiempos conformemente planos).

Interfaz común a toda región (la usa ``sprinkling.sample_uniform``):
  - ``dim``, ``volume`` (volumen físico, con la medida sqrt(-g))
  - ``contains(coords)``, ``bounding_box()``
  - ``weight(coords)`` en [0, 1]: densidad relativa de volumen físico respecto
    a la medida de coordenadas (1 en Minkowski; (a / a_max)^d en métricas
    conformemente planas g = a^2 eta).
  - ``top_vertex``: el vértice futuro del diamante (punto en el que se evalúa
    el operador BD cuando todo su pasado está sprinkleado).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def unit_ball_volume(n: int) -> float:
    """Volumen de la bola unidad en R^n: pi^(n/2) / Gamma(n/2 + 1)."""
    return float(np.pi ** (n / 2) / gamma(n / 2 + 1))


def unit_sphere_area(n: int) -> float:
    """Área de la esfera unidad S^n (en R^(n+1)): 2 pi^((n+1)/2) / Gamma((n+1)/2)."""
    return float(2 * np.pi ** ((n + 1) / 2) / gamma((n + 1) / 2))


def diamond_volume(dim: int, half_height: float) -> float:
    """Volumen de un diamante causal plano de semialtura h (dos conos)."""
    return 2.0 * half_height**dim * unit_ball_volume(dim - 1) / dim


@dataclass(frozen=True)
class CausalDiamond:
    """Diamante causal J+(p) ∩ J-(q) en Minkowski d-dimensional.

    Los vértices son p = (-h, 0, ..., 0) y q = (+h, 0, ..., 0), con
    h = ``half_height``. Un punto está dentro si |t| + |x| <= h.

    Volumen: dos conos de altura h y base una bola (d-1)-dimensional de
    radio h, V = 2 * (h / d) * V_{d-1}(1) * h^(d-1).
    """

    dim: int
    half_height: float = 1.0

    def __post_init__(self):
        if self.dim < 2:
            raise ValueError("dim debe ser >= 2")
        if self.half_height <= 0:
            raise ValueError("half_height debe ser > 0")

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
class ConformalDiamond:
    """Diamante en coordenadas conformes de una métrica g = a(eta)^2 eta_Minkowski.

    En coordenadas (eta, x) el diamante es |eta - eta_c| + |x| <= h, y como la
    métrica es conforme a Minkowski, su estructura causal es la de Minkowski
    (el diamante sigue siendo J+(p) ∩ J-(q)). Lo que cambia es la medida de
    volumen: dV = a(eta)^d d eta d^(d-1)x.

    ``scale_factor`` debe ser positiva y monótona en el intervalo temporal del
    diamante (así el máximo está en un extremo).
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
    """a(eta) = -1 / (H eta), eta < 0: parche plano (cosmológico) de de Sitter.

    ds^2 = a(eta)^2 (-d eta^2 + dx^2), curvatura escalar R = d(d-1) H^2
    (R = 2H^2 en 2D, R = 12H^2 en 4D).
    """

    def a(eta):
        eta = np.asarray(eta, dtype=float)
        if np.any(eta >= 0):
            raise ValueError("el parche plano de de Sitter requiere eta < 0")
        return -1.0 / (hubble * eta)

    return a


class DeSitterIntervalVolume:
    """Volumen físico de un intervalo causal de de Sitter en función de su
    tiempo propio tau (o, equivalentemente, del invariante Z = cosh(H tau)).

    De Sitter es maximalmente simétrico, así que el volumen de I(y, x) sólo
    depende de tau(y, x). Se calcula en el par equivalente sobre el eje,
    x' = (-1/H, 0), y' = (-e^{H tau}/H, 0), cuyo intervalo es un
    ``ConformalDiamond``, se tabula en H tau y se interpola en V^(1/d) (casi
    lineal en tau para intervalos pequeños, donde V ~ c_d tau^d).
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
            raise ValueError("H tau fuera del rango tabulado")
        return np.interp(h_tau, self._grid, self._root) ** self.dim

    def from_invariant(self, Z) -> np.ndarray:
        """V(Z) con Z = cosh(H tau) >= 1."""
        Z = np.maximum(np.asarray(Z, dtype=float), 1.0)
        return self.from_h_tau(np.arccosh(Z))


def de_sitter_invariant(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Z(y, x) = (eta_y^2 + eta_x^2 - |dx|^2) / (2 eta_y eta_x) en el parche plano.
    Para y, x con separación temporal, Z = cosh(H tau(y, x)) > 1."""
    y, x = np.atleast_2d(y), np.atleast_2d(x)
    dx2 = np.sum((y[:, 1:] - x[:, 1:]) ** 2, axis=1)
    return (y[:, 0] ** 2 + x[:, 0] ** 2 - dx2) / (2 * y[:, 0] * x[:, 0])


def de_sitter_interval_volume(past: np.ndarray, x: np.ndarray, hubble: float, dim: int) -> np.ndarray:
    """Volumen físico de I(y, x) en de Sitter para cada par de filas (y, x)
    (``x`` puede ser un único punto). Devuelve 0 si y no precede a x."""
    past = np.atleast_2d(past)
    x = np.broadcast_to(np.atleast_2d(x), past.shape)
    Z = de_sitter_invariant(past, x)
    related = (Z > 1) & (past[:, 0] < x[:, 0])
    h_tau = np.arccosh(np.where(related, Z, 1.0))
    vol = DeSitterIntervalVolume(hubble, dim, h_tau.max() * 1.001 + 1e-9)
    return np.where(related, vol.from_h_tau(h_tau), 0.0)
