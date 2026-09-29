"""Regiones del espacio-tiempo donde se hace el sprinkling.

Convención: coordenadas ``(t, x_1, ..., x_{d-1})`` con c = 1; la columna 0 es
siempre el tiempo.

Por ahora sólo el diamante causal (intervalo de Alexandrov) de Minkowski, que
es la región estándar para el estimador de Myrheim-Meyer (el número esperado de
relaciones sólo se conoce en forma cerrada para intervalos). Las fases
siguientes (de Sitter, FLRW) añadirán regiones en coordenadas conformes; toda
región debe exponer la misma interfaz: ``dim``, ``volume``, ``contains`` y
``bounding_box``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma


def unit_ball_volume(n: int) -> float:
    """Volumen de la bola unidad en R^n: pi^(n/2) / Gamma(n/2 + 1)."""
    return float(np.pi ** (n / 2) / gamma(n / 2 + 1))


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
        h, d = self.half_height, self.dim
        return 2.0 * h**d * unit_ball_volume(d - 1) / d

    def contains(self, coords: np.ndarray) -> np.ndarray:
        coords = np.atleast_2d(coords)
        t = coords[:, 0]
        r = np.linalg.norm(coords[:, 1:], axis=1)
        return np.abs(t) + r <= self.half_height

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        h = self.half_height
        return np.full(self.dim, -h), np.full(self.dim, h)
