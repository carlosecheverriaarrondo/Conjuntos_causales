"""Sprinkling de Poisson en una región del espacio-tiempo.

Un sprinkling de densidad rho en una región de volumen V consiste en
(i) extraer N ~ Poisson(rho * V) y (ii) colocar N puntos uniformemente
distribuidos (respecto a la medida de volumen) en la región. Condicionado a N,
los puntos son i.i.d. uniformes, así que ``sample_uniform`` sirve también para
sprinklings de N fijo.

Referencia: Bombelli, Lee, Meyer & Sorkin, Phys. Rev. Lett. 59, 521 (1987);
revisión en Surya, Living Rev. Relativ. 22, 5 (2019), sec. 3.

La muestra uniforme se hace por rechazo desde la caja envolvente de la región,
lo que sólo requiere ``contains`` y ``bounding_box``: vale para cualquier
región con esa interfaz. Los puntos se devuelven ordenados por tiempo, de modo
que la matriz causal resultante es estrictamente triangular superior.
"""

from __future__ import annotations

import numpy as np


def sample_uniform(region, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` puntos uniformes en ``region``, ordenados por la coordenada t."""
    if n < 0:
        raise ValueError("n debe ser >= 0")
    lo, hi = region.bounding_box()
    box_volume = float(np.prod(hi - lo))
    acceptance = region.volume / box_volume

    accepted = []
    remaining = n
    while remaining > 0:
        # Sobremuestreo con margen para no iterar muchas veces.
        batch = int(remaining / acceptance * 1.2) + 16
        candidates = rng.uniform(lo, hi, size=(batch, region.dim))
        inside = candidates[region.contains(candidates)][:remaining]
        accepted.append(inside)
        remaining -= len(inside)

    points = np.concatenate(accepted) if accepted else np.empty((0, region.dim))
    return points[np.argsort(points[:, 0], kind="stable")]


def sprinkle(region, density: float, rng: np.random.Generator) -> np.ndarray:
    """Sprinkling de Poisson de densidad ``density`` en ``region``."""
    if density <= 0:
        raise ValueError("density debe ser > 0")
    n = rng.poisson(density * region.volume)
    return sample_uniform(region, n, rng)
