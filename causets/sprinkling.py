"""Sprinkling de Poisson en una región del espacio-tiempo.

Un sprinkling de densidad rho en una región de volumen V consiste en
(i) extraer N ~ Poisson(rho * V) y (ii) colocar N puntos distribuidos
uniformemente respecto a la medida de volumen física sqrt(-g) d^dx.
Condicionado a N, los puntos son i.i.d., así que ``sample_uniform`` sirve
también para sprinklings de N fijo.

Referencia: Bombelli, Lee, Meyer & Sorkin, Phys. Rev. Lett. 59, 521 (1987);
revisión en Surya, Living Rev. Relativ. 22, 5 (2019), sec. 3.

Muestreo por rechazo desde la caja envolvente de la región: se acepta un
candidato si está en la región y, además, con probabilidad ``region.weight``
(la densidad de volumen relativa; 1 en Minkowski). Sólo requiere la interfaz
descrita en ``regions.py``. Los puntos se devuelven ordenados por tiempo, de
modo que la matriz causal resultante es estrictamente triangular superior.
"""

from __future__ import annotations

import numpy as np


def sample_uniform(region, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` puntos distribuidos según el volumen físico de ``region``,
    ordenados por la coordenada temporal."""
    if n < 0:
        raise ValueError("n debe ser >= 0")
    lo, hi = region.bounding_box()

    accepted = []
    remaining = n
    tried = kept = 0
    while remaining > 0:
        # Tasa de aceptación estimada con lo observado hasta ahora.
        rate = max(kept / tried, 1e-3) if tried else 0.1
        batch = int(remaining / rate * 1.2) + 64
        candidates = rng.uniform(lo, hi, size=(batch, region.dim))
        candidates = candidates[region.contains(candidates)]
        candidates = candidates[rng.random(len(candidates)) < region.weight(candidates)]
        tried += batch
        kept += len(candidates)
        accepted.append(candidates[:remaining])
        remaining -= len(accepted[-1])

    points = np.concatenate(accepted) if accepted else np.empty((0, region.dim))
    return points[np.argsort(points[:, 0], kind="stable")]


def sprinkle(region, density: float, rng: np.random.Generator) -> np.ndarray:
    """Sprinkling de Poisson de densidad ``density`` en ``region``."""
    if density <= 0:
        raise ValueError("density debe ser > 0")
    n = rng.poisson(density * region.volume)
    return sample_uniform(region, n, rng)
