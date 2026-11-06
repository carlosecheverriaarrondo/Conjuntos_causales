"""Poisson sprinkling in a region of spacetime.

A sprinkling of density rho in a region of volume V consists of
(i) drawing N ~ Poisson(rho * V) and (ii) placing N points distributed
uniformly with respect to the physical volume measure sqrt(-g) d^dx.
Conditioned on N, the points are i.i.d., so ``sample_uniform`` also serves
for sprinklings with fixed N.

Reference: Bombelli, Lee, Meyer & Sorkin, Phys. Rev. Lett. 59, 521 (1987);
review in Surya, Living Rev. Relativ. 22, 5 (2019), sec. 3.

Rejection sampling from the bounding box of the region: a candidate is
accepted if it lies in the region and, in addition, with probability
``region.weight`` (the relative volume density; 1 in Minkowski). It only
requires the interface described in ``regions.py``. The points are returned
sorted by time, so that the resulting causal matrix is strictly upper
triangular.
"""

from __future__ import annotations

import numpy as np


def sample_uniform(region, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points distributed according to the physical volume of ``region``,
    sorted by the time coordinate."""
    if n < 0:
        raise ValueError("n must be >= 0")
    lo, hi = region.bounding_box()

    accepted = []
    remaining = n
    tried = kept = 0
    while remaining > 0:
        # Acceptance rate estimated from what has been observed so far.
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
    """Poisson sprinkling of density ``density`` in ``region``."""
    if density <= 0:
        raise ValueError("density must be > 0")
    n = rng.poisson(density * region.volume)
    return sample_uniform(region, n, rng)
