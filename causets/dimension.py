"""Myrheim-Meyer dimension estimator.

For N uniform points in a causal interval of d-dimensional Minkowski, the
probability that a randomly chosen pair is causally related is

    f(d) = Gamma(d + 1) * Gamma(d / 2) / (2 * Gamma(3d / 2)),

so that the expected number of relations is E[R] = f(d) * N(N-1)/2.
Control cases: f(2) = 1/2, f(4) = 1/10.

The Myrheim-Meyer estimator inverts this relation: given the observed
ordering fraction f_obs = R / (N choose 2), the estimated dimension is the d
that solves f(d) = f_obs. f is strictly decreasing in d, so the root is
unique.

References:
- Myrheim, J. (1978), CERN preprint TH-2538 — ordering fraction and dimension.
- Meyer, D. A. (1988), PhD thesis, MIT — closed form of f(d).
- Reid, D. D. (2003), "Manifold dimension of a causal set: Tests in
  conformally flat spacetimes", Phys. Rev. D 67, 024034 (arXiv:gr-qc/0207103)
  — practical formulation of the estimator.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

# Search range for the dimension. f(1) = 1 (total chain) and f -> 0 as
# d -> infinity, so any f_obs in (0, 1) has a root at d > 1.
_D_MIN, _D_MAX = 1.0, 20.0


def ordering_fraction_expected(d: float) -> float:
    """f(d): expected fraction of related pairs in a d-dimensional interval."""
    return float(np.exp(gammaln(d + 1) + gammaln(d / 2) - gammaln(1.5 * d)) / 2)


def ordering_fraction(C: np.ndarray) -> float:
    """R / (N choose 2) for the causal matrix ``C``."""
    n = C.shape[0]
    if n < 2:
        raise ValueError("at least 2 elements are needed")
    return np.count_nonzero(C) / (n * (n - 1) / 2)


def myrheim_meyer_from_fraction(f: float) -> float:
    """Dimension d such that f(d) = f."""
    if not 0 < f < 1:
        raise ValueError(f"ordering fraction outside (0, 1): {f}")
    f_min = ordering_fraction_expected(_D_MAX)
    if f <= f_min:
        return _D_MAX
    return brentq(lambda d: ordering_fraction_expected(d) - f, _D_MIN, _D_MAX)


def myrheim_meyer_dimension(C: np.ndarray) -> float:
    """Myrheim-Meyer dimension of a causal set (causal matrix ``C``)."""
    return myrheim_meyer_from_fraction(ordering_fraction(C))
