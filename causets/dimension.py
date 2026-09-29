"""Estimador de dimensión de Myrheim-Meyer.

Para N puntos uniformes en un intervalo causal de Minkowski d-dimensional, la
probabilidad de que un par elegido al azar esté causalmente relacionado es

    f(d) = Gamma(d + 1) * Gamma(d / 2) / (2 * Gamma(3d / 2)),

de modo que el número esperado de relaciones es E[R] = f(d) * N(N-1)/2.
Casos de control: f(2) = 1/2, f(4) = 1/10.

El estimador de Myrheim-Meyer invierte esta relación: dado el ordering
fraction observado f_obs = R / (N choose 2), la dimensión estimada es la d que
resuelve f(d) = f_obs. f es estrictamente decreciente en d, así que la raíz es
única.

Referencias:
- Myrheim, J. (1978), CERN preprint TH-2538 — ordering fraction y dimensión.
- Meyer, D. A. (1988), tesis doctoral, MIT — expresión cerrada de f(d).
- Reid, D. D. (2003), "Manifold dimension of a causal set: Tests in
  conformally flat spacetimes", Phys. Rev. D 67, 024034 (arXiv:gr-qc/0207103)
  — formulación práctica del estimador.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

# Rango de búsqueda de la dimensión. f(1) = 1 (cadena total) y f -> 0 cuando
# d -> infinito, así que cualquier f_obs en (0, 1) tiene raíz en d > 1.
_D_MIN, _D_MAX = 1.0, 20.0


def ordering_fraction_expected(d: float) -> float:
    """f(d): fracción esperada de pares relacionados en un intervalo d-dim."""
    return float(np.exp(gammaln(d + 1) + gammaln(d / 2) - gammaln(1.5 * d)) / 2)


def ordering_fraction(C: np.ndarray) -> float:
    """R / (N choose 2) para la matriz causal ``C``."""
    n = C.shape[0]
    if n < 2:
        raise ValueError("se necesitan al menos 2 elementos")
    return np.count_nonzero(C) / (n * (n - 1) / 2)


def myrheim_meyer_from_fraction(f: float) -> float:
    """Dimensión d tal que f(d) = f."""
    if not 0 < f < 1:
        raise ValueError(f"ordering fraction fuera de (0, 1): {f}")
    f_min = ordering_fraction_expected(_D_MAX)
    if f <= f_min:
        return _D_MAX
    return brentq(lambda d: ordering_fraction_expected(d) - f, _D_MIN, _D_MAX)


def myrheim_meyer_dimension(C: np.ndarray) -> float:
    """Dimensión de Myrheim-Meyer de un conjunto causal (matriz causal ``C``)."""
    return myrheim_meyer_from_fraction(ordering_fraction(C))
