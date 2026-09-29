"""Relación causal entre puntos de un sprinkling.

Convención: ``C[i, j] = True`` si y sólo si i ≺ j estrictamente (i está en el
pasado causal de j, i != j). Se usa la relación causal cerrada J+ (incluye
separaciones nulas), aunque con probabilidad 1 ningún par de un sprinkling es
exactamente nulo.

En Minkowski, i ≺ j  <=>  t_j - t_i > 0  y  (t_j - t_i)^2 >= |x_j - x_i|^2.

Nota para fases posteriores: de Sitter y FLRW son conformemente planos, así que
en coordenadas conformes la relación causal es la de Minkowski y esta función
se reutiliza tal cual (lo que cambia es la medida de volumen del sprinkling).
"""

from __future__ import annotations

import numpy as np

_CHUNK = 1024


def minkowski_causal_matrix(coords: np.ndarray) -> np.ndarray:
    """Matriz causal (bool, N x N) de puntos en Minkowski d-dimensional.

    Se calcula por bloques de filas para acotar la memoria (O(N * chunk)
    temporales en lugar de O(N^2 * d)).
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    t, x = coords[:, 0], coords[:, 1:]
    C = np.zeros((n, n), dtype=bool)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        dt = t[None, :] - t[start:stop, None]
        dx2 = np.zeros_like(dt)
        for k in range(x.shape[1]):
            dx2 += (x[None, :, k] - x[start:stop, None, k]) ** 2
        C[start:stop] = (dt > 0) & (dt * dt >= dx2)
    return C


def partial_order_violations(C: np.ndarray) -> dict[str, int]:
    """Cuenta las violaciones de los axiomas de orden parcial estricto.

    - reflexividad: elementos con i ≺ i
    - antisimetría: pares con i ≺ j y j ≺ i
    - transitividad: pares (i, k) con i ≺ j ≺ k para algún j pero no i ≺ k

    Un orden parcial válido da cero en las tres.
    """
    C = np.asarray(C, dtype=bool)
    Ci = C.astype(np.float32)
    # (C @ C)[i, k] = número de j con i ≺ j ≺ k (exacto en float32 para N < 2^24)
    two_step = (Ci @ Ci) > 0
    return {
        "reflexive": int(np.trace(C)),
        "antisymmetric": int(np.count_nonzero(C & C.T)) // 2,
        "transitive": int(np.count_nonzero(two_step & ~C)),
    }


def is_partial_order(C: np.ndarray) -> bool:
    return not any(partial_order_violations(C).values())
