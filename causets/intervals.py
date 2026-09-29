"""Cardinalidades de intervalos y abundancias de intervalos.

Para un par relacionado y ≺ x, n(y, x) = |{z : y ≺ z ≺ x}| es el número de
elementos estrictamente entre ambos. La abundancia N_k es el número de pares
relacionados con n(y, x) = k (N_0 = número de links). Son la materia prima de
la acción de Benincasa-Dowker (ver ``action.py``).
"""

from __future__ import annotations

import numpy as np


def interval_cardinalities(C: np.ndarray) -> np.ndarray:
    """Matriz entera ``n[i, j]`` = |{z : i ≺ z ≺ j}| (sólo con sentido donde C[i, j]).

    Se calcula como (C @ C)[i, j]. En float32 el producto es exacto mientras
    N < 2^24.
    """
    Cf = np.asarray(C, dtype=np.float32)
    return (Cf @ Cf).astype(np.int64)


def interval_abundances(C: np.ndarray, kmax: int | None = None) -> np.ndarray:
    """Array ``N`` con N[k] = número de pares relacionados con k elementos entre ellos.

    Si ``kmax`` es None se devuelven todas las k hasta la máxima observada.
    """
    C = np.asarray(C, dtype=bool)
    n = interval_cardinalities(C)[C]
    counts = np.bincount(n, minlength=0 if kmax is None else kmax + 1)
    return counts if kmax is None else counts[: kmax + 1]
