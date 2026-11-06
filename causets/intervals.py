"""Interval cardinalities and interval abundances.

For a related pair y ≺ x, n(y, x) = |{z : y ≺ z ≺ x}| is the number of
elements strictly between the two. The abundance N_k is the number of related
pairs with n(y, x) = k (N_0 = number of links). They are the raw material of
the Benincasa-Dowker action (see ``action.py``).
"""

from __future__ import annotations

import numpy as np


def interval_cardinalities(C: np.ndarray) -> np.ndarray:
    """Integer matrix ``n[i, j]`` = |{z : i ≺ z ≺ j}| (only meaningful where C[i, j]).

    Computed as (C @ C)[i, j]. In float32 the product is exact as long as
    N < 2^24.
    """
    Cf = np.asarray(C, dtype=np.float32)
    return (Cf @ Cf).astype(np.int64)


def interval_abundances(C: np.ndarray, kmax: int | None = None) -> np.ndarray:
    """Array ``N`` with N[k] = number of related pairs with k elements between them.

    If ``kmax`` is None, all k up to the largest observed one are returned.
    """
    C = np.asarray(C, dtype=bool)
    n = interval_cardinalities(C)[C]
    counts = np.bincount(n, minlength=0 if kmax is None else kmax + 1)
    return counts if kmax is None else counts[: kmax + 1]
