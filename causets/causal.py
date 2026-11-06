"""Causal relation between the points of a sprinkling.

Convention: ``C[i, j] = True`` if and only if i ≺ j strictly (i lies in the
causal past of j, i != j). The closed causal relation J+ is used (it includes
null separations), although with probability 1 no pair of a sprinkling is
exactly null.

In Minkowski, i ≺ j  <=>  t_j - t_i > 0  and  (t_j - t_i)^2 >= |x_j - x_i|^2.

Note for later phases: de Sitter and FLRW are conformally flat, so in conformal
coordinates the causal relation is that of Minkowski and this function is
reused as is (what changes is the volume measure of the sprinkling).
"""

from __future__ import annotations

import numpy as np

_CHUNK = 1024


def minkowski_causal_matrix(coords: np.ndarray) -> np.ndarray:
    """Causal matrix (bool, N x N) of points in d-dimensional Minkowski.

    Computed in blocks of rows to bound the memory (O(N * chunk) temporaries
    instead of O(N^2 * d)).
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


def minkowski_future_counts(coords: np.ndarray) -> np.ndarray:
    """For each point y, the number of points z of the set with y ≺ z.

    Equivalent to ``minkowski_causal_matrix(coords).sum(axis=1)`` but without
    storing the N x N matrix (memory O(N * chunk)). If all the points lie in
    the past of a vertex x (sprinkling of the interval J-(x)), this number is
    the cardinality of the open interval between y and x.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    t, x = coords[:, 0], coords[:, 1:]
    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        dt = t[None, :] - t[start:stop, None]
        dx2 = np.zeros_like(dt)
        for k in range(x.shape[1]):
            dx2 += (x[None, :, k] - x[start:stop, None, k]) ** 2
        counts[start:stop] = np.count_nonzero((dt > 0) & (dt * dt >= dx2), axis=1)
    return counts


def partial_order_violations(C: np.ndarray) -> dict[str, int]:
    """Count the violations of the axioms of a strict partial order.

    - reflexivity: elements with i ≺ i
    - antisymmetry: pairs with i ≺ j and j ≺ i
    - transitivity: pairs (i, k) with i ≺ j ≺ k for some j but not i ≺ k

    A valid partial order gives zero in all three.
    """
    C = np.asarray(C, dtype=bool)
    Ci = C.astype(np.float32)
    # (C @ C)[i, k] = number of j with i ≺ j ≺ k (exact in float32 for N < 2^24)
    two_step = (Ci @ Ci) > 0
    return {
        "reflexive": int(np.trace(C)),
        "antisymmetric": int(np.count_nonzero(C & C.T)) // 2,
        "transitive": int(np.count_nonzero(two_step & ~C)),
    }


def is_partial_order(C: np.ndarray) -> bool:
    return not any(partial_order_violations(C).values())
