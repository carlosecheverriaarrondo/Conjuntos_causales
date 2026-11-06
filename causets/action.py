"""D'Alembertian operator and Benincasa-Dowker (BD) action, local and smeared.

Operator (smeared/non-local version, which contains the local one as eps = 1):

    (B_eps phi)(x) = alpha_d / l_eps^2 * [ beta_d phi(x)
                      + eps * sum_{y ≺ x} f_d(n(y, x), eps) phi(y) ]

    f_d(n, eps) = sum_k C_k binom(n, k) eps^k (1 - eps)^(n - k)

with l = rho^(-1/d) the discreteness scale, l_eps = l * eps^(-1/d) and

    d = 2:  alpha = 4,         beta = -1/2,  C = (1, -2, 1)
    d = 4:  alpha = 4/sqrt(6), beta = -1,    C = (1, -9, 16, -8)

With eps = 1 only the k = n terms survive and the local operator is recovered
(sums over the "layers" L_{k+1} of elements with k elements between y and x).
The smearing is equivalent to averaging the local operator over a random
"thinning" of the set with retention probability eps.

In the continuum limit  <B_eps phi>(x) -> (box - R/2) phi(x).

Action: S_eps = -s_d * sum_x l^2 (B_eps 1)(x), with s_2 = 1/2 and s_4 = 1, giving

    d = 2:  S = eps N - 2 eps^2 sum_n N_n f_2(n, eps)
            (eps = 1:  S = N - 2 N_0 + 4 N_1 - 2 N_2)
    d = 4:  S = (4 sqrt(eps) / sqrt(6)) [N - eps sum_n N_n f_4(n, eps)]
            (eps = 1:  S = (4/sqrt 6) [N - N_0 + 9 N_1 - 16 N_2 + 8 N_3])

in units hbar = 1 and (l / l_Planck)^2 = 1 in 4D.

References:
- Benincasa, D. M. T. & Dowker, F. (2010), Phys. Rev. Lett. 104, 181301,
  arXiv:1001.2725 — operator and action in 4D.
- Sorkin, R. D. (2007), arXiv:gr-qc/0703099 — 2D operator and eps smearing.
- Benincasa, Dowker & Schmitzer (2011), Class. Quantum Grav. 28, 105018,
  arXiv:1011.5191 — 2D action and its mean in intervals.
- Dowker, F. & Glaser, L. (2013), Class. Quantum Grav. 30, 195016,
  arXiv:1305.2588 — general form of f_d(n, eps) in dimension d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from .intervals import interval_cardinalities


@dataclass(frozen=True)
class BDCoefficients:
    dim: int
    alpha: float
    beta: float
    layers: tuple[float, ...]  # C_k, k = 0, 1, ...
    action_scale: float  # s_d in S = -s_d * sum_x l^2 B1(x)


BD = {
    2: BDCoefficients(2, 4.0, -0.5, (1.0, -2.0, 1.0), 0.5),
    4: BDCoefficients(4, 4.0 / np.sqrt(6.0), -1.0, (1.0, -9.0, 16.0, -8.0), 1.0),
}


def coefficients(dim: int) -> BDCoefficients:
    try:
        return BD[dim]
    except KeyError:
        raise ValueError(f"BD coefficients implemented only for d in {sorted(BD)}") from None


def _check_eps(eps: float) -> None:
    if not 0 < eps <= 1:
        raise ValueError("eps must lie in (0, 1]")


def smearing_function(n, dim: int, eps: float = 1.0) -> np.ndarray:
    """f_d(n, eps) = sum_k C_k binom(n, k) eps^k (1-eps)^(n-k)."""
    _check_eps(eps)
    n = np.asarray(n)
    return sum(c * binom.pmf(k, n, eps) for k, c in enumerate(coefficients(dim).layers))


def discreteness_scale_sq(dim: int, density: float, eps: float = 1.0) -> float:
    """l_eps^2 = (eps * rho)^(-2/d)."""
    return (eps * density) ** (-2.0 / dim)


def bd_on_constant_from_counts(n_past: np.ndarray, dim: int, density: float, eps: float = 1.0) -> float:
    """(B_eps 1)(x) given the cardinalities n(y, x) of all y ≺ x."""
    c = coefficients(dim)
    bracket = c.beta + eps * np.sum(smearing_function(n_past, dim, eps))
    return c.alpha * bracket / discreteness_scale_sq(dim, density, eps)


def bd_operator(C: np.ndarray, phi: np.ndarray, dim: int, density: float, eps: float = 1.0) -> np.ndarray:
    """(B_eps phi)(x) for all elements (full causal matrix ``C``).

    Near the past boundary of the region the result does not approximate
    box - R/2, because the past of the point is missing: the operator is non-local.
    """
    c = coefficients(dim)
    C = np.asarray(C, dtype=bool)
    n = interval_cardinalities(C)
    table = smearing_function(np.arange(max(C.shape[0] - 1, 1)), dim, eps)
    F = np.where(C, table[np.where(C, n, 0)], 0.0)
    bracket = c.beta * phi + eps * (F.T @ phi)
    return c.alpha * bracket / discreteness_scale_sq(dim, density, eps)


def bd_action_from_pairs(n_elements: int, pair_counts: np.ndarray, dim: int, eps: float = 1.0) -> float:
    """BD action given N and pair_counts[n] = number of pairs with n elements between them."""
    c = coefficients(dim)
    _check_eps(eps)
    ns = np.arange(len(pair_counts))
    weighted = float(np.sum(pair_counts * smearing_function(ns, dim, eps)))
    # S = -s_d * sum_x l^2 B1(x),  l^2 / l_eps^2 = eps^(2/d)
    return -c.action_scale * eps ** (2.0 / dim) * c.alpha * (c.beta * n_elements + eps * weighted)


def bd_action(C: np.ndarray, dim: int, eps: float = 1.0) -> float:
    """(Dimensionless) BD action of the causal set with causal matrix ``C``."""
    C = np.asarray(C, dtype=bool)
    n = interval_cardinalities(C)[C]
    return bd_action_from_pairs(C.shape[0], np.bincount(n), dim, eps)
