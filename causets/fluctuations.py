"""Fluctuations of the Benincasa-Dowker (BD) action: formalism of
Moradi, Yazdi & Zilhão (MYZ), "Fluctuations and Correlations in Causal Set
Theory", Class. Quantum Grav. 42 (2025) 045017, arXiv:2407.03395
(equation references: arXiv version v2).

WHICH ACTION
------------
The module was written for the **LOCAL** BD action (eps = 1): the one MYZ
write in (4.1)-(4.4) and whose variance they decompose in sec. 5. The action
is a finite linear combination of abundances,

    S = a_N N + sum_{k=0}^{n_d - 1} a_k N_k,

where N_k is the number of pairs y ≺ x with k elements between them
(MYZ: N_k = sum_x sum_{y in ◇_{k+1}(x)} 1) and N ≡ N_{-1} the number of elements.
With the normalisation of Goal 2 (``action.py``):
    2D:  a = (1, -2, 4, -2)                  [S = N - 2N_0 + 4N_1 - 2N_2]
    4D:  a = (4/sqrt6) (1, -1, 9, -16, 8)    [S = (4/sqrt6)(N - N_0 + 9N_1 - 16N_2 + 8N_3)]
MYZ normalise S = -(2/alpha_d) sum_i C_i N_{i-1} (5.4); e.g. in 4D their S is
-(sqrt6/2) times ours, so their variance is 3/2 ours. The decomposition does
not depend on that constant: (Delta S)^2 = a^T M a (MYZ 5.9).
The smeared action (eps < 1) involves all the N_n; it is handled with
thinned abundances (``eps`` argument, ``thinned_joint_probability``), see
the paper, appendix B.

FORMALISM
---------
- Cardinality indicator zeta_i(x, y) (MYZ 3.2, 3.4): 1 if y ≺ x and there are
  exactly i elements in I(y, x). ``cardinality_indicator``.
- Occupation indicator chi(R) (MYZ 3.16): 1 if the region R contains some
  element. ``occupation_indicator``.
- Correlation matrix M_ij = K_ij - L_i L_j = Cov(N_i, N_j),
  i, j = -1, ..., n_d - 1 (MYZ 5.7, 5.10), with L_i = <N_i> (5.5, 5.15).
- Components:
    M_{-1,-1} = rho V_M                                            (MYZ 5.20)
    M_{-1,j}  = K_{-1,j} - rho V_M L_j                              (MYZ 5.26-5.27)
              = (j + 2) L_j - (j + 1) L_{j+1}   [integrating 5.26 with
                z P_{j-1}(z) = j P_j(z); exact identity in any M]
    K_ij = J1 + ... + J6 (MYZ 5.29-5.36), according to which points coincide:
      J1: x1 = x2, y1 = y2   ->  delta_ij L_i                        (5.47)
      J2: y1 = y2 (x1 != x2)     J3: x1 = x2 (y1 != y2)
      J4: x1 = y2                J5: x2 = y1   (J4_ij = J5_ji, 5.37)
      J6: all four distinct
  Each integrand is evaluated with the rules of sec. 3.3: chi^n = chi
  (3.30); each point of the configuration contained in the interval of the
  other zeta lowers the required number of elements by 1 (3.31); and the
  zetas of overlapping regions are decomposed into disjoint regions (3.15, 3.28):

    <zeta_a(A) zeta_b(B)> = sum_beta P_{a-beta}(rho V_{A\\B}) P_beta(rho V_{A∩B}) P_{b-beta}(rho V_{B\\A}).

  The 11 subregions of D_6 (5.44) and those of D_2, D_3 (5.40-5.43) are
  exactly the patterns "which points fall in which interval", which here are
  detected configuration by configuration instead of being enumerated by hand.

- Connected part: in J6, L_i L_j is subtracted inside the integrand,
      J6c_ij = rho^4 ∫ [<zeta_i zeta_j chi..chi> / rho^4 - P_i(rho V_A) P_j(rho V_B)],
  which is non-zero only when I(y1,x1) ∩ I(y2,x2) ≠ ∅. Thus
      M_ij = delta_ij L_i + J2 + J3 + J4 + J5 + J6c   (i, j >= 0)
  without cancellations between huge numbers.

NUMERICAL EVALUATION OF THE INTEGRALS (what MYZ leave as future work)
---------------------------------------------------------------------
Monte Carlo integration with importance sampling: the intervals are
generated around a point with timelike displacements
y = x - tau (cosh chi, sinh chi n), sampled according to the volume measure
itself (see ``_Proposal``), so that the small and nearly null intervals
(those that dominate N_0..N_3) are well sampled and the weights have no
heavy tails. J6 is split into patterns (``term_J6_connected``): if an
endpoint of one interval falls inside the other, that point is sampled
inside the interval; if there is only an overlap, one integrates over
w ∈ A ∩ B with weight 1/|A ∩ B| (identity ∫ dB f = ∫_A dw ∫_{B ∋ w} dB f / |A ∩ B|),
where the connected integrand vanishes smoothly as |A ∩ B| -> 0.

- 2D Minkowski: |A ∩ B| has a closed form (intervals = rectangles in
  light-cone coordinates), and the integrands are evaluated EXACTLY.
  Only error: the statistical one of the MC integration, which is reported.
- 4D and de Sitter: |A ∩ B| has no closed form. UNBIASED estimators are
  used: <zeta_a(A) zeta_b(B)> is estimated with a local Poisson realisation
  in A ∪ B (indicator that the counts match), and 1/|A ∩ B| with geometric
  trials (E[K] = |A| / |A ∩ B|). No bias, but more variance; the reported
  statistical error includes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln

from .intervals import interval_cardinalities
from .regions import unit_sphere_area

# ---------------------------------------------------------------------------
# Coefficients of the local action
# ---------------------------------------------------------------------------

LOCAL_ACTION_COEFFICIENTS = {
    2: np.array([1.0, -2.0, 4.0, -2.0]),
    4: 4.0 / np.sqrt(6.0) * np.array([1.0, -1.0, 9.0, -16.0, 8.0]),
}


def local_action_coefficients(dim: int) -> np.ndarray:
    """a = (a_N, a_0, ..., a_{n_d - 1}) of the local BD action (Goal 2 normalisation)."""
    return LOCAL_ACTION_COEFFICIENTS[dim].copy()


def smeared_action_coefficients(dim: int, eps: float) -> np.ndarray:
    """a = (a_N, a_0, ..., a_{n_d - 1}) of the smeared action written in terms
    of thinned abundances (see ``thinned_abundance_vector``):
    S_eps = a_N N + sum_k a_k Ñ_k, with a_N = -s_d alpha beta eps^(2/d) and
    a_k = -s_d alpha eps^(1+2/d) C_k (``action.bd_action_from_pairs``).
    With eps = 1 it agrees with ``local_action_coefficients``."""
    from .action import coefficients

    c = coefficients(dim)
    pref = -c.action_scale * c.alpha * eps ** (2.0 / dim)
    return np.concatenate([[pref * c.beta], pref * eps * np.asarray(c.layers)])


def action_coefficients(dim: int, eps: float = 1.0) -> np.ndarray:
    return local_action_coefficients(dim) if eps == 1 else smeared_action_coefficients(dim, eps)


# ---------------------------------------------------------------------------
# MYZ indicators on a concrete causal set
# ---------------------------------------------------------------------------


def cardinality_indicator(C: np.ndarray, i: int) -> np.ndarray:
    """zeta_i (MYZ 3.4): Z[y, x] = 1 if y ≺ x with exactly i elements between them."""
    C = np.asarray(C, dtype=bool)
    return C & (interval_cardinalities(C) == i)


def occupation_indicator(points: np.ndarray, contains) -> bool:
    """chi(R) (MYZ 3.16): True if some element falls in the region R
    (``contains`` is the membership function of R)."""
    return bool(np.any(contains(np.atleast_2d(points)))) if len(points) else False


def abundance_vector(C: np.ndarray, n_max: int) -> np.ndarray:
    """(N, N_0, ..., N_{n_max - 1}) of a causal set."""
    C = np.asarray(C, dtype=bool)
    n = interval_cardinalities(C)[C]
    counts = np.bincount(n, minlength=n_max)[:n_max]
    return np.concatenate([[C.shape[0]], counts]).astype(float)


def thinned_abundance_vector(C: np.ndarray, n_max: int, eps: float) -> np.ndarray:
    """(N, Ñ_0, ..., Ñ_{n_max - 1}) with Ñ_k = sum_{y ≺ x} B_k(n(y,x), eps),
    B_k(n, eps) = binom(n, k) eps^k (1-eps)^(n-k): expected number of pairs
    with k elements between them after thinning the interval with retention
    probability eps. f_d(n, eps) = sum_k C_k B_k(n, eps) (``action.py``), so
    that S_eps = a · Ñ with ``smeared_action_coefficients``. With eps = 1, Ñ_k = N_k."""
    from scipy.stats import binom

    if eps == 1:
        return abundance_vector(C, n_max)
    C = np.asarray(C, dtype=bool)
    counts = np.bincount(interval_cardinalities(C)[C])
    ns = np.arange(len(counts))
    thinned = [float(np.sum(counts * binom.pmf(k, ns, eps))) for k in range(n_max)]
    return np.array([C.shape[0], *thinned], dtype=float)


# ---------------------------------------------------------------------------
# Numerical side: variance over an ensemble of sprinklings
# ---------------------------------------------------------------------------


@dataclass
class EnsembleFluctuations:
    """Variance of the action and covariance of abundances over M sprinklings."""

    abundances: np.ndarray  # (M, n_max + 1): columns N, N_0, ...
    coefficients: np.ndarray

    @property
    def actions(self) -> np.ndarray:
        return self.abundances @ self.coefficients

    @property
    def variance(self) -> float:
        return float(np.var(self.actions, ddof=1))

    @property
    def variance_se(self) -> float:
        """Standard error of the sample variance: std(d^2) / sqrt(M), d = S - <S>."""
        d = self.actions - self.actions.mean()
        return float(np.std(d * d, ddof=1) / np.sqrt(len(d)))

    @property
    def covariance(self) -> np.ndarray:
        return np.cov(self.abundances, rowvar=False, ddof=1)

    @property
    def covariance_se(self) -> np.ndarray:
        d = self.abundances - self.abundances.mean(axis=0)
        prods = d[:, :, None] * d[:, None, :]
        return prods.std(axis=0, ddof=1) / np.sqrt(len(d))


def ensemble_fluctuations(geometry, density: float, n_samples: int, rng, coefficients=None, eps: float = 1.0) -> EnsembleFluctuations:
    """Independent Poisson sprinklings of ``geometry`` (no Markov chain: no
    thermalisation) and their abundances (thinned if eps < 1)."""
    from .causal import minkowski_causal_matrix
    from .sprinkling import sprinkle

    a = action_coefficients(geometry.dim, eps) if coefficients is None else np.asarray(coefficients)
    n_max = len(a) - 1
    rows = [
        thinned_abundance_vector(minkowski_causal_matrix(sprinkle(geometry.region, density, rng)), n_max, eps)
        for _ in range(n_samples)
    ]
    return EnsembleFluctuations(np.array(rows), a)


# ---------------------------------------------------------------------------
# Analytic side (MYZ formalism)
# ---------------------------------------------------------------------------


def _pmf_table(z: np.ndarray, n: int) -> np.ndarray:
    """P_k(z) for k = 0..n-1, shape (len(z), n)."""
    z = np.asarray(z, dtype=float)[:, None]
    k = np.arange(n)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = k * np.log(np.where(z > 0, z, 1.0)) - z - gammaln(k + 1)
    p = np.exp(logp)
    return np.where(z > 0, p, (k == 0).astype(float))


def _shifted(P: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """G[m, i, j] = P[m, i - sa[m], j - sb[m]] (0 if either index is negative)."""
    m, n, _ = P.shape
    i = np.arange(n)[None, :, None] - sa[:, None, None]
    j = np.arange(n)[None, None, :] - sb[:, None, None]
    ok = (i >= 0) & (j >= 0)
    rows = np.arange(m)[:, None, None]
    return np.where(ok, P[rows, np.clip(i, 0, None), np.clip(j, 0, None)], 0.0)


def joint_cardinality_probability(za, zb, zc, n: int) -> np.ndarray:
    """<zeta_a(A) zeta_b(B)> for a, b = 0..n-1 (MYZ 3.15, 3.28, 3.34):
    sum_beta P_{a-beta}(za) P_beta(zb) P_{b-beta}(zc), with za = rho|A\\B|,
    zb = rho|A∩B|, zc = rho|B\\A|. Shape (m, n, n)."""
    pa, pb, pc = _pmf_table(za, n), _pmf_table(zb, n), _pmf_table(zc, n)
    out = np.zeros((len(pa), n, n))
    for beta in range(n):
        # a = alpha + beta, b = gamma + beta
        out[:, beta:, beta:] += pb[:, beta, None, None] * pa[:, : n - beta, None] * pc[:, None, : n - beta]
    return out


def thinned_joint_probability(zA, zB, zAB, sA, sB, n: int, eps: float = 1.0) -> np.ndarray:
    """<zeta_{i-sA}(A) zeta_{j-sB}(B)> for i, j = 0..n-1 after thinning A and B
    independently with retention probability eps, shape (m, n, n).
    zA = rho|A|, zB = rho|B|, zAB = rho|A∩B|; sA (sB) is the number of points
    of the configuration contained in A (B).

    With eps = 1 this is MYZ (3.15, 3.31). For eps < 1: the elements of A∩B
    are split (thinning of a Poisson process) into independent Poisson
    processes retained by both (eps^2), only by A or only by B
    (eps(1-eps) each), so the structure of (3.15) is preserved with
        zA' = eps zA - eps^2 zAB,  zAB' = eps^2 zAB,  zB' = eps zB - eps^2 zAB,
    and each fixed contained point is retained with probability eps:
    the shift sA becomes Binomial(sA, eps)."""
    zA, zB, zAB = (np.asarray(v, float) for v in (zA, zB, zAB))
    sA, sB = np.asarray(sA, int), np.asarray(sB, int)
    if eps == 1:
        return _shifted(joint_cardinality_probability(zA - zAB, zAB, zB - zAB, n), sA, sB)
    from scipy.stats import binom

    e2 = eps * eps
    P = joint_cardinality_probability(np.maximum(eps * zA - e2 * zAB, 0.0), e2 * zAB, np.maximum(eps * zB - e2 * zAB, 0.0), n)
    out = np.zeros_like(P)
    for u in range(int(sA.max(initial=0)) + 1):
        wu = binom.pmf(u, sA, eps)
        for v in range(int(sB.max(initial=0)) + 1):
            w = wu * binom.pmf(v, sB, eps)
            if np.any(w > 0):
                out += w[:, None, None] * _shifted(P, np.full(len(P), u), np.full(len(P), v))
    return out


def _sinh_power_integral(d: int, X):
    """S_d(X) = int_0^X sinh^{d-2}(chi) d chi."""
    X = np.asarray(X, dtype=float)
    if d == 2:
        return X
    if d == 3:
        return np.cosh(X) - 1.0
    if d == 4:
        return np.sinh(2 * X) / 4 - X / 2
    raise ValueError("proposal implemented for d = 2, 3, 4")


@dataclass
class _Proposal:
    """Future timelike displacements o = tau (cosh chi, sinh chi n).

    One samples z = rho_c c_d tau^d (the "coordinate volume in units of
    elements") with z ~ Gamma(2/d, z0), chi with density sinh^{d-2}(chi) on
    [0, ln(4 L / tau)] and n uniform on S^{d-2}. Since the volume measure in
    these variables is d^d o = dz sinh^{d-2}chi dchi dOmega / (d rho_c c_d),
    the weights 1/q are smooth (e^{z/z0} z^{1-2/d} S_d(chi_max)): there are no
    heavy tails. ``density`` returns q(o) in coordinate measure.

    ``z0`` may be a tuple of scales: then z is sampled from the equally
    weighted mixture of the Gammas ("defensive importance sampling", Hesterberg,
    Technometrics 37 (1995) 185), and the weight is bounded by that of the best
    component. Used with the smeared action, where the relevant intervals
    have rho V ~ 1/eps in addition to the small ones."""

    dim: int
    extent: float
    rho_c: float  # density in coordinate measure (rho in Minkowski)
    z0: float | tuple = 3.0

    def sample(self, m: int, rng, budget, local=1.0) -> tuple[np.ndarray, np.ndarray]:
        """``budget``: coordinate time available from each base point to the
        time boundary of M; tau cosh(chi) <= budget is imposed, i.e.
        chi <= arccosh(budget / tau) (necessary condition to remain in M).
        If tau >= budget the sample is invalid (q = inf, weight 0).
        ``local``: factor (per base point) multiplying rho_c, to adapt the
        scale of the intervals to the local physical density (FLRW). If it is
        a tuple of factors, each sample uses one at random and q is the density
        of the equally weighted mixture (as with ``z0``)."""
        from scipy.special import gamma as gamma_fn

        d = self.dim
        c = diamond_volume_unit(d)
        shape = 2.0 / d
        scales = np.atleast_1d(np.asarray(self.z0, dtype=float))
        z = rng.gamma(shape, scales[rng.integers(len(scales), size=m)])
        if isinstance(local, tuple):
            comps = [self.rho_c * np.broadcast_to(np.asarray(f, dtype=float), (m,)) for f in local]
            rho_c = np.choose(rng.integers(len(comps), size=m), comps)
            tau = (z / (rho_c * c)) ** (1.0 / d)

            def q_z(rc):
                """Density of the Gamma mixture at z' = rc c tau^d, times rc."""
                zz = rc * c * tau**d
                return rc * np.mean([zz ** (shape - 1) * np.exp(-zz / s) / (gamma_fn(shape) * s**shape) for s in scales], axis=0)

            p_z = np.mean([q_z(rc) for rc in comps], axis=0)  # includes the rho_c factor
            rho_q = 1.0
        else:
            rho_c = self.rho_c * np.broadcast_to(np.asarray(local, dtype=float), (m,))
            tau = (z / (rho_c * c)) ** (1.0 / d)
            p_z = np.mean([z ** (shape - 1) * np.exp(-z / s) / (gamma_fn(shape) * s**shape) for s in scales], axis=0)
            rho_q = rho_c
        budget = np.broadcast_to(np.asarray(budget, dtype=float), (m,))
        valid = tau < budget
        X = np.where(valid, np.arccosh(np.maximum(budget / tau, 1.0)), 1e-12)
        X = np.maximum(X, 1e-12)
        # chi ~ sinh^{d-2}: proposal e^{(d-2) chi} on [0, X] and acceptance (1 - e^{-2 chi})^{d-2}
        chi = np.zeros(m)
        todo = np.nonzero(valid)[0]  # the invalid ones do not need chi
        while len(todo):
            u = rng.random(len(todo))
            Xt = X[todo]
            if d == 2:
                cand = u * Xt
                acc = np.ones(len(todo), dtype=bool)
            else:
                k = d - 2
                cand = np.log1p(u * np.expm1(k * Xt)) / k
                acc = rng.random(len(todo)) < (-np.expm1(-2 * cand)) ** k
            chi[todo[acc]] = cand[acc]
            todo = todo[~acc]
        n = rng.normal(size=(m, d - 1))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        o = np.column_stack([tau * np.cosh(chi), (tau * np.sinh(chi))[:, None] * n])
        q = p_z * d * rho_q * c / (_sinh_power_integral(d, X) * unit_sphere_area(d - 2))
        return o, np.where(valid, q, np.inf)


def diamond_volume_unit(d: int) -> float:
    """Volume of a flat interval of unit proper time."""
    from .regions import diamond_volume

    return diamond_volume(d, 0.5)


@dataclass
class TermEstimate:
    """MC estimate of a matrix term T_ij and of its contraction a^T T a."""

    matrix: np.ndarray
    contraction: float
    contraction_se: float
    n_samples: int


@dataclass
class FluctuationPrediction:
    """MYZ prediction of Cov(N_i, N_j) and of (Delta S)^2 for the local action."""

    density: float
    coefficients: np.ndarray
    abundance_means: np.ndarray  # L_{-1}, L_0, ..., L_{n-1} (and L_n for M_{-1,n-1})
    covariance: np.ndarray  # M_ij, i, j = -1..n-1
    variance: float
    variance_se: float
    terms: dict = field(default_factory=dict)


class FluctuationKernel:
    """Evaluate the MYZ formalism on a geometry (see ``geometry.py``)."""

    def __init__(self, geometry, density: float, n_max: int, rng, z0=None,
                 overlap: str = "auto", overlap_points: int = 1024, eps: float = 1.0):
        """``eps``: smearing of the action (1 = local action). With eps < 1 the
        matrices are those of the thinned abundances Ñ_k
        (``thinned_abundance_vector``).

        ``overlap``: how to obtain |A ∩ B|.
          "exact"      deterministic, from the geometry: closed form in 2D
                       Minkowski, quadrature (``overlap.py``) in 4D Minkowski
          "plugin"     MC estimate with ``overlap_points`` points in A and the
                       exact formula for <zeta zeta> (bias O(1/overlap_points))
          "stochastic" unbiased estimators (local Poisson realisation and
                       geometric trials); much more variance
          "auto"       "exact" if available, otherwise "plugin"."""
        self.g = geometry
        self.rho = density
        self.n = n_max
        self.rng = rng
        lo, hi = geometry.region.bounding_box()
        extent = float(hi[0] - lo[0])
        rho_c = density * geometry.volume / geometry.coordinate_volume
        if z0 is None:  # proposal scale(s) (see ``_Proposal``)
            z0 = 3.0 if eps == 1 else (3.0, 3.0 / eps)
        self.prop = _Proposal(geometry.dim, extent, rho_c, z0)
        self._t_min, self._t_max = float(lo[0]), float(hi[0])
        has_exact = geometry.overlap_volume(*(np.zeros((1, geometry.dim)),) * 4) is not None
        if overlap == "auto":
            overlap = "exact" if has_exact else "plugin"
        if overlap == "exact" and not has_exact:
            raise ValueError("this geometry has no closed form for |A ∩ B|")
        if overlap == "stochastic" and eps != 1:
            raise ValueError("the stochastic mode is only implemented for eps = 1")
        if not 0 < eps <= 1:
            raise ValueError("eps must lie in (0, 1]")
        self.eps = float(eps)
        self.overlap_mode = overlap
        self.overlap_points = overlap_points
        self.exact_overlap = overlap == "exact"

    # -- utilities ------------------------------------------------------

    def _local(self, base, mixture=False):
        """Local density factor of the proposal: 1, except in geometries with
        ``local_proposal`` (FLRW): the relative physical density at the base
        point, bounded from below. Towards the future (``mixture``) the base
        point is the bottom vertex, which underestimates the density of the
        interval: it is mixed with the global one. The support of the proposal
        does not change."""
        if not getattr(self.g, "local_proposal", False):
            return 1.0
        rel = np.maximum(self.g.weight(base) * self.g.coordinate_volume / self.g.volume, 0.02)
        return (rel, 1.0) if mixture else rel

    def _past(self, base):
        """y = base - o with o a future displacement; returns (y, q)."""
        o, q = self.prop.sample(len(base), self.rng, base[:, 0] - self._t_min, self._local(base))
        return base - o, q

    def _future(self, base):
        o, q = self.prop.sample(len(base), self.rng, self._t_max - base[:, 0], self._local(base, mixture=True))
        return base + o, q

    def _inside(self, y, z, x):
        return self.g.precedes(y, z) & self.g.precedes(z, x)

    def _overlap(self, yA, xA, yB, xB, VA):
        """|A ∩ B|: exact or estimated with ``overlap_points`` points in A."""
        if self.exact_overlap:
            return self.g.overlap_volume(yA, xA, yB, xB)
        K = self.overlap_points
        m = len(VA)
        hits = np.zeros(m)
        chunk = max(1, 2_000_000 // K)
        for s in range(0, m, chunk):
            sl = slice(s, min(s + chunk, m))
            k = sl.stop - sl.start
            pts, owner = self.g.sample_in_intervals(yA[sl], xA[sl], np.full(k, K), self.rng)
            inB = self._inside(yB[sl][owner], pts, xB[sl][owner])
            hits[sl] = np.bincount(owner, weights=inB, minlength=k)
        return VA * hits / K

    def _joint(self, yA, xA, yB, xB, VA, VB, sA, sB, Vb=None):
        """<zeta_{i-sA}(A) zeta_{j-sB}(B)> for i, j = 0..n-1, shape (m, n, n).
        Exact if |A∩B| has a closed form; otherwise, indicator of a local
        Poisson realisation (unbiased estimator)."""
        rho, n = self.rho, self.n
        m = len(VA)
        if self.overlap_mode in ("exact", "plugin"):
            if Vb is None:
                Vb = self._overlap(yA, xA, yB, xB, VA)
            return thinned_joint_probability(rho * VA, rho * VB, rho * Vb, sA, sB, n, self.eps)
        rng = self.rng
        nA = rng.poisson(rho * VA)
        active = nA <= n  # if nA > n - 1 + sA it contributes to no i < n
        n_ab = np.zeros(m, dtype=int)
        idx = np.nonzero(active & (nA > 0))[0]
        if len(idx):
            pts, owner = self.g.sample_in_intervals(yA[idx], xA[idx], nA[idx], rng)
            inB = self._inside(yB[idx][owner], pts, xB[idx][owner])
            n_ab[idx] = np.bincount(owner, weights=inB, minlength=len(idx)).astype(int)
        m_c = np.zeros(m, dtype=int)
        nB = np.where(active, rng.poisson(rho * VB), 0)
        idx = np.nonzero(active & (nB > 0))[0]
        if len(idx):
            pts, owner = self.g.sample_in_intervals(yB[idx], xB[idx], nB[idx], rng)
            outA = ~self._inside(yA[idx][owner], pts, xA[idx][owner])
            m_c[idx] = np.bincount(owner, weights=outA, minlength=len(idx)).astype(int)
        cA, cB = nA + sA, n_ab + m_c + sB  # indices i, j it contributes to
        out = np.zeros((m, n, n))
        ok = active & (cA < n) & (cB < n)
        out[np.nonzero(ok)[0], cA[ok], cB[ok]] = 1.0
        return out

    def _inverse_overlap(self, yA, xA, yB, xB, VA):
        """Unbiased estimator of 1/|A∩B|: K/|A| with K ~ Geometric(|A∩B|/|A|)."""
        m = len(VA)
        K = np.zeros(m)
        pending = np.arange(m)
        rng = self.rng
        while len(pending):
            pts, owner = self.g.sample_in_intervals(yA[pending], xA[pending], np.full(len(pending), 32), rng)
            hit = self._inside(yB[pending][owner], pts, xB[pending][owner])
            # position (1..32) of the first hit in each interval
            pos = np.full(len(pending), 33)
            order = np.arange(len(owner)) - np.searchsorted(owner, owner) + 1
            np.minimum.at(pos, owner[hit], order[hit])
            found = pos <= 32
            K[pending] += np.where(found, pos, 32)
            pending = pending[~found]
        return K / VA

    def _contract(self, samples: np.ndarray, a: np.ndarray) -> TermEstimate:
        vals = np.einsum("mij,i,j->m", samples, a, a)
        return TermEstimate(samples.mean(axis=0), float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals))), len(vals))

    def _points_in_M(self, *pts):
        ok = np.ones(len(pts[0]), dtype=bool)
        for p in pts:
            ok &= self.g.contains(p)
        return ok

    # -- terms ------------------------------------------------------------

    def abundance_means(self, m: int, n_levels: int) -> tuple[np.ndarray, np.ndarray]:
        """L_k = <N_k> (MYZ 5.15), k = 0..n_levels-1, by MC (with eps < 1,
        <Ñ_k> = rho^2 ∫∫ P_k(eps rho V)). Returns (L, se)."""
        x = self.g.sample_M(m, self.rng)
        y, q = self._past(x)
        ok = self._points_in_M(y)
        V = self.g.interval_volume(y, x)
        w = np.where(ok, self.rho**2 * self.g.volume * self.g.weight(y) / q, 0.0)
        vals = w[:, None] * _pmf_table(self.eps * self.rho * V, n_levels)
        return vals.mean(axis=0), vals.std(axis=0, ddof=1) / np.sqrt(m)

    def term_J1(self, m: int, a: np.ndarray) -> TermEstimate:
        """J1_ij (MYZ 5.47): x1 = x2, y1 = y2. With eps = 1 it is delta_ij L_i; with
        eps < 1 the two factors Ñ_i, Ñ_j of the same pair are independent
        thinnings of the same interval: <..> = thinned_joint with A = B."""
        y, x, w = self._sample_pair(m)
        w = np.where(self._points_in_M(y), w, 0.0)
        V = self.g.interval_volume(y, x)
        zeros = np.zeros(m, dtype=int)
        G = thinned_joint_probability(self.rho * V, self.rho * V, self.rho * V, zeros, zeros, self.n, self.eps)
        return self._contract(w[:, None, None] * G, a)

    def term_J4(self, m: int, a: np.ndarray) -> TermEstimate:
        """J4_ij (MYZ 5.48): chain y1 ≺ x1 = y2 ≺ x2; disjoint intervals."""
        x1 = self.g.sample_M(m, self.rng)
        y1, q1 = self._past(x1)
        x2, q2 = self._future(x1)
        ok = self._points_in_M(y1, x2)
        w = np.where(ok, self.rho**3 * self.g.volume * self.g.weight(y1) * self.g.weight(x2) / (q1 * q2), 0.0)
        er = self.eps * self.rho
        Pi = _pmf_table(er * self.g.interval_volume(y1, x1), self.n)
        Pj = _pmf_table(er * self.g.interval_volume(x1, x2), self.n)
        return self._contract(w[:, None, None] * Pi[:, :, None] * Pj[:, None, :], a)

    def term_J2(self, m: int, a: np.ndarray) -> TermEstimate:
        """J2_ij (MYZ 5.52-5.56): common bottom vertex y, A = I(y,x1), B = I(y,x2)."""
        y = self.g.sample_M(m, self.rng)
        x1, q1 = self._future(y)
        x2, q2 = self._future(y)
        ok = self._points_in_M(x1, x2)
        w = np.where(ok, self.rho**3 * self.g.volume * self.g.weight(x1) * self.g.weight(x2) / (q1 * q2), 0.0)
        VA, VB = self.g.interval_volume(y, x1), self.g.interval_volume(y, x2)
        sA = self._inside(y, x2, x1).astype(int)  # x2 ∈ A
        sB = self._inside(y, x1, x2).astype(int)  # x1 ∈ B
        G = np.zeros((m, self.n, self.n))
        sel = np.nonzero(w != 0)[0]
        if len(sel):
            G[sel] = self._joint(y[sel], x1[sel], y[sel], x2[sel], VA[sel], VB[sel], sA[sel], sB[sel])
        return self._contract(w[:, None, None] * G, a)

    def term_J3(self, m: int, a: np.ndarray) -> TermEstimate:
        """J3_ij (MYZ 5.33): common top vertex x, A = I(y1,x), B = I(y2,x)."""
        x = self.g.sample_M(m, self.rng)
        y1, q1 = self._past(x)
        y2, q2 = self._past(x)
        ok = self._points_in_M(y1, y2)
        w = np.where(ok, self.rho**3 * self.g.volume * self.g.weight(y1) * self.g.weight(y2) / (q1 * q2), 0.0)
        VA, VB = self.g.interval_volume(y1, x), self.g.interval_volume(y2, x)
        sA = self._inside(y1, y2, x).astype(int)  # y2 ∈ A
        sB = self._inside(y2, y1, x).astype(int)  # y1 ∈ B
        G = np.zeros((m, self.n, self.n))
        sel = np.nonzero(w != 0)[0]
        if len(sel):
            G[sel] = self._joint(y1[sel], x[sel], y2[sel], x[sel], VA[sel], VB[sel], sA[sel], sB[sel])
        return self._contract(w[:, None, None] * G, a)

    def _j6_integrand(self, y1, x1, y2, x2, base, use_inverse_overlap):
        """(G - P_i P_j) * base [* 1/|A∩B|] for configurations with base != 0."""
        g, rho, n = self.g, self.rho, self.n
        m = len(base)
        out = np.zeros((m, n, n))
        sel = np.nonzero(base != 0)[0]
        if not len(sel):
            return out
        y1, x1, y2, x2 = y1[sel], x1[sel], y2[sel], x2[sel]
        VA, VB = g.interval_volume(y1, x1), g.interval_volume(y2, x2)
        sA = self._inside(y1, x2, x1).astype(int) + self._inside(y1, y2, x1)  # x2, y2 ∈ A
        sB = self._inside(y2, x1, x2).astype(int) + self._inside(y2, y1, x2)  # x1, y1 ∈ B
        Vb = self._overlap(y1, x1, y2, x2, VA) if self.overlap_mode != "stochastic" else None
        G = self._joint(y1, x1, y2, x2, VA, VB, sA, sB, Vb)
        er = self.eps * rho
        PP = _pmf_table(er * VA, n)[:, :, None] * _pmf_table(er * VB, n)[:, None, :]
        f = base[sel]
        if use_inverse_overlap:
            if Vb is None:
                inv = self._inverse_overlap(y1, x1, y2, x2, VA)
            else:
                # plugin: if no point falls in A ∩ B (overlap < |A|/K), it is
                # bounded by half a point; the bias is controlled by varying K
                inv = 1.0 / np.maximum(Vb, VA * 0.5 / self.overlap_points)
            f = f * inv
        out[sel] = f[:, None, None] * (G - PP)
        return out

    def _sample_pair(self, m):
        """Pair (y, x) with x uniform in M and y = x - o. Returns y, x, weight
        rho^2 V_M w(y) / q (without checking that y is in M)."""
        x = self.g.sample_M(m, self.rng)
        y, q = self._past(x)
        return y, x, self.rho**2 * self.g.volume * self.g.weight(y) / q

    def _anchor_in(self, y, x, m):
        """One uniform point (physical dV) in each I(y[k], x[k]); weight rho |I|.
        If y is outside M, the configuration has weight 0 (all uses require
        y in M) and x is returned without sampling: in FLRW an interval that
        crosses eta = 0 would make the rejection arbitrarily slow. (Only with
        ``local_proposal``, so as not to change the random sequence of the
        calculations already stored for Minkowski and de Sitter.)"""
        if not getattr(self.g, "local_proposal", False):
            pts, _ = self.g.sample_in_intervals(y, x, np.ones(m, dtype=int), self.rng)
            return pts, self.rho * self.g.interval_volume(y, x)
        ok = self.g.contains(y)
        pts = x.copy()
        if ok.any():
            pts[ok], _ = self.g.sample_in_intervals(y[ok], x[ok], np.ones(int(ok.sum()), dtype=int), self.rng)
        return pts, self.rho * self.g.interval_volume(y, x)

    def term_J6_connected(self, m: int, a: np.ndarray) -> TermEstimate:
        """J6_ij - L_i L_j (MYZ 5.57-5.60 minus 5.10). Partition of the domain
        (equivalent to the subregions of D_6, MYZ 5.44), without overlaps:
          c1: x2 ∈ A
          c2: y2 ∈ A, x2 ∉ A
          c3: x1 ∈ B, x2, y2 ∉ A
          c4: y1 ∈ B, x2, y2 ∉ A, x1 ∉ B
          c0: A ∩ B ≠ ∅ with no endpoint inside the other interval
        (outside these cases A and B are disjoint and the connected integrand is 0).
        In c1-c4 the contained point is sampled inside the interval; in c0 one
        integrates over w ∈ A ∩ B with weight 1/|A ∩ B|."""
        return self._contract(sum(self._j6_patterns(m)), a)

    def _j6_patterns(self, m):
        rng, g, rho = self.rng, self.g, self.rho
        inside = self._inside
        out = []
        # c1, c2: sample A and a point of B inside A
        for pattern in ("c1", "c2"):
            y1, x1, wA = self._sample_pair(m)
            z, wz = self._anchor_in(y1, x1, m)
            if pattern == "c1":
                y2, q = self._past(z)
                x2 = z
                w_other = g.weight(y2)
                excl = np.zeros(m, dtype=bool)
            else:
                x2, q = self._future(z)
                y2 = z
                w_other = g.weight(x2)
                excl = inside(y1, x2, x1)
            ok = self._points_in_M(y1, y2, x2) & ~excl
            base = np.where(ok, wA * wz * rho * w_other / q, 0.0)
            out.append(self._j6_integrand(y1, x1, y2, x2, base, False))
        # c3, c4: sample B and a point of A inside B
        for pattern in ("c3", "c4"):
            y2, x2, wB = self._sample_pair(m)
            z, wz = self._anchor_in(y2, x2, m)
            if pattern == "c3":
                y1, q = self._past(z)
                x1 = z
                w_other = g.weight(y1)
                excl = np.zeros(m, dtype=bool)
            else:
                x1, q = self._future(z)
                y1 = z
                w_other = g.weight(x1)
                excl = inside(y2, x1, x2)
            excl = excl | inside(y1, x2, x1) | inside(y1, y2, x1)
            ok = self._points_in_M(y1, x1, y2) & ~excl
            base = np.where(ok, wB * wz * rho * w_other / q, 0.0)
            out.append(self._j6_integrand(y1, x1, y2, x2, base, False))
        # c0: A as a pair; w uniform in A; B ∋ w with two displacements;
        # weight 1/|A ∩ B|. No endpoints inside the other interval.
        y1, x1, wA = self._sample_pair(m)
        w, ww = self._anchor_in(y1, x1, m)
        x2, q2 = self._future(w)
        y2, q3 = self._past(w)
        excl = inside(y1, x2, x1) | inside(y1, y2, x1) | inside(y2, x1, x2) | inside(y2, y1, x2)
        ok = self._points_in_M(y1, x2, y2) & ~excl
        # ww = rho |A|: |A| is used (density 1/|A| of w) and rho^2 for x2, y2
        base = np.where(ok, wA * (ww / rho) * rho**2 * g.weight(x2) * g.weight(y2) / (q2 * q3), 0.0)
        out.append(self._j6_integrand(y1, x1, y2, x2, base, True))
        return out

    # -- assembly -----------------------------------------------------------

    def predict(self, coefficients=None, m_terms: int = 200_000, m_means: int = 400_000, exact_means=None, batch: int = 50_000) -> FluctuationPrediction:
        """(Delta S)^2 = a^T M a. ``exact_means``: exact L_0..L_n if known
        (Minkowski: ``expected_thinned_abundance``); otherwise MC.

        With eps < 1, M is the covariance of (N, Ñ_0, ...); the row M_{-1,k}
        keeps the form (k+2) L_k - (k+1) L_{k+1} with L_k = <Ñ_k>: by the
        Mecke formula, Cov(N, sum_pairs h(n)) = rho^2 ∫∫ E[(2 + n - rho V) h(n)],
        and for h(n) = B_k(n, eps) one splits n = (retained) + (not retained)."""
        a = action_coefficients(self.g.dim, self.eps) if coefficients is None else np.asarray(coefficients, float)
        n = self.n
        assert len(a) == n + 1
        aN, ak = a[0], a[1:]
        rhoV = self.rho * self.g.volume

        if exact_means is not None:
            L = np.asarray(exact_means, float)[: n + 1]
            L_se = np.zeros(n + 1)
        else:
            L, L_se = self._batched_means(m_means, n + 1, batch)

        terms = {}
        fns = [("J2", self.term_J2), ("J3", self.term_J3), ("J4", self.term_J4), ("J6c", self.term_J6_connected)]
        if self.eps != 1:
            fns.insert(0, ("J1", self.term_J1))
        for name, fn in fns:
            terms[name] = self._batched(fn, m_terms, ak, batch)

        J1 = np.diag(L[:n]) if self.eps == 1 else terms["J1"].matrix
        Mkk = J1 + terms["J2"].matrix + terms["J3"].matrix + terms["J4"].matrix + terms["J4"].matrix.T + terms["J6c"].matrix
        Mkk = (Mkk + Mkk.T) / 2
        j = np.arange(n)
        MNk = (j + 2) * L[:n] - (j + 1) * L[1 : n + 1]
        cov = np.zeros((n + 1, n + 1))
        cov[0, 0] = rhoV
        cov[0, 1:] = cov[1:, 0] = MNk
        cov[1:, 1:] = Mkk

        var = float(a @ cov @ a)
        # error: independent MC terms + error of L (linear in L)
        se2 = sum(t.contraction_se**2 for name, t in terms.items() if name != "J4") + 4 * terms["J4"].contraction_se**2
        dvar_dL = np.zeros(n + 1)
        dvar_dL[:n] += (ak**2 if self.eps == 1 else 0.0) + 2 * aN * ak * (j + 2)
        dvar_dL[1 : n + 1] += -2 * aN * ak * (j + 1)
        se2 += float(np.sum((dvar_dL * L_se) ** 2))
        return FluctuationPrediction(self.rho, a, np.concatenate([[rhoV], L]), cov, var, float(np.sqrt(se2)), terms)

    def _batched(self, fn, m, a, batch):
        parts = [fn(min(batch, m - s), a) for s in range(0, m, batch)]
        tot = sum(p.n_samples for p in parts)
        mat = sum(p.matrix * p.n_samples for p in parts) / tot
        mean = sum(p.contraction * p.n_samples for p in parts) / tot
        # combine per-batch variances (same estimator): se^2 = sum (n_b^2 se_b^2) / N^2
        se = np.sqrt(sum((p.n_samples * p.contraction_se) ** 2 for p in parts)) / tot
        return TermEstimate(mat, mean, float(se), tot)

    def _batched_means(self, m, n_levels, batch):
        sums, sq = np.zeros(n_levels), np.zeros(n_levels)
        tot = 0
        for s in range(0, m, batch):
            k = min(batch, m - s)
            mean, se = self.abundance_means(k, n_levels)
            sums += mean * k
            sq += (se * k) ** 2
            tot += k
        return sums / tot, np.sqrt(sq) / tot


def expected_thinned_abundance(dim: int, lam: float, k: int, eps: float = 1.0) -> float:
    """Exact <Ñ_k> in a flat diamond with rho V = lam:
    rho^2 ∫∫ P_k(eps rho V) = L_k(eps lam) / eps^2 (substitution rho -> eps rho)."""
    from .continuum import expected_abundance

    return expected_abundance(dim, eps * lam, k) / eps**2


def pair_weight(dim: int, n, eps: float = 1.0) -> np.ndarray:
    """f_eps(n) = sum_k a_k B_k(n, eps): contribution to S_eps of a pair with n
    elements between its endpoints (without the a_N N term). With eps = 1,
    f(n) = a_n for n < n_d and 0 otherwise."""
    from scipy.stats import binom

    a = action_coefficients(dim, eps)[1:]
    n = np.asarray(n)
    return sum(a[k] * binom.pmf(k, n, eps) for k in range(len(a)))


def same_pair_log_coefficient(eps: float = 1.0, n_max: int = 20_000) -> float:
    """Coefficient b of (Delta S)^2 ≈ b rho V ln(rho V) + O(rho V) contributed
    by the "same pair" term (J¹ of MYZ) in a 2D Minkowski diamond.

    Derivation (Goal 3d, see README): in light-cone coordinates, with a
    diamond of side l (V = l²/2),
        <sum_pairs h(n)> = (rho²/4) ∫∫_0^l (l-a)(l-b) E_{n~Poisson(rho a b/2)}[h(n)] da db.
    At fixed ab = x, ∫_{x/l}^{l} da/a = ln(l²/x): the rapidity range of the
    nearly null intervals, cut off by the diamond. With ∫_0^∞ P_n(mu) dmu = 1,
        <sum_pairs h(n)> = rho V ln(rho V) sum_n h(n) + O(rho V).
    For h = f_eps² one gets b = sum_n f_eps(n)². The cross term with N carries
    sum_n f_eps(n) = 0 (because sum_k C_k = 0), and has no logarithm."""
    return float(np.sum(pair_weight(2, np.arange(n_max), eps) ** 2))


def smeared_pair_moment(dim: int, lam: float, eps: float = 1.0, power: int = 1) -> float:
    """Exact E[sum_{pairs y ≺ x} f_eps(n(y,x))^power] in a flat diamond
    with rho V = lam (``continuum._diamond_pair_integral`` with
    H(z) = E_{n~Poisson(z)}[f_eps(n)^power]). With power = 2 it is the
    "same pair" term J¹ of MYZ for the smeared action; with power = 1, the
    mean of the pair part of S_eps. Only d = 4 or 2 (kernel of ``continuum``)."""
    from scipy.stats import poisson

    from .continuum import _diamond_pair_integral

    n_max = int(80 / eps) + 200
    n = np.arange(n_max)
    g = pair_weight(dim, n, eps) ** power

    def H(z):
        lo = max(0, int(z - 12 * np.sqrt(z + 1) - 10))
        hi = min(n_max, int(z + 12 * np.sqrt(z + 1) + 20))
        return float(poisson.pmf(n[lo:hi], z) @ g[lo:hi]) if lo < n_max else 0.0

    return _diamond_pair_integral(H, dim, lam, decay=eps)


def smeared_pair_mecke_slope(dim: int, lam: float, eps: float = 1.0, h: float = 0.01) -> float:
    """g = d E[sum_pairs f_eps] / d(rho V). With the Mecke equation (Last &
    Penrose, "Lectures on the Poisson Process", Cambridge UP 2017, thm. 4.1),
    E[N F] = rho ∫ E[F(eta + δ_x)] dx, and the derivative with respect to the
    intensity, rho dE[F]/drho = rho ∫ E[F(eta + δ_x) - F(eta)] dx, one gets
    Cov(N, F) = rho dE[F]/drho = lam * g. With eps = 1 it reproduces the exact
    cross term (k+2)L_k - (k+1)L_{k+1} of Step 2 (test). Central difference
    with relative step h."""
    up = smeared_pair_moment(dim, lam * (1 + h), eps)
    dn = smeared_pair_moment(dim, lam * (1 - h), eps)
    return (up - dn) / (2 * h * lam)
