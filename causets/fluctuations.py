"""Fluctuaciones de la acción de Benincasa-Dowker (BD): formalismo de
Moradi, Yazdi & Zilhão (MYZ), "Fluctuations and Correlations in Causal Set
Theory", Class. Quantum Grav. 42 (2025) 045017, arXiv:2407.03395
(referencias a ecuaciones: versión v2 del arXiv).

QUÉ ACCIÓN
----------
Todo este módulo trata la acción BD **LOCAL** (eps = 1): la que MYZ escriben
en (4.1)-(4.4) y cuya varianza descomponen en la sec. 5. La acción es una
combinación lineal finita de abundancias,

    S = a_N N + sum_{k=0}^{n_d - 1} a_k N_k,

donde N_k es el número de pares y ≺ x con k elementos entre ellos
(MYZ: N_k = sum_x sum_{y in ◇_{k+1}(x)} 1) y N ≡ N_{-1} el número de elementos.
Con la normalización del Goal 2 (``action.py``):
    2D:  a = (1, -2, 4, -2)                  [S = N - 2N_0 + 4N_1 - 2N_2]
    4D:  a = (4/sqrt6) (1, -1, 9, -16, 8)    [S = (4/sqrt6)(N - N_0 + 9N_1 - 16N_2 + 8N_3)]
MYZ normalizan S = -(2/alpha_d) sum_i C_i N_{i-1} (5.4); p. ej. en 4D su S es
-(sqrt6/2) veces la nuestra, así que su varianza es 3/2 la nuestra. La
descomposición no depende de esa constante: (Delta S)^2 = a^T M a (MYZ 5.9).
La acción suavizada (eps < 1) involucra todas las N_n y queda como trabajo
futuro (ver README).

FORMALISMO
----------
- Indicador de cardinalidad zeta_i(x, y) (MYZ 3.2, 3.4): 1 si y ≺ x y hay
  exactamente i elementos en I(y, x). ``cardinality_indicator``.
- Indicador de ocupación chi(R) (MYZ 3.16): 1 si la región R contiene algún
  elemento. ``occupation_indicator``.
- Matriz de correlación M_ij = K_ij - L_i L_j = Cov(N_i, N_j),
  i, j = -1, ..., n_d - 1 (MYZ 5.7, 5.10), con L_i = <N_i> (5.5, 5.15).
- Componentes:
    M_{-1,-1} = rho V_M                                            (MYZ 5.20)
    M_{-1,j}  = K_{-1,j} - rho V_M L_j                              (MYZ 5.26-5.27)
              = (j + 2) L_j - (j + 1) L_{j+1}   [integrando 5.26 con
                z P_{j-1}(z) = j P_j(z); identidad exacta en cualquier M]
    K_ij = J1 + ... + J6 (MYZ 5.29-5.36), según qué puntos coinciden:
      J1: x1 = x2, y1 = y2   ->  delta_ij L_i                        (5.47)
      J2: y1 = y2 (x1 != x2)     J3: x1 = x2 (y1 != y2)
      J4: x1 = y2                J5: x2 = y1   (J4_ij = J5_ji, 5.37)
      J6: los cuatro distintos
  Cada integrando se evalúa con las reglas de la sec. 3.3: chi^n = chi
  (3.30); cada punto de la configuración contenido en el intervalo de la
  otra zeta reduce en 1 el número de elementos requerido (3.31); y las zetas
  de regiones solapadas se descomponen en regiones disjuntas (3.15, 3.28):

    <zeta_a(A) zeta_b(B)> = sum_beta P_{a-beta}(rho V_{A\\B}) P_beta(rho V_{A∩B}) P_{b-beta}(rho V_{B\\A}).

  Las 11 subregiones de D_6 (5.44) y las de D_2, D_3 (5.40-5.43) son
  exactamente los patrones "qué puntos caen en qué intervalo", que aquí se
  detectan configuración a configuración en lugar de enumerarse a mano.

- Parte conexa: en J6 se resta L_i L_j dentro del integrando,
      J6c_ij = rho^4 ∫ [<zeta_i zeta_j chi..chi> / rho^4 - P_i(rho V_A) P_j(rho V_B)],
  que sólo es distinto de 0 cuando I(y1,x1) ∩ I(y2,x2) ≠ ∅. Así
      M_ij = delta_ij L_i + J2 + J3 + J4 + J5 + J6c   (i, j >= 0)
  sin cancelaciones entre números enormes.

EVALUACIÓN NUMÉRICA DE LAS INTEGRALES (lo que MYZ dejan como trabajo futuro)
----------------------------------------------------------------------------
Integración Monte Carlo con muestreo de importancia: los intervalos se
generan alrededor de un punto con desplazamientos temporales
y = x - tau (cosh chi, sinh chi n), muestreados según la propia medida de
volumen (ver ``_Proposal``), de modo que los intervalos pequeños y casi
nulos (los que dominan N_0..N_3) están bien muestreados y los pesos no tienen
colas pesadas. J6 se parte en patrones (``term_J6_connected``): si un
extremo de un intervalo cae dentro del otro, ese punto se muestrea dentro del
intervalo; si sólo hay solapamiento, se integra sobre w ∈ A ∩ B con peso
1/|A ∩ B| (identidad ∫ dB f = ∫_A dw ∫_{B ∋ w} dB f / |A ∩ B|), donde el
integrando conexo se anula de forma suave cuando |A ∩ B| -> 0.

- Minkowski 2D: |A ∩ B| tiene forma cerrada (intervalos = rectángulos en
  coordenadas de cono de luz), y los integrandos se evalúan EXACTAMENTE.
  Único error: el estadístico de la integración MC, que se reporta.
- 4D y de Sitter: |A ∩ B| no tiene forma cerrada. Se usan estimadores
  INSESGADOS: <zeta_a(A) zeta_b(B)> se estima con una realización de Poisson
  local en A ∪ B (indicador de que los recuentos coinciden), y 1/|A ∩ B|
  con ensayos geométricos (E[K] = |A| / |A ∩ B|). Sin sesgo, pero con más
  varianza; el error estadístico reportado lo incluye.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln

from .intervals import interval_cardinalities
from .regions import unit_sphere_area

# ---------------------------------------------------------------------------
# Coeficientes de la acción local
# ---------------------------------------------------------------------------

LOCAL_ACTION_COEFFICIENTS = {
    2: np.array([1.0, -2.0, 4.0, -2.0]),
    4: 4.0 / np.sqrt(6.0) * np.array([1.0, -1.0, 9.0, -16.0, 8.0]),
}


def local_action_coefficients(dim: int) -> np.ndarray:
    """a = (a_N, a_0, ..., a_{n_d - 1}) de la acción BD local (normalización Goal 2)."""
    return LOCAL_ACTION_COEFFICIENTS[dim].copy()


def smeared_action_coefficients(dim: int, eps: float) -> np.ndarray:
    """a = (a_N, a_0, ..., a_{n_d - 1}) de la acción suavizada escrita con
    abundancias adelgazadas (ver ``thinned_abundance_vector``):
    S_eps = a_N N + sum_k a_k Ñ_k, con a_N = -s_d alpha beta eps^(2/d) y
    a_k = -s_d alpha eps^(1+2/d) C_k (``action.bd_action_from_pairs``).
    Con eps = 1 coincide con ``local_action_coefficients``."""
    from .action import coefficients

    c = coefficients(dim)
    pref = -c.action_scale * c.alpha * eps ** (2.0 / dim)
    return np.concatenate([[pref * c.beta], pref * eps * np.asarray(c.layers)])


def action_coefficients(dim: int, eps: float = 1.0) -> np.ndarray:
    return local_action_coefficients(dim) if eps == 1 else smeared_action_coefficients(dim, eps)


# ---------------------------------------------------------------------------
# Indicadores de MYZ sobre un conjunto causal concreto
# ---------------------------------------------------------------------------


def cardinality_indicator(C: np.ndarray, i: int) -> np.ndarray:
    """zeta_i (MYZ 3.4): Z[y, x] = 1 si y ≺ x con exactamente i elementos entre ellos."""
    C = np.asarray(C, dtype=bool)
    return C & (interval_cardinalities(C) == i)


def occupation_indicator(points: np.ndarray, contains) -> bool:
    """chi(R) (MYZ 3.16): True si algún elemento cae en la región R
    (``contains`` es la función de pertenencia de R)."""
    return bool(np.any(contains(np.atleast_2d(points)))) if len(points) else False


def abundance_vector(C: np.ndarray, n_max: int) -> np.ndarray:
    """(N, N_0, ..., N_{n_max - 1}) de un conjunto causal."""
    C = np.asarray(C, dtype=bool)
    n = interval_cardinalities(C)[C]
    counts = np.bincount(n, minlength=n_max)[:n_max]
    return np.concatenate([[C.shape[0]], counts]).astype(float)


def thinned_abundance_vector(C: np.ndarray, n_max: int, eps: float) -> np.ndarray:
    """(N, Ñ_0, ..., Ñ_{n_max - 1}) con Ñ_k = sum_{y ≺ x} B_k(n(y,x), eps),
    B_k(n, eps) = binom(n, k) eps^k (1-eps)^(n-k): número esperado de pares
    con k elementos entre ellos tras adelgazar el intervalo con probabilidad
    de retención eps. f_d(n, eps) = sum_k C_k B_k(n, eps) (``action.py``), así
    que S_eps = a · Ñ con ``smeared_action_coefficients``. Con eps = 1, Ñ_k = N_k."""
    from scipy.stats import binom

    if eps == 1:
        return abundance_vector(C, n_max)
    C = np.asarray(C, dtype=bool)
    counts = np.bincount(interval_cardinalities(C)[C])
    ns = np.arange(len(counts))
    thinned = [float(np.sum(counts * binom.pmf(k, ns, eps))) for k in range(n_max)]
    return np.array([C.shape[0], *thinned], dtype=float)


# ---------------------------------------------------------------------------
# Lado numérico: varianza sobre un conjunto de sprinklings
# ---------------------------------------------------------------------------


@dataclass
class EnsembleFluctuations:
    """Varianza de la acción y covarianza de abundancias sobre M sprinklings."""

    abundances: np.ndarray  # (M, n_max + 1): columnas N, N_0, ...
    coefficients: np.ndarray

    @property
    def actions(self) -> np.ndarray:
        return self.abundances @ self.coefficients

    @property
    def variance(self) -> float:
        return float(np.var(self.actions, ddof=1))

    @property
    def variance_se(self) -> float:
        """Error estándar de la varianza muestral: std(d^2) / sqrt(M), d = S - <S>."""
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
    """Sprinklings de Poisson independientes de ``geometry`` (sin cadena de
    Markov: no hay termalización) y sus abundancias (adelgazadas si eps < 1)."""
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
# Lado analítico (formalismo MYZ)
# ---------------------------------------------------------------------------


def _pmf_table(z: np.ndarray, n: int) -> np.ndarray:
    """P_k(z) para k = 0..n-1, forma (len(z), n)."""
    z = np.asarray(z, dtype=float)[:, None]
    k = np.arange(n)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = k * np.log(np.where(z > 0, z, 1.0)) - z - gammaln(k + 1)
    p = np.exp(logp)
    return np.where(z > 0, p, (k == 0).astype(float))


def _shifted(P: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """G[m, i, j] = P[m, i - sa[m], j - sb[m]] (0 si algún índice es negativo)."""
    m, n, _ = P.shape
    i = np.arange(n)[None, :, None] - sa[:, None, None]
    j = np.arange(n)[None, None, :] - sb[:, None, None]
    ok = (i >= 0) & (j >= 0)
    rows = np.arange(m)[:, None, None]
    return np.where(ok, P[rows, np.clip(i, 0, None), np.clip(j, 0, None)], 0.0)


def joint_cardinality_probability(za, zb, zc, n: int) -> np.ndarray:
    """<zeta_a(A) zeta_b(B)> para a, b = 0..n-1 (MYZ 3.15, 3.28, 3.34):
    sum_beta P_{a-beta}(za) P_beta(zb) P_{b-beta}(zc), con za = rho|A\\B|,
    zb = rho|A∩B|, zc = rho|B\\A|. Forma (m, n, n)."""
    pa, pb, pc = _pmf_table(za, n), _pmf_table(zb, n), _pmf_table(zc, n)
    out = np.zeros((len(pa), n, n))
    for beta in range(n):
        # a = alpha + beta, b = gamma + beta
        out[:, beta:, beta:] += pb[:, beta, None, None] * pa[:, : n - beta, None] * pc[:, None, : n - beta]
    return out


def thinned_joint_probability(zA, zB, zAB, sA, sB, n: int, eps: float = 1.0) -> np.ndarray:
    """<zeta_{i-sA}(A) zeta_{j-sB}(B)> para i, j = 0..n-1 tras adelgazar A y B
    de forma independiente con probabilidad de retención eps, forma (m, n, n).
    zA = rho|A|, zB = rho|B|, zAB = rho|A∩B|; sA (sB) es el número de puntos de
    la configuración contenidos en A (B).

    Con eps = 1 es MYZ (3.15, 3.31). Para eps < 1: los elementos de A∩B se
    reparten (adelgazamiento de un proceso de Poisson) en procesos de Poisson
    independientes retenidos por ambos (eps^2), sólo por A o sólo por B
    (eps(1-eps) cada uno), así que la estructura de (3.15) se mantiene con
        zA' = eps zA - eps^2 zAB,  zAB' = eps^2 zAB,  zB' = eps zB - eps^2 zAB,
    y cada punto fijo contenido se retiene con probabilidad eps:
    el desplazamiento sA pasa a ser Binomial(sA, eps)."""
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
    raise ValueError("propuesta implementada para d = 2, 3, 4")


@dataclass
class _Proposal:
    """Desplazamientos temporales futuros o = tau (cosh chi, sinh chi n).

    Se muestrea z = rho_c c_d tau^d (el "volumen coordenado en unidades de
    elementos") con z ~ Gamma(2/d, z0), chi con densidad sinh^{d-2}(chi) en
    [0, ln(4 L / tau)] y n uniforme en S^{d-2}. Como la medida de volumen en
    estas variables es d^d o = dz sinh^{d-2}chi dchi dOmega / (d rho_c c_d),
    los pesos 1/q son suaves (e^{z/z0} z^{1-2/d} S_d(chi_max)): no hay colas
    pesadas. ``density`` devuelve q(o) en medida coordenada.

    ``z0`` puede ser una tupla de escalas: entonces z se muestrea de la mezcla
    equiponderada de las Gammas ("defensive importance sampling", Hesterberg,
    Technometrics 37 (1995) 185), y el peso queda acotado por el de la mejor
    componente. Se usa con la acción suavizada, donde los intervalos
    relevantes tienen rho V ~ 1/eps además de los pequeños."""

    dim: int
    extent: float
    rho_c: float  # densidad en medida coordenada (rho en Minkowski)
    z0: float | tuple = 3.0

    def sample(self, m: int, rng, budget, local=1.0) -> tuple[np.ndarray, np.ndarray]:
        """``budget``: tiempo coordenado disponible desde cada punto base hasta
        el borde temporal de M; se impone tau cosh(chi) <= budget, i.e.
        chi <= arccosh(budget / tau) (condición necesaria para seguir en M).
        Si tau >= budget la muestra es inválida (q = inf, peso 0).
        ``local``: factor (por punto base) que multiplica rho_c, para adaptar
        la escala de los intervalos a la densidad física local (FLRW). Si es
        una tupla de factores, cada muestra usa uno al azar y q es la densidad
        de la mezcla equiponderada (como con ``z0``)."""
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
                """Densidad de la mezcla de Gammas en z' = rc c tau^d, por rc."""
                zz = rc * c * tau**d
                return rc * np.mean([zz ** (shape - 1) * np.exp(-zz / s) / (gamma_fn(shape) * s**shape) for s in scales], axis=0)

            p_z = np.mean([q_z(rc) for rc in comps], axis=0)  # incluye el factor rho_c
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
        # chi ~ sinh^{d-2}: propuesta e^{(d-2) chi} en [0, X] y aceptación (1 - e^{-2 chi})^{d-2}
        chi = np.zeros(m)
        todo = np.nonzero(valid)[0]  # las inválidas no necesitan chi
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
    """Volumen de un intervalo plano de tiempo propio 1."""
    from .regions import diamond_volume

    return diamond_volume(d, 0.5)


@dataclass
class TermEstimate:
    """Estimación MC de un término matricial T_ij y de su contracción a^T T a."""

    matrix: np.ndarray
    contraction: float
    contraction_se: float
    n_samples: int


@dataclass
class FluctuationPrediction:
    """Predicción MYZ de Cov(N_i, N_j) y de (Delta S)^2 para la acción local."""

    density: float
    coefficients: np.ndarray
    abundance_means: np.ndarray  # L_{-1}, L_0, ..., L_{n-1} (y L_n para M_{-1,n-1})
    covariance: np.ndarray  # M_ij, i, j = -1..n-1
    variance: float
    variance_se: float
    terms: dict = field(default_factory=dict)


class FluctuationKernel:
    """Evalúa el formalismo MYZ sobre una geometría (ver ``geometry.py``)."""

    def __init__(self, geometry, density: float, n_max: int, rng, z0=None,
                 overlap: str = "auto", overlap_points: int = 1024, eps: float = 1.0):
        """``eps``: suavizado de la acción (1 = acción local). Con eps < 1 las
        matrices son las de las abundancias adelgazadas Ñ_k
        (``thinned_abundance_vector``).

        ``overlap``: cómo obtener |A ∩ B|.
          "exact"      determinista, de la geometría: forma cerrada en 2D
                       Minkowski, cuadratura (``overlap.py``) en 4D Minkowski
          "plugin"     estimación MC con ``overlap_points`` puntos en A y fórmula
                       exacta de <zeta zeta> (sesgo O(1/overlap_points))
          "stochastic" estimadores insesgados (realización de Poisson local y
                       ensayos geométricos); mucha más varianza
          "auto"       "exact" si existe, si no "plugin"."""
        self.g = geometry
        self.rho = density
        self.n = n_max
        self.rng = rng
        lo, hi = geometry.region.bounding_box()
        extent = float(hi[0] - lo[0])
        rho_c = density * geometry.volume / geometry.coordinate_volume
        if z0 is None:  # escala(s) de la propuesta (ver ``_Proposal``)
            z0 = 3.0 if eps == 1 else (3.0, 3.0 / eps)
        self.prop = _Proposal(geometry.dim, extent, rho_c, z0)
        self._t_min, self._t_max = float(lo[0]), float(hi[0])
        has_exact = geometry.overlap_volume(*(np.zeros((1, geometry.dim)),) * 4) is not None
        if overlap == "auto":
            overlap = "exact" if has_exact else "plugin"
        if overlap == "exact" and not has_exact:
            raise ValueError("esta geometría no tiene |A ∩ B| en forma cerrada")
        if overlap == "stochastic" and eps != 1:
            raise ValueError("el modo estocástico sólo está implementado para eps = 1")
        if not 0 < eps <= 1:
            raise ValueError("eps debe estar en (0, 1]")
        self.eps = float(eps)
        self.overlap_mode = overlap
        self.overlap_points = overlap_points
        self.exact_overlap = overlap == "exact"

    # -- utilidades -----------------------------------------------------

    def _local(self, base, mixture=False):
        """Factor de densidad local de la propuesta: 1, salvo en geometrías con
        ``local_proposal`` (FLRW): la densidad física relativa en el punto base,
        acotada por debajo. Hacia el futuro (``mixture``) el punto base es el
        vértice inferior, que infravalora la densidad del intervalo: se mezcla
        con la global. El soporte de la propuesta no cambia."""
        if not getattr(self.g, "local_proposal", False):
            return 1.0
        rel = np.maximum(self.g.weight(base) * self.g.coordinate_volume / self.g.volume, 0.02)
        return (rel, 1.0) if mixture else rel

    def _past(self, base):
        """y = base - o con o del futuro; devuelve (y, q)."""
        o, q = self.prop.sample(len(base), self.rng, base[:, 0] - self._t_min, self._local(base))
        return base - o, q

    def _future(self, base):
        o, q = self.prop.sample(len(base), self.rng, self._t_max - base[:, 0], self._local(base, mixture=True))
        return base + o, q

    def _inside(self, y, z, x):
        return self.g.precedes(y, z) & self.g.precedes(z, x)

    def _overlap(self, yA, xA, yB, xB, VA):
        """|A ∩ B|: exacto o estimado con ``overlap_points`` puntos en A."""
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
        """<zeta_{i-sA}(A) zeta_{j-sB}(B)> para i, j = 0..n-1, forma (m, n, n).
        Exacto si hay |A∩B| cerrado; si no, indicador de una realización de
        Poisson local (estimador insesgado)."""
        rho, n = self.rho, self.n
        m = len(VA)
        if self.overlap_mode in ("exact", "plugin"):
            if Vb is None:
                Vb = self._overlap(yA, xA, yB, xB, VA)
            return thinned_joint_probability(rho * VA, rho * VB, rho * Vb, sA, sB, n, self.eps)
        rng = self.rng
        nA = rng.poisson(rho * VA)
        active = nA <= n  # si nA > n - 1 + sA no contribuye a ningún i < n
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
        cA, cB = nA + sA, n_ab + m_c + sB  # índices i, j a los que contribuye
        out = np.zeros((m, n, n))
        ok = active & (cA < n) & (cB < n)
        out[np.nonzero(ok)[0], cA[ok], cB[ok]] = 1.0
        return out

    def _inverse_overlap(self, yA, xA, yB, xB, VA):
        """Estimador insesgado de 1/|A∩B|: K/|A| con K ~ Geométrica(|A∩B|/|A|)."""
        m = len(VA)
        K = np.zeros(m)
        pending = np.arange(m)
        rng = self.rng
        while len(pending):
            pts, owner = self.g.sample_in_intervals(yA[pending], xA[pending], np.full(len(pending), 32), rng)
            hit = self._inside(yB[pending][owner], pts, xB[pending][owner])
            # posición (1..32) del primer acierto en cada intervalo
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

    # -- términos ---------------------------------------------------------

    def abundance_means(self, m: int, n_levels: int) -> tuple[np.ndarray, np.ndarray]:
        """L_k = <N_k> (MYZ 5.15), k = 0..n_levels-1, por MC (con eps < 1,
        <Ñ_k> = rho^2 ∫∫ P_k(eps rho V)). Devuelve (L, se)."""
        x = self.g.sample_M(m, self.rng)
        y, q = self._past(x)
        ok = self._points_in_M(y)
        V = self.g.interval_volume(y, x)
        w = np.where(ok, self.rho**2 * self.g.volume * self.g.weight(y) / q, 0.0)
        vals = w[:, None] * _pmf_table(self.eps * self.rho * V, n_levels)
        return vals.mean(axis=0), vals.std(axis=0, ddof=1) / np.sqrt(m)

    def term_J1(self, m: int, a: np.ndarray) -> TermEstimate:
        """J1_ij (MYZ 5.47): x1 = x2, y1 = y2. Con eps = 1 es delta_ij L_i; con
        eps < 1 los dos factores Ñ_i, Ñ_j del mismo par son adelgazamientos
        independientes del mismo intervalo: <..> = thinned_joint con A = B."""
        y, x, w = self._sample_pair(m)
        w = np.where(self._points_in_M(y), w, 0.0)
        V = self.g.interval_volume(y, x)
        zeros = np.zeros(m, dtype=int)
        G = thinned_joint_probability(self.rho * V, self.rho * V, self.rho * V, zeros, zeros, self.n, self.eps)
        return self._contract(w[:, None, None] * G, a)

    def term_J4(self, m: int, a: np.ndarray) -> TermEstimate:
        """J4_ij (MYZ 5.48): cadena y1 ≺ x1 = y2 ≺ x2; intervalos disjuntos."""
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
        """J2_ij (MYZ 5.52-5.56): vértice inferior común y, A = I(y,x1), B = I(y,x2)."""
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
        """J3_ij (MYZ 5.33): vértice superior común x, A = I(y1,x), B = I(y2,x)."""
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
        """(G - P_i P_j) * base [* 1/|A∩B|] para configuraciones con base != 0."""
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
                # plugin: si ningún punto cae en A ∩ B (solapamiento < |A|/K), se
                # acota con medio punto; el sesgo se controla variando K
                inv = 1.0 / np.maximum(Vb, VA * 0.5 / self.overlap_points)
            f = f * inv
        out[sel] = f[:, None, None] * (G - PP)
        return out

    def _sample_pair(self, m):
        """Par (y, x) con x uniforme en M e y = x - o. Devuelve y, x, peso
        rho^2 V_M w(y) / q (sin comprobar que y esté en M)."""
        x = self.g.sample_M(m, self.rng)
        y, q = self._past(x)
        return y, x, self.rho**2 * self.g.volume * self.g.weight(y) / q

    def _anchor_in(self, y, x, m):
        """Un punto uniforme (dV físico) en cada I(y[k], x[k]); peso rho |I|.
        Si y está fuera de M, la configuración tiene peso 0 (todos los usos
        exigen y en M) y se devuelve x sin muestrear: en FLRW un intervalo que
        cruza eta = 0 haría el rechazo arbitrariamente lento. (Sólo con
        ``local_proposal``, para no cambiar la secuencia aleatoria de los
        cálculos ya guardados en Minkowski y de Sitter.)"""
        if not getattr(self.g, "local_proposal", False):
            pts, _ = self.g.sample_in_intervals(y, x, np.ones(m, dtype=int), self.rng)
            return pts, self.rho * self.g.interval_volume(y, x)
        ok = self.g.contains(y)
        pts = x.copy()
        if ok.any():
            pts[ok], _ = self.g.sample_in_intervals(y[ok], x[ok], np.ones(int(ok.sum()), dtype=int), self.rng)
        return pts, self.rho * self.g.interval_volume(y, x)

    def term_J6_connected(self, m: int, a: np.ndarray) -> TermEstimate:
        """J6_ij - L_i L_j (MYZ 5.57-5.60 menos 5.10). Partición del dominio
        (equivalente a las subregiones de D_6, MYZ 5.44), sin solapamientos:
          c1: x2 ∈ A
          c2: y2 ∈ A, x2 ∉ A
          c3: x1 ∈ B, x2, y2 ∉ A
          c4: y1 ∈ B, x2, y2 ∉ A, x1 ∉ B
          c0: A ∩ B ≠ ∅ sin ningún extremo dentro del otro intervalo
        (fuera de estos casos A y B son disjuntos y el integrando conexo es 0).
        En c1-c4 el punto contenido se muestrea dentro del intervalo; en c0 se
        integra sobre w ∈ A ∩ B con peso 1/|A ∩ B|."""
        return self._contract(sum(self._j6_patterns(m)), a)

    def _j6_patterns(self, m):
        rng, g, rho = self.rng, self.g, self.rho
        inside = self._inside
        out = []
        # c1, c2: se muestrea A y un punto de B dentro de A
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
        # c3, c4: se muestrea B y un punto de A dentro de B
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
        # c0: A como par; w uniforme en A; B ∋ w con dos desplazamientos;
        # peso 1/|A ∩ B|. Sin extremos dentro del otro intervalo.
        y1, x1, wA = self._sample_pair(m)
        w, ww = self._anchor_in(y1, x1, m)
        x2, q2 = self._future(w)
        y2, q3 = self._past(w)
        excl = inside(y1, x2, x1) | inside(y1, y2, x1) | inside(y2, x1, x2) | inside(y2, y1, x2)
        ok = self._points_in_M(y1, x2, y2) & ~excl
        # ww = rho |A|: se usa |A| (densidad 1/|A| de w) y rho^2 para x2, y2
        base = np.where(ok, wA * (ww / rho) * rho**2 * g.weight(x2) * g.weight(y2) / (q2 * q3), 0.0)
        out.append(self._j6_integrand(y1, x1, y2, x2, base, True))
        return out

    # -- ensamblado ---------------------------------------------------------

    def predict(self, coefficients=None, m_terms: int = 200_000, m_means: int = 400_000, exact_means=None, batch: int = 50_000) -> FluctuationPrediction:
        """(Delta S)^2 = a^T M a. ``exact_means``: L_0..L_n exactos si se conocen
        (Minkowski: ``expected_thinned_abundance``); si no, MC.

        Con eps < 1, M es la covarianza de (N, Ñ_0, ...); la fila M_{-1,k}
        conserva la forma (k+2) L_k - (k+1) L_{k+1} con L_k = <Ñ_k>: por la
        fórmula de Mecke, Cov(N, sum_pares h(n)) = rho^2 ∫∫ E[(2 + n - rho V) h(n)],
        y para h(n) = B_k(n, eps) se separa n = (retenidos) + (no retenidos)."""
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
        # error: términos MC independientes + error de L (lineal en L)
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
        # combinar varianzas por lote (mismo estimador): se^2 = sum (n_b^2 se_b^2) / N^2
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
    """<Ñ_k> exacto en un diamante plano con rho V = lam:
    rho^2 ∫∫ P_k(eps rho V) = L_k(eps lam) / eps^2 (cambio rho -> eps rho)."""
    from .continuum import expected_abundance

    return expected_abundance(dim, eps * lam, k) / eps**2


def pair_weight(dim: int, n, eps: float = 1.0) -> np.ndarray:
    """f_eps(n) = sum_k a_k B_k(n, eps): contribución a S_eps de un par con n
    elementos entre sus extremos (sin el término a_N N). Con eps = 1,
    f(n) = a_n para n < n_d y 0 si no."""
    from scipy.stats import binom

    a = action_coefficients(dim, eps)[1:]
    n = np.asarray(n)
    return sum(a[k] * binom.pmf(k, n, eps) for k in range(len(a)))


def same_pair_log_coefficient(eps: float = 1.0, n_max: int = 20_000) -> float:
    """Coeficiente b de (Delta S)^2 ≈ b rho V ln(rho V) + O(rho V) que aporta
    el término "mismo par" (J¹ de MYZ) en un diamante de Minkowski 2D.

    Derivación (Goal 3d, ver README): en coordenadas de cono de luz, con
    diamante de lado l (V = l²/2),
        <sum_pares h(n)> = (rho²/4) ∫∫_0^l (l-a)(l-b) E_{n~Poisson(rho a b/2)}[h(n)] da db.
    A ab = x fijo, ∫_{x/l}^{l} da/a = ln(l²/x): el rango de rapidez de los
    intervalos casi nulos, cortado por el diamante. Con ∫_0^∞ P_n(mu) dmu = 1,
        <sum_pares h(n)> = rho V ln(rho V) sum_n h(n) + O(rho V).
    Para h = f_eps² sale b = sum_n f_eps(n)². El término cruzado con N lleva
    sum_n f_eps(n) = 0 (porque sum_k C_k = 0), y no tiene logaritmo."""
    return float(np.sum(pair_weight(2, np.arange(n_max), eps) ** 2))


def smeared_pair_moment(dim: int, lam: float, eps: float = 1.0, power: int = 1) -> float:
    """E[sum_{pares y ≺ x} f_eps(n(y,x))^power] exacto en un diamante plano
    con rho V = lam (``continuum._diamond_pair_integral`` con
    H(z) = E_{n~Poisson(z)}[f_eps(n)^power]). Con power = 2 es el término
    "mismo par" J¹ de MYZ para la acción suavizada; con power = 1, la media de
    la parte de pares de S_eps. Sólo d = 4 o 2 (núcleo de ``continuum``)."""
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
    """g = d E[sum_pares f_eps] / d(rho V). Con la ecuación de Mecke (Last &
    Penrose, "Lectures on the Poisson Process", Cambridge UP 2017, teor. 4.1),
    E[N F] = rho ∫ E[F(eta + δ_x)] dx, y la derivada respecto a la intensidad,
    rho dE[F]/drho = rho ∫ E[F(eta + δ_x) - F(eta)] dx, sale
    Cov(N, F) = rho dE[F]/drho = lam * g. Con eps = 1 reproduce el cruzado
    exacto (k+2)L_k - (k+1)L_{k+1} del Paso 2 (test). Diferencia central
    relativa de paso h."""
    up = smeared_pair_moment(dim, lam * (1 + h), eps)
    dn = smeared_pair_moment(dim, lam * (1 - h), eps)
    return (up - dn) / (2 * h * lam)
