"""Tests del Goal 3: fluctuaciones de la acción BD LOCAL (formalismo de
Moradi, Yazdi & Zilhão, Class. Quantum Grav. 42 (2025) 045017, arXiv:2407.03395).

Margen estadístico (fijado a priori, no ajustado a los resultados):
  |simulación - predicción| < K_SIGMA * sqrt(se_sim^2 + se_pred^2),  K_SIGMA = 4,
donde se_sim es el error estándar del estimador sobre sprinklings
independientes y se_pred el error estadístico de la integración Monte Carlo
de las integrales de MYZ. Semillas fijas: los tests son deterministas.
La predicción de Minkowski 2D es exacta salvo el error MC (|A ∩ B| en forma
cerrada); ver ``causets/fluctuations.py``.
"""

import numpy as np
import pytest

from causets import sprinkle
from causets.continuum import expected_abundance
from causets.fluctuations import (
    FluctuationKernel,
    abundance_vector,
    cardinality_indicator,
    ensemble_fluctuations,
    joint_cardinality_probability,
    local_action_coefficients,
    occupation_indicator,
)
from causets.geometry import MinkowskiDiamondGeometry, minkowski_precedes
from causets.action import bd_action
from causets.causal import minkowski_causal_matrix

K_SIGMA = 4.0


def _poisson(k, z):
    from scipy.stats import poisson

    return poisson.pmf(k, z)


# ------------------------------------------------------------ indicadores


def test_cardinality_indicator_partitions_relations():
    g = MinkowskiDiamondGeometry(2)
    C = minkowski_causal_matrix(sprinkle(g.region, 100.0, np.random.default_rng(1)))
    total = sum(cardinality_indicator(C, i).astype(int) for i in range(len(C)))
    np.testing.assert_array_equal(total, C.astype(int))  # cada relación tiene una única cardinalidad


def test_occupation_indicator():
    pts = np.array([[0.0, 0.0], [0.5, 0.1]])
    assert occupation_indicator(pts, lambda z: np.abs(z[:, 0] - 0.5) < 0.01)
    assert not occupation_indicator(pts, lambda z: z[:, 0] > 1.0)


def test_abundance_vector_gives_local_action():
    g = MinkowskiDiamondGeometry(4)
    C = minkowski_causal_matrix(sprinkle(g.region, 150.0, np.random.default_rng(2)))
    a = local_action_coefficients(4)
    assert abundance_vector(C, 4) @ a == pytest.approx(bd_action(C, 4), rel=1e-12)


# ------------------------------------------- reglas de correlación (sec. 3.3)


def _counts_in(points, y, x):
    return int(np.count_nonzero(minkowski_precedes(y, points) & minkowski_precedes(points, x)))


def test_zeta_chi_rule():
    """MYZ (3.31): con un elemento fijo z ∈ I(y, x), P(zeta_i) = P_{i-1}(rho V)."""
    g = MinkowskiDiamondGeometry(2)
    rho = 30.0
    y, x, z = np.array([-0.6, 0.0]), np.array([0.4, 0.1]), np.array([0.0, 0.05])
    V = float(g.interval_volume(y[None], x[None])[0])
    rng = np.random.default_rng(3)
    m = 20000
    counts = np.array([_counts_in(np.vstack([sprinkle(g.region, rho, rng), z]), y, x) for _ in range(m)])
    for i in range(1, 6):
        p = _poisson(i - 1, rho * V)
        freq = np.mean(counts == i)
        assert abs(freq - p) < K_SIGMA * np.sqrt(p * (1 - p) / m), (i, freq, p)


def test_zeta_zeta_overlapping_diamonds():
    """MYZ (3.34): dos diamantes solapados, <zeta_i zeta_j> = sum P_a P_b P_c."""
    g = MinkowskiDiamondGeometry(2)
    rho = 20.0
    y1, x1 = np.array([-0.6, -0.1]), np.array([0.2, -0.1])
    y2, x2 = np.array([-0.5, 0.2]), np.array([0.3, 0.2])
    VA = float(g.interval_volume(y1[None], x1[None])[0])
    VB = float(g.interval_volume(y2[None], x2[None])[0])
    Vb = float(g.overlap_volume(y1[None], x1[None], y2[None], x2[None])[0])
    assert 0 < Vb < min(VA, VB)
    n = 8
    P = joint_cardinality_probability(np.array([rho * (VA - Vb)]), np.array([rho * Vb]), np.array([rho * (VB - Vb)]), n)[0]
    rng = np.random.default_rng(4)
    m = 20000
    freq = np.zeros((n, n))
    for _ in range(m):
        pts = sprinkle(g.region, rho, rng)
        a, b = _counts_in(pts, y1, x1), _counts_in(pts, y2, x2)
        if a < n and b < n:
            freq[a, b] += 1
    freq /= m
    tol = K_SIGMA * np.sqrt(P * (1 - P) / m) + 1e-4
    assert np.all(np.abs(freq - P) < tol), np.max(np.abs(freq - P) / tol)


# ------------------------------------------------- varianza de la acción


@pytest.fixture(scope="module")
def minkowski_2d_ensembles():
    """Simulación y predicción MYZ en Minkowski 2D para rho V = 50 y 100."""
    g = MinkowskiDiamondGeometry(2)
    out = {}
    for lam in (50.0, 100.0):
        rho = lam / g.volume
        K = FluctuationKernel(g, rho, 3, np.random.default_rng(int(lam) + 7))
        pred = K.predict(m_terms=600_000, exact_means=[expected_abundance(2, lam, k) for k in range(4)], batch=100_000)
        sim = ensemble_fluctuations(g, rho, 4000, np.random.default_rng(int(lam) + 8))
        out[lam] = (pred, sim)
    return out


@pytest.mark.parametrize("lam", [50.0, 100.0])
def test_action_variance_2d_matches_myz_prediction(minkowski_2d_ensembles, lam):
    pred, sim = minkowski_2d_ensembles[lam]
    sigma = np.hypot(sim.variance_se, pred.variance_se)
    assert abs(sim.variance - pred.variance) < K_SIGMA * sigma, (
        f"rho V={lam}: sim={sim.variance:.1f}±{sim.variance_se:.1f}, "
        f"MYZ={pred.variance:.1f}±{pred.variance_se:.1f}"
    )


@pytest.mark.parametrize("lam", [50.0, 100.0])
def test_covariance_with_N_matches(minkowski_2d_ensembles, lam):
    """M_{-1,j} = (j+2) L_j - (j+1) L_{j+1} (de MYZ 5.26-5.27) y M_{-1,-1} = rho V (5.20)."""
    pred, sim = minkowski_2d_ensembles[lam]
    cov, se = sim.covariance, sim.covariance_se
    for j in range(4):
        assert abs(cov[0, j] - pred.covariance[0, j]) < K_SIGMA * se[0, j], (j, cov[0, j], pred.covariance[0, j])


def test_variance_is_super_poissonian_2d(minkowski_2d_ensembles):
    """Observación (no una predicción de MYZ): (Delta S)^2 >> rho V y la razón
    crece con rho V, i.e. las fluctuaciones de la acción local no escalan como
    las de N (sqrt V). Relevante para la conexión con Everpresent Lambda."""
    ratios = [minkowski_2d_ensembles[lam][0].variance / lam for lam in (50.0, 100.0)]
    assert ratios[0] > 10 and ratios[1] > ratios[0]
