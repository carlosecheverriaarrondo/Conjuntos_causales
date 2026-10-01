"""Tests de las fluctuaciones de la acción BD SUAVIZADA (eps < 1): extensión
del formalismo de Moradi, Yazdi & Zilhão (arXiv:2407.03395) con abundancias
adelgazadas (ver ``causets/fluctuations.py``).

Margen estadístico fijado a priori (igual que en el Goal 3):
  |simulación - predicción| < 4 sqrt(se_sim^2 + se_pred^2).
Semillas fijas.
"""

import numpy as np
import pytest
from scipy.stats import binom

from causets import sprinkle
from causets.action import bd_action
from causets.causal import minkowski_causal_matrix
from causets.continuum import expected_bd_action_diamond
from causets.fluctuations import (
    FluctuationKernel,
    ensemble_fluctuations,
    expected_thinned_abundance,
    local_action_coefficients,
    smeared_action_coefficients,
    thinned_abundance_vector,
    thinned_joint_probability,
)
from causets.geometry import MinkowskiDiamondGeometry, minkowski_precedes

K_SIGMA = 4.0


@pytest.mark.parametrize("dim", [2, 4])
def test_thinned_abundances_give_smeared_action(dim):
    g = MinkowskiDiamondGeometry(dim)
    C = minkowski_causal_matrix(sprinkle(g.region, 150.0 / g.volume, np.random.default_rng(1)))
    for eps in (1.0, 0.5, 0.15):
        a = smeared_action_coefficients(dim, eps)
        assert thinned_abundance_vector(C, len(a) - 1, eps) @ a == pytest.approx(bd_action(C, dim, eps), rel=1e-10)
    np.testing.assert_allclose(smeared_action_coefficients(dim, 1.0), local_action_coefficients(dim), rtol=1e-14)


@pytest.mark.parametrize("dim", [2, 4])
def test_expected_thinned_abundances_give_mean_action(dim):
    lam, eps = 100.0, 0.3
    a = smeared_action_coefficients(dim, eps)
    L = [lam] + [expected_thinned_abundance(dim, lam, k, eps) for k in range(len(a) - 1)]
    assert np.dot(a, L) == pytest.approx(expected_bd_action_diamond(dim, lam, eps), rel=1e-7)


def test_thinned_joint_probability_against_sprinklings():
    """<B_i(n_A, eps) B_j(n_B, eps)> con un extremo de B dentro de A (sA = 1):
    la fórmula de adelgazamiento contra sprinklings directos."""
    g = MinkowskiDiamondGeometry(2)
    rho, eps, n = 20.0, 0.4, 4
    y1, x1 = np.array([-0.6, -0.1]), np.array([0.2, -0.1])
    y2, x2 = np.array([-0.5, 0.2]), np.array([0.1, 0.0])  # x2 ∈ A
    assert minkowski_precedes(y1, x2) and minkowski_precedes(x2, x1)
    VA = g.interval_volume(y1[None], x1[None])
    VB = g.interval_volume(y2[None], x2[None])
    Vb = g.overlap_volume(y1[None], x1[None], y2[None], x2[None])
    P = thinned_joint_probability(rho * VA, rho * VB, rho * Vb, np.array([1]), np.array([0]), n, eps)[0]
    rng = np.random.default_rng(5)
    m = 20000
    acc = np.zeros((m, n, n))
    k = np.arange(n)
    for s in range(m):
        pts = np.vstack([sprinkle(g.region, rho, rng), x2])
        nA = np.count_nonzero(minkowski_precedes(y1, pts) & minkowski_precedes(pts, x1))
        nB = np.count_nonzero(minkowski_precedes(y2, pts) & minkowski_precedes(pts, x2))
        acc[s] = binom.pmf(k, nA, eps)[:, None] * binom.pmf(k, nB, eps)[None, :]
    mean, se = acc.mean(axis=0), acc.std(axis=0, ddof=1) / np.sqrt(m)
    assert np.all(np.abs(mean - P) < K_SIGMA * se + 1e-5), np.max(np.abs(mean - P) / (se + 1e-12))


@pytest.fixture(scope="module")
def smeared_2d():
    g = MinkowskiDiamondGeometry(2)
    lam, eps = 100.0, 0.5
    rho = lam / g.volume
    K = FluctuationKernel(g, rho, 3, np.random.default_rng(21), eps=eps)
    pred = K.predict(m_terms=600_000, exact_means=[expected_thinned_abundance(2, lam, k, eps) for k in range(4)], batch=100_000)
    sim = ensemble_fluctuations(g, rho, 3000, np.random.default_rng(22), eps=eps)
    return pred, sim


def test_smeared_action_variance_2d(smeared_2d):
    pred, sim = smeared_2d
    sigma = np.hypot(pred.variance_se, sim.variance_se)
    assert abs(sim.variance - pred.variance) < K_SIGMA * sigma, (sim.variance, sim.variance_se, pred.variance, pred.variance_se)


def test_smeared_covariance_with_N(smeared_2d):
    """M_{-1,k} = (k+2) L_k - (k+1) L_{k+1} también para las abundancias adelgazadas."""
    pred, sim = smeared_2d
    for j in range(4):
        assert abs(sim.covariance[0, j] - pred.covariance[0, j]) < K_SIGMA * sim.covariance_se[0, j], j
