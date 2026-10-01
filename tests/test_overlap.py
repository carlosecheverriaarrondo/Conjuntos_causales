"""Tests de la cuadratura determinista de |I(y1,x1) ∩ I(y2,x2)| en Minkowski 4D
(``causets/overlap.py``).

Márgenes fijados a priori: casos exactos con tolerancia relativa 1e-3 (error
de cuadratura); comparación con Monte Carlo con
|cuadratura - MC| < 4 sigma_MC + 1e-3 |A|.
"""

import numpy as np
import pytest

from causets.geometry import MinkowskiDiamondGeometry, minkowski_precedes
from causets.overlap import disk_intersection_area, interval_intersection_volume_4d

K_SIGMA = 4.0
QUAD_TOL = 1e-3


def test_disk_lens_area():
    # dos discos unidad a distancia 1: lente de área 2 arccos(1/2) - sqrt(3)/2
    area = disk_intersection_area(np.array([[[0.0, 0.0], [1.0, 0.0]]]), np.array([[1.0, 1.0]]))
    assert area[0] == pytest.approx(2 * np.arccos(0.5) - np.sqrt(3) / 2, rel=1e-12)


def test_disk_intersection_special_cases():
    c = np.array([[[0.0, 0.0], [0.2, 0.1], [5.0, 0.0], [0.0, 0.0]]])
    # disco contenido en otro; disco disjunto; disco duplicado; disco vacío
    assert disk_intersection_area(c[:, :2], np.array([[1.0, 0.3]]))[0] == pytest.approx(np.pi * 0.09, rel=1e-12)
    assert disk_intersection_area(c[:, [0, 2]], np.array([[1.0, 1.0]]))[0] == 0.0
    assert disk_intersection_area(c[:, [0, 3]], np.array([[1.0, 1.0]]))[0] == pytest.approx(np.pi, rel=1e-12)
    assert disk_intersection_area(c[:, :2], np.array([[1.0, 0.0]]))[0] == 0.0


def test_exact_cases():
    g = MinkowskiDiamondGeometry(4)
    y, x = np.array([[-0.5, 0.1, 0.0, 0.0]]), np.array([[0.5, 0.2, 0.1, 0.0]])
    yb, xb = np.array([[-0.2, 0.15, 0.0, 0.0]]), np.array([[0.3, 0.15, 0.05, 0.0]])
    VA, VB = g.interval_volume(y, x)[0], g.interval_volume(yb, xb)[0]
    assert interval_intersection_volume_4d(y, x, y, x)[0] == pytest.approx(VA, rel=QUAD_TOL)  # A = A
    assert interval_intersection_volume_4d(y, x, yb, xb)[0] == pytest.approx(VB, rel=QUAD_TOL)  # B ⊂ A
    far = np.array([[0.0, 3.0, 0.0, 0.0]])
    assert interval_intersection_volume_4d(y, x, y + far, x + far)[0] == 0.0  # disjuntos
    assert g.overlap_volume(y, x, yb, xb)[0] == pytest.approx(VB, rel=QUAD_TOL)  # cableado en la geometría


def _random_configs(rng, m):
    y1 = rng.normal(0, 0.2, (m, 4))
    y1[:, 0] = -0.5
    x1 = y1 + np.column_stack([np.full(m, 0.9), rng.normal(0, 0.15, (m, 3))])
    y2 = y1 + np.column_stack([rng.uniform(-0.1, 0.3, m), rng.normal(0, 0.15, (m, 3))])
    x2 = y2 + np.column_stack([np.full(m, 0.8), rng.normal(0, 0.15, (m, 3))])
    # dos con vértice compartido (como en J2 y J3, conos coincidentes)
    y2[0] = y1[0]
    x2[1] = x1[1]
    return y1, x1, y2, x2


def test_matches_monte_carlo():
    g = MinkowskiDiamondGeometry(4)
    rng = np.random.default_rng(11)
    y1, x1, y2, x2 = _random_configs(rng, 6)
    quad = interval_intersection_volume_4d(y1, x1, y2, x2)
    VA = g.interval_volume(y1, x1)
    n_mc = 1_000_000
    for k in range(len(VA)):
        pts, _ = g.sample_in_intervals(y1[k : k + 1], x1[k : k + 1], np.array([n_mc]), rng)
        p = np.mean(minkowski_precedes(y2[k], pts) & minkowski_precedes(pts, x2[k]))
        mc, se = VA[k] * p, VA[k] * np.sqrt(p * (1 - p) / n_mc)
        assert 0 < quad[k] < VA[k]
        assert abs(quad[k] - mc) < K_SIGMA * se + QUAD_TOL * VA[k], (k, quad[k], mc, se)
