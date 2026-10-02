"""Punto 1b: fluctuaciones de la acción BD en el cono pasado hasta el big bang
de un FLRW plano 2D, a(eta) = eta^alpha (``scripts/flrw_fluctuations.py``).

Márgenes fijados ANTES de ver los resultados:
  F1 (volúmenes en forma cerrada frente a integración numérica): relativo 1e-9;
  F2, F3, F4: 4σ (estándar del proyecto).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import dblquad

from causets import sprinkle
from causets.geometry import FLRWConeGeometry, MinkowskiConeGeometry, power_law_rectangle_volume

K_SIGMA = 4.0
FIG = Path(__file__).resolve().parents[1] / "figures"


def _numeric_rectangle(s_u, s_v, du, dv, p):
    return dblquad(lambda v, u: ((u + v) / 2) ** p / 2, s_u, s_u + du, s_v, s_v + dv, epsrel=1e-13, epsabs=0)[0]


@pytest.mark.parametrize("p", [0, 1, 2, 4])
def test_F1_rectangle_volume(p):
    for s_u, s_v, du, dv in [(0.1, 0.05, 0.5, 0.3), (0.0, 0.0, 1.0, 1.0), (0.7, -0.2, 0.01, 0.8)]:
        exact = power_law_rectangle_volume(s_u + s_v, du, dv, p)
        assert exact == pytest.approx(_numeric_rectangle(s_u, s_v, du, dv, p), rel=1e-9)


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_F1_region_volume_and_overlap(alpha):
    g = FLRWConeGeometry(alpha, 1.3)
    p = int(2 * alpha)
    assert g.volume == pytest.approx(2 * 1.3 ** (p + 2) / ((p + 1) * (p + 2)), rel=1e-10)
    rng = np.random.default_rng(5)
    pts = sprinkle(g.region, 40 / g.volume, rng)
    y1, x1, y2, x2 = pts[0:1], pts[-1:], pts[1:2], pts[-2:-1]
    # la intersección de un intervalo consigo mismo es él mismo
    assert g.overlap_volume(y1, x1, y1, x1) == pytest.approx(g.interval_volume(y1, x1), rel=1e-12)
    assert g.overlap_volume(y1, x1, y2, x2) == pytest.approx(g.overlap_volume(y2, x2, y1, x1), rel=1e-12)


def test_F1_alpha_zero_is_minkowski_cone():
    rng = np.random.default_rng(6)
    g0, gm = FLRWConeGeometry(0.0), MinkowskiConeGeometry(2)
    pts = sprinkle(gm.region, 200, rng)
    y, x = pts[:50], pts[50:100]
    assert np.allclose(g0.interval_volume(y, x), gm.interval_volume(y, x), rtol=1e-12, atol=0)
    assert g0.volume == pytest.approx(gm.volume, rel=1e-12)


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_sprinkling_density(alpha):
    """<N> = rho V (4σ, Poisson)."""
    g = FLRWConeGeometry(alpha)
    rng = np.random.default_rng(7)
    lam, n = 60.0, 400
    counts = [len(sprinkle(g.region, lam / g.volume, rng)) for _ in range(n)]
    assert abs(np.mean(counts) - lam) < K_SIGMA * np.sqrt(lam / n)


# --- resultados guardados por scripts/flrw_fluctuations.py (fallan si faltan) ---


@pytest.fixture(scope="module")
def results():
    path = FIG / "flrw_results.json"
    assert path.exists(), "ejecuta scripts/flrw_fluctuations.py sim, pred y fit"
    return json.loads(path.read_text())


def test_F2_means_match_sprinklings(results):
    assert len(results["F2"]) >= 8
    for r in results["F2"]:
        assert max(abs(z) for z in r["z"]) < K_SIGMA, r


def test_F3_myz_prediction_matches_simulation(results):
    assert len(results["F3"]) >= 8
    for r in results["F3"]:
        assert abs(r["z"]) < K_SIGMA, r


@pytest.mark.parametrize("alpha", ["1.0", "2.0"])
def test_F4_log_coefficient_matches_diamond(results, alpha):
    assert abs(results["fits"][alpha]["z_vs_diamond"]) < K_SIGMA, results["fits"][alpha]
