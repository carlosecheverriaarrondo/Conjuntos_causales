"""Goal 3c: fluctuaciones de la acción BD suavizada en el régimen eps rho V > 100
(Minkowski 2D).

Márgenes fijados ANTES de ver los resultados (mismo criterio que en el Goal 3):
  |simulación - predicción| < 4 sigma_combinada, punto a punto;
  |p_sim - p_pred| < 4 sigma_combinada para el exponente ΔS ∝ (rho V)^p.

``test_asymptotic_point`` se calcula en el propio test (semillas fijas).
``test_long_run_*`` comprueban los resultados guardados por
``scripts/goal3c_asymptotic.py`` (rho V = 800..3200, demasiado caro para la
suite): fallan si el fichero no existe, en vez de saltarse.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from causets.fluctuations import FluctuationKernel, ensemble_fluctuations, expected_thinned_abundance
from causets.geometry import MinkowskiDiamondGeometry

K_SIGMA = 4.0
RESULTS = Path(__file__).resolve().parents[1] / "figures" / "goal3c_results.json"


def test_asymptotic_point():
    g = MinkowskiDiamondGeometry(2)
    lam, eps = 600.0, 0.2
    assert eps * lam > 100
    rho = lam / g.volume
    K = FluctuationKernel(g, rho, 3, np.random.default_rng(301), eps=eps)
    pred = K.predict(m_terms=3_000_000, exact_means=[expected_thinned_abundance(2, lam, k, eps) for k in range(4)], batch=100_000)
    sim = ensemble_fluctuations(g, rho, 1000, np.random.default_rng(302), eps=eps)
    sigma = np.hypot(pred.variance_se, sim.variance_se)
    assert abs(sim.variance - pred.variance) < K_SIGMA * sigma, (sim.variance, sim.variance_se, pred.variance, pred.variance_se)


@pytest.fixture(scope="module")
def long_run():
    assert RESULTS.exists(), f"falta {RESULTS}: ejecuta scripts/goal3c_asymptotic.py (sim, pred, fit)"
    return json.loads(RESULTS.read_text())


def test_long_run_points_agree(long_run):
    pts = long_run["points"]
    assert len(pts) == 9 and all(p["eps"] * p["rhoV"] > 100 for p in pts)
    bad = [(p["eps"], p["rhoV"], p["z"]) for p in pts if abs(p["z"]) >= K_SIGMA]
    assert not bad, bad


def test_long_run_exponents_agree(long_run):
    for eps, ex in long_run["exponents"].items():
        s, p = ex["alto_sim"], ex["alto_pred"]
        z = (s["p"] - p["p"]) / np.hypot(s["se"], p["se"])
        assert abs(z) < K_SIGMA, (eps, s, p)
