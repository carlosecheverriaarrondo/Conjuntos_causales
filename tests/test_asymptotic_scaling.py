"""Goal 3c: fluctuations of the smeared BD action in the regime eps rho V > 100
(2D Minkowski).

Tolerances fixed BEFORE seeing the results (same criterion as in Goal 3):
  |simulation - prediction| < 4 sigma_combined, point by point;
  |p_sim - p_pred| < 4 sigma_combined for the exponent ΔS ∝ (rho V)^p.

``test_asymptotic_point`` is computed within the test itself (fixed seeds).
``test_long_run_*`` check the results saved by
``scripts/goal3c_asymptotic.py`` (rho V = 800..3200, too expensive for the
suite): they fail if the file does not exist, rather than being skipped.
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
    assert RESULTS.exists(), f"missing {RESULTS}: run scripts/goal3c_asymptotic.py (sim, pred, fit)"
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
