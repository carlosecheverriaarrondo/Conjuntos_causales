"""Item 1c: scaling of (Delta S)^2 in the truncated past cone of 4D Minkowski
(``scripts/cone4d_scaling.py``).

Tolerances fixed BEFORE seeing the cone results:
  2D cone integral vs. the exact formula of item 1a: relative 1e-9;
  G of the cone (closed form) vs. Monte Carlo: 4σ;
  H1, H3: 4σ; H2: monotonic convergence and < 3 % at rho V = 1e8.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from causets.continuum import expected_abundance_cone, expected_abundance_cone_d
from causets.regions import TruncatedPastCone
from causets.sprinkling import sample_uniform

K_SIGMA = 4.0
FIG = Path(__file__).resolve().parents[1] / "figures"


@pytest.mark.parametrize("lam,k", [(50.0, 0), (800.0, 2)])
def test_cone_integral_reduces_to_2d_formula(lam, k):
    assert expected_abundance_cone_d(2, lam, k) == pytest.approx(expected_abundance_cone(lam, k), rel=1e-9)


def test_cone_G_closed_form():
    """G/V^1.5 = 4π ∫ t² dV / V^1.5 (l = t in the cone) vs. Monte Carlo."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from cone4d_scaling import g_cone

    cone = TruncatedPastCone(4, 1.0)
    t = sample_uniform(cone, 400_000, np.random.default_rng(8))[:, 0]
    mc = 4 * np.pi * np.mean(t**2) * cone.volume / cone.volume**1.5
    se = 4 * np.pi * np.std(t**2) / np.sqrt(len(t)) * cone.volume / cone.volume**1.5
    assert abs(g_cone() - mc) < K_SIGMA * se


# --- results saved by scripts/cone4d_scaling.py (fail if missing) ---


@pytest.fixture(scope="module")
def results():
    path = FIG / "cone4d_results.json"
    assert path.exists(), "run scripts/cone4d_scaling.py sim and fit"
    return json.loads(path.read_text())


def test_H1_exact_means_match_sprinklings(results):
    assert len(results["H1"]) >= 8
    for r in results["H1"]:
        assert max(abs(z) for z in r["z"]) < K_SIGMA, r


def test_H2_asymptotic_prefactor(results):
    assert results["H2"]["monotonic"], results["H2"]
    assert results["H2"]["max_dev_last"] < 0.03, results["H2"]


def test_H3_correlations_constant_fraction_of_J1(results):
    assert abs(results["H3"]["z"]) < K_SIGMA, results["H3"]
