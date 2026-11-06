"""Item 1a: fluctuations of the BD action in the truncated past light cone
of 2D Minkowski (``scripts/cone_fluctuations.py``).

Tolerances fixed BEFORE seeing the results:
  exact pair formula vs. direct integration: relative 1e-8;
  logarithm coefficient (C2): relative 1e-3 between rho V = 1e5 and 1e6;
  C1, C3, C4: 4σ (project standard).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.stats import poisson

from causets import sprinkle
from causets.causal import minkowski_causal_matrix
from causets.continuum import cone_pair_integral_2d, expected_abundance_cone
from causets.fluctuations import pair_weight, same_pair_log_coefficient
from causets.geometry import MinkowskiConeGeometry
from causets.regions import TruncatedPastCone

K_SIGMA = 4.0
FIG = Path(__file__).resolve().parents[1] / "figures"


@pytest.mark.parametrize("dim", [2, 4])
def test_cone_volume_by_sampling(dim):
    """Analytic volume vs. the fraction of bounding-box points inside (4σ)."""
    cone = TruncatedPastCone(dim, 1.3)
    lo, hi = cone.bounding_box()
    rng = np.random.default_rng(3)
    n = 400_000
    frac = cone.contains(rng.uniform(lo, hi, size=(n, dim))).mean()
    box = np.prod(hi - lo)
    assert abs(frac * box - cone.volume) < K_SIGMA * box * np.sqrt(frac * (1 - frac) / n)


def test_cone_is_causally_convex():
    """Every point of I(y, x) with y, x in the cone lies in the cone."""
    g = MinkowskiConeGeometry(2)
    rng = np.random.default_rng(4)
    pts = sprinkle(g.region, 300, rng)
    C = minkowski_causal_matrix(pts)
    i, j = np.nonzero(C)
    inner, _ = g.sample_in_intervals(pts[i], pts[j], np.full(len(i), 3), rng)
    assert g.contains(inner).all()


@pytest.mark.parametrize("k", [0, 2])
def test_cone_pair_formula_against_direct_integral(k):
    """(1/4) ∫ H(mu) I(mu) dmu = (1/8) ∫∫_{a+b<=2h} (2h-a-b)² H(ab/2) da db."""
    lam = 30.0
    h = math.sqrt(lam)
    direct = dblquad(lambda b, a: (2 * h - a - b) ** 2 * poisson.pmf(k, a * b / 2) / 8,
                     0, 2 * h, 0, lambda a: 2 * h - a, epsrel=1e-11, epsabs=0)[0]
    assert expected_abundance_cone(lam, k) == pytest.approx(direct, rel=1e-8)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_cone_abundances_grow_as_rhoV_log_rhoV(k):
    """<N_k> = rho V ln(rho V) + c_k rho V + o(rho V) also in the cone."""
    r = [expected_abundance_cone(lam, k) / lam - np.log(lam) for lam in (1e5, 1e6)]
    assert abs(r[1] - r[0]) < 1e-3, r


@pytest.mark.parametrize("eps", [1.0, 0.5])
def test_C2_cone_same_pair_log_coefficient(eps):
    """C2: <sum_pairs f_eps(n)²>/rho V grows as b(eps) ln(rho V) in the cone,
    with the same b(eps) = sum f_eps² as in the diamond (exact integral)."""
    n = np.arange(3000)
    f2 = pair_weight(2, n, eps) ** 2
    y = [cone_pair_integral_2d(lambda z: float(np.sum(poisson.pmf(n, z) * f2)), lam) / lam for lam in (1e5, 1e6)]
    assert (y[1] - y[0]) / np.log(10) == pytest.approx(same_pair_log_coefficient(eps), rel=1e-3)


# --- results saved by scripts/cone_fluctuations.py (fail if missing) ---


@pytest.fixture(scope="module")
def results():
    path = FIG / "cone_results.json"
    assert path.exists(), "run scripts/cone_fluctuations.py sim, pred and fit"
    return json.loads(path.read_text())


def test_C1_exact_means_match_sprinklings(results):
    assert len(results["C1"]) >= 8
    for r in results["C1"]:
        assert max(abs(z) for z in r["z"]) < K_SIGMA, r


def test_C3_myz_prediction_matches_simulation(results):
    assert len(results["C3"]) >= 8
    for r in results["C3"]:
        assert abs(r["z"]) < K_SIGMA, r


def test_C4_cone_log_coefficient_matches_diamond(results):
    assert abs(results["C4"]["z"]) < K_SIGMA, results["C4"]
