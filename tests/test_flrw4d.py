"""Item 1d: scaling of (Delta S)^2 in the past cone down to the big bang of a
flat 4D FLRW spacetime (``scripts/flrw4d_scaling.py``).

Tolerances fixed BEFORE seeing the results:
  K1: volumes vs. Monte Carlo (4σ), alpha = 0 vs. Minkowski
      (relative 1e-12), g_flrw(0) = g_cone (relative 1e-10);
  K2, K4: 4σ; K3: < 3 % at rho V = 1e8 and closer to 1 than at 1e6.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from causets import sprinkle
from causets.geometry import FLRWConeGeometry, conformal_interval_volume_4d, diamond_volume, proper_time
from causets.regions import power_law_scale_factor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cone4d_scaling import g_cone, g_flrw  # noqa: E402

K_SIGMA = 4.0


def _pairs(n, seed):
    g = FLRWConeGeometry(0, dim=4)
    pts = sprinkle(g.region, 40 / g.volume, np.random.default_rng(seed))
    i, j = np.triu_indices(len(pts), 1)
    ok = g.precedes(pts[i], pts[j])
    return pts[i][ok][:n], pts[j][ok][:n]


def test_K1_alpha_zero_is_minkowski():
    y, x = _pairs(50, 1)
    assert np.allclose(conformal_interval_volume_4d(y, x, power_law_scale_factor(0)),
                       diamond_volume(4, proper_time(y, x) / 2), rtol=1e-12, atol=0)


@pytest.mark.parametrize("alpha", [1, 2])
def test_K1_interval_volume_against_monte_carlo(alpha):
    g = FLRWConeGeometry(alpha, dim=4)
    y, x = _pairs(3, 2)
    rng = np.random.default_rng(3)
    for k in range(len(y)):
        pts, _ = g._sample_coordinate_uniform(y[k:k + 1], x[k:k + 1], np.array([400_000]), rng)
        w = g.a(pts[:, 0]) ** 4
        vc = diamond_volume(4, proper_time(y[k:k + 1], x[k:k + 1]) / 2)[0]
        exact = g.interval_volume(y[k:k + 1], x[k:k + 1])[0]
        assert abs(exact - vc * w.mean()) < K_SIGMA * vc * w.std() / np.sqrt(len(w))


def test_K1_flrw_prefactor_reduces_to_minkowski():
    assert g_flrw(0) == pytest.approx(g_cone(), rel=1e-10)


@pytest.mark.parametrize("alpha", [1, 2])
def test_region_volume(alpha):
    """V = 4π ∫ eta^(4 alpha) (h - eta)³/3 deta = 4π B(4 alpha + 1, 4) / 3 · 6 / ... (closed form)."""
    from scipy.special import beta

    g = FLRWConeGeometry(alpha, 1.0, 4)
    assert g.volume == pytest.approx(4 * np.pi / 3 * beta(4 * alpha + 1, 4), rel=1e-10)


@pytest.mark.parametrize("alpha", [1, 2])
def test_exact_sampler_matches_rejection(alpha):
    """sample_M (eta ~ h Beta(4 alpha + 1, 4), ball of radius h - eta) vs. the
    generic rejection sampler: KS in eta, r and x_1 at level 1e-3 (fixed
    beforehand)."""
    from scipy.stats import ks_2samp

    from causets.sprinkling import sample_uniform

    g = FLRWConeGeometry(alpha, dim=4)
    rng = np.random.default_rng(9)
    a, b = g.sample_M(100_000, rng), sample_uniform(g.region, 20_000, rng)
    assert g.contains(a).all()
    for f in (lambda p: p[:, 0], lambda p: np.linalg.norm(p[:, 1:], axis=1), lambda p: p[:, 1]):
        assert ks_2samp(f(a), f(b)).pvalue > 1e-3


# --- results saved by scripts/flrw4d_scaling.py (fail if missing) ---


@pytest.fixture(scope="module")
def results():
    path = ROOT / "figures" / "flrw4d_results.json"
    assert path.exists(), "run scripts/flrw4d_scaling.py sim/means (alpha = 1, 2) and fit"
    return json.loads(path.read_text())


def test_K2_means_match_sprinklings(results):
    assert len(results["K2"]) >= 16
    assert results["K2_max"] < K_SIGMA


@pytest.mark.parametrize("alpha", ["1.0", "2.0"])
def test_K3_asymptotic_prefactor(results, alpha):
    k3 = results["K3"][alpha]
    assert k3["max_dev_last"] < 0.03, k3
    assert k3["closer_than_1e6"], k3


@pytest.mark.parametrize("alpha", ["1.0", "2.0"])
def test_K4_correlations_constant_fraction_of_J1(results, alpha):
    assert abs(results["K4"][alpha]["z"]) < K_SIGMA, results["K4"][alpha]
