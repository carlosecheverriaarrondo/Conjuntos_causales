"""Goal 3d: origin of the rho V ln(rho V) term in (Delta S)^2 (2D Minkowski).

Tolerances fixed BEFORE seeing the results:
  convergence of <N_k>/rho V - ln(rho V): |Δ| < 1e-3 between rho V = 1e5 and
  1e6 (deterministic computation);
  pair_weight identities: round-off tolerance (1e-10).
"""

import numpy as np
import pytest

from causets.continuum import expected_abundance
from causets.fluctuations import local_action_coefficients, pair_weight, same_pair_log_coefficient


@pytest.mark.parametrize("k", [0, 1, 2])
def test_abundances_grow_as_rhoV_log_rhoV(k):
    """<N_k> = rho V ln(rho V) + c_k rho V + o(rho V): unit log coefficient."""
    r = [expected_abundance(2, lam, k) / lam - np.log(lam) for lam in (1e5, 1e6)]
    assert abs(r[1] - r[0]) < 1e-3, r


def test_local_log_coefficient_is_sum_of_squares():
    a = local_action_coefficients(2)[1:]
    assert same_pair_log_coefficient(1.0) == pytest.approx(np.sum(a**2), rel=1e-12)
    assert same_pair_log_coefficient(1.0) == pytest.approx(24.0, rel=1e-12)


@pytest.mark.parametrize("eps", [1.0, 0.5, 0.25, 0.1])
def test_cross_term_has_no_log(eps):
    """sum_n f_eps(n) = 0: the cross term with N contributes no rho V ln(rho V)."""
    f = pair_weight(2, np.arange(20_000), eps)
    assert abs(np.sum(f)) < 1e-10 * np.sum(np.abs(f))


def test_smeared_same_pair_log_coefficient_exact():
    """<sum_pairs f_eps(n)^2>/rho V grows as b(eps) ln(rho V): exact integral
    over the diamond (no Monte Carlo), relative tolerance 1e-3."""
    from scipy.stats import poisson

    from causets.continuum import _diamond_pair_integral

    eps, n = 0.5, np.arange(3000)
    f2 = pair_weight(2, n, eps) ** 2
    y = [_diamond_pair_integral(lambda z: float(np.sum(poisson.pmf(n, z) * f2)), 2, lam) / lam for lam in (1e5, 1e6)]
    assert (y[1] - y[0]) / np.log(10) == pytest.approx(same_pair_log_coefficient(eps), rel=1e-3)


# --- results saved by scripts/goal3d_log_origin.py (fail if missing) ---

RESULTS = __import__("pathlib").Path(__file__).resolve().parents[1] / "figures"
K_SIGMA = 4.0


@pytest.fixture(scope="module")
def long_run():
    import json

    terms, fit = RESULTS / "goal3d_terms.json", RESULTS / "goal3d_results.json"
    assert terms.exists() and fit.exists(), "run scripts/goal3d_log_origin.py terms and fit"
    return json.loads(terms.read_text()), json.loads(fit.read_text())


def test_decomposition_closes(long_run):
    for r in long_run[0]:
        assert abs(r["closure"]) < 1e-8 * abs(r["variance"]), r


def test_local_total_log_coefficient(long_run):
    """Local action: b of the total (ρV = 50..12800) compatible with sum a_k^2 = 24."""
    tot = long_run[1]["1.0"]["total"]
    assert abs(tot["b"] - 24.0) < K_SIGMA * tot["se"], tot


@pytest.mark.parametrize("eps", [0.5, 0.25])
def test_smeared_same_pair_increment(long_run, eps):
    """Last increment of J¹/ρV (ρV 3200 -> 12800) vs. b(eps) ln 4."""
    rows = {r["rhoV"]: r["parts"]["J1"] for r in long_run[0] if r["eps"] == eps}
    a, b = rows[3200], rows[12800]
    inc = b["value"] / 12800 - a["value"] / 3200
    se = np.hypot(a["se"] / 3200, b["se"] / 12800)
    assert abs(inc - same_pair_log_coefficient(eps) * np.log(4)) < K_SIGMA * se, (inc, se)
