"""Goal 3d: origen del término rho V ln(rho V) en (Delta S)^2 (Minkowski 2D).

Márgenes fijados ANTES de ver los resultados:
  convergencia de <N_k>/rho V - ln(rho V): |Δ| < 1e-3 entre rho V = 1e5 y 1e6
  (cálculo determinista);
  identidades de pair_weight: tolerancia de redondeo (1e-10).
"""

import numpy as np
import pytest

from causets.continuum import expected_abundance
from causets.fluctuations import local_action_coefficients, pair_weight, same_pair_log_coefficient


@pytest.mark.parametrize("k", [0, 1, 2])
def test_abundances_grow_as_rhoV_log_rhoV(k):
    """<N_k> = rho V ln(rho V) + c_k rho V + o(rho V): coeficiente 1 del log."""
    r = [expected_abundance(2, lam, k) / lam - np.log(lam) for lam in (1e5, 1e6)]
    assert abs(r[1] - r[0]) < 1e-3, r


def test_local_log_coefficient_is_sum_of_squares():
    a = local_action_coefficients(2)[1:]
    assert same_pair_log_coefficient(1.0) == pytest.approx(np.sum(a**2), rel=1e-12)
    assert same_pair_log_coefficient(1.0) == pytest.approx(24.0, rel=1e-12)


@pytest.mark.parametrize("eps", [1.0, 0.5, 0.25, 0.1])
def test_cross_term_has_no_log(eps):
    """sum_n f_eps(n) = 0: el término cruzado con N no aporta rho V ln(rho V)."""
    f = pair_weight(2, np.arange(20_000), eps)
    assert abs(np.sum(f)) < 1e-10 * np.sum(np.abs(f))


def test_smeared_same_pair_log_coefficient_exact():
    """<sum_pares f_eps(n)^2>/rho V crece como b(eps) ln(rho V): integral exacta
    sobre el diamante (sin Monte Carlo), tolerancia relativa 1e-3."""
    from scipy.stats import poisson

    from causets.continuum import _diamond_pair_integral

    eps, n = 0.5, np.arange(3000)
    f2 = pair_weight(2, n, eps) ** 2
    y = [_diamond_pair_integral(lambda z: float(np.sum(poisson.pmf(n, z) * f2)), 2, lam) / lam for lam in (1e5, 1e6)]
    assert (y[1] - y[0]) / np.log(10) == pytest.approx(same_pair_log_coefficient(eps), rel=1e-3)


# --- resultados guardados por scripts/goal3d_log_origin.py (fallan si faltan) ---

RESULTS = __import__("pathlib").Path(__file__).resolve().parents[1] / "figures"
K_SIGMA = 4.0


@pytest.fixture(scope="module")
def long_run():
    import json

    terms, fit = RESULTS / "goal3d_terms.json", RESULTS / "goal3d_results.json"
    assert terms.exists() and fit.exists(), "ejecuta scripts/goal3d_log_origin.py terms y fit"
    return json.loads(terms.read_text()), json.loads(fit.read_text())


def test_decomposition_closes(long_run):
    for r in long_run[0]:
        assert abs(r["closure"]) < 1e-8 * abs(r["variance"]), r


def test_local_total_log_coefficient(long_run):
    """Acción local: b del total (ρV = 50..12800) compatible con sum a_k^2 = 24."""
    tot = long_run[1]["1.0"]["total"]
    assert abs(tot["b"] - 24.0) < K_SIGMA * tot["se"], tot


@pytest.mark.parametrize("eps", [0.5, 0.25])
def test_smeared_same_pair_increment(long_run, eps):
    """Último incremento de J¹/ρV (ρV 3200 -> 12800) frente a b(eps) ln 4."""
    rows = {r["rhoV"]: r["parts"]["J1"] for r in long_run[0] if r["eps"] == eps}
    a, b = rows[3200], rows[12800]
    inc = b["value"] / 12800 - a["value"] / 3200
    se = np.hypot(a["se"] / 3200, b["se"] / 12800)
    assert abs(inc - same_pair_log_coefficient(eps) * np.log(4)) < K_SIGMA * se, (inc, se)
