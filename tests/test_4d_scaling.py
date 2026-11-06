"""Step 2: scaling of (Delta S)^2 for the local BD action in 4D Minkowski.

Prediction (README, "Step 2"): <N_k> ∝ (rho V)^1.5 Γ(k+½)/k!, the "same pair"
term J¹ = sum a_k² <N_k> gives p -> 0.75, and the cross term with N does not
grow. Deterministic computations (exact integrals). Tolerances fixed before
writing the tests, although the values had been seen in a previous
exploration: they are loose and check the structure (monotonic convergence).
"""

import numpy as np
import pytest
from scipy.special import gammaln

from causets.continuum import expected_abundance
from causets.fluctuations import local_action_coefficients

A = local_action_coefficients(4)[1:]
G = np.exp(gammaln(np.arange(4) + 0.5) - gammaln(np.arange(4) + 1))  # Γ(k+½)/k!


def test_cross_term_cancels_in_4d():
    assert abs(A @ G) < 1e-12 * np.abs(A) @ G


def _spread(lam):
    r = np.array([expected_abundance(4, lam, k) for k in range(4)]) / lam**1.5 / G
    return (r.max() - r.min()) / r.max()


def test_abundances_scale_as_rhoV_three_halves():
    s = [_spread(lam) for lam in (1e4, 1e5, 1e6)]
    assert s[0] > s[1] > s[2] and s[2] < 0.10, s


def _j1(lam):
    return float(A**2 @ [expected_abundance(4, lam, k) for k in range(4)])


def test_same_pair_exponent_approaches_three_quarters():
    j = [_j1(lam) for lam in (1e4, 1e5, 1e6)]
    p = [np.log(j[i + 1] / j[i]) / np.log(10) / 2 for i in range(2)]
    assert p[0] > p[1] and 0.75 < p[1] < 0.80, p


# --- results saved by scripts/step2_4d_scaling.py (fail if missing) ---
# Rule fixed before the second batch of simulations: if the correlations grow
# like J¹, the slope of (total - J¹ - NN - Nk)/J¹ against ln(rho V) is
# compatible with 0 at 4 sigma.

def test_correlations_scale_like_same_pair():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "figures" / "step2_4d_results.json"
    assert path.exists(), "run scripts/step2_4d_scaling.py sim, extra and fit"
    pts = json.loads(path.read_text())["points"]
    assert max(p["rhoV"] for p in pts) >= 6400
    lam = np.array([p["rhoV"] for p in pts], float)
    y = np.array([p["rest_over_J1"] for p in pts])
    w = 1 / np.array([p["rest_over_J1_se"] for p in pts])
    X = np.column_stack([np.ones_like(lam), np.log(lam)])
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    se = np.sqrt(np.linalg.inv((X * w[:, None]).T @ (X * w[:, None]))[1, 1])
    assert abs(coef[1]) < 4.0 * se, (coef[1], se)
