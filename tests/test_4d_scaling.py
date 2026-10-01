"""Paso 2: escalado de (Delta S)^2 de la acción BD local en Minkowski 4D.

Predicción (README, "Paso 2"): <N_k> ∝ (rho V)^1.5 Γ(k+½)/k!, el término
"mismo par" J¹ = sum a_k² <N_k> da p -> 0.75 y el cruzado con N no crece.
Cálculos deterministas (integrales exactas). Márgenes fijados antes de
escribir los tests, aunque los valores se habían visto en una exploración
previa: son holgados y comprueban la estructura (convergencia monótona).
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


# --- resultados guardados por scripts/step2_4d_scaling.py (fallan si faltan) ---
# Regla fijada antes de la segunda tanda de simulaciones: si las correlaciones
# crecen como J¹, la pendiente de (total - J¹ - NN - Nk)/J¹ frente a ln(rho V)
# es compatible con 0 a 4 sigma.

def test_correlations_scale_like_same_pair():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "figures" / "step2_4d_results.json"
    assert path.exists(), "ejecuta scripts/step2_4d_scaling.py sim, extra y fit"
    pts = json.loads(path.read_text())["points"]
    assert max(p["rhoV"] for p in pts) >= 6400
    lam = np.array([p["rhoV"] for p in pts], float)
    y = np.array([p["rest_over_J1"] for p in pts])
    w = 1 / np.array([p["rest_over_J1_se"] for p in pts])
    X = np.column_stack([np.ones_like(lam), np.log(lam)])
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    se = np.sqrt(np.linalg.inv((X * w[:, None]).T @ (X * w[:, None]))[1, 1])
    assert abs(coef[1]) < 4.0 * se, (coef[1], se)
