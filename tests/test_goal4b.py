"""Goal 4b: stochastic Λ(t) with back-reaction (``causets/stochastic_lambda.py``).

Tolerances fixed BEFORE seeing the results of the sweep:
  C1  constant Λ reproduces ΛCDM: relative 1e-3 (h, Omega_Λ, d_L, E(z));
  C2  correlation of the OU process: 4σ;
  C3  V(a = 1) vs. ``everpresent.past_cone_volume``: relative 1e-3.
External check added AFTER seeing the results (not a pre-registered
criterion): completion rates of Model 1 vs. those published by DNY
(sec. 5.1: 9990, 3228 and 535 out of 10 000 for alpha = 0.005, 0.01, 0.012),
at 4σ combined binomial. DNY document that the rate depends on the time step;
the process in this module is the continuum limit.
"""

import numpy as np
import pytest

from causets import stochastic_lambda as sl
from causets.everpresent import past_cone_volume

H, OL = 0.674, 0.685


@pytest.fixture(scope="module")
def constant_lambda():
    spec = sl.ProcessSpec(3 * OL * H**2, 0.0, 0.0, "Λ constante")
    return sl.evolve(spec, 2, np.random.default_rng(0), s_fixed=1.0)


def test_C1_constant_lambda_reproduces_lcdm(constant_lambda):
    r = constant_lambda
    dL_ref, E_ref, ol_ref = sl.lcdm_reference(r.z, OL, H)
    assert r.h[0] == pytest.approx(H, rel=1e-3)
    assert r.omega_lambda0[0] == pytest.approx(ol_ref, rel=1e-3)
    assert np.max(np.abs(r.dL[0] / dL_ref - 1)) < 1e-3
    assert np.max(np.abs(r.E[0] / E_ref - 1)) < 1e-3


def test_C3_volume_matches_past_cone_volume(constant_lambda):
    v_ref = past_cone_volume(1.0, sl.OMEGA_M / H**2, sl.OMEGA_R / H**2, 1 - (sl.OMEGA_M + sl.OMEGA_R) / H**2)
    assert constant_lambda.extra["V_today"][0] * H**4 == pytest.approx(v_ref, rel=1e-3)


@pytest.mark.parametrize("c", [0.5, 0.75, 0.6])
def test_C2_ou_correlation(c):
    """After steps adding up to Δu = ln(V2/V1), corr = e^(-c Δu) = (V1/V2)^c."""
    rng = np.random.default_rng(11)
    n = 200_000
    s0 = rng.standard_normal(n)
    s = s0.copy()
    for du in np.full(50, 0.04):  # Δu = 2
        s = sl.ou_step(s, du, c, rng)
    r = np.corrcoef(s0, s)[0, 1]
    target = np.exp(-c * 2.0)
    assert abs(r - target) < 4 * (1 - target**2) / np.sqrt(n)
    assert abs(np.var(s) - 1) < 4 * np.sqrt(2 / n)


@pytest.mark.parametrize("alpha,completed", [(0.005, 9990), (0.01, 3228), (0.012, 535)])
def test_model1_completion_matches_dny(alpha, completed):
    n = 10_000
    r = sl.evolve(sl.model1(alpha), n, np.random.default_rng(21))
    p_ours, p_dny = 1 - r.collapsed.mean(), completed / 10_000
    se = np.sqrt(p_ours * (1 - p_ours) / n + p_dny * (1 - p_dny) / 10_000) + 1e-4
    assert abs(p_ours - p_dny) < 4 * se, (p_ours, p_dny)


def test_bd_lambda_to_lk_inverts_goal4a():
    """With the amplitude that gives Λ_obs today, l_k matches that of Goal 4a
    (matter FLRW, r = 0.33, l = l_p): 2.63e-23 m (order of magnitude, ±10 %)."""
    from causets.everpresent import past_cone_volume as pcv

    v0 = pcv(1.0, 0.315, 9.1e-5, 1 - 0.315 - 9.1e-5) / H**4  # V today in units of H_100^-4
    lam = 3 * OL * H**2 * v0**0.25  # Λ_env(today) = Λ_obs
    assert sl.bd_lambda_to_lk(lam) == pytest.approx(2.63e-23, rel=0.1)
