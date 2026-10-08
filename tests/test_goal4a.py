"""Goal 4a: compatibilidad de las fluctuaciones de la acción BD con el Λ
observado (``causets/everpresent.py``, ``scripts/goal4_compatibility.py``).

Márgenes fijados ANTES de ver los resultados:
  identidades y formas cerradas: relativo 1e-9 (1e-12 donde es álgebra exacta);
  asíntota de la acción suavizada: relativo 1e-2 con eps = 1e-4;
  reglas de decisión D1 y D2 tal como están en el script.
"""

import json
from math import gamma
from pathlib import Path

import numpy as np
import pytest

from causets import everpresent as ep

FIG = Path(__file__).resolve().parents[1] / "figures"


def test_E1_einstein_de_sitter_cone_volume():
    assert ep.past_cone_volume(1.0, 1.0, 0.0, 0.0) == pytest.approx(ep.eds_past_cone_volume(), rel=1e-9)


def test_E1_cone_volume_matches_flrw_geometry():
    """En EdS, a = (H0 eta / 2)², y el cono es el de FLRWConeGeometry(alpha = 2)."""
    from causets.geometry import FLRWConeGeometry

    g = FLRWConeGeometry(2.0, height=2.0, dim=4)  # eta_0 = 2 / H0 con H0 = 1
    assert g.volume / 2**8 == pytest.approx(ep.eds_past_cone_volume(), rel=1e-9)  # a = eta²/4 -> a^4 = eta^8/2^8


@pytest.mark.parametrize("alpha", [0.005, 0.01, 0.012])
def test_E2_mapping_reproduces_model1(alpha):
    """delta Λ l² = Delta S / (rho V) aplicada a la acción de Poisson de Model 1
    da exactamente su ec. 3.13."""
    v = 3.7e240
    # cociente: los valores (~1e-121) están muy por debajo de la tolerancia
    # absoluta por defecto de pytest.approx (1e-12), que haría el test vacío
    ratio = ep.model1_from_action_mapping(v, alpha) / ep.model1_delta_lambda(v, alpha)
    assert ratio == pytest.approx(1.0, rel=1e-12, abs=0)


@pytest.mark.parametrize("eps", [1.0, 0.5, 0.1, 0.01])
def test_E3_cross_term_vanishes(eps):
    from causets.fluctuations import pair_weight

    n = np.arange(int(60 / eps) + 50)
    assert abs(ep.cross_sum(eps)) < 1e-10 * np.sum(np.abs(pair_weight(4, n, eps)))


def test_E4_smeared_amplitude_scales_as_eps_5_4():
    assert ep.same_pair_sum(1e-4) / 1e-4**2.5 == pytest.approx(ep.smeared_asymptotic_constant(), rel=1e-2)


def test_E5_local_same_pair_sum_closed_form():
    a = 4 / np.sqrt(6) * np.array([-1.0, 9.0, -16.0, 8.0])
    closed = sum(a[k] ** 2 * gamma(k + 0.5) / gamma(k + 1) for k in range(4))
    assert ep.same_pair_sum(1.0) == pytest.approx(closed, rel=1e-12)


# --- resultados de scripts/goal4_compatibility.py (fallan si faltan) ---


@pytest.fixture(scope="module")
def results():
    path = FIG / "goal4a_results.json"
    assert path.exists(), "ejecuta scripts/goal4_compatibility.py"
    return json.loads(path.read_text())


def test_D1_local_action_excluded(results):
    assert results["D1"]["excluded"], results["D1"]


def test_D2_smeared_scale_below_lhc_bound(results):
    assert results["D2"]["open"], results["D2"]["l_k_range_m"]


def test_D3_envelope_equations_of_state(results):
    """Model 1 sigue a la materia (w ≈ 0); la acción BD da w ≈ -1/2
    (delta Λ ∝ V^(-1/4) ∝ H en la era de materia). Tolerancia 0.05."""
    w = results["D3"]["w_eff_matter_era"]
    assert abs(w["Model 1 (α = 0.01)"]) < 0.05
    assert abs(w["BD suavizada (l_k ajustada hoy)"] + 0.5) < 0.05
