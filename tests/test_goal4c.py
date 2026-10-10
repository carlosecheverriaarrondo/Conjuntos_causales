"""Goal 4c: likelihood Pantheon+SH0ES (``causets/supernovae.py``).

Criterio V2 (fijado antes de calcular, ver ``scripts/goal4c_sn_fit.py``):
el chi² comprimido coincide con el directo, |Δ| < 0.01. Dos partes: (a) la
forma cuadrática es algebraicamente exacta para el modelo spline; (b) el
spline en los nodos reproduce mu(z) exacta de ΛCDM en cada SN.
Necesita los datos (``scripts/get_pantheon.sh``); si faltan, se salta.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from causets.supernovae import DATA_DIR, PantheonPlusSH0ES

pytestmark = pytest.mark.skipif(not (DATA_DIR / "PantheonPlusSH0ES.dat").exists(),
                                reason="faltan los datos de Pantheon+ (scripts/get_pantheon.sh)")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sn():
    return PantheonPlusSH0ES()


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("goal4c", ROOT / "scripts" / "goal4c_sn_fit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sample_size(sn):
    """Corte estándar: 1701 curvas de luz -> zHD > 0.01 o calibrador; 77 curvas de calibradores."""
    assert sn.cov.shape == (sn.n, sn.n)
    assert sn.calib.sum() == 77


def test_V2a_quadratic_form_is_exact(sn):
    rng = np.random.default_rng(3)
    for _ in range(3):
        m = 40 + 5 * np.log10(sn.nodes) + 0.05 * rng.standard_normal(len(sn.nodes))
        MB = -19.25 + 0.1 * rng.standard_normal()
        chi2_c, _ = sn.chi2_nodes(m, MB)
        assert abs(chi2_c - sn.chi2_direct(sn.P @ m, MB)) < 0.01


def test_V2a_profiled_MB_is_minimum(sn):
    m = 43.2 + 5 * np.log10(sn.nodes)
    chi2, MB = sn.chi2_nodes(m)
    for d in (-1e-3, 1e-3):
        assert sn.chi2_nodes(m, MB + d)[0] > chi2


def test_V2b_spline_reproduces_exact_lcdm(sn, script):
    om, h = 0.15, 0.73
    mu_nodes = script.lcdm_mu_nodes(om, h, sn.nodes)
    mu_exact = script.lcdm_mu_nodes(om, h, sn.z_hd)  # en cada SN (calibradores: no se usa)
    MB = -19.25
    assert abs(sn.chi2_nodes(mu_nodes, MB)[0] - sn.chi2_direct(mu_exact, MB)) < 0.01


def test_seed_noise_independent_of_batch():
    """El ajuste por semilla exige que una semilla dé la misma historia sea cual
    sea el lote y los parámetros de las demás realizaciones."""
    from causets import stochastic_lambda as sl

    n_steps = 300
    xi = np.random.default_rng(5).standard_normal((n_steps, 3))
    spec = sl.ProcessSpec(np.array([0.05, 0.05, 0.08]), 0.5, 0.5)
    r1 = sl.evolve(spec, 3, None, a_start=1e-5, n_steps=n_steps, omega_m=np.array([0.14, 0.2, 0.14]), noise=xi)
    spec2 = sl.ProcessSpec(np.array([0.05]), 0.5, 0.5)
    r2 = sl.evolve(spec2, 1, None, a_start=1e-5, n_steps=n_steps, omega_m=np.array([0.14]), noise=xi[:, :1])
    assert not r1.collapsed.any() and not r2.collapsed.any()
    assert r1.lam_today[0] == r2.lam_today[0]
    np.testing.assert_array_equal(r1.dL[0], r2.dL[0])
    assert r1.lam_today[1] != r2.lam_today[0]
