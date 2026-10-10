"""Goal 4d: BAO de DESI DR2 y priors de distancia del CMB (``causets/cmb_bao.py``).

Tolerancias fijadas antes de ejecutar estos tests. W1 y W2 (criterios del
script) se evalúan en ``scripts/goal4d_combined.py validate``; aquí se
comprueban las piezas. Necesita los datos (``scripts/get_pantheon.sh``); si
faltan, se salta lo que los usa.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from causets import cmb_bao as cb
from causets import stochastic_lambda as sl

HAVE_DESI = (cb.DESI_DIR / "desi_gaussian_bao_ALL_GCcomb_mean.txt").exists()


def _lcdm(om=0.1433, h=0.6727, n_steps=2000):
    return sl.evolve(sl.ProcessSpec(3 * (h**2 - om - sl.OMEGA_R), 0, 0), 1, None, a_start=1e-5,
                     n_steps=n_steps, s_fixed=1.0, omega_m=om, record_path=True, exact_start=True)


def test_sound_horizon_without_baryons_is_eta_over_sqrt3():
    real = _lcdm()
    rs = cb.sound_horizon_path(real.extra["a_grid"], real.extra["eta_path"], ob=0.0)
    np.testing.assert_allclose(rs[:, 0], real.extra["eta_path"][:, 0] / np.sqrt(3), rtol=1e-12)


def test_sound_horizon_matches_direct_integral():
    """r_s(z*) por la historia de evolve frente a Chen et al. ec. 3 con quad: 1e-4."""
    om, h, ob = 0.1433, 0.6727, cb.OMEGA_B
    real = _lcdm(om, h)
    r_star = cb.early_quantities(real, om)[0][0]
    a_s = 1 / (1 + cb.z_star(om))
    ol = h**2 - om - sl.OMEGA_R

    def integrand(a):
        H = np.sqrt(sl.OMEGA_R / a**4 + om / a**3 + ol)
        return 1 / (a**2 * H * np.sqrt(3 * (1 + 3 * ob / (4 * cb.OMEGA_GAMMA) * a)))

    direct = quad(integrand, 0, a_s, epsrel=1e-12, limit=200)[0]
    assert abs(r_star / direct - 1) < 1e-4


def test_z_star_and_z_drag_anchored_to_planck():
    assert cb.z_star(cb.PLANCK_BF["omega_m"]) == pytest.approx(1089.95, abs=1e-9)
    assert cb.z_drag(cb.PLANCK_BF["omega_m"]) == pytest.approx(1059.93, abs=1e-9)


def test_cmb_conditional_gaussian():
    cmb = cb.CMBPriors()
    assert cmb.chi2(*cb.CMB_MEAN[:2]) == pytest.approx(0, abs=1e-12)  # omega_b fijado en su media
    # sólo R desplazado: varianza condicionada a omega_b y minimizada en l_A = sigma_R² (1 - rho²_R,ob) ...
    # comprobación directa: χ² de un desplazamiento en R con l_A en su media condicional óptima
    c = np.linalg.inv(cmb.cinv)
    assert c[0, 0] == pytest.approx(cb.CMB_SIG[0] ** 2 * (1 - 0.66**2), rel=1e-12)


@pytest.mark.skipif(not HAVE_DESI, reason="faltan los datos de DESI DR2")
def test_bao_chi2_zero_at_data_and_size():
    bao = cb.DESIDR2()
    assert len(bao.value) == 13 and len(bao.z_unique) == 7
    # construir (dm, dh, r_d) que reproducen exactamente los datos
    r_d = np.array([1.0])
    dm = np.zeros((1, 7))
    dh = np.zeros((1, 7))
    for z, v, k in zip(bao.z, bao.value, bao.kind):
        j = int(np.searchsorted(bao.z_unique, z))
        if k == "DM_over_rs":
            dm[0, j] = v
        elif k == "DH_over_rs":
            dh[0, j] = v
    j = int(np.searchsorted(bao.z_unique, 0.295))
    dm[0, j], dh[0, j] = 7.94167639, 7.94167639 / 0.295  # D_V = (z DM² DH)^(1/3) = DM si DH = DM/z
    assert bao.chi2(dm, dh, r_d)[0] == pytest.approx(0, abs=1e-12)
