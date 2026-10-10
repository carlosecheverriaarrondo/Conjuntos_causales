"""BAO (DESI DR2) y priors de distancia del CMB (Planck 2018) para historias
de expansión arbitrarias (Goal 4d). Referencias en ``scripts/goal4d_combined.py``.

Todas las distancias en unidades c/H_100 (como ``stochastic_lambda``).
  r_s(a) = ∫_0^eta(a) c_s d eta,  c_s = 1/sqrt(3 (1 + R_b a)),  R_b = 3 omega_b/(4 omega_γ),
  D_M(z) = eta_0 - eta(z),  D_H(z) = 1/H(z),  D_V = (z D_M² D_H)^(1/3),
  R = sqrt(omega_m) D_M(z*),  l_A = π D_M(z*)/r_s(z*)
(Chen, Huang & Wang 2019, ecs. 1-3, con universo plano).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OMEGA_B = 0.02236
OMEGA_GAMMA = 2.4728e-5  # T_CMB = 2.7255 K
PLANCK_BF = dict(omega_b=0.02236, omega_m=0.14327, h=0.6727, z_star=1089.95, z_drag=1059.93,
                 r_star=144.39, r_drag=147.05)  # Planck 2018 VI, tabla 2, TT,TE,EE+lowE
CMB_MEAN = np.array([1.7502, 301.471, 0.02236])
CMB_SIG = np.array([0.0046, 0.0895, 0.00015])
CMB_CORR = np.array([[1.0, 0.46, -0.66], [0.46, 1.0, -0.33], [-0.66, -0.33, 1.0]])
DESI_DIR = Path(__file__).resolve().parents[1] / "data" / "desi_dr2"


def z_star_hs96(ob, om):
    g1 = 0.0783 * ob**-0.238 / (1 + 39.5 * ob**0.763)
    g2 = 0.560 / (1 + 21.1 * ob**1.81)
    return 1048 * (1 + 0.00124 * ob**-0.738) * (1 + g1 * om**g2)


def z_drag_eh98(ob, om):
    b1 = 0.313 * om**-0.419 * (1 + 0.607 * om**0.674)
    b2 = 0.238 * om**0.223
    return 1291 * om**0.251 / (1 + 0.659 * om**0.828) * (1 + b1 * ob**b2)


def z_star(om, ob=OMEGA_B):
    """HS96 desplazado para dar z* de Planck en su mejor ajuste."""
    return z_star_hs96(ob, om) + PLANCK_BF["z_star"] - z_star_hs96(PLANCK_BF["omega_b"], PLANCK_BF["omega_m"])


def z_drag(om, ob=OMEGA_B):
    return z_drag_eh98(ob, om) + PLANCK_BF["z_drag"] - z_drag_eh98(PLANCK_BF["omega_b"], PLANCK_BF["omega_m"])


def _interp_path(a_grid, path, a_target):
    """path (n+1, R) en la rejilla común a_grid; valor en a_target (R,) por realización (lineal en ln a)."""
    la = np.log(a_grid)
    lt = np.log(a_target)
    i = np.clip(np.searchsorted(la, lt) - 1, 0, len(la) - 2)
    w = (lt - la[i]) / (la[i + 1] - la[i])
    cols = np.arange(path.shape[1])
    return path[i, cols] * (1 - w) + path[i + 1, cols] * w


def sound_horizon_path(a_grid, eta_path, ob=OMEGA_B):
    """r_s(a) acumulado (n+1, R): trapecio en eta; antes de a_grid[0], c_s = c_s(a_grid[0])."""
    cs = 1 / np.sqrt(3 * (1 + 3 * ob / (4 * OMEGA_GAMMA) * a_grid))[:, None]
    inc = 0.5 * (cs[1:] + cs[:-1]) * np.diff(eta_path, axis=0)
    return np.concatenate([cs[:1] * eta_path[:1], cs[:1] * eta_path[:1] + np.cumsum(inc, axis=0)])


def early_quantities(real, om, ob=OMEGA_B):
    """(r_star, r_drag, R, l_A) por realización, desde ``evolve(..., record_path=True)``."""
    a_grid, eta_path = real.extra["a_grid"], real.extra["eta_path"]
    om = np.broadcast_to(np.asarray(om, float), (eta_path.shape[1],))
    rs = sound_horizon_path(a_grid, eta_path, ob)
    a_s, a_d = 1 / (1 + z_star(om, ob)), 1 / (1 + z_drag(om, ob))
    r_star = _interp_path(a_grid, rs, a_s)
    r_drag = _interp_path(a_grid, rs, a_d)
    dm_star = eta_path[-1] - _interp_path(a_grid, eta_path, a_s)
    return r_star, r_drag, np.sqrt(om) * dm_star, np.pi * dm_star / r_star


class CMBPriors:
    """Gaussiana de (R, l_A) condicionada a omega_b = OMEGA_B."""

    def __init__(self):
        cov = CMB_CORR * np.outer(CMB_SIG, CMB_SIG)
        s11, s12, s22 = cov[:2, :2], cov[:2, 2:], cov[2:, 2:]
        self.mean = CMB_MEAN[:2] + (s12 @ np.linalg.inv(s22) @ (np.array([OMEGA_B]) - CMB_MEAN[2:]))
        self.cinv = np.linalg.inv(s11 - s12 @ np.linalg.inv(s22) @ s12.T)

    def chi2(self, R, lA):
        d = np.stack([R, lA], axis=-1) - self.mean
        return np.einsum("...i,ij,...j->...", d, self.cinv, d)


class DESIDR2:
    def __init__(self, data_dir: Path = DESI_DIR):
        rows = [ln.split() for ln in (data_dir / "desi_gaussian_bao_ALL_GCcomb_mean.txt").read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
        self.z = np.array([float(r[0]) for r in rows])
        self.value = np.array([float(r[1]) for r in rows])
        self.kind = [r[2] for r in rows]
        self.cinv = np.linalg.inv(np.loadtxt(data_dir / "desi_gaussian_bao_ALL_GCcomb_cov.txt"))
        self.z_unique = np.unique(self.z)

    def chi2(self, dm, dh, r_d):
        """dm, dh: (R, len(z_unique)) en unidades c/H_100; r_d: (R,) en las mismas unidades."""
        j = np.searchsorted(self.z_unique, self.z)
        cols = []
        for k, (z, kind) in enumerate(zip(self.z, self.kind)):
            DM, DH = dm[:, j[k]], dh[:, j[k]]
            val = {"DM_over_rs": DM, "DH_over_rs": DH, "DV_over_rs": np.cbrt(z * DM**2 * DH)}[kind]
            cols.append(val / r_d)
        d = np.stack(cols, axis=1) - self.value
        return np.einsum("ri,ij,rj->r", d, self.cinv, d)
