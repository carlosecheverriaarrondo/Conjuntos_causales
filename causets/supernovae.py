"""Pantheon+SH0ES supernova likelihood (Goal 4c).

Data: Pantheon+ (Scolnic et al., ApJ 938 (2022) 113, arXiv:2112.03863;
Brout et al., ApJ 938 (2022) 110, arXiv:2202.04077) with the SH0ES Cepheid
calibrators (Riess et al., ApJL 934 (2022) L7, arXiv:2112.04510).
Files from https://github.com/PantheonPlusSH0ES/DataRelease
(``scripts/get_pantheon.sh``). Standard cut of the Pantheon+SH0ES
likelihood: zHD > 0.01 or IS_CALIBRATOR = 1.

Residual (Brout et al. 2022, eqs. 14-15):
    r_i = m_b_corr_i - M_B - mu_i,
    mu_i = CEPH_DIST_i              if i is a calibrator,
    mu_i = mu_model(zHD_i, zHEL_i)  otherwise,
    chi² = rᵀ C⁻¹ r,  C = STAT+SYS covariance.
mu_model = 5 log10[(1 + zHEL) chi(zHD) / Mpc] + 25, with chi the comoving
distance (flat universe), that is, mu(zHD) + 5 log10[(1 + zHEL)/(1 + zHD)].

EXACT COMPRESSION. mu(zHD) is interpolated with a natural cubic spline in
ln z over K nodes; the spline is linear in the nodal values, mu = P m.
Since r is linear in (m, M_B), chi² is a quadratic form of dimension
K + 1 with precomputed matrices: cost O(K²) per model instead of O(N²).
M_B is minimised analytically (profiled).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "pantheonplus"
C_KM_S = 299792.458
DL_UNIT_MPC = C_KM_S / 100.0  # c / H_100 in Mpc


def default_nodes() -> np.ndarray:
    return np.geomspace(0.0095, 2.3, 64)


class PantheonPlusSH0ES:
    def __init__(self, data_dir: Path = DATA_DIR, nodes: np.ndarray | None = None):
        names = (data_dir / "PantheonPlusSH0ES.dat").read_text().split("\n")[0].split()
        raw = np.genfromtxt(data_dir / "PantheonPlusSH0ES.dat", names=True, dtype=None, encoding=None)
        cov_flat = np.loadtxt(data_dir / "PantheonPlusSH0ES_STAT+SYS.cov")
        n_all = int(cov_flat[0])
        assert n_all == len(raw), (n_all, len(raw), names[:3])
        cov = cov_flat[1:].reshape(n_all, n_all)
        mask = (raw["zHD"] > 0.01) | (raw["IS_CALIBRATOR"] == 1)
        self.z_hd = raw["zHD"][mask]
        self.z_hel = raw["zHEL"][mask]
        self.m_b = raw["m_b_corr"][mask]
        self.calib = raw["IS_CALIBRATOR"][mask] == 1
        self.ceph = raw["CEPH_DIST"][mask]
        self.cov = cov[np.ix_(mask, mask)]
        self.cinv = np.linalg.inv(self.cov)
        self.n = int(mask.sum())
        self.nodes = default_nodes() if nodes is None else np.asarray(nodes)
        hub = ~self.calib
        assert self.z_hd[hub].min() >= self.nodes[0] and self.z_hd[hub].max() <= self.nodes[-1]
        # P: natural spline in ln z, linear in the nodal values
        K = len(self.nodes)
        P = np.zeros((self.n, K))
        ln_nodes = np.log(self.nodes)
        for k in range(K):
            e = np.zeros(K)
            e[k] = 1.0
            P[hub, k] = CubicSpline(ln_nodes, e, bc_type="natural")(np.log(self.z_hd[hub]))
        self.P = P
        # data vector with the fixed parts subtracted: d' = m_b - mu_ceph (calib.) - offset_hel (rest)
        offset = np.where(hub, 5 * np.log10((1 + self.z_hel) / (1 + self.z_hd)), self.ceph)
        dprime = self.m_b - offset
        # quadratic form in x = (m_1..m_K, M_B): r = d' - Q x. It is expanded
        # around a reference model x0 (approximate ΛCDM) so that all the terms
        # are small: expanded at x = 0 there are cancellations of terms ~10¹²
        # and round-off errors ~0.1 in chi².
        Q = np.column_stack([P, np.ones(self.n)])
        self._x0 = np.concatenate([_reference_mu(self.nodes), [-19.25]])
        r0 = dprime - Q @ self._x0
        CQ = self.cinv @ Q
        self._A = Q.T @ CQ  # (K+1, K+1)
        self._b = CQ.T @ r0  # Qᵀ C⁻¹ r0
        self._c = float(r0 @ self.cinv @ r0)

    # ---- direct chi² (to validate the compression) ----
    def chi2_direct(self, mu_hd: np.ndarray, M_B: float) -> float:
        """mu_hd: model mu(zHD) at each SN (ignored for calibrators)."""
        mu = np.where(self.calib, self.ceph, mu_hd + 5 * np.log10((1 + self.z_hel) / (1 + self.z_hd)))
        r = self.m_b - M_B - mu
        return float(r @ self.cinv @ r)

    # ---- compressed chi² ----
    def chi2_nodes(self, mu_nodes: np.ndarray, M_B: np.ndarray | float | None = None):
        """mu_nodes: (..., K) values of mu(z) at the nodes. Returns (chi², M_B);
        if M_B is None it is minimised analytically."""
        m = np.asarray(mu_nodes, float) - self._x0[:-1]
        A, b = self._A, self._b
        Amm, AmB, ABB = A[:-1, :-1], A[:-1, -1], A[-1, -1]
        bm, bB = b[:-1], b[-1]
        quad = np.einsum("...i,ij,...j->...", m, Amm, m)
        lin_m = m @ bm
        cross = m @ AmB
        if M_B is None:
            dB = (bB - cross) / ABB
        else:
            dB = np.asarray(M_B, float) - self._x0[-1]
        chi2 = self._c - 2 * (lin_m + dB * bB) + quad + 2 * dB * cross + dB**2 * ABB
        return chi2, dB + self._x0[-1]


def _reference_mu(z: np.ndarray, om: float = 0.143, h: float = 0.73) -> np.ndarray:
    """mu(z) of flat ΛCDM (without radiation), only as the expansion point."""
    zz = np.linspace(0, z.max(), 20001)
    inv_E = 1 / np.sqrt(om / h**2 * (1 + zz) ** 3 + 1 - om / h**2)
    chi = np.concatenate([[0], np.cumsum(0.5 * (inv_E[1:] + inv_E[:-1]) * np.diff(zz))]) / h
    return mu_from_dl(np.interp(z, zz, chi) * (1 + z))


def mu_from_dl(dl_h100: np.ndarray) -> np.ndarray:
    """mu from a luminosity distance in units of c/H_100."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 5 * np.log10(dl_h100 * DL_UNIT_MPC) + 25
