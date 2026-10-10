"""Λ(t) estocástico con retroacción sobre la expansión (Goal 4b-2).

MODELO
------
A lo largo de la línea de universo de un observador comóvil,
    Λ(t) = lam · V(t)^(-q) · s(t),
con V(t) el volumen del pasado causal (hasta el big bang) y s(t) un proceso
gaussiano de media 0 y varianza 1 con
    corr(s(t1), s(t2)) = (V1 / V2)^c       (t1 < t2),
es decir, un proceso de Ornstein-Uhlenbeck en u = ln V con tasa c
(``ou_step``).
  - Everpresent Λ, Model 1 (Das, Nasiri & Yazdi, JCAP 10 (2023) 047,
    arXiv:2304.03819, ecs. 3.13 y 3.15): Λ V es un paseo aleatorio con
    Var ∝ V, así que q = 1/2, c = 1/2 y lam = 8π alpha (en unidades H_100,
    c = 1: los factores de Planck se cancelan).
  - Acción BD (este proyecto, Goal 4a-4b): Λ V ∝ Delta S con
    Var(Delta S) ∝ V^(3/2), así que q = 1/4 y, si S(t) es un paseo aleatorio,
    c = 3/4. Con correlación de incrementos (Goal 4b-1), c = 3 beta / 4,
    beta <= 1. La amplitud lam se traduce en la escala de no localidad l_k
    con ``bd_lambda_to_lk``.

FONDO Y RETROACCIÓN
-------------------
Unidades: H_100 = 100 km/s/Mpc, c = 1. Densidades físicas fijas (Planck 2018,
Astron. Astrophys. 641, A6 (2020), arXiv:1807.06209): omega_m = 0.1430,
omega_r = 4.15e-5 (fotones + neutrinos sin masa). Friedmann:
    (H / H_100)² = omega_r / a^4 + omega_m / a³ + Λ / 3,
con Λ(t) evaluado sobre la propia historia. "Hoy" es a = 1, así que h = H(1)
y Omega_Λ0 = Λ(1) / (3 h²) salen de cada realización (como en DNY, sec. 4).
Si H² < 0 la realización colapsa y se marca.

Volumen del pasado causal (FLRW plano, ver ``everpresent.past_cone_volume``):
    V(eta) = (4π/3) [eta³ M0 - 3 eta² M1 + 3 eta M2 - M3],  M_k = ∫ a^4 eta'^k d eta',
acumulado paso a paso (O(1) por paso). Se arranca en la era de radiación
(a = sqrt(omega_r) eta, M_k = omega_r² eta^(5+k)/(5+k)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

OMEGA_M = 0.1430
OMEGA_R = 4.15e-5
H100_TP = 100e3 / 3.0856776e22 * 5.391247e-44  # H_100 en unidades de Planck


@dataclass
class ProcessSpec:
    """Λ = lam V^(-q) s, corr(s) = (V1/V2)^c."""

    lam: float
    q: float
    c: float
    name: str = ""


def model1(alpha: float) -> ProcessSpec:
    return ProcessSpec(8 * np.pi * alpha, 0.5, 0.5, f"Model 1 (α = {alpha:g})")


def bd_process(lam: float, beta: float = 1.0) -> ProcessSpec:
    return ProcessSpec(lam, 0.25, 0.75 * beta, f"BD (λ = {lam:g}, β = {beta:g})")


def bd_lambda_to_lk(lam: float, l_over_lp: float = 1.0, r: float = 0.33, g_over_v15: float = 1.4194) -> float:
    """l_k (m) de la acción suavizada que da Λ_env = lam V^(-1/4) (unidades H_100).

    Del Goal 4a: Λ l_p² = A (rho V)^(-1/4) (l_p/l)², con V en l_p^4; en
    unidades H_100, Λ = A / (l H100_TP) · V^(-1/4), así que A = lam l H100_TP,
    y A² = (1 + r) (G/V^1.5)/(8 √c4) · c_asym eps^(5/2), l_k = l eps^(-1/4)."""
    from .everpresent import C4, L_PLANCK_M, smeared_asymptotic_constant

    A = lam * l_over_lp * H100_TP
    eps = (A**2 * 8 * np.sqrt(C4) / ((1 + r) * g_over_v15 * smeared_asymptotic_constant())) ** 0.4
    return float(l_over_lp * L_PLANCK_M * eps**-0.25)


def ou_step(s: np.ndarray, du: float, c: float, rng) -> np.ndarray:
    """Un paso del proceso OU en u = ln V: corr(s(u), s(u + du)) = e^(-c du)."""
    rho = np.exp(-c * du)
    return rho * s + np.sqrt(1 - rho * rho) * rng.standard_normal(s.shape)


@dataclass
class Realizations:
    spec: ProcessSpec
    h: np.ndarray
    omega_lambda0: np.ndarray
    collapsed: np.ndarray
    z: np.ndarray  # rejilla de corrimientos al rojo
    dL: np.ndarray  # (R, nz) distancia de luminosidad (unidades c/H_100)
    E: np.ndarray  # (R, nz) H(z)/H0
    lam_today: np.ndarray
    extra: dict = field(default_factory=dict)


def evolve(spec: ProcessSpec, n_real: int, rng, a_start: float = 1e-6, n_steps: int = 4000,
           z_grid: np.ndarray | None = None, s_fixed: float | None = None, z_on: float = 1e4) -> Realizations:
    """Integra n_real realizaciones a la vez desde a_start hasta a = 1 (pasos
    uniformes en ln a; Λ constante dentro de cada paso, trapecio en eta).

    Convención de DNY (sec. 5.2): Λ = 0 antes de z_on = 1e4 (el volumen previo
    sólo cuenta materia y radiación) y la "acción" arranca en S0 = 0, es decir
    s = 0 en z_on. Para Model 1 esto da exactamente su paseo aleatorio desde
    S0 = 0: Var(Λ V) = lam² (V - V0). (Arrancar antes dispara los colapsos
    H² < 0, como documentan DNY.)
    ``s_fixed``: si se da, s(t) = s_fixed desde el principio (determinista; con
    q = c = 0 es un Λ constante, para validar frente a ΛCDM)."""
    if z_grid is None:
        z_grid = np.concatenate([np.geomspace(0.01, 2.3, 40)])
    a_grid = np.exp(np.linspace(np.log(a_start), 0.0, n_steps + 1))
    R = n_real
    eta = np.full(R, a_start / np.sqrt(OMEGA_R))
    M = np.array([np.full(R, OMEGA_R**2 * eta[0] ** (5 + k) / (5 + k)) for k in range(4)])

    def volume(eta, M):
        return 4 * np.pi / 3 * (eta**3 * M[0] - 3 * eta**2 * M[1] + 3 * eta * M[2] - M[3])

    s = np.full(R, s_fixed if s_fixed is not None else 0.0, float)
    a_on = 0.0 if s_fixed is not None else 1 / (1 + z_on)
    V = volume(eta, M)
    collapsed = np.zeros(R, dtype=bool)
    # eta en los a de la rejilla de z (para distancias) y H en esos puntos
    a_z = 1 / (1 + z_grid)
    eta_at = np.full((R, len(z_grid)), np.nan)
    H_at = np.full((R, len(z_grid)), np.nan)
    lam_now = spec.lam * V ** (-spec.q) * s

    def H2(a, lam):
        return OMEGA_R / a**4 + OMEGA_M / a**3 + lam / 3

    for i in range(n_steps):
        a0, a1 = a_grid[i], a_grid[i + 1]
        lam_now = spec.lam * V ** (-spec.q) * s if a0 >= a_on else np.zeros(R)
        h2_0 = H2(a0, lam_now)
        h2_1 = H2(a1, lam_now)
        bad = (h2_0 <= 0) | (h2_1 <= 0)
        collapsed |= bad
        h0_ = np.sqrt(np.where(bad, np.nan, h2_0))
        h1_ = np.sqrt(np.where(bad, np.nan, h2_1))
        # d eta = da / (a² H) = d ln a / (a H): trapecio en ln a
        dlna = np.log(a1 / a0)
        deta = 0.5 * dlna * (1 / (a0 * h0_) + 1 / (a1 * h1_))
        eta_new = eta + deta
        # momentos M_k: trapecio en eta de a^4 eta^k
        for k in range(4):
            M[k] = M[k] + 0.5 * deta * (a0**4 * eta**k + a1**4 * eta_new**k)
        eta = eta_new
        V_new = volume(eta, M)
        if s_fixed is None and a1 > a_on:
            s = ou_step(s, np.log(V_new / V), spec.c, rng)
        V = V_new
        # registrar los a de la rejilla que caen en (a0, a1]
        for jz in np.nonzero((a_z > a0) & (a_z <= a1))[0]:
            frac = np.log(a_z[jz] / a0) / dlna
            eta_at[:, jz] = eta - deta + frac * deta
            H_at[:, jz] = np.sqrt(np.maximum(H2(a_z[jz], spec.lam * V ** (-spec.q) * s), 0))

    lam_today = spec.lam * V ** (-spec.q) * s
    h = np.sqrt(np.where(collapsed, np.nan, np.maximum(H2(1.0, lam_today), 0)))
    chi = eta[:, None] - eta_at  # distancia comóvil
    dL = chi * (1 + z_grid[None, :])
    return Realizations(spec, h, lam_today / (3 * h**2), collapsed, z_grid, dL, H_at / h[:, None], lam_today,
                        dict(V_today=V))


def lcdm_reference(z_grid: np.ndarray, omega_lambda: float = 0.685, h: float = 0.674):
    """Distancias y E(z) de ΛCDM plano con omega_m, omega_r fijos (Λ constante)."""
    from scipy.integrate import quad

    om, orad = OMEGA_M / h**2, OMEGA_R / h**2
    ol = 1 - om - orad

    def E(z):
        return np.sqrt(orad * (1 + z) ** 4 + om * (1 + z) ** 3 + ol)

    dL = np.array([(1 + z) * quad(lambda x: 1 / E(x), 0, z, epsrel=1e-12)[0] / h for z in z_grid])
    return dL, np.array([E(z) for z in z_grid]), ol


def lcdm_like(real: Realizations, omega_l=(0.6, 0.8), h_range=(0.6, 0.8), dmu_max=0.1, e_tol=0.05):
    """Criterio "tipo ΛCDM" (fijado antes de calcular, ver README Goal 4b-2):
    Omega_Λ0 y h en los rangos; forma de mu(z) (restada la media, que absorbe H0
    y la magnitud absoluta de las SN) a menos de dmu_max de ΛCDM en
    0.01 < z < 2.3; |E(z)/E_ΛCDM(z) - 1| < e_tol en z = 0.5, 1, 2."""
    dL_ref, E_ref, _ = lcdm_reference(real.z)
    with np.errstate(invalid="ignore", divide="ignore"):
        dmu = 5 * np.log10(real.dL / dL_ref[None, :])
    dmu_shape = dmu - np.nanmean(dmu, axis=1, keepdims=True)
    E_ok = np.ones(len(real.h), dtype=bool)
    for zt in (0.5, 1.0, 2.0):
        j = int(np.argmin(abs(real.z - zt)))
        E_ok &= np.abs(real.E[:, j] / E_ref[j] - 1) < e_tol
    ok = (~real.collapsed & (real.omega_lambda0 > omega_l[0]) & (real.omega_lambda0 < omega_l[1])
          & (real.h > h_range[0]) & (real.h < h_range[1]) & (np.nanmax(np.abs(dmu_shape), axis=1) < dmu_max) & E_ok)
    return ok
