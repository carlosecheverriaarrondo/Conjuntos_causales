"""Tests del Goal 2 en de Sitter: el operador BD recupera R = d(d-1) H^2.

Estrategia (ver README, Goal 2):
1. Valor esperado exacto a densidad finita, E[(B_eps 1)(x)], por la fórmula de
   Mecke (integral numérica determinista, sin ruido de muestreo).
2. Convergencia: E[B 1] -> -R/2 cuando l H -> 0 (rho -> infinito) con la
   región fija. Tolerancias: |E + R/2| < 4 % de R/2 (4D) y < 1.5 % (2D) en
   l H = 0.05, y la distancia a -R/2 decrece monótonamente con l H.
3. Simulación: la media de B_eps 1 sobre los puntos de un sprinkling de
   de Sitter coincide con el valor exacto dentro de 4 SEM.
"""

import numpy as np
import pytest

from causets import minkowski_causal_matrix, sprinkle
from causets.action import bd_operator
from causets.continuum import expected_bd_on_constant_at_top, expected_bd_on_constant_de_sitter
from causets.regions import (
    ConformalDiamond,
    DeSitterIntervalVolume,
    de_sitter_interval_volume,
    de_sitter_invariant,
    de_sitter_scale_factor,
)

K_SIGMA = 4.0
H = 1.0


@pytest.mark.parametrize("dim", [2, 4])
def test_conformal_diamond_volume(dim):
    """Volumen en de Sitter frente a integración Monte Carlo con peso a^d."""
    a = de_sitter_scale_factor(H)
    region = ConformalDiamond(dim, 0.8, -1.5, a)
    rng = np.random.default_rng(60)
    lo, hi = region.bounding_box()
    pts = rng.uniform(lo, hi, size=(1_000_000, dim))
    vals = region.contains(pts) * a(pts[:, 0]) ** dim * np.prod(hi - lo)
    assert abs(vals.mean() - region.volume) < K_SIGMA * vals.std() / np.sqrt(len(vals))


def test_de_sitter_interval_volume_off_axis():
    """El volumen de I(y, x) sólo depende del invariante Z (simetría máxima):
    se comprueba un par fuera del eje contra Monte Carlo directo."""
    a = de_sitter_scale_factor(H)
    x, y = np.array([-1.0, 0.0, 0.0, 0.0]), np.array([-2.0, 0.5, 0.0, 0.0])
    rng = np.random.default_rng(61)
    lo, hi = np.array([-2.0, -0.5, -1.0, -1.0]), np.array([-1.0, 1.5, 1.0, 1.0])
    P = rng.uniform(lo, hi, size=(2_000_000, 4))

    def prec(p, q):
        dt = q[..., 0] - p[..., 0]
        return (dt > 0) & (dt**2 >= np.sum((q[..., 1:] - p[..., 1:]) ** 2, axis=-1))

    vals = (prec(y, P) & prec(P, x)) * a(P[:, 0]) ** 4 * np.prod(hi - lo)
    v = de_sitter_interval_volume(y, x, H, 4)[0]
    assert abs(vals.mean() - v) < K_SIGMA * vals.std() / np.sqrt(len(vals))


@pytest.mark.parametrize("dim", [2, 4])
def test_exact_expectation_reduces_to_flat_for_small_region(dim):
    """Región pequeña frente a 1/H (H tau = 0.02): la integral de de Sitter
    debe diferir de la plana (misma rho V) en aproximadamente -R/2. Es una
    comprobación de coherencia entre las dos implementaciones; la precisión
    de la aproximación a -R/2 se testea aparte."""
    h_tau = 0.02
    vol = float(DeSitterIntervalVolume(H, dim, h_tau * 1.01).from_h_tau(h_tau))
    rho = 50.0 / vol
    ds = expected_bd_on_constant_de_sitter(dim, H, rho, h_tau)
    flat = expected_bd_on_constant_at_top(dim, rho, vol)
    r_half = dim * (dim - 1) * H**2 / 2
    # la diferencia es la curvatura: -R/2 con precisión O(1) a esta densidad
    assert -1.5 * r_half < ds - flat < -0.5 * r_half


@pytest.mark.parametrize("dim, final_tol", [(2, 0.015), (4, 0.04)])
def test_exact_expectation_converges_to_minus_half_ricci(dim, final_tol):
    r_half = dim * (dim - 1) * H**2 / 2
    h_tau = 1.5
    distances = []
    for l_h in (0.4, 0.2, 0.1, 0.05):
        rho = l_h ** (-dim)
        e1 = expected_bd_on_constant_de_sitter(dim, H, rho, h_tau, 1.0, resolution=1)
        e2 = expected_bd_on_constant_de_sitter(dim, H, rho, h_tau, 1.0, resolution=2)
        assert abs(e1 - e2) < 1e-3 * r_half, "cuadratura no convergida"
        distances.append(abs(e2 + r_half))
    assert all(np.diff(distances) < 0), distances
    assert distances[-1] < final_tol * r_half, distances


@pytest.mark.parametrize(
    "dim, n, eps, eta_bot, eta_top, m",
    [(2, 2000, 0.05, -4.0, -0.3, 40), (4, 2500, 0.05, -3.0, -0.3, 40)],
)
def test_sprinkled_de_sitter_matches_exact_expectation(dim, n, eps, eta_bot, eta_top, m):
    a = de_sitter_scale_factor(H)
    region = ConformalDiamond(dim, (eta_top - eta_bot) / 2, (eta_top + eta_bot) / 2, a)
    rho = n / region.volume
    p = np.zeros(dim)
    p[0] = eta_bot
    h_tau_top = np.log(eta_bot / eta_top)
    vol = DeSitterIntervalVolume(H, dim, h_tau_top * 1.001)

    taus = np.linspace(0.05, h_tau_top, 12)
    exact_table = np.array([expected_bd_on_constant_de_sitter(dim, H, rho, t, eps) for t in taus])

    rng = np.random.default_rng(70 + dim)
    sim, pred = [], []
    for _ in range(m):
        pts = sprinkle(region, rho, rng)
        B = bd_operator(minkowski_causal_matrix(pts), np.ones(len(pts)), dim, rho, eps)
        h_tau = np.arccosh(np.maximum(de_sitter_invariant(np.broadcast_to(p, pts.shape), pts), 1.0))
        # sólo puntos cuyo pasado sprinkleado es grande frente a l_eps
        sel = eps * rho * vol.from_h_tau(h_tau) >= 30
        sim.append(B[sel].mean())
        pred.append(np.interp(h_tau[sel], taus, exact_table).mean())
    sim = np.array(sim)
    sem = sim.std(ddof=1) / np.sqrt(m)
    target = np.mean(pred)
    assert abs(sim.mean() - target) < K_SIGMA * sem, f"sim={sim.mean():.4f} ± {sem:.4f}, exacto={target:.4f}"
