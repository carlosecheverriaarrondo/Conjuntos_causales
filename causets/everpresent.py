"""Compatibility of the BD action fluctuations with the observed Λ
(Goal 4, reformulated; see README "Goal 4a").

CENTRAL ASSUMPTION (of this project, not of the Everpresent Λ model)
--------------------------------------------------------------------
The fluctuation of the BD action in the causal past of the observer is
interpreted as a fluctuation of the cosmological term. In the continuum the
BD action reproduces the Einstein-Hilbert one: with our normalisation
(``action.py``), <S> -> (1/(2 l²)) ∫ R dV, and S_EH/hbar = (1/(16π l_p²)) ∫ (R - 2Λ) dV,
so that S_EH/hbar = kappa S with kappa = l² / (8π l_p²). The cosmological
term is Λ V / (8π l_p²) (Das, Nasiri & Yazdi, JCAP 10 (2023) 047,
arXiv:2304.03819, eq. 3.12, with their sign convention), so that
    delta Λ · l² = Delta S / (rho V),        rho = l^-4,
independent of l_p. With the Poisson action of Model 1 (S_Λ ~ N(0, √N/2)
in units of kappa, eqs. 3.11-3.13 of DNY) one recovers their
delta Λ = 8π alpha / √V, alpha = (l_p / l)² / 2 (eq. 3.9): see
``model1_delta_lambda``.

AMPLITUDE OF THE BD ACTION (items 1c-1d)
----------------------------------------
Asymptotically (Delta S)² = (1 + r) J¹ with
    J¹ ≈ (rho V)^1.5 · (G / V^1.5) / (8 √c_4) · sum_n f(n)² Γ(n+½)/n!,
f(n) the weight of a pair in the action (``fluctuations.pair_weight``), G the
boost range of the region (``scripts/cone4d_scaling.g_flrw``) and r the
measured fraction of correlations (0.32-0.35 with the local action). Thus
    delta Λ · l² = A (rho V)^(-1/4),   A² = (1 + r) (G/V^1.5)/(8 √c_4) · Σ f² Γ(n+½)/n!.

SMEARED ACTION
--------------
With eps = (l / l_k)^4 (l_k: non-locality scale), f_eps(n) ≈
-(4/√6) eps^(3/2) g(eps n), g(x) = sum_k C_k x^k e^-x / k!, and
    sum_n f_eps² Γ(n+½)/n! -> (8/3) eps^(5/2) ∫_0^∞ g(x)² x^(-1/2) dx   (eps -> 0),
that is, A ∝ eps^(5/4) = (l / l_k)^5 (``same_pair_sum`` and ``smeared_asymptotic_constant``).

Constants: l_p = 1.616255e-35 m, t_p = 5.391247e-44 s (CODATA 2018);
1 Mpc = 3.0856776e22 m.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

L_PLANCK_M = 1.616255e-35
T_PLANCK_S = 5.391247e-44
MPC_M = 3.0856776e22
C4 = np.pi / 24  # volume of a 4D interval of unit proper time


def hubble_planck(h0_km_s_mpc: float) -> float:
    """H0 in Planck units (1 / t_p)."""
    return h0_km_s_mpc * 1e3 / MPC_M * T_PLANCK_S


def _E(a, om, orad, ol):
    return np.sqrt(om / a**3 + orad / a**4 + ol + (1 - om - orad - ol) / a**2)


def conformal_time(a, om, orad, ol) -> float:
    """eta(a) H0 = ∫_0^a da' / (a'² E(a'))  (c = 1, units of 1/H0)."""
    val, _ = quad(lambda x: 1.0 / (x * x * _E(x, om, orad, ol)), 0.0, a, epsabs=0, epsrel=1e-11, limit=400)
    return val


def past_cone_volume(a_obs: float, om: float, orad: float, ol: float) -> float:
    """V H0⁴ of the causal past (down to the big bang) of an observer at a_obs,
    flat FLRW:  V = (4π/3) ∫_0^{eta_obs} a(eta)^4 (eta_obs - eta)^3 d eta
               = (4π/3) ∫_0^{a_obs} a² (eta_obs - eta(a))³ / E(a) da."""
    e_obs = conformal_time(a_obs, om, orad, ol)

    def f(a):
        return a * a * (e_obs - conformal_time(a, om, orad, ol)) ** 3 / _E(a, om, orad, ol)

    edges = np.concatenate([[0.0], np.geomspace(1e-8 * a_obs, a_obs, 40)])
    return 4 * np.pi / 3 * sum(quad(f, lo, hi, epsabs=0, epsrel=1e-9, limit=200)[0] for lo, hi in zip(edges[:-1], edges[1:]))


def eds_past_cone_volume() -> float:
    """V H0⁴ in Einstein-de Sitter (a ∝ eta², H0 eta0 = 2): (4π/3) 16 B(9, 4)."""
    from scipy.special import beta

    return 4 * np.pi / 3 * 16 * beta(9, 4)


def same_pair_sum(eps: float = 1.0, n_max: int | None = None) -> float:
    """sum_n f_eps(n)² Γ(n+½)/n! for the 4D BD action (local with eps = 1).
    For eps < 1e-5 the direct sum is not viable (n ~ 60/eps terms) and the
    asymptote ``smeared_asymptotic_constant() · eps^(5/2)`` is used (relative
    error 4e-4 already at eps = 1e-4, decreasing with eps)."""
    from .fluctuations import pair_weight

    if eps < 1e-5:
        return smeared_asymptotic_constant() * eps**2.5
    if n_max is None:
        n_max = int(60 / eps) + 50
    n = np.arange(n_max)
    w = np.exp(gammaln(n + 0.5) - gammaln(n + 1))
    return float(np.sum(pair_weight(4, n, eps) ** 2 * w))


def cross_sum(eps: float = 1.0, n_max: int | None = None) -> float:
    """sum_n f_eps(n) Γ(n+½)/n!: coefficient of the cross term with N at order
    (rho V)^1.5 (it must vanish for (Delta S)² not to have a larger term)."""
    from .fluctuations import pair_weight

    if n_max is None:
        n_max = int(60 / eps) + 50
    n = np.arange(n_max)
    w = np.exp(gammaln(n + 0.5) - gammaln(n + 1))
    return float(np.sum(pair_weight(4, n, eps) * w))


def smeared_asymptotic_constant() -> float:
    """(8/3) ∫_0^∞ g(x)² x^(-1/2) dx with g(x) = sum_k C_k x^k e^-x/k!,
    C = (1, -9, 16, -8): sum_n f_eps² Γ(n+½)/n! ≈ const · eps^(5/2) (eps -> 0)."""
    from math import factorial

    from .action import coefficients

    C = coefficients(4).layers

    def g(x):
        return sum(c * x**k / factorial(k) for k, c in enumerate(C)) * np.exp(-x)

    # x = u²: ∫ g(x)² x^(-1/2) dx = 2 ∫ g(u²)² du
    val, _ = quad(lambda u: 2 * g(u * u) ** 2, 0, np.inf, epsabs=0, epsrel=1e-12, limit=400)
    return 8 / 3 * val


def amplitude(g_over_v15: float, eps: float = 1.0, r: float = 0.33) -> float:
    """A with delta Λ l² = A (rho V)^(-1/4)."""
    return float(np.sqrt((1 + r) * g_over_v15 / (8 * np.sqrt(C4)) * same_pair_sum(eps)))


def delta_lambda_planck(A: float, v_planck: float, l_over_lp: float = 1.0) -> float:
    """delta Λ l_p² for the BD action: A (rho V)^(-1/4) (l_p / l)², rho V = V / l^4."""
    rho_v = v_planck / l_over_lp**4
    return A * rho_v**-0.25 / l_over_lp**2


def model1_delta_lambda(v_planck: float, alpha: float) -> float:
    """delta Λ l_p² of Everpresent Λ, Model 1 (DNY eq. 3.13): 8π alpha / √V."""
    return 8 * np.pi * alpha / np.sqrt(v_planck)


def model1_from_action_mapping(v_planck: float, alpha: float) -> float:
    """The same, but with the conversion delta Λ l² = Delta S / (rho V) applied
    to the Poisson action of Model 1: Delta S_EH = √N / 2 (DNY 3.11-3.13), and
    Delta S = Delta S_EH / kappa with kappa = l²/(8π l_p²) = 1/(16π alpha)."""
    l_over_lp = np.sqrt(1 / (2 * alpha))
    n = v_planck / l_over_lp**4
    dS = np.sqrt(n) / 2 * 16 * np.pi * alpha
    return dS / n / l_over_lp**2
