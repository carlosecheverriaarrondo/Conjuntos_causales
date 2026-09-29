"""Tests del Goal 2: intervalos, operador y acción de Benincasa-Dowker.

Tolerancias:
- Comparaciones simulación vs. valor esperado exacto: |media - esperado| < 4 SEM
  (probabilidad de fallo espurio ~6e-5 por comparación; semillas fijas).
- Identidades algebraicas y límites analíticos: tolerancias numéricas
  indicadas en cada test.
"""

import numpy as np
import pytest

from causets import (
    CausalDiamond,
    minkowski_causal_matrix,
    sample_uniform,
    sprinkle,
)
from causets.action import (
    bd_action,
    bd_on_constant_from_counts,
    bd_operator,
    coefficients,
    discreteness_scale_sq,
    smearing_function,
)
from causets.causal import minkowski_future_counts
from causets.continuum import (
    expected_abundance,
    expected_bd_action_diamond,
    expected_bd_bracket_at_top,
    expected_bd_on_constant_at_top,
    past_weight,
    poisson_pmf,
)
from causets.intervals import interval_abundances, interval_cardinalities

K_SIGMA = 4.0


def _within(samples, expected, k=K_SIGMA):
    samples = np.asarray(samples, dtype=float)
    mean = samples.mean()
    sem = samples.std(ddof=1) / np.sqrt(len(samples))
    return abs(mean - expected) < k * sem, f"media={mean:.5g} ± {sem:.3g}, esperado={expected:.5g}"


# ---------------------------------------------------------------- intervalos


def test_interval_cardinalities_brute_force():
    pts = sample_uniform(CausalDiamond(3), 120, np.random.default_rng(20))
    C = minkowski_causal_matrix(pts)
    n = interval_cardinalities(C)
    for i, j in zip(*np.nonzero(C)):
        assert n[i, j] == np.count_nonzero(C[i, :] & C[:, j])


def test_interval_abundances_chain():
    """Cadena de 5 elementos: N_k = 5 - 1 - k pares con k elementos entre ellos."""
    chain = np.triu(np.ones((5, 5), dtype=bool), k=1)
    np.testing.assert_array_equal(interval_abundances(chain), [4, 3, 2, 1])


def test_future_counts_matches_matrix():
    pts = sample_uniform(CausalDiamond(4), 900, np.random.default_rng(21))
    np.testing.assert_array_equal(minkowski_future_counts(pts), minkowski_causal_matrix(pts).sum(axis=1))


# ------------------------------------------------------ coeficientes y acción


@pytest.mark.parametrize("dim", [2, 4])
def test_smearing_reduces_to_local_layers(dim):
    layers = coefficients(dim).layers
    f = smearing_function(np.arange(8), dim, 1.0)
    np.testing.assert_allclose(f, list(layers) + [0.0] * (8 - len(layers)), atol=1e-15)


def test_smearing_2d_matches_sorkin_closed_form():
    """f(n, eps) = (1-eps)^n [1 - 2 eps n/(1-eps) + eps^2 n(n-1) / (2 (1-eps)^2)]
    (Sorkin 2007, gr-qc/0703099)."""
    eps = 0.13
    n = np.arange(40)
    closed = (1 - eps) ** n * (1 - 2 * eps * n / (1 - eps) + eps**2 * n * (n - 1) / (2 * (1 - eps) ** 2))
    np.testing.assert_allclose(smearing_function(n, 2, eps), closed, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("dim", [2, 4])
def test_local_action_explicit_formula(dim):
    pts = sample_uniform(CausalDiamond(dim), 400, np.random.default_rng(22))
    C = minkowski_causal_matrix(pts)
    N = interval_abundances(C, 3)
    n = len(pts)
    if dim == 2:
        explicit = n - 2 * N[0] + 4 * N[1] - 2 * N[2]
    else:
        explicit = 4 / np.sqrt(6) * (n - N[0] + 9 * N[1] - 16 * N[2] + 8 * N[3])
    assert bd_action(C, dim) == pytest.approx(explicit, rel=1e-12)


@pytest.mark.parametrize("dim, eps", [(2, 1.0), (2, 0.2), (4, 1.0), (4, 0.2)])
def test_action_is_sum_of_operator(dim, eps):
    """S_eps = -s_d * sum_x l^2 (B_eps 1)(x)."""
    rho = 50.0
    pts = sprinkle(CausalDiamond(dim), rho, np.random.default_rng(23))
    C = minkowski_causal_matrix(pts)
    B = bd_operator(C, np.ones(len(pts)), dim, rho, eps)
    l2 = discreteness_scale_sq(dim, rho)
    expected = -coefficients(dim).action_scale * l2 * B.sum()
    assert bd_action(C, dim, eps) == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------- resultados analíticos exactos


@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("lam", [3.0, 40.0, 150.0])
def test_past_weight_of_one_is_volume(dim, lam):
    assert past_weight(lambda z: np.ones_like(z), dim, lam) == pytest.approx(lam, rel=1e-9)


@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("eps", [1.0, 0.1])
def test_expected_operator_vanishes_in_flat_limit(dim, eps):
    """E[l^2 B 1] -> 0 (R = 0) cuando eps rho V -> infinito: los términos en
    ln(Lambda) y sqrt(Lambda) se cancelan por la elección de los C_k."""
    for lam_eps in (100.0, 1e3, 1e5):
        assert abs(expected_bd_bracket_at_top(dim, lam_eps / eps, eps)) < 1e-6


def test_expected_abundances_2d_independent_derivation():
    """En 2D (coordenadas de cono de luz) E[N_k] = L^2 int int (1-s)(1-w) p_k(L s w)."""
    from scipy.integrate import dblquad

    lam = 30.0
    for k in range(4):
        alt = dblquad(lambda w, s: lam**2 * (1 - s) * (1 - w) * poisson_pmf(k, lam * s * w), 0, 1, 0, 1)[0]
        assert expected_abundance(2, lam, k) == pytest.approx(alt, rel=1e-8)


def test_expected_2d_action_of_diamond_tends_to_one():
    """Benincasa, Dowker & Schmitzer (2011): <S> -> V - E + F del teselado nulo;
    un diamante tiene 4 vértices, 4 aristas y 1 cara: <S> -> 1."""
    for lam in (100.0, 1000.0, 5000.0):
        assert expected_bd_action_diamond(2, lam) == pytest.approx(1.0, abs=1e-6)
    assert expected_bd_action_diamond(2, 1000.0, eps=0.2) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------- simulación vs. valores esperados


@pytest.mark.parametrize("dim", [2, 4])
def test_sprinkled_abundances_and_action_match_expectation(dim):
    lam, m = 300.0, 300
    region = CausalDiamond(dim)
    rho = lam / region.volume
    rng = np.random.default_rng(30 + dim)
    abund, actions = [], []
    for _ in range(m):
        C = minkowski_causal_matrix(sprinkle(region, rho, rng))
        abund.append(interval_abundances(C, 3))
        actions.append(bd_action(C, dim))
    abund = np.array(abund)
    for k in range(4):
        ok, msg = _within(abund[:, k], expected_abundance(dim, lam, k))
        assert ok, f"N_{k}: {msg}"
    ok, msg = _within(actions, expected_bd_action_diamond(dim, lam))
    assert ok, f"S: {msg}"


def test_sprinkled_2d_action_mean_is_one():
    """Media de la acción 2D de un diamante con rho V = 1000, frente a 1."""
    region = CausalDiamond(2)
    rho = 1000.0 / region.volume
    rng = np.random.default_rng(40)
    actions = [bd_action(minkowski_causal_matrix(sprinkle(region, rho, rng)), 2) for _ in range(150)]
    ok, msg = _within(actions, 1.0)
    assert ok, msg


@pytest.mark.parametrize("dim, lam, eps", [(2, 300.0, 1.0), (2, 1000.0, 0.1), (4, 300.0, 1.0), (4, 1000.0, 0.1)])
def test_operator_at_top_vertex_matches_expectation(dim, lam, eps):
    """(B_eps 1) en el vértice superior de un diamante plano (pasado completo
    sprinkleado): la media simulada coincide con la exacta, que es ~0 (R = 0)."""
    region = CausalDiamond(dim)
    rho = lam / region.volume
    rng = np.random.default_rng(50 + dim)
    values = [
        bd_on_constant_from_counts(minkowski_future_counts(sprinkle(region, rho, rng)), dim, rho, eps)
        for _ in range(300)
    ]
    exact = expected_bd_on_constant_at_top(dim, rho, region.volume, eps)
    lsq = discreteness_scale_sq(dim, rho, eps)
    assert abs(exact * lsq) < 1e-4
    ok, msg = _within(values, exact)
    assert ok, msg
