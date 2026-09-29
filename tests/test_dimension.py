"""Tests del estimador de Myrheim-Meyer: convergencia a d = 2 y d = 4.

Criterio de convergencia (tolerancia estadística explícita):
  para un conjunto de M sprinklings independientes de N puntos,
  (a) |<d_MM> - d| < 4 * SEM, con SEM = std(d_MM) / sqrt(M), y
  (b) |<d_MM> - d| < TOL_ABS[d] (cota absoluta, por si la SEM fuese grande).
Además, la dispersión debe decrecer con N como ~ N^(-1/2).
"""

import numpy as np
import pytest

from causets import (
    CausalDiamond,
    minkowski_causal_matrix,
    myrheim_meyer_dimension,
    myrheim_meyer_from_fraction,
    ordering_fraction,
    ordering_fraction_expected,
    sample_uniform,
)

K_SIGMA = 4.0
TOL_ABS = {2: 0.02, 4: 0.05}
N_SAMPLES = 20


def _ensemble(dim, n, m, seed):
    rng = np.random.default_rng(seed)
    region = CausalDiamond(dim)
    return np.array(
        [
            myrheim_meyer_dimension(minkowski_causal_matrix(sample_uniform(region, n, rng)))
            for _ in range(m)
        ]
    )


@pytest.mark.parametrize(
    "d, expected",
    [(1, 1.0), (2, 0.5), (3, 8 / 35), (4, 0.1)],
)
def test_ordering_fraction_closed_form(d, expected):
    assert ordering_fraction_expected(d) == pytest.approx(expected, rel=1e-12)


def test_ordering_fraction_monotonic():
    ds = np.linspace(1, 10, 200)
    fs = [ordering_fraction_expected(d) for d in ds]
    assert np.all(np.diff(fs) < 0)


@pytest.mark.parametrize("d", [1.5, 2.0, 2.7, 3.0, 4.0, 6.0])
def test_inversion_roundtrip(d):
    assert myrheim_meyer_from_fraction(ordering_fraction_expected(d)) == pytest.approx(
        d, abs=1e-9
    )


def test_degenerate_orders_rejected():
    n = 10
    chain = np.triu(np.ones((n, n), dtype=bool), k=1)
    antichain = np.zeros((n, n), dtype=bool)
    assert ordering_fraction(chain) == 1.0
    with pytest.raises(ValueError):
        myrheim_meyer_dimension(chain)
    with pytest.raises(ValueError):
        myrheim_meyer_dimension(antichain)


@pytest.mark.parametrize("dim, n", [(2, 1000), (2, 2000), (4, 1000), (4, 2000)])
def test_myrheim_meyer_converges_to_manifold_dimension(dim, n):
    ds = _ensemble(dim, n, N_SAMPLES, seed=100 + dim * 10_000 + n)
    mean = ds.mean()
    sem = ds.std(ddof=1) / np.sqrt(len(ds))
    err = abs(mean - dim)
    assert err < K_SIGMA * sem, f"d={dim}, N={n}: <d_MM>={mean:.4f} ± {sem:.4f}"
    assert err < TOL_ABS[dim], f"d={dim}, N={n}: |<d_MM>-d|={err:.4f} > {TOL_ABS[dim]}"


@pytest.mark.parametrize("dim", [2, 4])
def test_myrheim_meyer_spread_shrinks_with_n(dim):
    """std(d_MM) ~ N^(-1/2): al multiplicar N por 8, la std debe bajar
    claramente (esperado factor ~0.35; exigimos < 0.6)."""
    std_small = _ensemble(dim, 250, 40, seed=200 + dim).std(ddof=1)
    std_large = _ensemble(dim, 2000, 40, seed=300 + dim).std(ddof=1)
    assert std_large < 0.6 * std_small, (std_small, std_large)
