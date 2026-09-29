"""Tests de la región (diamante causal) y del sprinkling de Poisson.

Tolerancia estadística: las comprobaciones Monte Carlo exigen que la
desviación respecto al valor esperado sea < 4 errores estándar (probabilidad de
fallo espurio ~6e-5 si el código es correcto). Las semillas son fijas, así que
los tests son deterministas.
"""

import numpy as np
import pytest

from causets import CausalDiamond, sample_uniform, sprinkle

K_SIGMA = 4.0


@pytest.mark.parametrize(
    "dim, h, expected",
    [
        (2, 1.0, 2.0),  # cuadrado girado de diagonal 2h: 2h^2
        (2, 0.5, 0.5),
        (3, 1.0, 2 * np.pi / 3),  # dos conos de radio y altura h
        (4, 1.0, 2 * np.pi / 3),  # 2 * (h/4) * (4/3 pi h^3)
        (4, 2.0, 2 * np.pi / 3 * 16),
    ],
)
def test_diamond_volume_closed_form(dim, h, expected):
    assert CausalDiamond(dim, h).volume == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_diamond_volume_monte_carlo(dim):
    """La fracción de la caja envolvente dentro del diamante reproduce V."""
    region = CausalDiamond(dim)
    rng = np.random.default_rng(1)
    lo, hi = region.bounding_box()
    m = 400_000
    p = region.volume / np.prod(hi - lo)
    frac = region.contains(rng.uniform(lo, hi, size=(m, dim))).mean()
    assert abs(frac - p) < K_SIGMA * np.sqrt(p * (1 - p) / m)


@pytest.mark.parametrize("dim", [2, 4])
def test_sample_uniform_basic_properties(dim):
    region = CausalDiamond(dim)
    pts = sample_uniform(region, 3000, np.random.default_rng(2))
    assert pts.shape == (3000, dim)
    assert region.contains(pts).all()
    assert np.all(np.diff(pts[:, 0]) >= 0), "los puntos deben ir ordenados por t"


@pytest.mark.parametrize("dim", [2, 4])
def test_sample_uniform_is_uniform(dim):
    """Momentos de la distribución uniforme en el diamante.

    Por simetría E[t] = E[x_k] = 0. Además, P(|t| < h/2) = 1 - (1/2)^d,
    porque la región {|t| >= h/2} está formada por dos conos semejantes a los
    del diamante con razón 1/2 (volumen escalado por (1/2)^d).
    """
    region = CausalDiamond(dim)
    n = 20_000
    pts = sample_uniform(region, n, np.random.default_rng(3))

    for k in range(dim):
        col = pts[:, k]
        assert abs(col.mean()) < K_SIGMA * col.std() / np.sqrt(n)

    p = 1 - 0.5**dim
    frac = np.mean(np.abs(pts[:, 0]) < 0.5)
    assert abs(frac - p) < K_SIGMA * np.sqrt(p * (1 - p) / n)


def test_sprinkle_poisson_statistics():
    """N ~ Poisson(rho V): media y varianza iguales a rho V."""
    region = CausalDiamond(4)
    rho = 200.0
    lam = rho * region.volume
    rng = np.random.default_rng(4)
    counts = np.array([len(sprinkle(region, rho, rng)) for _ in range(400)])
    m = len(counts)
    assert abs(counts.mean() - lam) < K_SIGMA * np.sqrt(lam / m)
    # Var de la varianza muestral de una Poisson ~ (lam + 2 lam^2) / m
    assert abs(counts.var(ddof=1) - lam) < K_SIGMA * np.sqrt((lam + 2 * lam**2) / m)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        CausalDiamond(1)
    with pytest.raises(ValueError):
        CausalDiamond(2, -1.0)
    with pytest.raises(ValueError):
        sprinkle(CausalDiamond(2), 0.0, np.random.default_rng())
