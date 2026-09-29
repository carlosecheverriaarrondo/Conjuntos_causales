"""Tests de la relación causal y de la validez del orden parcial."""

import numpy as np
import pytest

from causets import (
    CausalDiamond,
    is_partial_order,
    minkowski_causal_matrix,
    partial_order_violations,
    sample_uniform,
)


def test_small_hand_examples_2d():
    pts = np.array(
        [
            [0.0, 0.0],  # a
            [1.0, 0.5],  # b: temporal respecto a a
            [1.0, 2.0],  # c: espacial respecto a a y b
            [3.0, 1.0],  # d: futuro de a, b y c
        ]
    )
    C = minkowski_causal_matrix(pts)
    expected = np.array(
        [
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(C, expected)


def test_null_separation_is_related():
    C = minkowski_causal_matrix(np.array([[0.0, 0.0, 0.0], [1.0, 0.6, 0.8]]))
    assert C[0, 1] and not C[1, 0]


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_matches_pointwise_definition(dim):
    """La versión vectorizada por bloques coincide con la definición par a par."""
    pts = sample_uniform(CausalDiamond(dim), 150, np.random.default_rng(10))
    C = minkowski_causal_matrix(pts)
    for i in range(len(pts)):
        for j in range(len(pts)):
            dt = pts[j, 0] - pts[i, 0]
            dx2 = np.sum((pts[j, 1:] - pts[i, 1:]) ** 2)
            assert C[i, j] == (dt > 0 and dt * dt >= dx2)


def test_chunking_consistent(monkeypatch):
    """El resultado no depende del tamaño de bloque."""
    import causets.causal as causal

    pts = sample_uniform(CausalDiamond(4), 700, np.random.default_rng(11))
    C_ref = minkowski_causal_matrix(pts)
    monkeypatch.setattr(causal, "_CHUNK", 97)
    np.testing.assert_array_equal(minkowski_causal_matrix(pts), C_ref)


@pytest.mark.parametrize("dim", [2, 4])
def test_lorentz_invariance(dim):
    """Un boost en la dirección x_1 no cambia la relación causal."""
    pts = sample_uniform(CausalDiamond(dim), 500, np.random.default_rng(12))
    v = 0.8
    g = 1 / np.sqrt(1 - v**2)
    boosted = pts.copy()
    boosted[:, 0] = g * (pts[:, 0] - v * pts[:, 1])
    boosted[:, 1] = g * (pts[:, 1] - v * pts[:, 0])
    np.testing.assert_array_equal(
        minkowski_causal_matrix(pts), minkowski_causal_matrix(boosted)
    )


@pytest.mark.parametrize("dim, n", [(2, 1500), (4, 1500)])
def test_sprinkled_causet_is_partial_order(dim, n):
    pts = sample_uniform(CausalDiamond(dim), n, np.random.default_rng(13))
    C = minkowski_causal_matrix(pts)
    assert partial_order_violations(C) == {
        "reflexive": 0,
        "antisymmetric": 0,
        "transitive": 0,
    }
    # Con los puntos ordenados por t, C es estrictamente triangular superior.
    assert not np.tril(C).any()


def test_violations_are_detected():
    """El verificador no es trivial: detecta cada tipo de violación."""
    refl = np.eye(3, dtype=bool)
    assert partial_order_violations(refl)["reflexive"] == 3

    anti = np.array([[0, 1], [1, 0]], dtype=bool)
    assert partial_order_violations(anti)["antisymmetric"] == 1

    # 0 ≺ 1 ≺ 2 pero falta 0 ≺ 2
    intrans = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    assert partial_order_violations(intrans)["transitive"] == 1
    assert not is_partial_order(intrans)

    chain = np.triu(np.ones((5, 5), dtype=bool), k=1)
    assert is_partial_order(chain)
