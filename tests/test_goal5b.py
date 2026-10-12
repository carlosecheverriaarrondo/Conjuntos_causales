"""Goal 5b: estadísticos de la descomposición de MYZ (``scripts/goal5b_decomposition.py``)
frente a un recuento explícito de pares de pares. Exactos (sin tolerancia estadística)."""

import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from causets import sprinkle
from causets.causal import minkowski_causal_matrix
from causets.fluctuations import pair_weight
from causets.geometry import MinkowskiDiamondGeometry
from causets.intervals import interval_cardinalities

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("goal5b", ROOT / "scripts" / "goal5b_decomposition.py")
g5b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g5b)


def test_statistics_match_explicit_pair_of_pairs_count():
    g = MinkowskiDiamondGeometry(4)
    C = minkowski_causal_matrix(sprinkle(g.region, 40 / g.volume, np.random.default_rng(3)))
    eps = 0.3
    n = interval_cardinalities(C)
    pairs = [(y, x, pair_weight(4, n[y, x], eps)) for y, x in zip(*np.nonzero(C))]
    t = dict(J1=0.0, J2=0.0, J3=0.0, J4=0.0, J6=0.0)
    for (y1, x1, f1), (y2, x2, f2) in product(pairs, pairs):
        if (y1, x1) == (y2, x2):
            t["J1"] += f1 * f2
        elif y1 == y2:
            t["J2"] += f1 * f2
        elif x1 == x2:
            t["J3"] += f1 * f2
        elif x1 == y2 or x2 == y1:
            t["J4"] += f1 * f2
        else:
            t["J6"] += f1 * f2
    s = g5b.pair_statistics(C, eps)
    assert s["F"] == pytest.approx(sum(p[2] for p in pairs), rel=1e-10)
    assert s["S2"] == pytest.approx(t["J1"], rel=1e-10)
    assert s["R2"] - s["S2"] == pytest.approx(t["J2"], rel=1e-10)
    assert s["C2"] - s["S2"] == pytest.approx(t["J3"], rel=1e-10)
    assert 2 * s["CR"] == pytest.approx(t["J4"], rel=1e-10)
    total = s["F"] ** 2 - (s["R2"] + s["C2"] - s["S2"] + 2 * s["CR"])
    assert total == pytest.approx(t["J6"], rel=1e-9)
