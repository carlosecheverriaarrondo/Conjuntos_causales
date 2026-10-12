"""Goal 4e: variantes de la dinámica de Λ (``scripts/goal4e_robustness.py``).

Tolerancias fijadas antes de ejecutar estos tests. Necesita los datos
(``scripts/get_pantheon.sh``) y los resultados del Goal 4d; si faltan, se salta.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("goal4e", ROOT / "scripts" / "goal4e_robustness.py")
g4e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g4e)

SEEDS_4D = ROOT / "figures" / "goal4d_seeds_bd.json"
HAVE = SEEDS_4D.exists() and (ROOT / "data" / "desi_dr2").exists() and (ROOT / "data" / "pantheonplus").exists()


def test_amplitude_grid_scaling():
    k = g4e.register(0.25, 0.675)
    np.testing.assert_array_equal(g4e.g4c.AMP_GRID[k], g4e.g4c.AMP_GRID["bd"])
    k = g4e.register(0.5, 0.5)
    np.testing.assert_allclose(g4e.g4c.AMP_GRID[k] / g4e.g4c.AMP_GRID["bd"], g4e.V_REF**0.25, rtol=1e-14)
    assert g4e.parse(g4e.key(0.375, 0.75)) == (0.375, 0.75)


def test_reference_volume_matches_planck_lcdm():
    sl = g4e.sl
    real = sl.evolve(sl.ProcessSpec(3 * (0.6727**2 - 0.14327 - sl.OMEGA_R), 0, 0), 1, None, a_start=1e-5,
                     n_steps=2000, s_fixed=1.0, omega_m=0.14327)
    assert abs(real.extra["V_today"][0] / g4e.V_REF - 1) < 1e-3


@pytest.mark.skipif(not HAVE, reason="faltan datos o resultados del Goal 4d")
def test_E0_reproduces_goal4d_per_seed():
    rows4d = json.loads(SEEDS_4D.read_text())["rows"][:4]
    v = g4e.Variant()
    rows = g4e.fit_seeds(v, g4e.register(0.25, 0.75 * 0.9), [r["seed"] for r in rows4d])
    for a, b in zip(rows, rows4d):
        assert a["seed"] == b["seed"]
        assert abs(a["chi2"] - b["chi2"]) < 1e-6
        assert a["amp"] == pytest.approx(b["amp"], rel=1e-12, abs=0)
