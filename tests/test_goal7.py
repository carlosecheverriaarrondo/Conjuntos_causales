"""Goal 7: computations requested by the third external review (``scripts/goal7_referee3.py``).

Statistical tolerance: 4σ combined, the project standard. Criteria R6-R8 were
fixed before computing (script docstring). The tests independently recompute
(np.polyfit, np.corrcoef) what the JSON files claim. The 3 % tolerance on the
logarithm coefficient of the correction (deterministic, no statistical error;
the remainder is of higher order) was fixed after seeing the 1.9 % of the
worst case: it is a regression check, not a criterion.
Skipped if results or data are missing.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
K = 4.0


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slope(x, y, se):
    coef, cov = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1, w=1 / np.asarray(se), cov="unscaled")
    return coef[0], np.sqrt(cov[0, 0])


@pytest.mark.parametrize("block", [64, 200, 2048])
def test_histogram_tri_matches_full_product(block):
    from causets import sprinkle
    from causets.causal import minkowski_causal_matrix
    from causets.geometry import MinkowskiDiamondGeometry

    g7, g5 = _load("goal7_referee3"), _load("goal5_smeared4d")
    g = MinkowskiDiamondGeometry(4)
    x = sprinkle(g.region, 800 / g.volume, np.random.default_rng(7))
    x = x[np.argsort(x[:, 0])]
    C = minkowski_causal_matrix(x)
    assert not np.tril(C).any()  # sorted by t: strictly upper triangular
    np.testing.assert_array_equal(g7.histogram_tri(C, block=block), g5.histogram(C))


@pytest.mark.skipif(not (FIG / "goal7_extras.json").exists(), reason="goal7_extras.json missing")
def test_extras_beta_correction_r2():
    e = json.loads((FIG / "goal7_extras.json").read_text())
    for g in ("mink4", "flrw4"):
        S = np.array(json.loads((FIG / f"goal4b_sim_{g}.json").read_text())["S"])
        r, v = np.corrcoef(S.T), S.var(axis=0, ddof=1)
        b = [2 * np.log(r[i, j]) / np.log(v[i] / v[j]) for i in range(4) for j in range(i + 1, 4)]
        np.testing.assert_allclose(e["beta"][g]["pairs"], b, rtol=1e-10)
        x = e["beta"][g]
        assert min(b) - 1e-12 <= x["gls"] <= max(b) + 1e-12 and 0 < x["gls_se"] < min(x["se"])
    for k, c in e["correction"].items():
        assert abs(c["rel"]) < 0.03, (k, c)
    d2 = json.loads((FIG / "goal3d_results.json").read_text())
    assert e["r2_2d"]["1.0"]["r2"] == pytest.approx(d2["1.0"]["no_J1"]["b"] / 24.0, rel=1e-10)


@pytest.mark.skipif(not (FIG / "goal7_results.json").exists(), reason="goal7_results.json missing")
def test_R6_R7_slopes_recomputed():
    r = json.loads((FIG / "goal7_results.json").read_text())
    R6 = r["R6"]
    pts = R6["points"]
    assert pts[-1]["rhoV"] == 25_600 and R6["M"] == 500
    b, b_se = _slope(np.log([p["rhoV"] for p in pts]), [p["r"] for p in pts], [p["se"] for p in pts])
    assert b == pytest.approx(R6["slope"]["slope"], rel=1e-8) and b_se == pytest.approx(R6["slope"]["slope_se"], rel=1e-8)
    z = (R6["r_new"] - 0.324) / np.hypot(R6["r_new_se"], 0.012)
    assert R6["passed"] == bool(abs(z) < K and abs(b / b_se) < K)
    for eps in ("0.5", "0.2", "0.1"):
        v = r["R7"][eps]
        pts = v["points"]
        b, b_se = _slope(np.log([p["rhoV"] for p in pts]), [p["r"] for p in pts], [p["se"] for p in pts])
        assert b == pytest.approx(v["slope"]["slope"], rel=1e-8)
        assert v["passed"] == bool(abs(v["z_new"]) < K and abs(b / b_se) < K)
        # the bootstrap error of the new point confirms the formula error (not a criterion: only |ratio - 1| < 25 %)
        assert abs(v["r_new_se_boot"] / v["r_new_se"] - 1) < 0.25


REFINE = sorted(p for p in FIG.glob("goal7_refine_*.json") if p.stem != "goal7_refine_parts")


@pytest.mark.skipif(not REFINE, reason="goal7_refine_*.json missing")
def test_R8_refined_fits_consistent():
    for p in REFINE:
        d = json.loads(p.read_text())
        s, rows = d["summary"], d["rows"]
        assert len(rows) == s["n_refined"] == 50
        chi = np.array([x["chi2"] for x in rows])
        grid = np.array([x["chi2_grid"] for x in rows])
        assert np.all(chi <= grid + 1e-9)
        assert s["dchi2_min_refined"] == pytest.approx(chi.min() - s["ref"], rel=1e-12)
        assert s["n_good"]["14.7"] == int(np.sum(chi - s["ref"] < 14.7))
        assert s["safe"], p.name  # the 50th seed minus the largest improvement stays above the threshold


@pytest.mark.skipif(not (REFINE and (ROOT / "data" / "desi_dr2").exists()), reason="data missing")
def test_R8_refined_best_seed_reproduced():
    g7 = _load("goal7_referee3")
    d = json.loads(REFINE[0].read_text())
    run = d["summary"]["run"]
    _, _, ev, _, g4c = g7._run_setup(run)
    b = min(d["rows"], key=lambda x: x["chi2"])
    nz = np.random.default_rng(g4c.SEED0 + b["seed"]).standard_normal(g4c.N_STEPS)[:, None]
    if b["chi2"] < b["chi2_grid"]:
        assert float(ev(nz, np.array([b["omega_m"]]), np.array([b["amp"]]))[0]) == pytest.approx(b["chi2"], abs=1e-6)


PARTS = FIG / "goal7_refine_parts.json"


@pytest.mark.skipif(not (PARTS.exists() and REFINE), reason="goal7_refine_parts.json missing")
def test_R8_parts_add_up_to_refined_minimum():
    # table 2 of the paper: the Δχ² of the best refitted seed, split by probe, adds up to the refitted minimum
    parts = json.loads(PARTS.read_text())
    assert set(parts) == {p.stem.removeprefix("goal7_refine_") for p in REFINE}
    for p in REFINE:
        s = json.loads(p.read_text())["summary"]
        v = parts[s["run"]]
        assert sum(v["parts"].values()) == pytest.approx(s["dchi2_min_refined"], abs=1e-6)
        assert v["dchi2"] == pytest.approx(s["dchi2_min_refined"], abs=1e-9)
