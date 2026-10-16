"""Goal 6: cálculos pedidos por la revisión externa (``scripts/goal6_referee.py``).

Tolerancia: 4σ combinada, la del proyecto. Los criterios R1-R5 se fijaron antes
de calcular (docstring del script); estos tests recalculan de forma
independiente (np.polyfit) lo que afirma ``figures/goal6_results.json`` y
comprueban el ajuste SN + BAO. Los valores de los logs se vieron antes de
escribir los tests (la tolerancia es la estándar, no ajustada a ellos).
Si faltan resultados o datos, se saltan.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = FIG / "goal6_results.json"
HAVE_RES = RES.exists()
HAVE_DATA = (ROOT / "data" / "desi_dr2").exists() and (ROOT / "data" / "pantheonplus").exists()
K = 4.0


def _load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("goal6", ROOT / "scripts" / "goal6_referee.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slope(x, y, se):
    """Pendiente y su error con np.polyfit (independiente de goal5_smeared4d._slope)."""
    coef, cov = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1, w=1 / np.asarray(se), cov="unscaled")
    return coef[0], np.sqrt(cov[0, 0])


@pytest.mark.skipif(not HAVE_RES, reason="falta figures/goal6_results.json")
def test_R1_R2_slopes_recomputed():
    r = json.loads(RES.read_text())
    R1 = r["R1"]
    pts = R1["points"]
    assert [p["rhoV"] for p in pts] == [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    b, b_se = _slope(np.log([p["rhoV"] for p in pts]), [p["r"] for p in pts], [p["se"] for p in pts])
    assert b == pytest.approx(R1["slope"]["slope"], rel=1e-8)
    assert b_se == pytest.approx(R1["slope"]["slope_se"], rel=1e-8)
    assert abs(b / b_se) < K
    z = (R1["r_12800"] - 0.317) / np.hypot(R1["r_12800_se"], 0.014)
    assert abs(z) < K and R1["M"] == 500
    assert R1["passed"]
    for eps in ("0.5", "0.2", "0.1"):
        v = r["R2"][eps]
        pts = v["points"]
        b, b_se = _slope(np.log([p["rhoV"] for p in pts]), [p["r"] for p in pts], [p["se"] for p in pts])
        assert b == pytest.approx(v["slope"]["slope"], rel=1e-8)
        assert abs(b / b_se) < K and v["passed"]


@pytest.mark.skipif(not HAVE_RES, reason="falta figures/goal6_results.json")
def test_R3_flrw_6400():
    sys.path.insert(0, str(ROOT / "scripts"))
    import flrw4d_scaling as f4

    r = json.loads(RES.read_text())["R3"]
    for alpha, ref in (("1.0", (0.348, 0.016)), ("2.0", (0.317, 0.016))):
        new = json.loads((FIG / f"goal6_flrw_a{float(alpha):g}.json").read_text())
        s, m = new["sim"], new["means"]
        # Las abundancias simuladas <N_k> (k = 0..3) coinciden con las medias MC L_k.
        z = (np.array(s["means"][1:]) - np.array(m["L"][:4])) / np.hypot(s["means_se"][1:], m["L_se"][:4])
        assert np.all(np.abs(z) < K), z
        assert abs(s["means"][0] / 6400 - 1) < K * np.sqrt(6400 / s["n_sprinklings"]) / 6400
        v = r[alpha]
        # resto/J¹ recalculado: Var = NN + cruzado N-k + J¹ + resto (descomposición del punto 1d).
        pa = f4.parts(m["L"], m["L_se"], 6400)
        assert v["r_6400"] == pytest.approx((s["sim"] - pa["NN"] - pa["Nk"] - pa["J1"]) / pa["J1"], rel=1e-10)
        assert abs((v["r_6400"] - ref[0]) / np.hypot(v["r_6400_se"], ref[1])) < K
        assert abs(v["slope"]["z"]) < K and v["passed"]


@pytest.mark.skipif(not HAVE_RES, reason="falta figures/goal6_results.json")
def test_R4_R5_counts_from_rows():
    r = json.loads(RES.read_text())
    rows = json.loads((FIG / "goal4e_q0.125_c0.25_n10000.json").read_text())["rows"]
    lcdm = json.loads((FIG / "goal4e_results.json").read_text())["lcdm_chi2"]
    d = np.array([x["chi2"] for x in rows]) - lcdm
    assert len(d) == 10_000 and r["R4"]["n_good"] == int(np.sum(d < 14.7)) == 0
    ref = json.loads((FIG / "goal6_snbao_lcdm.json").read_text())
    assert ref["chi2"] == pytest.approx(ref["sn"] + ref["bao"], rel=1e-10)
    for model in ("model1", "bd"):
        rows = json.loads((FIG / f"goal6_snbao_{model}.json").read_text())["rows"]
        assert len(rows) == 10_000 and len({x["seed"] for x in rows}) == 10_000
        chi = np.array([x["chi2"] for x in rows])
        # χ² sin CMB: es exactamente SN + BAO en cada semilla.
        np.testing.assert_allclose(chi, [x["sn"] + x["bao"] for x in rows], rtol=1e-9)
        assert r["R5"][model]["n_good"] == int(np.sum(chi < ref["chi2"] + 14.7))
        assert r["R5"][model]["dchi2_min"] == pytest.approx(chi.min() - ref["chi2"], rel=1e-12)


@pytest.mark.skipif(not (HAVE_RES and HAVE_DATA), reason="faltan datos o resultados")
def test_snbao_lcdm_is_a_minimum():
    g6 = _load_script()
    c, _, _ = g6._snbao()
    ref = json.loads((FIG / "goal6_snbao_lcdm.json").read_text())
    p = c.lcdm(ref["omega_m"], ref["h"])
    assert float(p["sn"][0] + p["bao"][0]) == pytest.approx(ref["chi2"], rel=1e-9)
    for dom, dh in ((0.003, 0), (-0.003, 0), (0, 0.005), (0, -0.005)):
        q = c.lcdm(ref["omega_m"] + dom, ref["h"] + dh)
        assert float(q["sn"][0] + q["bao"][0]) > ref["chi2"]
