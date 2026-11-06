"""Goal 6: calculations requested by the external review of the draft (2026-10-15).

The review (independent referee) pointed out three points that can be
strengthened with more data, without changing any earlier result:
  (a) the extrapolation p → 3/4 rests on a constant rest/J¹ only up to
      ρV = 6400 (diamond) and 3200 (FLRW);
  (b) with the smeared action (ε ≤ 0.2) the constancy of the unprojected rest
      is an a posteriori conclusion of Goal 5b (the preregistered F2 failed);
  (c) the best cell of the (q, c) map of Goal 4e (q = 1/8, c = 1/4, Δχ² = +17)
      was measured with only 1000 seeds.

Calculations (new outputs; the JSON files of the earlier goals are untouched):
  diamond   4D Minkowski diamond, ρV = 12 800, 500 sprinklings; full histogram
            of cardinalities per sprinkling (data/goal5/, same format as
            Goal 5, new seeds) -> ε = 1, 0.5, 0.2, 0.1
  pred      exact J¹_ε, E[F], g with ρV = 12 800 -> figures/goal6_pred.json
  flrw A    4D FLRW past cone, a = η^A (A = 1, 2), ρV = 6400, 500
            sprinklings and MC means (1.6e7 samples), as in point 1d
            -> figures/goal6_flrw_aA.json
  cell      Goal 4e, cell (q, c) = (1/8, 1/4) with 10 000 seeds
            -> figures/goal4e_q0.125_c0.25_n10000.json
  snbao M   fit per seed with SN + BAO (without CMB), M = model1 | bd
            -> figures/goal6_snbao_lcdm.json, figures/goal6_snbao_M.json
  fit       criteria -> figures/goal6_results.json

PREREGISTERED CRITERIA (2026-10-15, before computing; 4σ combined):
  R1 (diamond, local action). All independent sprinklings with ε = 1 are
     combined: Step 2 (ρV = 25-6400), Goal 5 (400-6400) and the new point
     (12 800); at each ρV they are averaged with weights 1/se².
     (i) rest/J¹ at ρV = 12 800 compatible with the weighted mean of
     Step 2 (0.317 ± 0.014); (ii) the slope of rest/J¹ versus ln ρV
     over the whole range is compatible with 0. The bound on the differential
     exponent δ of the correlations, (rest/J¹) ∝ (ρV)^δ, is also reported at
     2σ and at 4σ.
  R2 (smeared action, ε = 0.5, 0.2, 0.1). UNPROJECTED rest,
     r = (Var F − J¹)/J¹. Goal 5 and Goal 5b (independent sprinklings,
     400-6400) and the new point (12 800) are combined. The slope of r
     versus ln ρV is compatible with 0. This is the a posteriori claim of
     Goal 5b; here it is tested with a point that doubles ερV.
  R3 (4D FLRW, α = 1, 2). rest/J¹ at ρV = 6400 compatible with the weighted
     mean of point 1d (0.348 ± 0.016 and 0.317 ± 0.016), and the slope
     over ρV = 25-6400 remains compatible with 0.
  R4 (map of Goal 4e). The number of good seeds (Δχ² < 14.7) of the cell
     (1/8, 1/4) with 10 000 seeds is counted. If there is any, the sentence
     "no cell has good seeds" is corrected in the paper.
  R5 (added the same day, also before computing). The CMB distance priors
     are not applicable to Model 1 (Λ/3H² ~ 0.5 at z = 1100).
     The fit per seed of Goal 4d is repeated WITHOUT the CMB (χ² = SN + BAO),
     with the ΛCDM reference also fitted to SN + BAO, for Model 1 and for
     the BD action (β = 0.9), same 10 000 seeds. The number of good seeds
     (Δχ² < 14.7) and the minimum Δχ² are reported. Descriptive: it replaces
     in table 2 of the paper the +364 of Model 1, which included an
     inapplicable CMB.

Usage:  python scripts/goal6_referee.py diamond | pred | flrw A | cell | snbao M | fit
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import goal5_smeared4d as g5  # noqa: E402
from causets import sprinkle  # noqa: E402
from causets.causal import minkowski_causal_matrix  # noqa: E402
from causets.fluctuations import (  # noqa: E402
    EnsembleFluctuations,
    FluctuationKernel,
    abundance_vector,
    local_action_coefficients,
    pair_weight,
    smeared_pair_mecke_slope,
    smeared_pair_moment,
)
from causets.geometry import FLRWConeGeometry, MinkowskiDiamondGeometry  # noqa: E402

warnings.filterwarnings("ignore")
LAM_D, M_D = 12_800, 500
LAM_F, M_F = 6400, 500
PRED = ROOT / "figures" / "goal6_pred.json"
OUT = ROOT / "figures" / "goal6_results.json"
EPS = (1.0, 0.5, 0.2, 0.1)
STEP2_R = (0.317, 0.014)
FLRW_R = {1.0: (0.348, 0.016), 2.0: (0.317, 0.016)}
K = 4.0


def flrw_path(alpha):
    return ROOT / "figures" / f"goal6_flrw_a{alpha:g}.json"


def diamond():
    """Histograms with ρV = 12 800 (same format and folder as Goal 5;
    seeds [93_000, ρV, block], different from those of Goal 5)."""
    g = MinkowskiDiamondGeometry(4)
    g5.DATA.mkdir(parents=True, exist_ok=True)
    for c in range(M_D // g5.CHUNK):
        path = g5.DATA / f"hist_{LAM_D}_{c:03d}.npz"
        if path.exists():
            continue
        rng = np.random.default_rng([93_000, LAM_D, c])
        t0 = time.time()
        N, H = [], []
        for _ in range(g5.CHUNK):
            C = minkowski_causal_matrix(sprinkle(g.region, LAM_D / g.volume, rng))
            N.append(C.shape[0])
            H.append(g5.histogram(C))
            del C
        hist = np.zeros((g5.CHUNK, max(len(h) for h in H)), dtype=np.int64)
        for i, h in enumerate(H):
            hist[i, : len(h)] = h
        np.savez_compressed(path, N=np.array(N), hist=hist)
        print(f"ρV={LAM_D} block {c} {time.time() - t0:.0f}s", flush=True)


def pred():
    out = json.loads(PRED.read_text()) if PRED.exists() else {}
    for eps in EPS:
        key = f"{eps}_{LAM_D}"
        if key in out:
            continue
        t0 = time.time()
        out[key] = dict(eps=eps, rhoV=LAM_D, J1=smeared_pair_moment(4, LAM_D, eps, 2),
                        EF=smeared_pair_moment(4, LAM_D, eps, 1), g=smeared_pair_mecke_slope(4, LAM_D, eps))
        PRED.write_text(json.dumps(out, indent=1))
        print(key, out[key], f"{time.time() - t0:.0f}s", flush=True)


def flrw(alpha: float):
    """Sprinklings and MC means with ρV = 6400, as in ``flrw4d_scaling`` (new seeds)."""
    path = flrw_path(alpha)
    res = json.loads(path.read_text()) if path.exists() else {}
    g = FLRWConeGeometry(alpha, dim=4)
    if "sim" not in res:
        rng = np.random.default_rng(131_000 + LAM_F + int(10 * alpha))
        t0 = time.time()
        ab = np.array([abundance_vector(minkowski_causal_matrix(sprinkle(g.region, LAM_F / g.volume, rng)), 4)
                       for _ in range(M_F)])
        E = EnsembleFluctuations(ab, local_action_coefficients(4))
        res["sim"] = dict(alpha=alpha, rhoV=LAM_F, sim=E.variance, sim_se=E.variance_se, n_sprinklings=M_F,
                          means=ab.mean(axis=0).tolist(), means_se=(ab.std(axis=0, ddof=1) / np.sqrt(M_F)).tolist(),
                          t=time.time() - t0)
        path.write_text(json.dumps(res, indent=1))
        print("sim", res["sim"], flush=True)
    if "means" not in res:
        import flrw4d_scaling as f4

        t0 = time.time()
        Kn = FluctuationKernel(g, LAM_F / g.volume, 4, np.random.default_rng(141_000 + int(10 * alpha)))
        L, se = Kn._batched_means(f4.M_MEANS, 5, 100_000)
        res["means"] = dict(alpha=alpha, rhoV=LAM_F, L=L.tolist(), L_se=se.tolist(), t=time.time() - t0)
        path.write_text(json.dumps(res, indent=1))
        print("means", res["means"], flush=True)


def cell():
    import goal4e_robustness as g4e

    g4e.run_variant(g4e.Variant(), g4e.register(0.125, 0.25), 10_000)


def _snbao():
    import goal4e_robustness as g4e

    g4c, g4d = g4e.g4c, g4e.g4d

    class SNBAO(g4d.Combined):
        def evaluator(self, model, noise, om, amp):
            real = self.run(g4c.spec_for(model, amp), om, noise=noise)
            p = self.parts(real, om)
            return p["sn"] + p["bao"], p["M_B"], real.h, real.omega_lambda0

    return SNBAO(), g4c, g4d


def snbao_lcdm(c):
    from scipy.optimize import minimize

    path = ROOT / "figures" / "goal6_snbao_lcdm.json"
    if path.exists():
        return json.loads(path.read_text())
    import causets.stochastic_lambda as sl

    def f(x):
        om, h = x
        if not (0.02 < om < 0.8 and 0.4 < h < 1 and om + sl.OMEGA_R < h**2):
            return 1e9
        p = c.lcdm(om, h)
        return float(p["sn"][0] + p["bao"][0])

    best = min((minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-6))
                for x0 in [(0.14, 0.68), (0.15, 0.70), (0.13, 0.72)]), key=lambda r: r.fun)
    om, h = best.x
    p = c.lcdm(om, h)
    out = dict(omega_m=float(om), h=float(h), Omega_m=float(om / h**2), chi2=float(best.fun),
               sn=float(p["sn"][0]), bao=float(p["bao"][0]))
    path.write_text(json.dumps(out, indent=1))
    print("ΛCDM SN+BAO", out, flush=True)
    return out


def snbao(model):
    c, g4c, _ = _snbao()
    snbao_lcdm(c)
    path = ROOT / "figures" / f"goal6_snbao_{model}.json"
    rows = json.loads(path.read_text())["rows"] if path.exists() else []
    t0 = time.time()
    while len(rows) < g4c.N_SEEDS:
        start = len(rows)
        batch = list(range(start, min(start + g4c.BATCH, g4c.N_SEEDS)))
        new = g4c.fit_batch(c.sn, model, batch, evaluator=c.evaluator)
        noise = np.stack([np.random.default_rng(g4c.SEED0 + s).standard_normal(g4c.N_STEPS) for s in batch], axis=1)
        om = np.array([r["omega_m"] for r in new])
        p = c.parts(c.run(g4c.spec_for(model, np.array([r["amp"] for r in new])), om, noise=noise), om)
        for k, r in enumerate(new):
            r.update(sn=float(p["sn"][k]), bao=float(p["bao"][k]))
        rows += new
        path.write_text(json.dumps(dict(model=model, probes="SN+BAO", rows=rows)))
        print(f"{model} SN+BAO: {len(rows)} seeds, {time.time() - t0:.0f} s", flush=True)


def _wmean(y, se):
    w = 1 / np.asarray(se) ** 2
    return float(np.sum(w * y) / w.sum()), float(1 / np.sqrt(w.sum()))


def _slope(x, y, se):
    b, b_se, chi2, dof = g5._slope(np.asarray(x, float), np.asarray(y, float), np.asarray(se, float))
    return dict(slope=b, slope_se=b_se, z=b / b_se, chi2=chi2, dof=dof)


def _combine(points):
    """{ρV: [(r, se), ...]} -> rows with the weighted mean at each ρV."""
    rows = []
    for lam in sorted(points):
        r, se = _wmean([p[0] for p in points[lam]], [p[1] for p in points[lam]])
        rows.append(dict(rhoV=lam, r=r, se=se, n_sets=len(points[lam])))
    return rows


def _rest_new(eps, P):
    """Unprojected rest/J¹ with ρV = 12 800 from the new histograms."""
    N, hist = g5.load(LAM_D)
    F = hist @ pair_weight(4, np.arange(hist.shape[1]), eps)
    var, var_se = g5._var_se(F)
    p = P[f"{eps}_{LAM_D}"]
    return (var - p["J1"]) / p["J1"], var_se / p["J1"], len(F)


def fit():
    import flrw4d_scaling as f4

    P = json.loads(PRED.read_text())
    res = {}
    # --- R1: diamond, local action
    pts = {}
    for p in json.loads((ROOT / "figures" / "step2_4d_results.json").read_text())["points"]:
        pts.setdefault(p["rhoV"], []).append((p["rest_over_J1"], p["rest_over_J1_se"]))
    g5pts = json.loads((ROOT / "figures" / "goal5_results.json").read_text())["points"]
    for p in g5pts:
        if p["eps"] == 1.0:
            pts.setdefault(p["rhoV"], []).append((p["rest_over_J1"], p["rest_over_J1_se"]))
    r_new, se_new, m_new = _rest_new(1.0, P)
    pts.setdefault(LAM_D, []).append((r_new, se_new))
    # total Var S with ρV = 12 800 (for figure 1 of the paper; added in the second review)
    N12, h12 = g5.load(LAM_D)
    varS12 = g5._var_se(g5.action_coefficients(4, 1.0)[0] * N12 + h12 @ pair_weight(4, np.arange(h12.shape[1]), 1.0))
    rows = _combine(pts)
    z1 = (r_new - STEP2_R[0]) / np.hypot(se_new, STEP2_R[1])
    sl = _slope(np.log([r["rhoV"] for r in rows]), [r["r"] for r in rows], [r["se"] for r in rows])
    mean, mean_se = _wmean([r["r"] for r in rows], [r["se"] for r in rows])
    # bound on δ: if the correlations ∝ J¹ (ρV)^δ, then r ∝ (ρV)^δ and d r / d ln ρV = δ r
    delta = [(sl["slope"] - k * sl["slope_se"]) / mean for k in (4, 2)] + \
            [(sl["slope"] + k * sl["slope_se"]) / mean for k in (2, 4)]
    res["R1"] = dict(points=rows, r_12800=r_new, r_12800_se=se_new, M=m_new, z_12800=float(z1),
                     varS_12800=varS12[0], varS_12800_se=varS12[1],
                     slope=sl, mean=mean, mean_se=mean_se,
                     delta_4sigma=[delta[0], delta[3]], delta_2sigma=[delta[1], delta[2]],
                     passed=bool(abs(z1) < K and abs(sl["z"]) < K))
    # --- R2: smeared action, unprojected rest
    g5b = json.loads((ROOT / "figures" / "goal5b_results.json").read_text())["points"]
    res["R2"] = {}
    for eps in (0.5, 0.2, 0.1):
        pts = {}
        for p in g5pts:
            if p["eps"] == eps:
                pts.setdefault(p["rhoV"], []).append((p["rest_over_J1"], p["rest_over_J1_se"]))
        for p in g5b:
            if p["eps"] == eps:
                pts.setdefault(p["rhoV"], []).append((p["rest_over_J1"], p["rest_over_J1_se"]))
        r_new, se_new, _ = _rest_new(eps, P)
        pts.setdefault(LAM_D, []).append((r_new, se_new))
        rows = _combine(pts)
        sl = _slope(np.log([r["rhoV"] for r in rows]), [r["r"] for r in rows], [r["se"] for r in rows])
        res["R2"][str(eps)] = dict(points=rows, r_12800=r_new, r_12800_se=se_new, slope=sl,
                                   passed=bool(abs(sl["z"]) < K))
    # --- R3: FLRW 6400
    res["R3"] = {}
    old = json.loads((ROOT / "figures" / "flrw4d_results.json").read_text())
    for alpha in (1.0, 2.0):
        new = json.loads(flrw_path(alpha).read_text())
        pa = f4.parts(new["means"]["L"], new["means"]["L_se"], LAM_F)
        s = new["sim"]
        r = (s["sim"] - pa["NN"] - pa["Nk"] - pa["J1"]) / pa["J1"]
        r_se = np.hypot(s["sim_se"], pa["parts_se"]) / pa["J1"]
        z = (r - FLRW_R[alpha][0]) / np.hypot(r_se, FLRW_R[alpha][1])
        old_pts = old["points"][str(alpha)]
        x = [np.log(p["rhoV"]) for p in old_pts] + [np.log(LAM_F)]
        y = [p["rest_over_J1"] for p in old_pts] + [r]
        se = [p["rest_over_J1_se"] for p in old_pts] + [r_se]
        sl = _slope(x, y, se)
        res["R3"][str(alpha)] = dict(r_6400=float(r), r_6400_se=float(r_se), z=float(z), slope=sl,
                                     passed=bool(abs(z) < K and abs(sl["z"]) < K))
    # --- R4: map cell with 10 000 seeds
    import goal4e_robustness as g4e

    rows = json.loads(g4e.path_for(g4e.register(0.125, 0.25), 10_000).read_text())["rows"]
    lcdm = json.loads((ROOT / "figures" / "goal4e_results.json").read_text())["lcdm_chi2"]
    d = np.array([r["chi2"] for r in rows]) - lcdm  # same χ² as goal4e_robustness.summary
    res["R4"] = dict(n=len(d), n_good=int(np.sum(d < 14.7)), dchi2_min=float(d.min()))
    # --- R5: SN + BAO without CMB
    ref = json.loads((ROOT / "figures" / "goal6_snbao_lcdm.json").read_text())
    res["R5"] = dict(lcdm=ref)
    for model in ("model1", "bd"):
        rows = json.loads((ROOT / "figures" / f"goal6_snbao_{model}.json").read_text())["rows"]
        chi = np.array([r["chi2"] for r in rows])
        b = rows[int(np.argmin(chi))]
        res["R5"][model] = dict(n=len(rows), n_good=int(np.sum(chi < ref["chi2"] + 14.7)),
                                dchi2_min=float(chi.min() - ref["chi2"]),
                                best_sn=b["sn"] - ref["sn"], best_bao=b["bao"] - ref["bao"],
                                best_omega_m=b["omega_m"], best_h=b.get("h"))
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if k != "R1" else {kk: vv for kk, vv in v.items() if kk != "points"})
                      for k, v in res.items()}, indent=1, default=str))


BOOT = ROOT / "figures" / "goal6_bootstrap.json"


def boot(B: int = 2000):
    """A POSTERIORI (second review, 2026-10-16): bootstrap of the errors of r.

    The error std(d²)/√M of the variance may be underestimated with heavy
    tails. For each set with per-sprinkling data (Goal 5, Goal 5b and
    ρV = 12 800) the sprinklings are resampled (B replicas) and the bootstrap
    error of r = (Var F − J¹)/J¹ is compared with that of the formula. Then the
    slopes of R1 and R2 are repeated with the larger of the two errors, with
    the error scaled by sqrt(χ²/dof) and restricting the range (ρV ≥ 400,
    ≥ 800). This is not a criterion: it is a check of the robustness of the
    bounds.
    """
    import goal5b_decomposition as g5b

    rng = np.random.default_rng(161_016)
    P5 = json.loads(g5.PRED.read_text())
    P6 = json.loads(PRED.read_text())
    ratios = []

    def r_boot(F, J1):
        M = len(F)
        idx = rng.integers(0, M, size=(B, M))
        v = np.var(F[idx], axis=1, ddof=1)
        return float(np.std((v - J1) / J1, ddof=1)), g5._var_se(F)[1] / J1

    se_boot = {}
    for lam in list(g5.PLAN) + [LAM_D]:
        N, hist = g5.load(lam)
        P = P6 if lam == LAM_D else P5
        for eps in EPS:
            F = hist @ pair_weight(4, np.arange(hist.shape[1]), eps)
            b, f = r_boot(F, P[f"{eps}_{lam}"]["J1"])
            src = "goal6" if lam == LAM_D else "goal5"
            se_boot[(src, eps, lam)] = (b, f)
            ratios.append(dict(set=src, eps=eps, rhoV=lam, se_boot=b, se_formula=f, ratio=b / f))
    Pb = json.loads(g5b.PRED.read_text()) if hasattr(g5b, "PRED") else P5
    for lam in g5b.PLAN:
        st = g5b.load(lam)
        for j, eps in enumerate(g5b.EPS):
            b, f = r_boot(st[:, j, 1], Pb[f"{eps}_{lam}"]["J1"])
            se_boot[("goal5b", eps, lam)] = (b, f)
            ratios.append(dict(set="goal5b", eps=eps, rhoV=lam, se_boot=b, se_formula=f, ratio=b / f))
    out = dict(B=B, ratios=ratios)
    res = json.loads(OUT.read_text())

    def variants(rows_by_set):
        """rows_by_set: {ρV: [(r, se_formula, se_boot or None), ...]} -> slopes."""
        v = {}
        for name, use_boot, lmin in (("formula", False, 0), ("max_boot", True, 0),
                                     ("max_boot_ge400", True, 400), ("max_boot_ge800", True, 800)):
            pts = {lam: [(r, max(sf, sb) if (use_boot and sb) else sf) for r, sf, sb in lst]
                   for lam, lst in rows_by_set.items() if lam >= lmin}
            rows = _combine(pts)
            sl = _slope(np.log([r["rhoV"] for r in rows]), [r["r"] for r in rows], [r["se"] for r in rows])
            mean = _wmean([r["r"] for r in rows], [r["se"] for r in rows])[0]
            scale = max(1.0, np.sqrt(sl["chi2"] / sl["dof"]))
            v[name] = dict(slope=sl, mean=mean, chi2_scale=float(scale),
                           delta_2sigma=[(sl["slope"] - 2 * sl["slope_se"]) / mean,
                                         (sl["slope"] + 2 * sl["slope_se"]) / mean],
                           delta_2sigma_scaled=[(sl["slope"] - 2 * scale * sl["slope_se"]) / mean,
                                                (sl["slope"] + 2 * scale * sl["slope_se"]) / mean])
        return v

    # R1: Step 2 (no per-sprinkling data: formula error only) + Goal 5 (ε = 1) + 12 800
    sets = {}
    for p in json.loads((ROOT / "figures" / "step2_4d_results.json").read_text())["points"]:
        sets.setdefault(p["rhoV"], []).append((p["rest_over_J1"], p["rest_over_J1_se"], None))
    for p in json.loads((ROOT / "figures" / "goal5_results.json").read_text())["points"]:
        if p["eps"] == 1.0:
            sets.setdefault(p["rhoV"], []).append((p["rest_over_J1"], p["rest_over_J1_se"],
                                                   se_boot[("goal5", 1.0, p["rhoV"])][0]))
    sets.setdefault(LAM_D, []).append((res["R1"]["r_12800"], res["R1"]["r_12800_se"], se_boot[("goal6", 1.0, LAM_D)][0]))
    out["R1"] = variants(sets)
    out["R2"] = {}
    g5pts = json.loads((ROOT / "figures" / "goal5_results.json").read_text())["points"]
    g5bpts = json.loads((ROOT / "figures" / "goal5b_results.json").read_text())["points"]
    for eps in (0.5, 0.2, 0.1):
        sets = {}
        for src, pts in (("goal5", g5pts), ("goal5b", g5bpts)):
            for p in pts:
                if p["eps"] == eps:
                    sets.setdefault(p["rhoV"], []).append((p["rest_over_J1"], p["rest_over_J1_se"],
                                                           se_boot[(src, eps, p["rhoV"])][0]))
        R2 = res["R2"][str(eps)]
        sets.setdefault(LAM_D, []).append((R2["r_12800"], R2["r_12800_se"], se_boot[("goal6", eps, LAM_D)][0]))
        out["R2"][str(eps)] = variants(sets)
    BOOT.write_text(json.dumps(out, indent=1))
    rr = np.array([x["ratio"] for x in ratios])
    print(f"se_boot/se_formula: min {rr.min():.3f} median {np.median(rr):.3f} max {rr.max():.3f}")
    for k in ("R1",):
        for name, v in out[k].items():
            print(k, name, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v["slope"].items()},
                  "δ2σ", np.round(v["delta_2sigma"], 3), "scaled", np.round(v["delta_2sigma_scaled"], 3))
    for e, vs in out["R2"].items():
        for name, v in vs.items():
            print("R2", e, name, round(v["slope"]["slope"], 3), "±", round(v["slope"]["slope_se"], 3),
                  "z", round(v["slope"]["z"], 2), "δ2σ", np.round(v["delta_2sigma"], 3))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "flrw":
        flrw(float(sys.argv[2]))
    elif cmd == "snbao":
        snbao(sys.argv[2])
    else:
        {"diamond": diamond, "pred": pred, "cell": cell, "fit": fit, "boot": boot}[cmd]()
