"""Goal 6: cálculos pedidos por la revisión externa del borrador (2026-10-15).

La revisión (árbitro independiente) señaló tres puntos que se pueden reforzar
con más datos, sin cambiar ningún resultado anterior:
  (a) la extrapolación p → 3/4 se apoya en resto/J¹ constante sólo hasta
      ρV = 6400 (diamante) y 3200 (FLRW);
  (b) con la acción suavizada (ε ≤ 0.2) la constancia del resto sin proyectar
      es una conclusión a posteriori del Goal 5b (el F2 prerregistrado falló);
  (c) la mejor celda del mapa (q, c) del Goal 4e (q = 1/8, c = 1/4, Δχ² = +17)
      se midió con sólo 1000 semillas.

Cálculos (salidas nuevas; los JSON de los goals anteriores no se tocan):
  diamond   diamante de Minkowski 4D, ρV = 12 800, 500 sprinklings; histograma
            completo de cardinalidades por sprinkling (data/goal5/, mismo
            formato que el Goal 5, semillas nuevas) -> ε = 1, 0.5, 0.2, 0.1
  pred      J¹_ε, E[F], g exactos con ρV = 12 800 -> figures/goal6_pred.json
  flrw A    cono pasado FLRW 4D, a = η^A (A = 1, 2), ρV = 6400, 500
            sprinklings y medias MC (1.6e7 muestras), como el punto 1d
            -> figures/goal6_flrw_aA.json
  cell      Goal 4e, celda (q, c) = (1/8, 1/4) con 10 000 semillas
            -> figures/goal4e_q0.125_c0.25_n10000.json
  snbao M   ajuste por semilla con SN + BAO (sin CMB), M = model1 | bd
            -> figures/goal6_snbao_lcdm.json, figures/goal6_snbao_M.json
  fit       criterios -> figures/goal6_results.json

CRITERIOS PRERREGISTRADOS (2026-10-15, antes de calcular; 4σ combinada):
  R1 (diamante, acción local). Se combinan todos los sprinklings
     independientes con ε = 1: Paso 2 (ρV = 25-6400), Goal 5 (400-6400) y el
     punto nuevo (12 800); en cada ρV se promedian ponderando por 1/se².
     (i) resto/J¹ con ρV = 12 800 compatible con la media ponderada del
     Paso 2 (0.317 ± 0.014); (ii) la pendiente de resto/J¹ frente a ln ρV
     sobre todo el rango es compatible con 0. Se reporta además la cota sobre
     el exponente diferencial δ de las correlaciones, (resto/J¹) ∝ (ρV)^δ, a
     2σ y a 4σ.
  R2 (acción suavizada, ε = 0.5, 0.2, 0.1). Resto SIN proyectar,
     r = (Var F − J¹)/J¹. Se combinan Goal 5 y Goal 5b (sprinklings
     independientes, 400-6400) y el punto nuevo (12 800). La pendiente de r
     frente a ln ρV es compatible con 0. Esta es la afirmación a posteriori del
     Goal 5b; aquí se pone a prueba con un punto que duplica ερV.
  R3 (FLRW 4D, α = 1, 2). resto/J¹ con ρV = 6400 compatible con la media
     ponderada del punto 1d (0.348 ± 0.016 y 0.317 ± 0.016), y la pendiente
     sobre ρV = 25-6400 sigue compatible con 0.
  R4 (mapa del Goal 4e). Se cuenta el número de semillas buenas
     (Δχ² < 14.7) de la celda (1/8, 1/4) con 10 000 semillas. Si hay alguna,
     la frase "ninguna celda tiene semillas buenas" se corrige en el paper.
  R5 (añadido el mismo día, también antes de calcular). Los priors de
     distancia del CMB no son aplicables a Model 1 (Λ/3H² ~ 0.5 en z = 1100).
     Se repite el ajuste por semilla del Goal 4d SIN el CMB (χ² = SN + BAO),
     con la referencia ΛCDM ajustada también a SN + BAO, para Model 1 y para
     la acción BD (β = 0.9), mismas 10 000 semillas. Se reporta el número de
     semillas buenas (Δχ² < 14.7) y el Δχ² mínimo. Descriptivo: sustituye en
     la tabla 2 del paper al +364 de Model 1, que incluía un CMB no aplicable.

Uso:  python scripts/goal6_referee.py diamond | pred | flrw A | cell | snbao M | fit
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
    """Histogramas con ρV = 12 800 (mismo formato y carpeta que el Goal 5;
    semillas [93_000, ρV, bloque], distintas de las del Goal 5)."""
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
        print(f"ρV={LAM_D} bloque {c} {time.time() - t0:.0f}s", flush=True)


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
    """Sprinklings y medias MC con ρV = 6400, como ``flrw4d_scaling`` (semillas nuevas)."""
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
        print(f"{model} SN+BAO: {len(rows)} semillas, {time.time() - t0:.0f} s", flush=True)


def _wmean(y, se):
    w = 1 / np.asarray(se) ** 2
    return float(np.sum(w * y) / w.sum()), float(1 / np.sqrt(w.sum()))


def _slope(x, y, se):
    b, b_se, chi2, dof = g5._slope(np.asarray(x, float), np.asarray(y, float), np.asarray(se, float))
    return dict(slope=b, slope_se=b_se, z=b / b_se, chi2=chi2, dof=dof)


def _combine(points):
    """{ρV: [(r, se), ...]} -> filas con la media ponderada en cada ρV."""
    rows = []
    for lam in sorted(points):
        r, se = _wmean([p[0] for p in points[lam]], [p[1] for p in points[lam]])
        rows.append(dict(rhoV=lam, r=r, se=se, n_sets=len(points[lam])))
    return rows


def _rest_new(eps, P):
    """resto/J¹ sin proyectar con ρV = 12 800 a partir de los histogramas nuevos."""
    N, hist = g5.load(LAM_D)
    F = hist @ pair_weight(4, np.arange(hist.shape[1]), eps)
    var, var_se = g5._var_se(F)
    p = P[f"{eps}_{LAM_D}"]
    return (var - p["J1"]) / p["J1"], var_se / p["J1"], len(F)


def fit():
    import flrw4d_scaling as f4

    P = json.loads(PRED.read_text())
    res = {}
    # --- R1: diamante, acción local
    pts = {}
    for p in json.loads((ROOT / "figures" / "step2_4d_results.json").read_text())["points"]:
        pts.setdefault(p["rhoV"], []).append((p["rest_over_J1"], p["rest_over_J1_se"]))
    g5pts = json.loads((ROOT / "figures" / "goal5_results.json").read_text())["points"]
    for p in g5pts:
        if p["eps"] == 1.0:
            pts.setdefault(p["rhoV"], []).append((p["rest_over_J1"], p["rest_over_J1_se"]))
    r_new, se_new, m_new = _rest_new(1.0, P)
    pts.setdefault(LAM_D, []).append((r_new, se_new))
    rows = _combine(pts)
    z1 = (r_new - STEP2_R[0]) / np.hypot(se_new, STEP2_R[1])
    sl = _slope(np.log([r["rhoV"] for r in rows]), [r["r"] for r in rows], [r["se"] for r in rows])
    mean, mean_se = _wmean([r["r"] for r in rows], [r["se"] for r in rows])
    # cota sobre δ: si las correlaciones ∝ J¹ (ρV)^δ, r ∝ (ρV)^δ y d r / d ln ρV = δ r
    delta = [(sl["slope"] - k * sl["slope_se"]) / mean for k in (4, 2)] + \
            [(sl["slope"] + k * sl["slope_se"]) / mean for k in (2, 4)]
    res["R1"] = dict(points=rows, r_12800=r_new, r_12800_se=se_new, M=m_new, z_12800=float(z1),
                     slope=sl, mean=mean, mean_se=mean_se,
                     delta_4sigma=[delta[0], delta[3]], delta_2sigma=[delta[1], delta[2]],
                     passed=bool(abs(z1) < K and abs(sl["z"]) < K))
    # --- R2: acción suavizada, resto sin proyectar
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
    # --- R4: celda del mapa con 10 000 semillas
    import goal4e_robustness as g4e

    rows = json.loads(g4e.path_for(g4e.register(0.125, 0.25), 10_000).read_text())["rows"]
    lcdm = json.loads((ROOT / "figures" / "goal4e_results.json").read_text())["lcdm_chi2"]
    d = np.array([r["chi2"] for r in rows]) - lcdm  # mismo χ² que goal4e_robustness.summary
    res["R4"] = dict(n=len(d), n_good=int(np.sum(d < 14.7)), dchi2_min=float(d.min()))
    # --- R5: SN + BAO sin CMB
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


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "flrw":
        flrw(float(sys.argv[2]))
    elif cmd == "snbao":
        snbao(sys.argv[2])
    else:
        {"diamond": diamond, "pred": pred, "cell": cell, "fit": fit}[cmd]()
