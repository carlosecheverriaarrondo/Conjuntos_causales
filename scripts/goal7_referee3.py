"""Goal 7: cálculos pedidos por la tercera revisión externa (2026-10-23).

El árbitro (lectura nueva, recomendación "major revision") pide, entre otras
cosas, un punto de la acción suavizada con ερV mayor que 1280, p. ej.
ε = 0.5 con ρV = 25 600, porque la constancia de r = (Var F − J¹)/J¹ con
suavizado sólo se ha visto en 1.2 décadas. Los mismos sprinklings sirven
para todos los ε.

Cálculos (salidas nuevas; los JSON anteriores no se tocan):
  diamond   diamante de Minkowski 4D, ρV = 25 600, 500 sprinklings;
            histograma completo de cardinalidades por sprinkling (data/goal5/,
            mismo formato que el Goal 5; semillas [95_000, ρV, bloque]).
            Los puntos se ordenan por t y el producto C·C se hace por bloques
            triangulares (``histogram_tri``): mismo resultado que
            ``goal5_smeared4d.histogram`` (comprobado en los tests), ~6 veces
            menos operaciones y ~3 GB de memoria.
  pred      J¹_ε, E[F], g exactos con ρV = 25 600 -> figures/goal7_pred.json
  fit       criterios -> figures/goal7_results.json

CRITERIOS PRERREGISTRADOS (2026-10-23, antes de calcular; 4σ combinada):
  R6 (acción local, ε = 1). (i) r con ρV = 25 600 compatible con la media
     ponderada del Goal 6 sobre 25-12 800 (0.324 ± 0.012); (ii) la pendiente
     de r frente a ln ρV sobre 25-25 600 (puntos del Goal 6 más el nuevo) es
     compatible con 0. Se reporta la cota sobre δ a 2σ y 4σ.
  R7 (acción suavizada, ε = 0.5, 0.2, 0.1; ερV = 12 800, 5120, 2560). (i) La
     pendiente de r frente a ln ρV sobre 400-25 600 (puntos del Goal 6 más el
     nuevo) es compatible con 0; (ii) r con ρV = 25 600 es compatible con la
     media ponderada de los puntos 400-12 800 del mismo ε. Se reportan además
     (descriptivo) la pendiente con ρV ≥ 800 y la cota superior de δ a 2σ.
  Si R6 o R7 fallan, el paper lo dice y la afirmación sobre el exponente con
  suavizado se rebaja en consecuencia.
  R8 (ajuste por semilla, pedido por el árbitro: la rejilla del Goal 4c tiene
     una resolución final de 1/8 de paso, ~5 % en ω_m). Para cada conjunto de
     10⁴ semillas de la tabla 2 y de la celda (1/8, 1/4), las 50 semillas con
     menor χ² de rejilla se reajustan con Nelder-Mead en (ln ω_m, ln amp),
     partiendo del mejor punto de la rejilla. Se reportan el Δχ² mínimo
     reajustado, la mayor mejora ΔΔχ² y el número de semillas buenas
     (Δχ² < 14.7). Si alguna pasa a ser buena, el paper lo dice y la frase
     "ninguna semilla es buena" se corrige. Se comprueba además que la semilla
     50 de la rejilla menos la mayor mejora sigue por encima del umbral (si
     no, se reajustan más semillas). Se reportan también (descriptivo) los
     recuentos con márgenes 25 y 50.

  extras    (a posteriori, sin simulaciones) β con errores, corrección
            (ρV)^(-1/2) ln ρV de las medias, r₂ en 2D -> figures/goal7_extras.json

Uso:  python scripts/goal7_referee3.py diamond | pred | fit | extras | refine RUN | bestparts
      (RUN: snbao_bd, snbao_model1, full_bd, cell, beta0.88, beta0.97, beta0.5, beta1;
       full_model1 añadido a posteriori, al ver la mejora de full_bd;
       map_q{q}_c{c} (celdas del mapa con q <= 3/8, 1000 semillas) añadido a
       posteriori, al ver que la celda (1/8, 1/4) reajustada tiene semillas buenas)
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
import goal6_referee as g6  # noqa: E402
from causets import sprinkle  # noqa: E402
from causets.causal import minkowski_causal_matrix  # noqa: E402
from causets.fluctuations import pair_weight, smeared_pair_mecke_slope, smeared_pair_moment  # noqa: E402
from causets.geometry import MinkowskiDiamondGeometry  # noqa: E402

warnings.filterwarnings("ignore")
LAM, M = 25_600, 500
PRED = ROOT / "figures" / "goal7_pred.json"
OUT = ROOT / "figures" / "goal7_results.json"
EPS = (1.0, 0.5, 0.2, 0.1)
G6_MEAN = (0.324, 0.012)
K = 4.0


def histogram_tri(C: np.ndarray, block: int = 2048) -> np.ndarray:
    """Como ``goal5_smeared4d.histogram``, para C estrictamente triangular superior
    (puntos ordenados por t): (C·C)[i, j] sólo recibe k con i < k < j, así que
    el bloque (I, J) sólo necesita los bloques K = I..J."""
    n = len(C)
    Cf = C.astype(np.float32)
    edges = list(range(0, n, block)) + [n]
    nb = len(edges) - 1
    h = np.zeros(1, dtype=np.int64)
    for I in range(nb):
        i0, i1 = edges[I], edges[I + 1]
        for J in range(I, nb):
            j0, j1 = edges[J], edges[J + 1]
            mask = C[i0:i1, j0:j1]
            if not mask.any():
                continue
            P = np.zeros((i1 - i0, j1 - j0), dtype=np.float32)
            for Kb in range(I, J + 1):
                k0, k1 = edges[Kb], edges[Kb + 1]
                P += Cf[i0:i1, k0:k1] @ Cf[k0:k1, j0:j1]
            c = np.bincount(P[mask].astype(np.int64))
            if len(c) > len(h):
                h = np.pad(h, (0, len(c) - len(h)))
            h[: len(c)] += c
    return h


def diamond():
    g = MinkowskiDiamondGeometry(4)
    g5.DATA.mkdir(parents=True, exist_ok=True)
    for c in range(M // g5.CHUNK):
        path = g5.DATA / f"hist_{LAM}_{c:03d}.npz"
        if path.exists():
            continue
        rng = np.random.default_rng([95_000, LAM, c])
        t0 = time.time()
        N, H = [], []
        for _ in range(g5.CHUNK):
            x = sprinkle(g.region, LAM / g.volume, rng)
            C = minkowski_causal_matrix(x[np.argsort(x[:, 0])])
            N.append(C.shape[0])
            H.append(histogram_tri(C))
            del C
        hist = np.zeros((g5.CHUNK, max(len(h) for h in H)), dtype=np.int64)
        for i, h in enumerate(H):
            hist[i, : len(h)] = h
        np.savez_compressed(path, N=np.array(N), hist=hist)
        print(f"ρV={LAM} bloque {c} {time.time() - t0:.0f}s", flush=True)


def pred():
    out = json.loads(PRED.read_text()) if PRED.exists() else {}
    for eps in EPS:
        key = f"{eps}_{LAM}"
        if key in out:
            continue
        t0 = time.time()
        out[key] = dict(eps=eps, rhoV=LAM, J1=smeared_pair_moment(4, LAM, eps, 2),
                        EF=smeared_pair_moment(4, LAM, eps, 1), g=smeared_pair_mecke_slope(4, LAM, eps))
        PRED.write_text(json.dumps(out, indent=1))
        print(key, out[key], f"{time.time() - t0:.0f}s", flush=True)


def _rest_new(eps, P):
    N, hist = g5.load(LAM)
    F = hist @ pair_weight(4, np.arange(hist.shape[1]), eps)
    var, var_se = g5._var_se(F)
    J1 = P[f"{eps}_{LAM}"]["J1"]
    rng = np.random.default_rng(171_023)
    idx = rng.integers(0, len(F), size=(2000, len(F)))
    se_boot = float(np.std((np.var(F[idx], axis=1, ddof=1) - J1) / J1, ddof=1))
    return (var - J1) / J1, var_se / J1, se_boot, len(F)


def _delta(sl, mean):
    return dict(delta_2sigma=[(sl["slope"] - 2 * sl["slope_se"]) / mean, (sl["slope"] + 2 * sl["slope_se"]) / mean],
                delta_4sigma=[(sl["slope"] - 4 * sl["slope_se"]) / mean, (sl["slope"] + 4 * sl["slope_se"]) / mean])


def fit():
    P = json.loads(PRED.read_text())
    r6 = json.loads(g6.OUT.read_text())
    res = {}
    # --- R6: acción local
    r, se, sb, m = _rest_new(1.0, P)
    rows = r6["R1"]["points"] + [dict(rhoV=LAM, r=r, se=se, n_sets=1)]
    sl = g6._slope(np.log([p["rhoV"] for p in rows]), [p["r"] for p in rows], [p["se"] for p in rows])
    mean, mean_se = g6._wmean([p["r"] for p in rows], [p["se"] for p in rows])
    z = (r - G6_MEAN[0]) / np.hypot(se, G6_MEAN[1])
    res["R6"] = dict(points=rows, r_new=r, r_new_se=se, r_new_se_boot=sb, M=m, z_new=float(z), slope=sl,
                     mean=mean, mean_se=mean_se, **_delta(sl, mean),
                     passed=bool(abs(z) < K and abs(sl["z"]) < K))
    # --- R7: acción suavizada
    res["R7"] = {}
    for eps in (0.5, 0.2, 0.1):
        r, se, sb, _ = _rest_new(eps, P)
        old = r6["R2"][str(eps)]["points"]
        m_old, m_old_se = g6._wmean([p["r"] for p in old], [p["se"] for p in old])
        z = (r - m_old) / np.hypot(se, m_old_se)
        rows = old + [dict(rhoV=LAM, r=r, se=se, n_sets=1)]
        sl = g6._slope(np.log([p["rhoV"] for p in rows]), [p["r"] for p in rows], [p["se"] for p in rows])
        mean = g6._wmean([p["r"] for p in rows], [p["se"] for p in rows])[0]
        hi = [p for p in rows if p["rhoV"] >= 800]
        sl800 = g6._slope(np.log([p["rhoV"] for p in hi]), [p["r"] for p in hi], [p["se"] for p in hi])
        res["R7"][str(eps)] = dict(points=rows, r_new=r, r_new_se=se, r_new_se_boot=sb, mean_old=m_old,
                                   mean_old_se=m_old_se, z_new=float(z), slope=sl, slope_ge800=sl800,
                                   mean=mean, **_delta(sl, mean),
                                   passed=bool(abs(z) < K and abs(sl["z"]) < K))
    # Var S total (figura 1)
    N, h = g5.load(LAM)
    res["varS_new"] = g5._var_se(g5.action_coefficients(4, 1.0)[0] * N + h @ pair_weight(4, np.arange(h.shape[1]), 1.0))
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "points"} if k == "R6" else v)
                      for k, v in res.items()}, indent=1, default=str))


EXTRAS = ROOT / "figures" / "goal7_extras.json"


def extras():
    """A POSTERIORI (tercera revisión), sin simulaciones nuevas:
    (a) memoria β de los conos anidados del Goal 4b: β por par de conos,
        errores bootstrap (2000 réplicas, semilla 171_024) y media GLS con la
        covarianza bootstrap; asimetría y curtosis de S y de sus incrementos.
    (b) corrección de orden (ρV)^(-1/2) ln ρV a ⟨N_k⟩ ≈ asintótico (apéndice C
        del paper): coeficiente del logaritmo ajustado a las medias exactas del
        cono de Minkowski (10⁶, 10⁷, 10⁸) frente a la predicción
        a_k = π k! / (√c₄ (G/V^1.5) Γ(k+½)).
    (c) r₂ en 2D: b(resto)/Σf² a partir de figures/goal3d_results.json.
    """
    from scipy.special import gamma
    from scipy.stats import kurtosis, skew

    out = dict(beta={}, correction={}, r2_2d={})
    rng = np.random.default_rng(171_024)
    for g in ("mink4", "flrw4"):
        S = np.array(json.loads((ROOT / "figures" / f"goal4b_sim_{g}.json").read_text())["S"])

        def betas(X):
            C = np.cov(X.T)
            v = np.diag(C)
            return np.array([2 * np.log(C[i, j] / np.sqrt(v[i] * v[j])) / np.log(v[i] / v[j])
                             for i in range(4) for j in range(i + 1, 4)])

        b0 = betas(S)
        B = np.array([betas(S[rng.integers(0, len(S), len(S))]) for _ in range(2000)])
        w = np.linalg.solve(np.cov(B.T), np.ones(len(b0)))
        out["beta"][g] = dict(pairs=b0.tolist(), se=B.std(axis=0, ddof=1).tolist(), gls=float(w @ b0 / w.sum()),
                              gls_se=float(1 / np.sqrt(w.sum())), skew=skew(S).tolist(), exkurt=kurtosis(S).tolist(),
                              incr_exkurt=kurtosis(np.diff(S, axis=1)).tolist(), kurt_se=float(np.sqrt(24 / len(S))))
    c = json.loads((ROOT / "figures" / "cone4d_results.json").read_text())
    R = c["H2"]["ratios"]
    xs = sorted(R, key=float)
    x = np.array([float(v) for v in xs])
    c4 = np.pi / 24
    for k in range(4):
        y = np.array([1 - R[v][k] for v in xs]) * np.sqrt(x)
        a, b = np.linalg.lstsq(np.vstack([np.log(x), np.ones_like(x)]).T, y, rcond=None)[0]
        pred = np.pi / (np.sqrt(c4) * c["G_cone"]) * gamma(k + 1) / gamma(k + 0.5)
        out["correction"][str(k)] = dict(a_fit=float(a), b_fit=float(b), a_pred=float(pred), rel=float(a / pred - 1))
    d2 = json.loads((ROOT / "figures" / "goal3d_results.json").read_text())
    for eps, v in d2.items():
        out["r2_2d"][eps] = dict(r2=v["no_J1"]["b"] / v["b_pred"], r2_se=v["no_J1"]["se"] / v["b_pred"],
                                 b_total=v["total"]["b"], b_J1=v["J1"]["b"], b_J6c=v["J6c"]["b"], sum_f2=v["b_pred"])
    EXTRAS.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


REFINE_N = 50
MARGINS = (14.7, 25.0, 50.0)


def _run_setup(run):
    """(filas, χ² ΛCDM de referencia, evaluador(noise, om, amp) -> χ², clave del modelo)."""
    import goal4c_sn_fit as g4c
    import goal4d_combined as g4d
    import goal4e_robustness as g4e

    fig = ROOT / "figures"
    if run.startswith("snbao_"):
        model = run.split("_")[1]
        c, _, _ = g6._snbao()
        rows = json.loads((fig / f"goal6_snbao_{model}.json").read_text())["rows"]
        ref = json.loads((fig / "goal6_snbao_lcdm.json").read_text())["chi2"]
        return rows, ref, (lambda nz, om, amp: c.evaluator(model, nz, om, amp)[0]), model, g4c
    ref = json.loads((fig / "goal4d_lcdm.json").read_text())["chi2"]
    if run.startswith("full_"):
        model = run.split("_")[1]
        c = g4d.Combined()
        rows = json.loads((fig / f"goal4d_seeds_{model}.json").read_text())["rows"]
        return rows, ref, (lambda nz, om, amp: c.evaluator(model, nz, om, amp)[0]), model, g4c
    # goal4e carga su propia copia de goal4c (vía goal4d): las variantes se dan de alta en ella
    v = g4e.Variant()
    if run.startswith("map_"):  # celda del mapa (q, c) con 1000 semillas, p. ej. map_q0.125_c0.5
        k = g4e.register(*g4e.parse(run[4:]))
        rows = json.loads(g4e.path_for(k, g4e.MAP_SEEDS).read_text())["rows"]
        return rows, ref, (lambda nz, om, amp: v.evaluator(k, nz, om, amp)[0]), k, g4e.g4c
    k = g4e.register(0.125, 0.25) if run == "cell" else g4e.register(0.25, 0.75 * float(run[4:]))
    rows = json.loads(g4e.path_for(k, 10_000).read_text())["rows"]
    return rows, ref, (lambda nz, om, amp: v.evaluator(k, nz, om, amp)[0]), k, g4e.g4c


def refine(run):
    from scipy.optimize import minimize

    path = ROOT / "figures" / f"goal7_refine_{run}.json"
    rows, ref, ev, key, g4c = _run_setup(run)
    lo_om, hi_om = np.log(g4c.OMEGA_M_GRID[[0, -1]])
    lo_a, hi_a = np.log(g4c.AMP_GRID[key][[0, -1]])
    chi = np.array([r["chi2"] for r in rows])
    order = np.argsort(chi)[:REFINE_N]
    out = []
    t0 = time.time()
    for i in order:
        r = rows[i]
        nz = np.random.default_rng(g4c.SEED0 + r["seed"]).standard_normal(g4c.N_STEPS)[:, None]

        def f(x):
            if not (lo_om <= x[0] <= hi_om and lo_a <= x[1] <= hi_a):
                return 1e9
            v = float(ev(nz, np.exp(x[:1]), np.exp(x[1:]))[0])
            return v if np.isfinite(v) else 1e9

        x0 = np.log([r["omega_m"], r["amp"]])
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-3, initial_simplex=[x0, x0 + [0.05, 0], x0 + [0, 0.05]]))
        out.append(dict(seed=r["seed"], chi2_grid=r["chi2"], chi2=min(float(res.fun), r["chi2"]),
                        omega_m=float(np.exp(res.x[0])), amp=float(np.exp(res.x[1])), nfev=int(res.nfev)))
    d_grid = chi[order] - ref
    d_ref = np.array([o["chi2"] for o in out]) - ref
    gain = d_grid - d_ref
    summ = dict(run=run, ref=ref, n=len(rows), n_refined=len(out), dchi2_min_grid=float(d_grid.min()),
                dchi2_min_refined=float(d_ref.min()), max_gain=float(gain.max()), median_gain=float(np.median(gain)),
                last_grid=float(d_grid[-1]), safe=bool(d_grid[-1] - gain.max() > MARGINS[0]),
                n_good={str(m): int(np.sum(d_ref < m)) for m in MARGINS},
                n_good_grid={str(m): int(np.sum(chi - ref < m)) for m in MARGINS}, t=time.time() - t0)
    path.write_text(json.dumps(dict(summary=summ, rows=out), indent=1))
    print(json.dumps(summ), flush=True)


def bestparts():
    """Δχ² por sonda (SN / BAO / CMB) de la mejor semilla reajustada de cada
    goal7_refine_*.json, respecto a las partes del ΛCDM de referencia
    (figures/goal7_refine_parts.json; para la tabla 2 del paper)."""
    fig = ROOT / "figures"
    out = {}
    for path in sorted(fig.glob("goal7_refine_*.json")):
        run = path.stem.removeprefix("goal7_refine_")
        if run == "parts":
            continue
        d = json.loads(path.read_text())
        b = min(d["rows"], key=lambda x: x["chi2"])
        rows, ref, ev, key, g4c = _run_setup(run)
        snbao = run.startswith("snbao_")
        lc = json.loads((fig / ("goal6_snbao_lcdm.json" if snbao else "goal4d_lcdm.json")).read_text())
        if snbao:
            c, _, _ = g6._snbao()
            spec = g4c.spec_for(key, b["amp"])
        elif run.startswith("full_"):
            import goal4d_combined as g4d
            c = g4d.Combined()
            spec = g4c.spec_for(key, b["amp"])
        else:
            import goal4e_robustness as g4e
            c = g4e.Variant()
            spec = c.spec(key, b["amp"])
        nz = np.random.default_rng(g4c.SEED0 + b["seed"]).standard_normal(g4c.N_STEPS)[:, None]
        om = np.array([b["omega_m"]])
        p = c.parts(c.run(spec, om, noise=nz), om)
        probes = ("sn", "bao") if snbao else ("sn", "bao", "cmb")
        parts = {k: float(p[k][0] - lc[k]) for k in probes}
        total = sum(float(p[k][0]) for k in probes)
        assert abs(total - b["chi2"]) < 1e-6, (run, total, b["chi2"])
        out[run] = dict(seed=b["seed"], dchi2=b["chi2"] - ref, omega_m=b["omega_m"], amp=b["amp"], parts=parts)
        print(run, json.dumps(out[run]), flush=True)
    (fig / "goal7_refine_parts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "refine":
        refine(sys.argv[2])
    else:
        {"diamond": diamond, "pred": pred, "fit": fit, "extras": extras, "bestparts": bestparts}[sys.argv[1]]()
