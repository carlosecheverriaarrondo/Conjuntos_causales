"""Punto 1a: fluctuaciones de la acción BD en el cono de luz pasado truncado
J^-(q) ∩ {t >= 0} de Minkowski 2D (``TruncatedPastCone``), el análogo plano
de la región que usa Everpresent Λ.

Hipótesis (formulada ANTES de calcular): el término "mismo par" (J¹) da
<sum_pares h(n)> = rho V ln(rho V) sum_n h(n) + O(rho V) para cualquier
región compacta: el coeficiente del logaritmo sale del rango de rapidez
ln(rho) de los intervalos casi nulos, y la forma de la región sólo cambia la
constante. Por tanto, en el cono, (Delta S)² ⊇ b(eps) rho V ln(rho V) con el
mismo b(eps) = sum_n f_eps(n)² que en el diamante.

Criterios de éxito (fijados antes de ver los resultados):
  C1  <N_k> exacto en el cono (``continuum.expected_abundance_cone``) frente a
      sprinklings: |z| < 4 para k = 0..3 en cada densidad.
  C2  aumento por década de <sum f_eps²>/rho V entre rho V = 1e5 y 1e6
      (integral exacta en el cono) dividido por ln 10 = b(eps), tolerancia
      relativa 1e-3 (test).
  C3  predicción MYZ (``FluctuationKernel`` sobre ``MinkowskiConeGeometry``)
      frente a la varianza de los sprinklings, eps = 1 y 0.5,
      rho V = 50, 200, 800, 3200: todos los puntos con |z| < 4 (error combinado).
  C4  ajuste (Delta S)²/rho V = c + b ln(rho V) de las predicciones MYZ con
      eps = 1, rho V = 50 .. 12800: b compatible a 4σ con el del diamante
      (24.43 ± 0.24, Goal 3d).

Fases:
  sim      sprinklings del cono; abundancias con eps = 1 y 0.5 de los mismos
           sprinklings -> figures/cone_sim.json
  pred     predicción MYZ y descomposición por términos -> figures/cone_pred.json
  fit      criterios, ajustes y figura -> figures/cone_results.json,
           figures/cone_fluctuations.png

Uso:  python scripts/cone_fluctuations.py sim | pred | fit
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causets import sprinkle  # noqa: E402
from causets.causal import minkowski_causal_matrix  # noqa: E402
from causets.continuum import expected_abundance_cone  # noqa: E402
from causets.fluctuations import (  # noqa: E402
    EnsembleFluctuations,
    FluctuationKernel,
    action_coefficients,
    same_pair_log_coefficient,
    thinned_abundance_vector,
)
from causets.geometry import MinkowskiConeGeometry  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from goal3d_log_origin import decompose, fit_log  # noqa: E402

warnings.filterwarnings("ignore")
SIM = ROOT / "figures" / "cone_sim.json"
PRED = ROOT / "figures" / "cone_pred.json"
RESULTS = ROOT / "figures" / "cone_results.json"
EPS = (1.0, 0.5)
SIM_LAMS = {50: 4000, 200: 4000, 800: 3000, 3200: 2000}  # rho V: sprinklings
PRED_LAMS = (50, 200, 800, 3200, 12800)
M_TERMS = {1.0: 2_000_000, 0.5: 4_000_000}
DIAMOND_B = (24.43, 0.24)  # b total del diamante, eps = 1 (Goal 3d)
K_SIGMA = 4.0
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def exact_means(lam: float, eps: float) -> list[float]:
    return [expected_abundance_cone(lam, k, eps) for k in range(4)]


def sim():
    g = MinkowskiConeGeometry(2)
    rows = []
    for lam, n in SIM_LAMS.items():
        t0 = time.time()
        rng = np.random.default_rng(71_000 + lam)
        ab = {eps: [] for eps in EPS}
        for _ in range(n):
            C = minkowski_causal_matrix(sprinkle(g.region, lam / g.volume, rng))
            for eps in EPS:
                ab[eps].append(thinned_abundance_vector(C, 3, eps))
        for eps in EPS:
            E = EnsembleFluctuations(np.array(ab[eps]), action_coefficients(2, eps))
            A = E.abundances
            row = dict(rhoV=lam, eps=eps, n=n, variance=E.variance, variance_se=E.variance_se,
                       means=A.mean(axis=0).tolist(), means_se=(A.std(axis=0, ddof=1) / np.sqrt(n)).tolist(),
                       t=time.time() - t0)
            print(json.dumps(row), flush=True)
            rows.append(row)
        SIM.write_text(json.dumps(rows, indent=1))


def pred():
    g = MinkowskiConeGeometry(2)
    rows = []
    for eps in EPS:
        for lam in PRED_LAMS:
            t0 = time.time()
            K = FluctuationKernel(g, lam / g.volume, 3, np.random.default_rng(81_000 + lam + int(100 * eps)), eps=eps)
            P = K.predict(m_terms=M_TERMS[eps], exact_means=exact_means(lam, eps), batch=100_000)
            parts = decompose(P, eps)
            total = sum(p["value"] for p in parts.values())
            row = dict(eps=eps, rhoV=lam, variance=P.variance, variance_se=P.variance_se, parts=parts,
                       closure=total - P.variance, t=time.time() - t0)
            print(json.dumps(row), flush=True)
            rows.append(row)
            PRED.write_text(json.dumps(rows, indent=1))


def fit():
    sims, preds = json.loads(SIM.read_text()), json.loads(PRED.read_text())
    out = {"C1": [], "C3": []}
    for s in sims:
        L = exact_means(s["rhoV"], s["eps"])
        z = [(m - l) / se for m, l, se in zip(s["means"][1:], L, s["means_se"][1:])]
        out["C1"].append(dict(rhoV=s["rhoV"], eps=s["eps"], exact=L, z=z))
        p = next(p for p in preds if p["rhoV"] == s["rhoV"] and p["eps"] == s["eps"])
        z3 = (s["variance"] - p["variance"]) / np.hypot(s["variance_se"], p["variance_se"])
        out["C3"].append(dict(rhoV=s["rhoV"], eps=s["eps"], sim=s["variance"], sim_se=s["variance_se"],
                              pred=p["variance"], pred_se=p["variance_se"], z=z3))
    out["fits"] = {}
    for eps in EPS:
        sel = sorted((p for p in preds if p["eps"] == eps), key=lambda r: r["rhoV"])
        lam = np.array([r["rhoV"] for r in sel], float)
        y = np.array([r["variance"] for r in sel]) / lam
        se = np.array([r["variance_se"] for r in sel]) / lam
        b, b_se, c, chi2 = fit_log(lam, y, se)
        j1 = np.array([r["parts"]["J1"]["value"] for r in sel]) / lam
        j1_inc = (np.diff(j1) / np.diff(np.log(lam))).tolist()  # -> b(eps)
        rest = y - j1 - np.array([r["parts"]["NN"]["value"] + r["parts"]["Nk"]["value"] for r in sel]) / lam
        p_loc = (np.diff(np.log(y * lam)) / np.diff(np.log(lam)) / 2).tolist()
        out["fits"][str(eps)] = dict(b=b, b_se=b_se, c=c, chi2dof=chi2, b_pred_J1=same_pair_log_coefficient(eps),
                                     J1_increase_over_ln4=j1_inc, rest_over_rhoV=rest.tolist(),
                                     local_exponent_of_DeltaS=p_loc)
    f1 = out["fits"]["1.0"]
    out["C4"] = dict(b_cone=f1["b"], b_cone_se=f1["b_se"], b_diamond=DIAMOND_B[0], b_diamond_se=DIAMOND_B[1],
                     z=(f1["b"] - DIAMOND_B[0]) / np.hypot(f1["b_se"], DIAMOND_B[1]))
    RESULTS.write_text(json.dumps(out, indent=1))

    print("C1 (|z| de <N_k>):", max(max(abs(v) for v in r["z"]) for r in out["C1"]))
    for r in out["C3"]:
        print(f"C3 eps={r['eps']:<4} rhoV={r['rhoV']:>5}: sim {r['sim']:.5g} ± {r['sim_se']:.2g}   "
              f"pred {r['pred']:.5g} ± {r['pred_se']:.2g}   z = {r['z']:+.2f}")
    for eps, f in out["fits"].items():
        print(f"eps={eps}: b = {f['b']:.4g} ± {f['b_se']:.2g} (chi2/dof {f['chi2dof']:.2f}), b(J¹) predicho {f['b_pred_J1']:.4g}")
        print("   aumentos de J¹/rhoV / ln4:", [f"{v:.4g}" for v in f["J1_increase_over_ln4"]])
        print("   resto/rhoV:", [f"{v:.4g}" for v in f["rest_over_rhoV"]])
        print("   exponente local de ΔS:", [f"{v:.3f}" for v in f["local_exponent_of_DeltaS"]])
    print("C4:", out["C4"])

    # figura: (Delta S)²/rho V frente a rho V, cono (MYZ y simulación)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, eps in zip(axes, EPS):
        sel = sorted((p for p in preds if p["eps"] == eps), key=lambda r: r["rhoV"])
        lam = np.array([r["rhoV"] for r in sel], float)
        ax.errorbar(lam, [r["variance"] / r["rhoV"] for r in sel], [r["variance_se"] / r["rhoV"] for r in sel],
                    fmt="o-", color=BLUE, label="predicción MYZ", markersize=4, linewidth=1.4, capsize=2)
        ax.plot(lam, [r["parts"]["J1"]["value"] / r["rhoV"] for r in sel], "--", color=INK_MUTED, label="sólo J¹", linewidth=1.2)
        ss = sorted((s for s in sims if s["eps"] == eps), key=lambda r: r["rhoV"])
        ax.errorbar([s["rhoV"] for s in ss], [s["variance"] / s["rhoV"] for s in ss], [s["variance_se"] / s["rhoV"] for s in ss],
                    fmt="s", color=ORANGE, label="sprinklings", markersize=5, capsize=2)
        f = out["fits"][str(eps)]
        xx = np.geomspace(lam[0], lam[-1], 100)
        ax.plot(xx, f["c"] + f["b"] * np.log(xx), ":", color=BLUE, linewidth=1, label=f"c + b ln ρV, b = {f['b']:.3g}")
        ax.set_xscale("log")
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("ρV", color=INK)
        ax.set_ylabel("(ΔS)² / ρV", color=INK)
        ax.set_title(f"cono pasado truncado 2D, ε = {eps:g}  (Σf² = {f['b_pred_J1']:.3g})", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    path = ROOT / "figures" / "cone_fluctuations.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figura", path)


if __name__ == "__main__":
    {"sim": sim, "pred": pred, "fit": fit}[sys.argv[1]]()
