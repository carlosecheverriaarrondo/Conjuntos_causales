"""Punto 1b: fluctuaciones de la acción BD en el cono de luz pasado hasta el
big bang de un FLRW plano 2D, a(eta) = eta^alpha (``FLRWConeGeometry``),
con alpha = 1 y 2. Es la región de Everpresent Λ.

Hipótesis (formulada ANTES de calcular): en coordenadas conformes un par casi
nulo con lados Δu, Δv en eta tiene volumen físico ≈ a(eta)² Δu Δv / 2. A
volumen físico mu fijo, el rango de rapidez es ln(rho a(eta)² ... / mu): el
coeficiente de ln rho sigue siendo rho ∫ dV sum h, y a(eta) sólo aporta
rho ∫ ln a² dV a la constante (finito para potencias de eta). Predicción:
(Delta S)² = b rho V ln(rho V) + O(rho V) con el MISMO b que en Minkowski
(24 por J¹, 24.43 ± 0.24 total en el diamante). El riesgo es el big bang
(a -> 0): si el corte de rapidez cambiara allí, b cambiaría.

Criterios de éxito (fijados antes de ver los resultados), eps = 1:
  F1  volúmenes de intervalos e intersecciones en forma cerrada frente a
      integración numérica: relativo 1e-9 (test).
  F2  <N_k> (MC del formalismo) frente a sprinklings: |z| < 4.
  F3  predicción MYZ frente a la varianza de los sprinklings, alpha = 1, 2 y
      rho V = 50, 200, 800, 3200: todos los puntos con |z| < 4.
  F4  ajuste (Delta S)²/rho V = c + b ln(rho V) de las predicciones MYZ
      (rho V = 50 .. 12800), para cada alpha: b compatible a 4σ con el del
      diamante (24.43 ± 0.24).

Fases:
  sim      sprinklings -> figures/flrw_sim.json
  pred     predicción MYZ (medias por MC) -> figures/flrw_pred.json
  fit      criterios, ajustes y figura -> figures/flrw_results.json,
           figures/flrw_fluctuations.png

Uso:  python scripts/flrw_fluctuations.py sim | pred | fit
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
from causets.fluctuations import (  # noqa: E402
    EnsembleFluctuations,
    FluctuationKernel,
    action_coefficients,
    thinned_abundance_vector,
)
from causets.geometry import FLRWConeGeometry  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from goal3d_log_origin import decompose, fit_log  # noqa: E402

warnings.filterwarnings("ignore")
SIM = ROOT / "figures" / "flrw_sim.json"
PRED = ROOT / "figures" / "flrw_pred.json"
RESULTS = ROOT / "figures" / "flrw_results.json"
ALPHAS = (1.0, 2.0)
SIM_LAMS = {50: 4000, 200: 4000, 800: 3000, 3200: 2000}  # rho V: sprinklings
PRED_LAMS = (50, 200, 800, 3200, 12800)
M_TERMS, M_MEANS = 4_000_000, 4_000_000
DIAMOND_B = (24.43, 0.24)  # b total del diamante de Minkowski, eps = 1 (Goal 3d)
B_J1 = 24.0  # sum_k a_k²
K_SIGMA = 4.0
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def sim():
    rows = []
    for alpha in ALPHAS:
        g = FLRWConeGeometry(alpha)
        for lam, n in SIM_LAMS.items():
            t0 = time.time()
            rng = np.random.default_rng(91_000 + lam + int(10 * alpha))
            ab = [thinned_abundance_vector(minkowski_causal_matrix(sprinkle(g.region, lam / g.volume, rng)), 3, 1.0)
                  for _ in range(n)]
            E = EnsembleFluctuations(np.array(ab), action_coefficients(2))
            A = E.abundances
            row = dict(alpha=alpha, rhoV=lam, n=n, variance=E.variance, variance_se=E.variance_se,
                       means=A.mean(axis=0).tolist(), means_se=(A.std(axis=0, ddof=1) / np.sqrt(n)).tolist(),
                       t=time.time() - t0)
            print(json.dumps(row), flush=True)
            rows.append(row)
            SIM.write_text(json.dumps(rows, indent=1))


def pred():
    rows = []
    for alpha in ALPHAS:
        g = FLRWConeGeometry(alpha)
        for lam in PRED_LAMS:
            t0 = time.time()
            K = FluctuationKernel(g, lam / g.volume, 3, np.random.default_rng(97_000 + lam + int(10 * alpha)))
            P = K.predict(m_terms=M_TERMS, m_means=M_MEANS, batch=100_000)
            L, L_se = K._batched_means(M_MEANS, 4, 100_000)  # medias independientes para F2
            parts = decompose(P, 1.0)
            total = sum(p["value"] for p in parts.values())
            row = dict(alpha=alpha, rhoV=lam, variance=P.variance, variance_se=P.variance_se, parts=parts,
                       means=P.abundance_means[1:5].tolist(), means_check=L.tolist(), means_check_se=L_se.tolist(),
                       closure=total - P.variance, t=time.time() - t0)
            print(json.dumps(row), flush=True)
            rows.append(row)
            PRED.write_text(json.dumps(rows, indent=1))


def fit():
    sims, preds = json.loads(SIM.read_text()), json.loads(PRED.read_text())
    out = {"F2": [], "F3": [], "fits": {}}
    for s in sims:
        p = next(p for p in preds if p["rhoV"] == s["rhoV"] and p["alpha"] == s["alpha"])
        z = [(m - l) / np.hypot(se, lse) for m, se, l, lse in
             zip(s["means"][1:], s["means_se"][1:], p["means_check"][:3], p["means_check_se"][:3])]
        out["F2"].append(dict(alpha=s["alpha"], rhoV=s["rhoV"], z=z))
        z3 = (s["variance"] - p["variance"]) / np.hypot(s["variance_se"], p["variance_se"])
        out["F3"].append(dict(alpha=s["alpha"], rhoV=s["rhoV"], sim=s["variance"], sim_se=s["variance_se"],
                              pred=p["variance"], pred_se=p["variance_se"], z=z3))
    for alpha in ALPHAS:
        sel = sorted((p for p in preds if p["alpha"] == alpha), key=lambda r: r["rhoV"])
        lam = np.array([r["rhoV"] for r in sel], float)
        y = np.array([r["variance"] for r in sel]) / lam
        se = np.array([r["variance_se"] for r in sel]) / lam
        b, b_se, c, chi2 = fit_log(lam, y, se)
        j1 = np.array([r["parts"]["J1"]["value"] for r in sel]) / lam
        p_loc = (np.diff(np.log(y * lam)) / np.diff(np.log(lam)) / 2).tolist()
        out["fits"][str(alpha)] = dict(b=b, b_se=b_se, c=c, chi2dof=chi2,
                                       J1_increase_over_ln=(np.diff(j1) / np.diff(np.log(lam))).tolist(),
                                       J6c_over_rhoV=[r["parts"]["J6c"]["value"] / r["rhoV"] for r in sel],
                                       local_exponent_of_DeltaS=p_loc,
                                       z_vs_diamond=(b - DIAMOND_B[0]) / np.hypot(b_se, DIAMOND_B[1]))
    RESULTS.write_text(json.dumps(out, indent=1))

    print("F2 (|z| de <N_k>):", max(max(abs(v) for v in r["z"]) for r in out["F2"]))
    for r in out["F3"]:
        print(f"F3 alpha={r['alpha']:<3} rhoV={r['rhoV']:>5}: sim {r['sim']:.5g} ± {r['sim_se']:.2g}   "
              f"pred {r['pred']:.5g} ± {r['pred_se']:.2g}   z = {r['z']:+.2f}")
    for alpha, f in out["fits"].items():
        print(f"alpha={alpha}: b = {f['b']:.4g} ± {f['b_se']:.2g} (chi2/dof {f['chi2dof']:.2f}), z frente al diamante {f['z_vs_diamond']:+.2f}")
        print("   aumentos de J¹/rhoV por unidad de ln rhoV:", [f"{v:.4g}" for v in f["J1_increase_over_ln"]])
        print("   J6c/rhoV:", [f"{v:.4g}" for v in f["J6c_over_rhoV"]])
        print("   exponente local de ΔS:", [f"{v:.3f}" for v in f["local_exponent_of_DeltaS"]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, alpha in zip(axes, ALPHAS):
        sel = sorted((p for p in preds if p["alpha"] == alpha), key=lambda r: r["rhoV"])
        lam = np.array([r["rhoV"] for r in sel], float)
        ax.errorbar(lam, [r["variance"] / r["rhoV"] for r in sel], [r["variance_se"] / r["rhoV"] for r in sel],
                    fmt="o-", color=BLUE, label="predicción MYZ", markersize=4, linewidth=1.4, capsize=2)
        ax.plot(lam, [r["parts"]["J1"]["value"] / r["rhoV"] for r in sel], "--", color=INK_MUTED, label="sólo J¹", linewidth=1.2)
        ss = sorted((s for s in sims if s["alpha"] == alpha), key=lambda r: r["rhoV"])
        ax.errorbar([s["rhoV"] for s in ss], [s["variance"] / s["rhoV"] for s in ss], [s["variance_se"] / s["rhoV"] for s in ss],
                    fmt="s", color=ORANGE, label="sprinklings", markersize=5, capsize=2)
        f = out["fits"][str(alpha)]
        xx = np.geomspace(lam[0], lam[-1], 100)
        ax.plot(xx, f["c"] + f["b"] * np.log(xx), ":", color=BLUE, linewidth=1, label=f"c + b ln ρV, b = {f['b']:.3g}")
        ax.set_xscale("log")
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("ρV", color=INK)
        ax.set_ylabel("(ΔS)² / ρV", color=INK)
        ax.set_title(f"cono pasado hasta el big bang, FLRW 2D, a ∝ η^{alpha:g}", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    path = ROOT / "figures" / "flrw_fluctuations.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figura", path)


if __name__ == "__main__":
    {"sim": sim, "pred": pred, "fit": fit}[sys.argv[1]]()
