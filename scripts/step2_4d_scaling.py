"""Paso 2: escalado de (Delta S)^2 de la acción BD local en Minkowski 4D.

Predicción (derivada antes de calcular, ver README "Paso 2"): en d dimensiones
    <sum_pares h(n)> ∝ (rho V)^(2-2/d) sum_n h(n) Γ(n+2/d)/n!,
así que en 4D el término "mismo par" J¹ = sum_k a_k² <N_k> crece como
(rho V)^1.5 (p = 0.75) y el término cruzado con N no crece (sum_k a_k Γ(k+½)/k! = 0).
J¹ se calcula exactamente (``continuum.expected_abundance``); la varianza
total se mide sobre sprinklings, y la diferencia es la suma de los términos
de correlación (J², J³, J⁴, J⁶ de MYZ).

Fases:
  sim   rho V = 200 .. 3200 -> figures/step2_4d_sim.json (incremental)
  extra segunda tanda: 1500 más con rho V = 3200 y 500 con 6400
        -> figures/step2_4d_sim_extra.json
  fit   comparación con J¹ exacto, figura figures/step2_4d_scaling.png y
        resumen figures/step2_4d_results.json

Uso:  python scripts/step2_4d_scaling.py sim | fit
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
from causets.continuum import expected_abundance  # noqa: E402
from causets.fluctuations import EnsembleFluctuations, abundance_vector, local_action_coefficients  # noqa: E402
from causets.geometry import MinkowskiDiamondGeometry  # noqa: E402

warnings.filterwarnings("ignore")
SIM = ROOT / "figures" / "step2_4d_sim.json"
SIM_EXTRA = ROOT / "figures" / "step2_4d_sim_extra.json"
FIT = ROOT / "figures" / "step2_4d_results.json"
PLAN = {200: 2000, 400: 2000, 800: 2000, 1600: 1000, 3200: 500}  # rho V -> sprinklings
PLAN_EXTRA = {3200: 1500, 6400: 500}  # segunda tanda (semillas distintas)
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def exact_parts(lam: float) -> dict:
    """NN, cruzado N-k y J¹ exactos (acción local 4D)."""
    a = local_action_coefficients(4)
    aN, ak = a[0], a[1:]
    L = np.array([expected_abundance(4, lam, k) for k in range(5)])
    j = np.arange(4)
    MNk = (j + 2) * L[:4] - (j + 1) * L[1:]
    return dict(NN=aN**2 * lam, Nk=2 * aN * float(ak @ MNk), J1=float(ak**2 @ L[:4]))


def sim(plan=PLAN, out=SIM, seed=71_000):
    g = MinkowskiDiamondGeometry(4)
    a = local_action_coefficients(4)
    rows = []
    for lam, m in plan.items():
        rng = np.random.default_rng(seed + lam)
        t0 = time.time()
        ab = [abundance_vector(minkowski_causal_matrix(sprinkle(g.region, lam / g.volume, rng)), 4) for _ in range(m)]
        E = EnsembleFluctuations(np.array(ab), a)
        row = dict(rhoV=lam, sim=E.variance, sim_se=E.variance_se, n_sprinklings=m, t=time.time() - t0)
        print(json.dumps(row), flush=True)
        rows.append(row)
        out.write_text(json.dumps(rows, indent=1))


def _combine(rows):
    """Une tandas de la misma densidad: media de las varianzas ponderada por el
    número de sprinklings (errores combinados en consecuencia)."""
    by = {}
    for r in rows:
        by.setdefault(r["rhoV"], []).append(r)
    out = []
    for lam, rs in sorted(by.items()):
        n = np.array([r["n_sprinklings"] for r in rs], float)
        w = n / n.sum()
        out.append(dict(rhoV=lam, sim=float(w @ [r["sim"] for r in rs]),
                        sim_se=float(np.sqrt(w**2 @ np.array([r["sim_se"] for r in rs]) ** 2)), n_sprinklings=int(n.sum())))
    return out


def fit():
    old = [dict(rhoV=r["rhoV"], sim=r["sim"], sim_se=r["sim_se"], n_sprinklings=r["n_sprinklings"])
           for r in json.loads((ROOT / "figures" / "smeared_results_4d.json").read_text())]
    extra = json.loads(SIM_EXTRA.read_text()) if SIM_EXTRA.exists() else []
    rows = _combine(old + json.loads(SIM.read_text()) + extra)
    for r in rows:
        r.update(exact_parts(r["rhoV"]))
        r["rest"] = r["sim"] - r["NN"] - r["Nk"] - r["J1"]
        r["rest_over_J1"] = r["rest"] / r["J1"]
        r["rest_over_J1_se"] = r["sim_se"] / r["J1"]
    lam = np.array([r["rhoV"] for r in rows], float)
    # exponentes locales (pares consecutivos) de la simulación y de J¹
    p_sim = [dict(rhoV=(rows[i]["rhoV"], rows[i + 1]["rhoV"]),
                  p=float(np.log(rows[i + 1]["sim"] / rows[i]["sim"]) / np.log(lam[i + 1] / lam[i]) / 2),
                  se=float(np.hypot(rows[i]["sim_se"] / rows[i]["sim"], rows[i + 1]["sim_se"] / rows[i + 1]["sim"])
                           / np.log(lam[i + 1] / lam[i]) / 2)) for i in range(len(rows) - 1)]
    lam_j = np.geomspace(25, 1e6, 25)
    j1 = np.array([exact_parts(x)["J1"] for x in lam_j])
    p_j1 = np.log(j1[1:] / j1[:-1]) / np.log(lam_j[1:] / lam_j[:-1]) / 2
    FIT.write_text(json.dumps(dict(points=rows, p_local_sim=p_sim,
                                   J1_curve=dict(rhoV=lam_j.tolist(), J1=j1.tolist())), indent=1))
    for r in rows:
        print(f"ρV={r['rhoV']:<5} total/ρV={r['sim']/r['rhoV']:9.1f}±{r['sim_se']/r['rhoV']:<6.1f} J1/ρV={r['J1']/r['rhoV']:9.1f}"
              f"  resto/J1={r['rest_over_J1']:+.3f}±{r['rest_over_J1_se']:.3f}")
    for p in p_sim:
        print(f"p_local sim {p['rhoV']}: {p['p']:.3f} ± {p['se']:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax1.plot(lam_j, j1 / lam_j, "-", color=BLUE, linewidth=2, label="J¹ exacto (mismo par)")
    ax1.errorbar(lam, [r["sim"] / r["rhoV"] for r in rows], yerr=[r["sim_se"] / r["rhoV"] for r in rows], fmt="o",
                 color=INK, markerfacecolor="white", capsize=3, label="(ΔS)² total, simulación")
    ref = lam_j[lam_j >= 1e4]
    ax1.plot(ref, j1[-1] / lam_j[-1] * (ref / lam_j[-1]) ** 0.5, ":", color=INK_MUTED, label="pendiente p = 0.75")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("ρV", color=INK)
    ax1.set_ylabel("(ΔS)² / ρV", color=INK)
    ax1.set_title("Minkowski 4D, acción BD local", color=INK, fontsize=10)
    ax1.legend(frameon=False, fontsize=8)
    mid = np.sqrt(lam_j[1:] * lam_j[:-1])
    ax2.plot(mid, p_j1, "-", color=BLUE, linewidth=2, label="J¹ exacto")
    ax2.errorbar([np.sqrt(p["rhoV"][0] * p["rhoV"][1]) for p in p_sim], [p["p"] for p in p_sim],
                 yerr=[p["se"] for p in p_sim], fmt="o", color=INK, markerfacecolor="white", capsize=3, label="total, simulación")
    ax2.axhline(0.75, color=BLUE, linestyle=":", linewidth=1.2)
    ax2.axhline(0.5, color=ORANGE, linestyle=":", linewidth=1.2)
    ax2.text(1.2e5, 0.76, "0.75 (asintótico J¹)", color=BLUE, fontsize=8)
    ax2.text(1.2e5, 0.51, "0.5 (Poisson)", color=ORANGE, fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("ρV", color=INK)
    ax2.set_ylabel("exponente local p", color=INK)
    ax2.set_title("Exponente local de ΔS ∝ (ρV)^p", color=INK, fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    path = ROOT / "figures" / "step2_4d_scaling.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figura", path)


if __name__ == "__main__":
    {"sim": sim, "extra": lambda: sim(PLAN_EXTRA, SIM_EXTRA, 81_000), "fit": fit}[sys.argv[1]]()
