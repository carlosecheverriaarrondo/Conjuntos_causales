"""Fluctuaciones (Delta S)^2 de la acción BD SUAVIZADA (eps < 1) y de la
acción local en 4D con |A∩B| exacto (cuadratura de ``causets/overlap.py``):
simulación frente al formalismo de Moradi-Yazdi-Zilhão (arXiv:2407.03395)
extendido a abundancias adelgazadas (``causets/fluctuations.py``).

Fases (cada una guarda su propio JSON en figures/, se pueden lanzar en
paralelo):
  run2d  Minkowski 2D, eps = 0.5 y 0.25, rho V = 25..400 (|A∩B| exacto).
         Los valores con eps = 1 se toman de figures/goal3_results.json.
  run4d  Minkowski 4D, acción local, rho V = 25, 50, 100 con la cuadratura
         exacta de |A∩B| (sustituye a la predicción "plugin" del Goal 3).
  plot   figura figures/smeared_fluctuations.png a partir de los JSON.

Uso:  python scripts/smeared_fluctuations.py run2d | run4d | plot
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

from causets.fluctuations import FluctuationKernel, ensemble_fluctuations, expected_thinned_abundance  # noqa: E402
from causets.geometry import MinkowskiDiamondGeometry  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS_2D = ROOT / "figures" / "smeared_results_2d.json"
RESULTS_4D = ROOT / "figures" / "smeared_results_4d.json"
GOAL3 = ROOT / "figures" / "goal3_results.json"
LAMS_2D = (25, 50, 100, 200, 400)
EPS_2D = {0.5: 4_000_000, 0.25: 10_000_000}  # eps -> muestras MC por término
LAMS_4D = {50: 1_000_000, 25: 500_000, 100: 500_000}
COLORS = {1.0: "#2a78d6", 0.5: "#eb6834", 0.25: "#1baf7a", "4D": "#8a4fd6"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _row(label, lam, eps, P, E, t_pred, t_sim, **extra):
    z = (E.variance - P.variance) / np.hypot(E.variance_se, P.variance_se)
    row = dict(label=label, rhoV=lam, eps=eps, pred=P.variance, pred_se=P.variance_se, sim=E.variance,
               sim_se=E.variance_se, z=z, n_sprinklings=len(E.actions), t_pred=t_pred, t_sim=t_sim, **extra)
    print(json.dumps(row), flush=True)
    return row


def run2d():
    rows = []
    g = MinkowskiDiamondGeometry(2)
    for eps, m in EPS_2D.items():
        for lam in LAMS_2D:
            rho = lam / g.volume
            seed = int(1000 * eps) * 1000 + lam
            t0 = time.time()
            K = FluctuationKernel(g, rho, 3, np.random.default_rng(seed), eps=eps)
            P = K.predict(m_terms=m, exact_means=[expected_thinned_abundance(2, lam, k, eps) for k in range(4)], batch=100_000)
            t1 = time.time()
            E = ensemble_fluctuations(g, rho, 4000, np.random.default_rng(seed + 1), eps=eps)
            rows.append(_row("2D", lam, eps, P, E, t1 - t0, time.time() - t1, overlap="exact", m_terms=m))
            RESULTS_2D.write_text(json.dumps(rows, indent=1))


def run4d():
    rows = []
    g = MinkowskiDiamondGeometry(4)
    for lam, m in LAMS_4D.items():
        rho = lam / g.volume
        t0 = time.time()
        K = FluctuationKernel(g, rho, 4, np.random.default_rng(7000 + lam), overlap="exact")
        P = K.predict(m_terms=m, exact_means=[expected_thinned_abundance(4, lam, k) for k in range(5)], batch=20_000)
        t1 = time.time()
        E = ensemble_fluctuations(g, rho, 6000, np.random.default_rng(8000 + lam))
        rows.append(_row("4D", lam, 1.0, P, E, t1 - t0, time.time() - t1, overlap="exact", m_terms=m))
        RESULTS_4D.write_text(json.dumps(rows, indent=1))


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)


def _fit_exponent(lam, var, se):
    """Ajuste ponderado log(var) = 2 p log(lam) + c; devuelve (p, se_p) con
    ΔS ∝ (rho V)^p."""
    x, y, w = np.log(lam), np.log(var), (var / se) ** 2
    (slope, _), cov = np.polyfit(x, y, 1, w=np.sqrt(w), cov="unscaled")
    return slope / 2, np.sqrt(cov[0, 0]) / 2


def plot():
    rows = [dict(r, eps=1.0) for r in json.loads(GOAL3.read_text()) if r["label"] == "2D"]
    rows += json.loads(RESULTS_2D.read_text()) if RESULTS_2D.exists() else []
    rows4 = sorted(json.loads(RESULTS_4D.read_text()), key=lambda r: r["rhoV"]) if RESULTS_4D.exists() else []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    fits = {}
    for eps in (1.0, 0.5, 0.25):
        sel = sorted((r for r in rows if r["eps"] == eps), key=lambda r: r["rhoV"])
        if not sel:
            continue
        lam = np.array([r["rhoV"] for r in sel], float)
        sim, sim_se = np.array([r["sim"] for r in sel]), np.array([r["sim_se"] for r in sel])
        p, p_se = _fit_exponent(lam, sim, sim_se)
        fits[eps] = (p, p_se)
        c = COLORS[eps]
        ax1.errorbar(lam, [r["pred"] for r in sel], yerr=[r["pred_se"] for r in sel], fmt="-", color=c, linewidth=2,
                     capsize=3, label=f"ε = {eps:g}: predicción  (ΔS ∝ (ρV)^{p:.2f}±{p_se:.2f})")
        ax1.errorbar(lam, sim, yerr=sim_se, fmt="o", color=c, markerfacecolor="white", markersize=6, capsize=3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(LAMS_2D, [str(v) for v in LAMS_2D])
    ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    _style(ax1)
    ax1.set_xlabel("ρV (número medio de elementos)", color=INK)
    ax1.set_ylabel("(ΔS)²", color=INK)
    ax1.set_title("Acción BD suavizada, Minkowski 2D (líneas: MYZ; círculos: simulación)", color=INK, fontsize=10)
    ax1.legend(frameon=False, labelcolor=INK, fontsize=8)

    allrows = rows + rows4
    labels = [f"{r['label']} ε={r['eps']:g} ρV={int(r['rhoV'])}" for r in allrows]
    y = np.arange(len(allrows))[::-1]
    ax2.axvspan(-4, 4, color=GRID, alpha=0.6, linewidth=0)
    ax2.axvline(0, color=INK_MUTED, linewidth=1)
    ax2.scatter([r["z"] for r in allrows], y, color=[COLORS["4D"] if r["label"] == "4D" else COLORS[r["eps"]] for r in allrows],
                s=40, zorder=3)
    ax2.set_yticks(y, labels, fontsize=8)
    _style(ax2)
    ax2.set_xlabel("z = (sim − pred) / σ_combinada   (banda: |z| < 4)", color=INK)
    ax2.set_title("Acuerdo simulación–predicción (4D: |A∩B| por cuadratura)", color=INK, fontsize=10)

    path = ROOT / "figures" / "smeared_fluctuations.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print(f"figura guardada en {path}")
    for eps, (p, p_se) in fits.items():
        print(f"eps={eps:g}: ΔS ∝ (rho V)^{p:.3f} ± {p_se:.3f}")


if __name__ == "__main__":
    {"run2d": run2d, "run4d": run4d, "plot": plot}[sys.argv[1] if len(sys.argv) > 1 else "plot"]()
