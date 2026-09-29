"""Goal 3: fluctuaciones (Delta S)^2 de la acción BD LOCAL, simulación frente al
formalismo de Moradi-Yazdi-Zilhão (arXiv:2407.03395).

Fase "run" (lenta, guarda figures/goal3_results.json):
  - Minkowski 2D: predicción exacta (|A∩B| en forma cerrada) vs simulación,
    para varios rho V.
  - Minkowski 4D (preliminar): predicción "plugin" (|A∩B| estimado con K
    puntos) para K = 256 y 1024, vs simulación.
  - de Sitter 2D (exploratorio): predicción plugin vs simulación.
Fase "plot": figura figures/goal3_fluctuations.png a partir del JSON.

Uso:  python scripts/goal3_fluctuations.py run   |   python scripts/goal3_fluctuations.py plot
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

from causets.continuum import expected_abundance  # noqa: E402
from causets.fluctuations import FluctuationKernel, ensemble_fluctuations  # noqa: E402
from causets.geometry import DeSitterGeometry, MinkowskiDiamondGeometry  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = ROOT / "figures" / "goal3_results.json"
COLORS = {"2D": "#2a78d6", "4D": "#eb6834", "dS2D": "#1baf7a"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _row(label, lam, P, E, t_pred, t_sim, **extra):
    z = (E.variance - P.variance) / np.hypot(E.variance_se, P.variance_se)
    row = dict(label=label, rhoV=lam, pred=P.variance, pred_se=P.variance_se, sim=E.variance,
               sim_se=E.variance_se, z=z, n_sprinklings=len(E.actions), t_pred=t_pred, t_sim=t_sim, **extra)
    print(json.dumps(row), flush=True)
    return row


def run(only_curved: bool = False):
    rows = json.loads(RESULTS.read_text()) if only_curved else []
    rows = [r for r in rows if r["label"] == "2D"]
    # --- Minkowski 2D, exacto --------------------------------------------
    g = MinkowskiDiamondGeometry(2)
    for lam in (() if only_curved else (25, 50, 100, 200, 400)):
        rho = lam / g.volume
        t0 = time.time()
        K = FluctuationKernel(g, rho, 3, np.random.default_rng(1000 + lam))
        P = K.predict(m_terms=1_000_000, exact_means=[expected_abundance(2, lam, k) for k in range(4)], batch=100_000)
        t1 = time.time()
        E = ensemble_fluctuations(g, rho, 4000, np.random.default_rng(2000 + lam))
        rows.append(_row("2D", lam, P, E, t1 - t0, time.time() - t1, overlap="exact"))
        RESULTS.write_text(json.dumps(rows, indent=1))

    # --- de Sitter 2D, plugin (exploratorio) ------------------------------
    gd = DeSitterGeometry(2, -4.0, -0.3)
    for lam in (50, 100):
        rho = lam / gd.volume
        t0 = time.time()
        K = FluctuationKernel(gd, rho, 3, np.random.default_rng(3000 + lam), overlap="plugin", overlap_points=256)
        P = K.predict(m_terms=300_000, m_means=1_000_000, batch=50_000)
        t1 = time.time()
        E = ensemble_fluctuations(gd, rho, 4000, np.random.default_rng(4000 + lam))
        rows.append(_row("dS2D", lam, P, E, t1 - t0, time.time() - t1, overlap="plugin", K=256))
        RESULTS.write_text(json.dumps(rows, indent=1))

    # --- Minkowski 4D, plugin (preliminar) --------------------------------
    g4 = MinkowskiDiamondGeometry(4)
    lam = 50
    rho = lam / g4.volume
    E = ensemble_fluctuations(g4, rho, 6000, np.random.default_rng(5000))
    for Kp in (256, 1024):
        t0 = time.time()
        K = FluctuationKernel(g4, rho, 4, np.random.default_rng(6000 + Kp), overlap="plugin", overlap_points=Kp)
        P = K.predict(m_terms=100_000, exact_means=[expected_abundance(4, lam, k) for k in range(5)], batch=20_000)
        rows.append(_row("4D", lam, P, E, time.time() - t0, 0.0, overlap="plugin", K=Kp))
        RESULTS.write_text(json.dumps(rows, indent=1))


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)


def plot():
    rows = json.loads(RESULTS.read_text())
    two = [r for r in rows if r["label"] == "2D"]
    lam = np.array([r["rhoV"] for r in two])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)

    ax1.errorbar(lam, [r["pred"] for r in two], yerr=[r["pred_se"] for r in two], fmt="-", color=COLORS["2D"],
                 linewidth=2, capsize=3, label="predicción MYZ (exacta en 2D)")
    ax1.errorbar(lam, [r["sim"] for r in two], yerr=[r["sim_se"] for r in two], fmt="o", color=COLORS["2D"],
                 markerfacecolor="white", markersize=6, capsize=3, label="simulación (4000 sprinklings)")
    ax1.plot(lam, lam, color=INK_MUTED, linestyle="--", linewidth=1, label="ρV (sólo el término de Poisson de N)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(lam, [str(int(v)) for v in lam])
    ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    _style(ax1)
    ax1.set_xlabel("ρV (número medio de elementos)", color=INK)
    ax1.set_ylabel("(ΔS)²", color=INK)
    ax1.set_title("Acción BD local, diamante de Minkowski 2D", color=INK, fontsize=11)
    ax1.legend(frameon=False, labelcolor=INK)

    labels, zs, cols = [], [], []
    for r in rows:
        tag = r["label"] + f" ρV={int(r['rhoV'])}" + (f" K={r['K']}" if "K" in r else "")
        labels.append(tag)
        zs.append(r["z"])
        cols.append(COLORS[r["label"]])
    y = np.arange(len(rows))[::-1]
    ax2.axvspan(-4, 4, color=GRID, alpha=0.6, linewidth=0)
    ax2.axvline(0, color=INK_MUTED, linewidth=1)
    ax2.scatter(zs, y, color=cols, s=40, zorder=3)
    ax2.set_yticks(y, labels)
    _style(ax2)
    ax2.set_xlabel("z = (sim − pred) / σ_combinada   (banda: |z| < 4)", color=INK)
    ax2.set_title("Acuerdo simulación–predicción", color=INK, fontsize=11)

    path = ROOT / "figures" / "goal3_fluctuations.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print(f"figura guardada en {path}")


if __name__ == "__main__":
    {"run": run, "run_curved": lambda: run(True), "plot": plot}[sys.argv[1] if len(sys.argv) > 1 else "plot"]()
