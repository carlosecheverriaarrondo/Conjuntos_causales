"""Goal 4b-2: Λ(t) estocástico con retroacción, Everpresent Λ (Model 1) frente
a las fluctuaciones de la acción BD (``causets/stochastic_lambda.py``).

Criterios (fijados antes de calcular):
  C1  validación: Λ constante reproduce ΛCDM (h, Omega_Λ, d_L, E(z)) con error
      relativo < 1e-3; V(a=1) frente a ``everpresent.past_cone_volume`` < 1e-3.
  C2  validación: el proceso OU da corr = (V1/V2)^c (4σ).
  Comparación (misma para los dos modelos): fracción de realizaciones "tipo
  ΛCDM" (``stochastic_lambda.lcdm_like``: Omega_Λ0 en [0.6, 0.8], h en
  [0.6, 0.8], forma de mu(z) a < 0.1 mag de ΛCDM en 0.01 < z < 2.3,
  |E/E_ΛCDM - 1| < 5 % en z = 0.5, 1, 2) frente a la amplitud.
  No es un ajuste a datos reales: compara los dos modelos con el mismo
  criterio.

Uso:  python scripts/goal4b_lambda.py run | plot
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causets import stochastic_lambda as sl  # noqa: E402

OUT = ROOT / "figures" / "goal4b_lambda.json"
N_REAL = 2000
ALPHAS = (0.003, 0.005, 0.008, 0.012, 0.02, 0.03)
BD_LAMS = (0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0)
BETAS = (0.7, 0.85, 1.0)
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREEN, BLUE, ORANGE, PURPLE = "#1baf7a", "#2a78d6", "#eb6834", "#8a4fd6"


def summarize(real, **meta):
    ok = sl.lcdm_like(real)
    good = ~real.collapsed
    n = len(ok)
    frac = ok.mean()
    q = lambda x, p: float(np.nanpercentile(x[good], p)) if good.any() else float("nan")  # noqa: E731
    return dict(**meta, n=n, frac_lcdm_like=float(frac), frac_se=float(np.sqrt(frac * (1 - frac) / n)),
                frac_collapsed=float(real.collapsed.mean()),
                omega_l_pct=[q(real.omega_lambda0, p) for p in (5, 16, 50, 84, 95)],
                h_pct=[q(real.h, p) for p in (5, 16, 50, 84, 95)],
                frac_omega_l_in_range=float(np.mean(good & (real.omega_lambda0 > 0.6) & (real.omega_lambda0 < 0.8))))


def run():
    rows = []
    rng = np.random.default_rng(141_000)
    for a in ALPHAS:
        t0 = time.time()
        r = summarize(sl.evolve(sl.model1(a), N_REAL, rng), model="Model 1", alpha=a, lam=8 * np.pi * a)
        r["t"] = time.time() - t0
        print(json.dumps(r), flush=True)
        rows.append(r)
    for beta in BETAS:
        for lam in BD_LAMS:
            t0 = time.time()
            r = summarize(sl.evolve(sl.bd_process(lam, beta), N_REAL, rng), model="BD", beta=beta, lam=lam,
                          l_k_m=sl.bd_lambda_to_lk(lam))
            r["t"] = time.time() - t0
            print(json.dumps(r), flush=True)
            rows.append(r)
    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def plot():
    rows = json.loads(OUT.read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    m1 = [r for r in rows if r["model"] == "Model 1"]
    ax1.errorbar([r["alpha"] for r in m1], [r["frac_lcdm_like"] for r in m1], [r["frac_se"] for r in m1],
                 fmt="o-", color=GREEN, capsize=3, label="Model 1 (DNY)")
    ax1.set_xscale("log")
    ax1.set_xlabel("α (Model 1)", color=INK)
    ax1.set_ylabel("fracción de realizaciones tipo ΛCDM", color=INK)
    ax1.set_title("Everpresent Λ, Model 1", color=INK, fontsize=10)
    for beta, col in zip(BETAS, (PURPLE, ORANGE, BLUE)):
        bd = [r for r in rows if r["model"] == "BD" and r["beta"] == beta]
        ax2.errorbar([r["lam"] for r in bd], [r["frac_lcdm_like"] for r in bd], [r["frac_se"] for r in bd],
                     fmt="o-", color=col, capsize=3, label=f"acción BD, β = {beta:g}")
    ax2.set_xscale("log")
    ax2.set_xlabel("λ (amplitud de la acción BD; λ ↔ ℓ_k)", color=INK)
    ax2.set_title("Fluctuaciones de la acción BD", color=INK, fontsize=10)
    top = ax2.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    ticks = [0.3, 1.0, 6.0]
    top.set_xticks(ticks)
    top.set_xticklabels([f"{sl.bd_lambda_to_lk(t):.1e} m" for t in ticks], fontsize=7)
    top.set_xlabel("ℓ_k (ℓ = ℓ_p, r = 0.33)", color=INK_MUTED, fontsize=8)
    for ax in (ax1, ax2):
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    path = ROOT / "figures" / "goal4b_lambda.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figura", path)


if __name__ == "__main__":
    {"run": run, "plot": plot}[sys.argv[1]]()
