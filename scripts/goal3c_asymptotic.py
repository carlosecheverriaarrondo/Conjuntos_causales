"""Goal 3c: escalado de (Delta S)^2 de la acción BD suavizada en el régimen
eps rho V > 100 (Minkowski 2D), simulación frente al formalismo MYZ con
abundancias adelgazadas (``causets/fluctuations.py``).

Fases (cada una guarda su JSON en figures/; "sim" y "pred" se pueden lanzar
en paralelo):
  sim   rho V = 800, 1600, 3200; 2000 sprinklings por densidad. Las mismas
        configuraciones sirven para eps = 1, 0.5 y 0.25 (sólo cambia el
        adelgazamiento de las abundancias).
  pred  predicción MYZ para las mismas (eps, rho V).
  fit   z de cada punto, exponentes p (Delta S ∝ (rho V)^p) en el régimen
        bajo (rho V = 25..400, Goal 3 y su ampliación) y alto (800..3200), y
        figura figures/goal3c_asymptotic.png.

Reglas de decisión fijadas ANTES de ver los resultados:
  acuerdo sim-pred: |z| < 4 (sigma combinada);
  p compatible con 0.5: |p - 0.5| < 4 sigma_p;
  p estable entre regímenes: |p_alto - p_bajo| < 4 sigma combinada.

Uso:  python scripts/goal3c_asymptotic.py sim | pred | fit
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
    expected_thinned_abundance,
    thinned_abundance_vector,
)
from causets.geometry import MinkowskiDiamondGeometry  # noqa: E402

warnings.filterwarnings("ignore")
SIM = ROOT / "figures" / "goal3c_sim.json"
PRED = ROOT / "figures" / "goal3c_pred.json"
FIT = ROOT / "figures" / "goal3c_results.json"
LAMS = (800, 1600, 3200)
EPS = (1.0, 0.5, 0.25)
N_SPRINKLINGS = 2000
M_TERMS = {1.0: 2_000_000, 0.5: 8_000_000, 0.25: 20_000_000}
K_SIGMA = 4.0
COLORS = {1.0: "#2a78d6", 0.5: "#eb6834", 0.25: "#1baf7a"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def sim():
    g = MinkowskiDiamondGeometry(2)
    rows = []
    for lam in LAMS:
        rng = np.random.default_rng(31_000 + lam)
        t0 = time.time()
        ab = {eps: [] for eps in EPS}
        for _ in range(N_SPRINKLINGS):
            C = minkowski_causal_matrix(sprinkle(g.region, lam / g.volume, rng))
            for eps in EPS:
                ab[eps].append(thinned_abundance_vector(C, 3, eps))
        for eps in EPS:
            E = EnsembleFluctuations(np.array(ab[eps]), action_coefficients(2, eps))
            row = dict(rhoV=lam, eps=eps, sim=E.variance, sim_se=E.variance_se, mean_action=float(E.actions.mean()),
                       n_sprinklings=N_SPRINKLINGS, t_sim=time.time() - t0)
            print(json.dumps(row), flush=True)
            rows.append(row)
        SIM.write_text(json.dumps(rows, indent=1))


def pred():
    g = MinkowskiDiamondGeometry(2)
    rows = []
    for lam in LAMS:
        for eps in EPS:
            t0 = time.time()
            K = FluctuationKernel(g, lam / g.volume, 3, np.random.default_rng(41_000 + lam + int(100 * eps)), eps=eps)
            P = K.predict(m_terms=M_TERMS[eps], exact_means=[expected_thinned_abundance(2, lam, k, eps) for k in range(4)],
                          batch=100_000)
            row = dict(rhoV=lam, eps=eps, pred=P.variance, pred_se=P.variance_se, m_terms=M_TERMS[eps], t_pred=time.time() - t0)
            print(json.dumps(row), flush=True)
            rows.append(row)
            PRED.write_text(json.dumps(rows, indent=1))


def fit_exponent(lam, var, se):
    """Ajuste ponderado log(var) = 2p log(rho V) + c: devuelve (p, se_p)."""
    x, y, w = np.log(lam), np.log(var), var / se
    (slope, _), cov = np.polyfit(x, y, 1, w=w, cov="unscaled")
    return float(slope / 2), float(np.sqrt(cov[0, 0]) / 2)


def low_regime_rows():
    """Resultados previos rho V = 25..400 (Goal 3 y su ampliación)."""
    rows = [dict(r, eps=1.0) for r in json.loads((ROOT / "figures" / "goal3_results.json").read_text()) if r["label"] == "2D"]
    return rows + json.loads((ROOT / "figures" / "smeared_results_2d.json").read_text())


def merged_high_rows():
    preds = {(r["rhoV"], r["eps"]): r for r in json.loads(PRED.read_text())}
    out = []
    for r in json.loads(SIM.read_text()):
        p = preds.get((r["rhoV"], r["eps"]))
        if p is not None:
            z = (r["sim"] - p["pred"]) / np.hypot(r["sim_se"], p["pred_se"])
            out.append(dict(r, pred=p["pred"], pred_se=p["pred_se"], z=float(z)))
    return out


def fit():
    low, high = low_regime_rows(), merged_high_rows()
    summary = dict(points=high, exponents={})
    for eps in EPS:
        res = {}
        for name, rows in (("bajo", low), ("alto", high)):
            sel = sorted((r for r in rows if r["eps"] == eps), key=lambda r: r["rhoV"])
            if len(sel) < 2:
                continue
            lam = np.array([r["rhoV"] for r in sel], float)
            for src in ("sim", "pred"):
                p, se = fit_exponent(lam, np.array([r[src] for r in sel]), np.array([r[src + "_se"] for r in sel]))
                res[f"{name}_{src}"] = dict(p=p, se=se, rhoV=[int(v) for v in lam], z_vs_half=(p - 0.5) / se)
        if "bajo_sim" in res and "alto_sim" in res:
            a, b = res["bajo_sim"], res["alto_sim"]
            res["z_alto_vs_bajo"] = (b["p"] - a["p"]) / np.hypot(a["se"], b["se"])
        summary["exponents"][str(eps)] = res
    FIT.write_text(json.dumps(summary, indent=1))
    for r in high:
        print(f"eps={r['eps']:<5g} rhoV={r['rhoV']:<5d} pred={r['pred']:10.2f}±{r['pred_se']:<8.2f} "
              f"sim={r['sim']:10.2f}±{r['sim_se']:<8.2f} z={r['z']:+.2f}")
    for eps, res in summary["exponents"].items():
        for k, v in res.items():
            print(f"eps={eps} {k}: {v if not isinstance(v, dict) else f'p={v['p']:.3f}±{v['se']:.3f} (z vs 0.5: {v['z_vs_half']:+.1f})'}")
    _plot(low, high, summary)


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)


def _plot(low, high, summary):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for eps in EPS:
        c = COLORS[eps]
        for rows, filled in ((low, False), (high, True)):
            sel = sorted((r for r in rows if r["eps"] == eps), key=lambda r: r["rhoV"])
            lam = [r["rhoV"] for r in sel]
            ax1.errorbar(lam, [r["sim"] for r in sel], yerr=[r["sim_se"] for r in sel], fmt="o", color=c,
                         markerfacecolor=c if filled else "white", markersize=5, capsize=2)
            ax1.plot(lam, [r["pred"] for r in sel], "-", color=c, linewidth=1.6)
        ex = summary["exponents"][str(eps)]
        lab = f"ε = {eps:g}: p = {ex['bajo_sim']['p']:.2f} (ρV ≤ 400)"
        if "alto_sim" in ex:
            lab += f" → {ex['alto_sim']['p']:.2f}±{ex['alto_sim']['se']:.2f} (ρV ≥ 800)"
        ax1.plot([], [], "-o", color=c, label=lab)
    lam_ref = np.array([25, 3200.0])
    ax1.plot(lam_ref, 0.6 * lam_ref, ":", color=INK_MUTED, linewidth=1.2, label="pendiente de Poisson (p = 0.5)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    _style(ax1)
    ax1.set_xlabel("ρV", color=INK)
    ax1.set_ylabel("(ΔS)²", color=INK)
    ax1.set_title("Minkowski 2D: abiertos ρV ≤ 400, rellenos ρV ≥ 800; líneas: MYZ", color=INK, fontsize=10)
    ax1.legend(frameon=False, labelcolor=INK, fontsize=8)

    y = np.arange(len(EPS))
    for i, eps in enumerate(EPS):
        ex = summary["exponents"][str(eps)]
        for key, dy, mk in (("bajo_sim", 0.12, "o"), ("alto_sim", -0.12, "s")):
            if key in ex:
                ax2.errorbar(ex[key]["p"], y[i] + dy, xerr=K_SIGMA * ex[key]["se"], fmt=mk, color=COLORS[eps], capsize=3,
                             markerfacecolor="white" if key == "bajo_sim" else COLORS[eps])
    ax2.axvline(0.5, color=INK_MUTED, linestyle=":", linewidth=1.2)
    ax2.text(0.505, len(EPS) - 0.6, "Poisson (0.5)", color=INK_MUTED, fontsize=8)
    ax2.set_yticks(y, [f"ε = {e:g}" for e in EPS])
    ax2.set_ylim(-0.6, len(EPS) - 0.4)
    _style(ax2)
    ax2.set_xlabel("p en ΔS ∝ (ρV)^p   (barras: ±4σ; círculo: ρV ≤ 400, cuadrado: ρV ≥ 800)", color=INK)
    ax2.set_title("Exponente de escalado por régimen", color=INK, fontsize=10)
    path = ROOT / "figures" / "goal3c_asymptotic.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print(f"figura guardada en {path}")


if __name__ == "__main__":
    {"sim": sim, "pred": pred, "fit": fit}[sys.argv[1]]()
