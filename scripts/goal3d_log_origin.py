"""Paso 1 (Goal 3d): origen del término rho V ln(rho V) en (Delta S)^2,
Minkowski 2D.

Hipótesis (formulada antes del cálculo): el logaritmo viene del término
"mismo par" (J¹ de MYZ, ruido de Poisson del número de intervalos), porque
<N_k> = rho V ln(rho V) + O(rho V) en 2D: los intervalos casi nulos de
volumen fijo aparecen con todas las rapideces hasta el corte del diamante.
Predice (Delta S)² = b rho V ln(rho V) + O(rho V) con
    b(eps) = sum_n f_eps(n)^2,
f_eps(n) = peso suavizado de un par con n elementos (``b_coefficient``), y
que los términos de correlación (J², J³, J⁴, J⁶) y el cruzado con N son
O(rho V) sin logaritmo.

Fases:
  terms     descomposición de la predicción MYZ por términos,
            eps = 1, 0.5, 0.25 y rho V = 50 .. 12800 -> figures/goal3d_terms.json
  rapidity  sobre sprinklings (rho V = 3200), contribución de los pares a
            sum f(n)^2 por intervalo de rapidez eta = ½ ln(Δu/Δv)
            -> figures/goal3d_rapidity.json
  fit       ajustes T/rho V = c + b ln(rho V) por término y figura
            figures/goal3d_log_origin.png

Uso:  python scripts/goal3d_log_origin.py terms | rapidity | fit
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
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causets import sprinkle  # noqa: E402
from causets.causal import minkowski_causal_matrix  # noqa: E402
from causets.fluctuations import FluctuationKernel, action_coefficients, expected_thinned_abundance  # noqa: E402
from causets.geometry import MinkowskiDiamondGeometry  # noqa: E402
from causets.intervals import interval_cardinalities  # noqa: E402

warnings.filterwarnings("ignore")
TERMS = ROOT / "figures" / "goal3d_terms.json"
RAPIDITY = ROOT / "figures" / "goal3d_rapidity.json"
FIT = ROOT / "figures" / "goal3d_results.json"
LAMS = (50, 200, 800, 3200, 12800)
EPS = (1.0, 0.5, 0.25)
M_TERMS = {1.0: 2_000_000, 0.5: 4_000_000, 0.25: 8_000_000}
NAMES = ("NN", "Nk", "J1", "J2", "J3", "J4", "J6c")
COLORS = {"J1": "#2a78d6", "J2": "#eb6834", "J3": "#1baf7a", "J4": "#8a4fd6", "J6c": "#d6456b", "Nk": "#7a7a74", "NN": "#b5b3ac"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def pair_weight(eps: float, n: np.ndarray) -> np.ndarray:
    """f_eps(n) = sum_k a_k B_k(n, eps): contribución de un par con n elementos
    a la acción (sin el término a_N N)."""
    a = action_coefficients(2, eps)[1:]
    return sum(a[k] * binom.pmf(k, n, eps) for k in range(len(a)))


def b_coefficient(eps: float, n_max: int = 20_000) -> float:
    """Coeficiente de rho V ln(rho V) predicho: sum_n f_eps(n)^2 (usa
    ∫_0^∞ P_n(mu) dmu = 1 en la asintótica de <sum_pares f(n)^2>)."""
    return float(np.sum(pair_weight(eps, np.arange(n_max)) ** 2))


def decompose(P, eps: float) -> dict:
    """Contribuciones a a^T M a por término (sumadas dan P.variance)."""
    a = P.coefficients
    aN, ak = a[0], a[1:]
    n = len(ak)
    L = P.abundance_means[1:]
    t = P.terms
    out = {
        "NN": (aN**2 * P.abundance_means[0], 0.0),
        "Nk": (2 * aN * float(ak @ P.covariance[0, 1:]), 0.0),
        "J1": (float(np.sum(ak**2 * L[:n])), 0.0) if eps == 1 else (t["J1"].contraction, t["J1"].contraction_se),
        "J2": (t["J2"].contraction, t["J2"].contraction_se),
        "J3": (t["J3"].contraction, t["J3"].contraction_se),
        "J4": (2 * t["J4"].contraction, 2 * t["J4"].contraction_se),
        "J6c": (t["J6c"].contraction, t["J6c"].contraction_se),
    }
    return {k: dict(value=v, se=s) for k, (v, s) in out.items()}


def terms():
    g = MinkowskiDiamondGeometry(2)
    rows = []
    for eps in EPS:
        for lam in LAMS:
            t0 = time.time()
            K = FluctuationKernel(g, lam / g.volume, 3, np.random.default_rng(51_000 + lam + int(100 * eps)), eps=eps)
            P = K.predict(m_terms=M_TERMS[eps], exact_means=[expected_thinned_abundance(2, lam, k, eps) for k in range(4)],
                          batch=100_000)
            parts = decompose(P, eps)
            total = sum(p["value"] for p in parts.values())
            row = dict(eps=eps, rhoV=lam, variance=P.variance, variance_se=P.variance_se, parts=parts,
                       closure=total - P.variance, t=time.time() - t0)
            print(json.dumps(row), flush=True)
            rows.append(row)
            TERMS.write_text(json.dumps(rows, indent=1))


def rapidity(lam: float = 3200, n_sprinklings: int = 20, n_bins: int = 40):
    """Histograma de sum f(n)^2 en eta = ½ ln(Δu/Δv) sobre sprinklings."""
    g = MinkowskiDiamondGeometry(2)
    rng = np.random.default_rng(61_000)
    edges = np.linspace(-6, 6, n_bins + 1)
    out = {str(eps): np.zeros(n_bins) for eps in EPS}
    for _ in range(n_sprinklings):
        pts = sprinkle(g.region, lam / g.volume, rng)
        C = minkowski_causal_matrix(pts)
        nmat = interval_cardinalities(C)
        i, j = np.nonzero(C)
        du = (pts[j, 0] + pts[j, 1]) - (pts[i, 0] + pts[i, 1])
        dv = (pts[j, 0] - pts[j, 1]) - (pts[i, 0] - pts[i, 1])
        eta = 0.5 * np.log(du / dv)
        n = nmat[i, j]
        for eps in EPS:
            out[str(eps)] += np.histogram(eta, edges, weights=pair_weight(eps, n) ** 2)[0]
    res = dict(rhoV=lam, n_sprinklings=n_sprinklings, edges=edges.tolist(),
               per_unit_eta={k: (v / n_sprinklings / np.diff(edges) / lam).tolist() for k, v in out.items()})
    RAPIDITY.write_text(json.dumps(res, indent=1))
    print("guardado", RAPIDITY)


def fit_log(lam, y, se):
    """y = T/rho V = c + b ln(rho V), mínimos cuadrados ponderados. Devuelve
    (b, se_b, c, chi2/dof)."""
    X = np.column_stack([np.ones_like(lam), np.log(lam)])
    w = 1 / np.maximum(se, 1e-12 * np.abs(y).max())
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    cov = np.linalg.inv((X * w[:, None]).T @ (X * w[:, None]))
    chi2 = float(np.sum(((y - X @ coef) * w) ** 2))
    return float(coef[1]), float(np.sqrt(cov[1, 1])), float(coef[0]), chi2 / max(len(y) - 2, 1)


def fit():
    rows = json.loads(TERMS.read_text())
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, eps in zip(axes, EPS):
        sel = sorted((r for r in rows if r["eps"] == eps), key=lambda r: r["rhoV"])
        if not sel:
            continue
        lam = np.array([r["rhoV"] for r in sel], float)
        res = dict(b_pred=b_coefficient(eps))
        # total: el error de la predicción completa
        tot = np.array([r["variance"] for r in sel]) / lam
        tot_se = np.array([r["variance_se"] for r in sel]) / lam
        res["total"] = dict(zip(("b", "se", "c", "chi2dof"), fit_log(lam, tot, tot_se)))
        corr_v = np.zeros(len(sel))
        corr_se2 = np.zeros(len(sel))
        for name in NAMES:
            y = np.array([r["parts"][name]["value"] for r in sel]) / lam
            s = np.array([r["parts"][name]["se"] for r in sel]) / lam
            if name not in ("J1",):
                corr_v += y
                corr_se2 += s**2
            if name in ("NN", "Nk") or (name == "J1" and eps == 1):
                s = np.full_like(y, 1e-9)  # exactos
            res[name] = dict(zip(("b", "se", "c", "chi2dof"), fit_log(lam, y, s)))
            ax.plot(lam, y, "o-", color=COLORS[name], label=name, markersize=4, linewidth=1.4)
        res["no_J1"] = dict(zip(("b", "se", "c", "chi2dof"), fit_log(lam, corr_v, np.sqrt(corr_se2) + 1e-9)))
        ax.plot(lam, tot, "s-", color=INK, label="total", markersize=4, linewidth=2)
        ax.set_xscale("log")
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("ρV", color=INK)
        ax.set_ylabel("contribución / ρV", color=INK)
        ax.set_title(f"ε = {eps:g}:  b(J¹) = {res['J1']['b']:.3g},  predicho Σf² = {res['b_pred']:.3g}", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        summary[str(eps)] = res
    FIT.write_text(json.dumps(summary, indent=1))
    path = ROOT / "figures" / "goal3d_log_origin.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    for eps, res in summary.items():
        print(f"--- eps={eps}  b predicho = {res['b_pred']:.4f}")
        for k, v in res.items():
            if isinstance(v, dict):
                print(f"  {k:6s} b={v['b']:+.4g} ± {v['se']:.2g}   c={v['c']:+.4g}   chi2/dof={v['chi2dof']:.2f}")
    print("figura", path)


if __name__ == "__main__":
    {"terms": terms, "rapidity": rapidity, "fit": fit}[sys.argv[1]]()
