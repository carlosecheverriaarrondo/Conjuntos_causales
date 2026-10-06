"""Punto 1d: escalado de (Delta S)^2 de la acción BD local en el cono de luz
pasado hasta el big bang de un FLRW plano 4D, a(eta) = eta^alpha, con
alpha = 1 (radiación) y 2 (materia). Es la región de Everpresent Λ.

Predicción (derivada antes de calcular, ver README "Punto 1d"): el argumento
de los boosts del punto 1c, con el volumen físico de un intervalo casi nulo
c_4 tau^4 w (tau conforme, w = media de a^4 con el perfil s(lambda - s)), da
    <N_k> ≈ (rho V)^1.5 · (G / V^1.5) / (8 sqrt(c_4)) · Γ(k+½)/k!,
con G de ``cone4d_scaling.g_flrw``: G/V^1.5 = 1.3145 (alpha = 1) y 1.4194
(alpha = 2), frente a 0.8187 en el cono de Minkowski. Mismo exponente:
ΔS ∝ (rho V)^0.75.

Las medias no tienen forma cerrada: <N_k> por el Monte Carlo del formalismo
(``FluctuationKernel.abundance_means``, propuesta local), validado con
alpha = 0 frente a la integral exacta del cono.

Criterios (fijados antes de ver los resultados; el de K3 es el del punto 1c,
sin cambiar, porque en la prueba de precisión del MC se vio de pasada
<N_0>/(rho V)^1.5 = 0.799 frente a 0.805 con alpha = 1 y rho V = 1e8):
  K1  volúmenes de intervalos frente a MC; alpha = 0 reproduce Minkowski;
      g_flrw(0) = g_cone (tests deterministas).
  K2  <N_k> (MC) frente a sprinklings: |z| < 4.
  K3  cociente <N_k> (MC) / asintótico, k = 0..3: |1 - r| < 3 % con
      rho V = 1e8, y menor que con 1e6.
  K4  resto/J¹: pendiente frente a ln(rho V) compatible con 0 a 4σ.

Fases (por alpha, para lanzarlas en paralelo):
  sim ALPHA     sprinklings -> figures/flrw4d_sim_aALPHA.json
  means ALPHA   medias MC en las densidades de la simulación y en 1e6..1e8
                -> figures/flrw4d_means_aALPHA.json
  fit           criterios, figura figures/flrw4d_scaling.png y resumen
                figures/flrw4d_results.json

Uso:  python scripts/flrw4d_scaling.py sim 1 | means 1 | fit
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
from causets.fluctuations import EnsembleFluctuations, FluctuationKernel, abundance_vector, local_action_coefficients  # noqa: E402
from causets.geometry import FLRWConeGeometry  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from cone4d_scaling import asymptotic_abundance, g_flrw  # noqa: E402
from goal3d_log_origin import fit_log  # noqa: E402

warnings.filterwarnings("ignore")
ALPHAS = (1.0, 2.0)
PLAN = {25: 4000, 50: 4000, 100: 4000, 200: 2000, 400: 2000, 800: 2000, 1600: 1000, 3200: 1000}
ASYMP_LAMS = (1e6, 1e7, 1e8)
M_MEANS = 16_000_000
K_SIGMA = 4.0
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def sim_path(alpha):
    return ROOT / "figures" / f"flrw4d_sim_a{alpha:g}.json"


def means_path(alpha):
    return ROOT / "figures" / f"flrw4d_means_a{alpha:g}.json"


def sim(alpha: float):
    g = FLRWConeGeometry(alpha, dim=4)
    a = local_action_coefficients(4)
    path = sim_path(alpha)
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["rhoV"] for r in rows}
    for lam, m in PLAN.items():
        if lam in done:
            continue
        rng = np.random.default_rng(111_000 + lam + int(10 * alpha))
        t0 = time.time()
        ab = np.array([abundance_vector(minkowski_causal_matrix(sprinkle(g.region, lam / g.volume, rng)), 4)
                       for _ in range(m)])
        E = EnsembleFluctuations(ab, a)
        row = dict(alpha=alpha, rhoV=lam, sim=E.variance, sim_se=E.variance_se, n_sprinklings=m,
                   means=ab.mean(axis=0).tolist(), means_se=(ab.std(axis=0, ddof=1) / np.sqrt(m)).tolist(),
                   t=time.time() - t0)
        print(json.dumps(row), flush=True)
        rows.append(row)
        path.write_text(json.dumps(rows, indent=1))


def means(alpha: float):
    g = FLRWConeGeometry(alpha, dim=4)
    path = means_path(alpha)
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {r["rhoV"] for r in rows}
    for lam in [*PLAN, *ASYMP_LAMS]:
        if lam in done:
            continue
        t0 = time.time()
        K = FluctuationKernel(g, lam / g.volume, 4, np.random.default_rng(121_000 + int(lam % 100_000) + int(10 * alpha)))
        L, se = K._batched_means(M_MEANS, 5, 100_000)
        row = dict(alpha=alpha, rhoV=lam, L=L.tolist(), L_se=se.tolist(), t=time.time() - t0)
        print(json.dumps(row), flush=True)
        rows.append(row)
        path.write_text(json.dumps(rows, indent=1))


def parts(L, L_se, lam):
    """NN, cruzado N-k y J¹ con las medias MC (y el error de J¹ y del cruzado)."""
    a = local_action_coefficients(4)
    aN, ak = a[0], a[1:]
    L, L_se = np.asarray(L), np.asarray(L_se)
    j = np.arange(4)
    # (Delta S)² ⊇ J¹ + cruzado = sum_k (a_k² + 2 aN a_k (k+2)) L_k - 2 aN sum_k a_k (k+1) L_{k+1}
    grad = np.zeros(5)
    grad[:4] += ak**2 + 2 * aN * ak * (j + 2)
    grad[1:] += -2 * aN * ak * (j + 1)
    MNk = (j + 2) * L[:4] - (j + 1) * L[1:]
    return dict(NN=aN**2 * lam, Nk=2 * aN * float(ak @ MNk), J1=float(ak**2 @ L[:4]),
                J1_se=float(np.sqrt(np.sum((ak**2 * L_se[:4]) ** 2))), parts_se=float(np.sqrt(np.sum((grad * L_se) ** 2))))


def fit():
    out = dict(G={}, K2=[], K3={}, K4={}, points={}, p_local_sim={})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = {1.0: BLUE, 2.0: ORANGE}
    for alpha in ALPHAS:
        sims = {r["rhoV"]: r for r in json.loads(sim_path(alpha).read_text())}
        mns = {r["rhoV"]: r for r in json.loads(means_path(alpha).read_text())}
        gf = g_flrw(alpha)
        out["G"][str(alpha)] = gf
        pts = []
        for lam in sorted(sims):
            s, m = sims[lam], mns[lam]
            z = [(x - l) / np.hypot(xs, ls) for x, xs, l, ls in zip(s["means"][1:], s["means_se"][1:], m["L"][:4], m["L_se"][:4])]
            out["K2"].append(dict(alpha=alpha, rhoV=lam, z=z))
            pr = parts(m["L"], m["L_se"], lam)
            rest = s["sim"] - pr["NN"] - pr["Nk"] - pr["J1"]
            pts.append(dict(rhoV=lam, sim=s["sim"], sim_se=s["sim_se"], **pr, rest_over_J1=rest / pr["J1"],
                            rest_over_J1_se=float(np.hypot(s["sim_se"], pr["parts_se"]) / pr["J1"])))
        out["points"][str(alpha)] = pts
        ratios = {str(lam): [mns[lam]["L"][k] / asymptotic_abundance(gf, lam, k) for k in range(4)] for lam in ASYMP_LAMS}
        ratios_se = {str(lam): [mns[lam]["L_se"][k] / asymptotic_abundance(gf, lam, k) for k in range(4)] for lam in ASYMP_LAMS}
        dev = {k: [abs(1 - v) for v in vs] for k, vs in ratios.items()}
        out["K3"][str(alpha)] = dict(ratios=ratios, ratios_se=ratios_se, max_dev_last=max(dev[str(1e8)]),
                                     closer_than_1e6=bool(all(d8 < d6 for d8, d6 in zip(dev[str(1e8)], dev[str(1e6)]))))
        lam = np.array([p["rhoV"] for p in pts], float)
        y = np.array([p["rest_over_J1"] for p in pts])
        se = np.array([p["rest_over_J1_se"] for p in pts])
        b, b_se, c, chi2 = fit_log(lam, y, se)
        w = 1 / se**2
        out["K4"][str(alpha)] = dict(slope=b, slope_se=b_se, z=b / b_se, chi2dof=chi2,
                                     mean=float(np.sum(w * y) / np.sum(w)), mean_se=float(1 / np.sqrt(np.sum(w))))
        tot = np.array([p["sim"] for p in pts])
        tot_se = np.array([p["sim_se"] for p in pts])
        out["p_local_sim"][str(alpha)] = [dict(rhoV=(int(lam[i]), int(lam[i + 1])),
                                               p=float(np.log(tot[i + 1] / tot[i]) / np.log(lam[i + 1] / lam[i]) / 2),
                                               se=float(np.hypot(tot_se[i] / tot[i], tot_se[i + 1] / tot[i + 1]) / np.log(lam[i + 1] / lam[i]) / 2))
                                          for i in range(len(lam) - 1)]
        # figura
        col = colors[alpha]
        ax1.errorbar(lam, tot / lam, yerr=tot_se / lam, fmt="o", color=col, markerfacecolor="white", capsize=3,
                     label=f"(ΔS)² simulación, α = {alpha:g}")
        ax1.plot(lam, [p["J1"] / p["rhoV"] for p in pts], "-", color=col, linewidth=1.4, label=f"J¹ (MC), α = {alpha:g}")
        lam_a = np.array(ASYMP_LAMS)
        j1a = [parts(mns[x]["L"], mns[x]["L_se"], x)["J1"] / x for x in ASYMP_LAMS]
        ax1.plot(lam_a, j1a, "s", color=col, markersize=4)
        asym = sum(ak**2 * asymptotic_abundance(gf, 1.0, k) for k, ak in enumerate(local_action_coefficients(4)[1:]))
        ref = np.geomspace(1e5, 1e8, 20)
        ax1.plot(ref, asym * ref**0.5, ":", color=col, linewidth=1.2)
        ax2.errorbar(lam * (1.04 if alpha == 2 else 1), y, yerr=se, fmt="o", color=col, markerfacecolor="white", capsize=3,
                     label=f"α = {alpha:g}: media {out['K4'][str(alpha)]['mean']:.3f}")
    out["K2_max"] = max(max(abs(v) for v in r["z"]) for r in out["K2"])
    (ROOT / "figures" / "flrw4d_results.json").write_text(json.dumps(out, indent=1))

    print("G/V^1.5:", out["G"], "  K2 max |z|:", round(out["K2_max"], 2))
    for alpha in ALPHAS:
        k3, k4 = out["K3"][str(alpha)], out["K4"][str(alpha)]
        print(f"--- alpha = {alpha:g}")
        for lam_s, vs in k3["ratios"].items():
            print(f"   K3 ρV={float(lam_s):.0e}: r =", [f"{v:.4f}±{e:.4f}" for v, e in zip(vs, k3["ratios_se"][lam_s])])
        print(f"   K3 max |1-r| a 1e8 = {k3['max_dev_last']:.4f}, más cerca que a 1e6: {k3['closer_than_1e6']}")
        for p in out["points"][str(alpha)]:
            print(f"   ρV={p['rhoV']:<5} total/ρV={p['sim'] / p['rhoV']:9.2f}±{p['sim_se'] / p['rhoV']:<7.2f} "
                  f"J1/ρV={p['J1'] / p['rhoV']:9.2f}  resto/J1={p['rest_over_J1']:+.3f}±{p['rest_over_J1_se']:.3f}")
        print(f"   K4: pendiente {k4['slope']:+.4f} ± {k4['slope_se']:.4f} (z = {k4['z']:+.2f}, chi2/dof {k4['chi2dof']:.2f});"
              f" media {k4['mean']:.3f} ± {k4['mean_se']:.3f}")
        print("   p local:", [f"{p['p']:.2f}±{p['se']:.2f}" for p in out["p_local_sim"][str(alpha)]])

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("ρV", color=INK)
    ax1.set_ylabel("(ΔS)² / ρV", color=INK)
    ax1.set_title("Cono pasado hasta el big bang, FLRW 4D (a ∝ η^α)", color=INK, fontsize=10)
    ax1.legend(frameon=False, fontsize=7)
    ax2.axhline(0.317, color=INK_MUTED, linestyle="--", linewidth=1, label="diamante de Minkowski: 0.317")
    ax2.axhline(0.342, color=GREEN, linestyle=":", linewidth=1, label="cono de Minkowski: 0.342")
    ax2.set_xscale("log")
    ax2.set_xlabel("ρV", color=INK)
    ax2.set_ylabel("correlaciones / J¹", color=INK)
    ax2.set_title("Fracción de correlaciones", color=INK, fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    path = ROOT / "figures" / "flrw4d_scaling.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figura", path)


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "fit":
        fit()
    else:
        {"sim": sim, "means": means}[phase](float(sys.argv[2]))
