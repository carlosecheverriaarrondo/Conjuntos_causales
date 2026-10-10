"""Goal 4b-1: correlación temporal de la acción BD entre los conos pasados de
varios instantes de la misma línea de universo.

Los conos pasados de los observadores q_j = (t_j, 0) son anidados: C_i ⊂ C_j
si t_i < t_j. Como la región es causalmente convexa, cada par de C_i es par de
C_j con la misma cardinalidad, así que S(t_j) = S(t_i) + incremento. Si el
incremento es independiente de S(t_i), S(t) es un paseo aleatorio con
"reloj" Var S(t) (∝ V^1.5 asintóticamente), como la recursión de Model 1
(Das-Nasiri-Yazdi 2023, ec. 3.15) pero con Var ∝ V^1.5 en lugar de V.

Con un solo sprinkling del cono mayor se obtiene S(t_j) de todos los
subconos (elementos con t + |x| <= t_j).

Criterios (fijados antes de calcular), acción local:
  B1  kappa_ij = Cov(S_i, S_j - S_i) / Var(S_i) compatible con 0 a 4σ, todos
      los pares (independencia exacta).
  B2  |kappa_ij| < 0.2 (independencia práctica: el paseo aleatorio cambia la
      covarianza menos de un 20 %).
  B3  Var(S_j) de los subconos frente a la varianza del cono completo de los
      puntos 1c / 1d a la misma rho V (Minkowski: rho V = 200, 3200; FLRW
      alpha = 2: rho V = 200, 3200): |z| < 4.

Fases:
  sim GEOM   GEOM = mink4 | flrw4: sprinklings -> figures/goal4b_sim_GEOM.json
             (S(t_j) de cada sprinkling, para cualquier análisis posterior)
  fit        criterios y figura -> figures/goal4b_results.json, figures/goal4b_correlations.png

Uso:  python scripts/goal4b_correlations.py sim mink4 | sim flrw4 | fit
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
from causets.fluctuations import local_action_coefficients  # noqa: E402
from causets.geometry import FLRWConeGeometry, MinkowskiConeGeometry  # noqa: E402
from causets.intervals import interval_cardinalities  # noqa: E402

warnings.filterwarnings("ignore")
LAMS = (200, 500, 1250, 3200)  # rho V de los subconos; el mayor es el cono completo
N_SPRINKLINGS = 4000  # ampliado de 1000 (B2 no concluyente con 1000; ver README)
GEOMS = {"mink4": lambda: MinkowskiConeGeometry(4), "flrw4": lambda: FLRWConeGeometry(2.0, dim=4)}
K_SIGMA = 4.0
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def sim_path(geom):
    return ROOT / "figures" / f"goal4b_sim_{geom}.json"


def subcone_heights(g, lams):
    """t_j tal que el subcono J^-(t_j) ∩ {t >= 0} tenga rho V = lams[j] (rho
    fijado por el cono mayor). Minkowski: V ∝ t^4; FLRW a = eta^alpha: V ∝ t^(4 + 4 alpha)."""
    h = g.region.height
    power = 4 + 4 * getattr(g, "alpha", 0.0)
    return [h * (lam / lams[-1]) ** (1 / power) for lam in lams]


def subcone_actions(pts, heights, a):
    """S(t_j) de la acción local 4D para cada subcono."""
    C = minkowski_causal_matrix(pts)
    n = interval_cardinalities(C)
    t, r = pts[:, 0], np.linalg.norm(pts[:, 1:], axis=1)
    out = []
    for tj in heights:
        inside = t + r <= tj
        # par y ≺ x con x en el subcono (y ya lo está: pasado de x)
        sel = C & inside[None, :]
        counts = np.bincount(n[sel], minlength=4)[:4]
        out.append(float(a @ np.concatenate([[inside.sum()], counts])))
    return out


def sim(geom: str):
    g = GEOMS[geom]()
    a = local_action_coefficients(4)
    heights = subcone_heights(g, LAMS)
    path = sim_path(geom)
    rows = json.loads(path.read_text())["S"] if path.exists() else []
    rng = np.random.default_rng(131_000 + len(rows) + (0 if geom == "mink4" else 7))
    t0 = time.time()
    for k in range(len(rows), N_SPRINKLINGS):
        rows.append(subcone_actions(sprinkle(g.region, LAMS[-1] / g.volume, rng), heights, a))
        if (k + 1) % 50 == 0:
            path.write_text(json.dumps(dict(geom=geom, lams=LAMS, heights=heights, S=rows)))
            print(f"{geom}: {k + 1} sprinklings, {time.time() - t0:.0f} s", flush=True)
    path.write_text(json.dumps(dict(geom=geom, lams=LAMS, heights=heights, S=rows)))


def kappa(S, i, j):
    """kappa_ij = Cov(S_i, S_j - S_i)/Var(S_i) y su error (delta method por
    lotes: error estándar de la media de d_i e_ij / Var(S_i))."""
    d = S[:, i] - S[:, i].mean()
    e = (S[:, j] - S[:, i]) - (S[:, j] - S[:, i]).mean()
    var = np.mean(d * d)
    k = np.mean(d * e) / var
    # error: linealización de cov/var
    z = (d * e - k * d * d) / var
    return float(k), float(np.std(z, ddof=1) / np.sqrt(len(d)))


def fit():
    ref = {
        "mink4": {p["rhoV"]: p for p in json.loads((ROOT / "figures" / "cone4d_results.json").read_text())["points"]},
        "flrw4": {p["rhoV"]: p for p in json.loads((ROOT / "figures" / "flrw4d_results.json").read_text())["points"]["2.0"]},
    }
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, geom in zip(axes, GEOMS):
        if not sim_path(geom).exists():
            continue
        dat = json.loads(sim_path(geom).read_text())
        S = np.array(dat["S"])
        M = len(S)
        var = S.var(axis=0, ddof=1)
        var_se = np.array([np.std((S[:, j] - S[:, j].mean()) ** 2, ddof=1) / np.sqrt(M) for j in range(len(LAMS))])
        kap = {f"{LAMS[i]}->{LAMS[j]}": kappa(S, i, j) for i in range(len(LAMS)) for j in range(i + 1, len(LAMS))}
        corr = np.corrcoef(S, rowvar=False)
        b3 = []
        for j, lam in enumerate(LAMS):
            if lam in ref[geom]:
                p = ref[geom][lam]
                b3.append(dict(rhoV=lam, sub=var[j], sub_se=var_se[j], ref=p["sim"], ref_se=p["sim_se"],
                               z=float((var[j] - p["sim"]) / np.hypot(var_se[j], p["sim_se"]))))
        out[geom] = dict(n=M, var=var.tolist(), var_se=var_se.tolist(), kappa=kap, corr=corr.tolist(),
                         B1=bool(all(abs(k) < K_SIGMA * s for k, s in kap.values())),
                         B2=bool(all(abs(k) < 0.2 for k, _ in kap.values())),
                         B3=b3, B3_ok=bool(all(abs(b["z"]) < K_SIGMA for b in b3)),
                         # predicción del paseo aleatorio: corr_ij = sqrt(Var_i / Var_j)
                         corr_random_walk=[[float(np.sqrt(var[min(i, j)] / var[max(i, j)])) for j in range(len(LAMS))]
                                           for i in range(len(LAMS))])
        print(f"--- {geom}: {M} sprinklings")
        print("   Var(S_j):", [f"{v:.4g} ± {s:.2g}" for v, s in zip(var, var_se)])
        for k, (v, s) in kap.items():
            print(f"   kappa {k:>10}: {v:+.3f} ± {s:.3f}  ({v / s:+.1f}σ)")
        print("   corr medida / paseo aleatorio:")
        for i in range(len(LAMS)):
            print("     ", " ".join(f"{corr[i, j]:.3f}/{out[geom]['corr_random_walk'][i][j]:.3f}" for j in range(len(LAMS))))
        for b in b3:
            print(f"   B3 ρV={b['rhoV']}: subcono {b['sub']:.4g} ± {b['sub_se']:.2g}, cono completo {b['ref']:.4g} ± {b['ref_se']:.2g}, z = {b['z']:+.2f}")
        print(f"   B1 {out[geom]['B1']}  B2 {out[geom]['B2']}  B3 {out[geom]['B3_ok']}")
        # figura: corr(S_i, S_j) frente a sqrt(Var_i/Var_j)
        xs, ys = [], []
        for i in range(len(LAMS)):
            for j in range(i + 1, len(LAMS)):
                xs.append(out[geom]["corr_random_walk"][i][j])
                ys.append(corr[i, j])
        ax.plot([0, 1], [0, 1], "--", color=INK_MUTED, linewidth=1, label="paseo aleatorio")
        ax.plot(xs, ys, "o", color=BLUE if geom == "mink4" else ORANGE, markersize=6, label="medido")
        ax.set_xlabel("√(Var S_i / Var S_j)", color=INK)
        ax.set_ylabel("corr(S_i, S_j)", color=INK)
        ax.set_title({"mink4": "cono de Minkowski 4D", "flrw4": "cono de FLRW 4D (materia)"}[geom], color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    (ROOT / "figures" / "goal4b_results.json").write_text(json.dumps(out, indent=1))
    path = ROOT / "figures" / "goal4b_correlations.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figura", path)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit()
    else:
        sim(sys.argv[2])
