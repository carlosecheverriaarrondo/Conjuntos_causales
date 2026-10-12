"""Goal 5b: ¿qué término de MYZ hace crecer resto/J¹ con suavizado fuerte?

El Goal 5 encontró que, con eps ≤ 0.2, resto⊥/J¹ crece con ρV (≈ ln ρV)
en Minkowski 4D. Aquí se descompone Var(F), F = sum_pares f_eps(n), en los
términos de Moradi, Yazdi & Zilhão (arXiv:2407.03395, ecs. 5.47-5.60),
estimados directamente sobre cada sprinkling con la matriz f(y, x) (0 si y ⊀ x),
sus sumas por filas R_y = sum_x f(y, x) y por columnas C_x = sum_y f(y, x):
  sum f²                      -> J¹  (mismo par)
  sum_y R_y² − sum f²          -> J²  (vértice inferior común)
  sum_x C_x² − sum f²          -> J³  (vértice superior común)
  sum_z C_z R_z                -> J⁴  (cadena y1 ≺ x1 = y2 ≺ x2; entra 2 veces)
  (F − <F>)² − (J¹+J²+J³+2J⁴)  -> J⁶ conexo (4 puntos distintos)
Son estimadores insesgados de cada término (salvo O(1/M) por la media de F).
También resto⊥ = Var(F − gN) − J¹ con g exacto (Goal 5, ``goal5_pred.json``).
Semillas nuevas, independientes de las del Goal 5. ρV = 400 … 6400 (2000,
2000, 1000, 1000, 500 sprinklings), eps = 0.5, 0.2, 0.1.

CRITERIOS PRERREGISTRADOS (2026-10-12, antes de simular; 4σ):
  G0 (exactitud): (a) J¹ estimado = J¹ exacto con |z| < 4 en las 15
     combinaciones (eps, ρV); (b) J⁴ con eps = 0.1, ρV = 400 compatible con la
     estimación independiente del integrador MYZ (``FluctuationKernel``,
     4·10⁵ muestras, semilla 1): 19.38 ± 2.1.
  G1 (reproducción): la pendiente de resto⊥/J¹ frente a ln ρV con eps = 0.1
     es compatible con la del Goal 5 (0.548 ± 0.082) a 4σ combinada.
  G2 (descriptivo, sin criterio): pendiente de cada término / J¹ frente a
     ln ρV para cada eps; dice qué término produce el crecimiento.

Uso:  python scripts/goal5b_decomposition.py sim LAM [LAM...] | fit | plot
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
from causets.fluctuations import pair_weight  # noqa: E402
from causets.geometry import MinkowskiDiamondGeometry  # noqa: E402
from causets.intervals import interval_cardinalities  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data" / "goal5b"
PRED = ROOT / "figures" / "goal5_pred.json"
FIT = ROOT / "figures" / "goal5b_results.json"
PLAN = {400: 2000, 800: 2000, 1600: 1000, 3200: 1000, 6400: 500}
EPS = (0.5, 0.2, 0.1)
CHUNK = 50
STATS = ("N", "F", "S2", "R2", "C2", "CR")
J4_MYZ = (19.38, 2.1)  # FluctuationKernel, eps = 0.1, ρV = 400 (ver docstring)
G5_SLOPE = (0.548, 0.082)  # Goal 5, F2 con eps = 0.1
TERMS = ("J2", "J3", "2J4", "J6c")
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"J2": "#2a78d6", "J3": "#1baf7a", "2J4": "#9b59b6", "J6c": "#eb6834", "rest": "#0b0b0b"}


def pair_statistics(C: np.ndarray, eps: float) -> dict:
    """N, F, sum f², sum R², sum C², sum C·R de un conjunto causal."""
    n = interval_cardinalities(C)
    table = pair_weight(4, np.arange(n.max() + 1), eps)
    f = np.where(C, table[n], 0.0)
    R, Col = f.sum(axis=1), f.sum(axis=0)
    return dict(N=C.shape[0], F=float(f.sum()), S2=float(np.sum(f * f)), R2=float(R @ R),
                C2=float(Col @ Col), CR=float(Col @ R))


def sim(lams):
    g = MinkowskiDiamondGeometry(4)
    DATA.mkdir(parents=True, exist_ok=True)
    for lam in lams:
        for c in range(PLAN[lam] // CHUNK):
            path = DATA / f"stats_{lam}_{c:03d}.npz"
            if path.exists():
                continue
            rng = np.random.default_rng([92_000, lam, c])
            t0 = time.time()
            out = np.zeros((CHUNK, len(EPS), len(STATS)))
            for i in range(CHUNK):
                C = minkowski_causal_matrix(sprinkle(g.region, lam / g.volume, rng))
                for j, eps in enumerate(EPS):
                    s = pair_statistics(C, eps)
                    out[i, j] = [s[k] for k in STATS]
            np.savez_compressed(path, stats=out)
            print(f"ρV={lam} chunk {c} {time.time() - t0:.0f}s", flush=True)


def load(lam):
    return np.concatenate([np.load(p)["stats"] for p in sorted(DATA.glob(f"stats_{lam}_*.npz"))])


def _mean_se(x):
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def _slope(x, y, se):
    w = 1 / se**2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a, b = cov @ (A.T @ (w * y))
    return float(b), float(np.sqrt(cov[1, 1]))


def decompose(st: np.ndarray, J1: float, g: float) -> dict:
    """Términos por sprinkling (columnas de ``STATS``) -> medias y errores."""
    N, F, S2, R2, C2, CR = st.T
    M = len(F)
    dF = (F - F.mean()) * np.sqrt(M / (M - 1))  # varianza insesgada
    P = F - g * N
    dP = (P - P.mean()) * np.sqrt(M / (M - 1))
    per = dict(J1=S2, J2=R2 - S2, J3=C2 - S2, **{"2J4": 2 * CR})
    per["J6c"] = dF**2 - (S2 + per["J2"] + per["J3"] + per["2J4"])
    per["rest"] = dF**2 - S2
    per["restP"] = dP**2 - S2
    out = {}
    for k, v in per.items():
        m, se = _mean_se(v)
        out[k], out[k + "_se"] = m, se
        if k != "J1":
            out[k + "_over_J1"], out[k + "_over_J1_se"] = m / J1, se / J1
    out["z_J1"] = (out["J1"] - J1) / out["J1_se"]
    return out


def fit():
    P = json.loads(PRED.read_text())
    rows = []
    for lam in PLAN:
        st = load(lam)
        for j, eps in enumerate(EPS):
            p = P[f"{eps}_{lam}"]
            rows.append(dict(eps=eps, rhoV=lam, M=len(st), J1_exact=p["J1"], **decompose(st[:, j], p["J1"], p["g"])))
    res = dict(points=rows)
    zs = [abs(r["z_J1"]) for r in rows]
    r0 = next(r for r in rows if r["eps"] == 0.1 and r["rhoV"] == 400)
    zJ4 = (r0["2J4"] / 2 - J4_MYZ[0]) / np.hypot(r0["2J4_se"] / 2, J4_MYZ[1])
    res["G0"] = dict(max_abs_z_J1=max(zs), n=len(zs), J4_400=r0["2J4"] / 2, J4_400_se=r0["2J4_se"] / 2,
                     z_J4=float(zJ4), passed=bool(max(zs) < 4 and abs(zJ4) < 4))
    res["G2"] = {}
    for eps in EPS:
        sel = [r for r in rows if r["eps"] == eps]
        x = np.log([r["rhoV"] for r in sel])
        res["G2"][str(eps)] = {}
        for k in ("restP", "rest", *TERMS):
            y = np.array([r[k + "_over_J1"] for r in sel])
            se = np.array([r[k + "_over_J1_se"] for r in sel])
            b, b_se = _slope(x, y, se)
            res["G2"][str(eps)][k] = dict(slope=b, slope_se=b_se, mean=float(np.sum(y / se**2) / np.sum(1 / se**2)))
    s = res["G2"]["0.1"]["restP"]
    z1 = (s["slope"] - G5_SLOPE[0]) / np.hypot(s["slope_se"], G5_SLOPE[1])
    res["G1"] = dict(slope=s["slope"], slope_se=s["slope_se"], goal5=G5_SLOPE, z=float(z1), passed=bool(abs(z1) < 4))
    FIT.write_text(json.dumps(res, indent=1))

    for r in rows:
        terms = "  ".join(f"{k}/J1={r[k + '_over_J1']:+.3f}±{r[k + '_over_J1_se']:.3f}" for k in (*TERMS, "rest", "restP"))
        print(f"ε={r['eps']:<4} ρV={r['rhoV']:<5} z_J1={r['z_J1']:+.2f}  {terms}")
    print("G0", res["G0"])
    print("G1", res["G1"])
    for eps in EPS:
        print(f"G2 ε={eps}: " + "  ".join(f"{k}: {v['slope']:+.3f}±{v['slope_se']:.3f}" for k, v in res["G2"][str(eps)].items()))
    plot(rows)


def plot(rows):
    fig, axes = plt.subplots(1, len(EPS), figsize=(15, 4.6), constrained_layout=True, sharey=False)
    for ax, eps in zip(axes, EPS):
        sel = [r for r in rows if r["eps"] == eps]
        lam = np.array([r["rhoV"] for r in sel], float)
        for i, k in enumerate((*TERMS, "rest")):
            ax.errorbar(lam * (1 + 0.025 * i), [r[k + "_over_J1"] for r in sel], yerr=[r[k + "_over_J1_se"] for r in sel],
                        fmt="o-", color=COLORS[k], markerfacecolor="white", capsize=3,
                        label={"rest": "resto (suma)", "J6c": "J⁶ conexo", "2J4": "2 J⁴", "J2": "J²", "J3": "J³"}[k])
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("ρV", color=INK)
        ax.set_title(f"ε = {eps}", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("término / J¹", color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    path = ROOT / "figures" / "goal5b_decomposition.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figura", path)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sim":
        sim([int(x) for x in sys.argv[2:]])
    else:
        {"fit": fit, "plot": lambda: plot(json.loads(FIT.read_text())["points"])}[cmd]()
