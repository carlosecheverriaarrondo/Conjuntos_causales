"""Goal 5b: which MYZ term makes rest/J¹ grow with strong smearing?

Goal 5 found that, with eps ≤ 0.2, rest⊥/J¹ grows with ρV (≈ ln ρV)
in 4D Minkowski. Here Var(F), F = sum_pairs f_eps(n), is decomposed into the
terms of Moradi, Yazdi & Zilhão (arXiv:2407.03395, eqs. 5.47-5.60),
estimated directly on each sprinkling with the matrix f(y, x) (0 if y ⊀ x),
its row sums R_y = sum_x f(y, x) and column sums C_x = sum_y f(y, x):
  sum f²                      -> J¹  (same pair)
  sum_y R_y² − sum f²          -> J²  (common lower vertex)
  sum_x C_x² − sum f²          -> J³  (common upper vertex)
  sum_z C_z R_z                -> J⁴  (chain y1 ≺ x1 = y2 ≺ x2; enters twice)
  (F − <F>)² − (J¹+J²+J³+2J⁴)  -> connected J⁶ (4 distinct points)
These are unbiased estimators of each term (up to O(1/M) from the mean of F).
Also rest⊥ = Var(F − gN) − J¹ with exact g (Goal 5, ``goal5_pred.json``).
New seeds, independent of those of Goal 5. ρV = 400 … 6400 (2000,
2000, 1000, 1000, 500 sprinklings), eps = 0.5, 0.2, 0.1.

PREREGISTERED CRITERIA (2026-10-12, before simulating; 4σ):
  G0 (exactness): (a) estimated J¹ = exact J¹ with |z| < 4 in the 15
     (eps, ρV) combinations; (b) J⁴ with eps = 0.1, ρV = 400 compatible with
     the independent estimate of the MYZ integrator (``FluctuationKernel``,
     4·10⁵ samples, seed 1): 19.38 ± 2.1.
  G1 (reproduction): the slope of rest⊥/J¹ versus ln ρV with eps = 0.1
     is compatible with that of Goal 5 (0.548 ± 0.082) at 4σ combined.
  G2 (descriptive, no criterion): slope of each term / J¹ versus
     ln ρV for each eps; it says which term produces the growth.

Usage:  python scripts/goal5b_decomposition.py sim LAM [LAM...] | fit | plot
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
J4_MYZ = (19.38, 2.1)  # FluctuationKernel, eps = 0.1, ρV = 400 (see docstring)
G5_SLOPE = (0.548, 0.082)  # Goal 5, F2 with eps = 0.1
TERMS = ("J2", "J3", "2J4", "J6c")
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"J2": "#2a78d6", "J3": "#1baf7a", "2J4": "#9b59b6", "J6c": "#eb6834", "rest": "#0b0b0b"}


def pair_statistics(C: np.ndarray, eps: float) -> dict:
    """N, F, sum f², sum R², sum C², sum C·R of a causal set."""
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
    """Terms per sprinkling (columns of ``STATS``) -> means and errors."""
    N, F, S2, R2, C2, CR = st.T
    M = len(F)
    dF = (F - F.mean()) * np.sqrt(M / (M - 1))  # unbiased variance
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
                        label={"rest": "rest (sum)", "J6c": "connected J⁶", "2J4": "2 J⁴", "J2": "J²", "J3": "J³"}[k])
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("ρV", color=INK)
        ax.set_title(f"ε = {eps}", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("term / J¹", color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    path = ROOT / "figures" / "goal5b_decomposition.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figure", path)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sim":
        sim([int(x) for x in sys.argv[2:]])
    else:
        {"fit": fit, "plot": lambda: plot(json.loads(FIT.read_text())["points"])}[cmd]()
