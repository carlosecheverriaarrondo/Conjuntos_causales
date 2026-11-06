"""Goal 5: scaling of the fluctuations of the SMEARED BD action in 4D Minkowski.

Question: does the asymptotic exponent p = 0.75 (ΔS ∝ (ρV)^p) of Step 2,
measured with the local action, survive with the smeared action (Sorkin 2007,
arXiv:gr-qc/0703099; Benincasa & Dowker 2010), which is the one used in Goal 4a?

Notation. S_eps = a_N N + F, F = sum_pairs f_eps(n) (``pair_weight``).
  J¹_eps = E[sum_pairs f_eps(n)²]              (``smeared_pair_moment``, power 2)
  g      = dE[F]/d(ρV), Cov(N, F) = ρV g        (Mecke, ``smeared_pair_mecke_slope``)
  Var(S) = a_N² ρV + 2 a_N ρV g + Var(F), so the "rest" of correlations
  (J², J³, J⁴, J⁶ of MYZ) is  rest = Var(F) − J¹  (as in Step 2), and the
  PROJECTED rest, without the part linear in N (Poisson-like, which for small
  eps dominates at low densities), is
      rest_⊥ = Var(F − g N) − J¹ = rest − ρV g².
  Var(F − g N) is measured directly (less noise than subtracting).

Simulation: 4D Minkowski diamond, ρV = 400, 800, 1600, 3200, 6400 with
2000, 2000, 1000, 1000, 500 sprinklings. The full histogram of interval
cardinalities of each sprinkling is stored (data/goal5/, not versioned),
so all eps come from the same sprinklings: eps = 1, 0.5, 0.2, 0.1.
(eps = 0.05 is left out: at ρV ≤ 6400 the part in N is 3-55 times J¹.)

PREREGISTERED CRITERIA (fixed on 2026-10-12, before simulating; 4σ):
  F0 (exactness): for each eps and ρV, the simulated mean of F and Cov(N, F)
     agree with the exact E[F] and ρV g with |z| < 4 (40 comparisons).
  F1 (reproduction): with eps = 1, the weighted mean of rest/J¹ over the 5
     densities is compatible with that of Step 2, 0.317 ± 0.014, at 4σ
     combined (these are new seeds).
  F2 (main): for eps = 0.5, 0.2 and 0.1, the slope of a weighted linear fit
     of rest_⊥/J¹ versus ln ρV is compatible with 0 at 4σ. If it holds,
     Var(S) − (part linear in N) ∝ J¹ ∝ (ρV)^1.5 asymptotically
     (exponent 0.75, Step 2) also with smearing.
  F3 (descriptive, no criterion): rest_⊥/J¹ as a function of eps (Goal 4a
     used r = 0.33; ℓ_k only depends on r as (1 + r)^0.1).
Limit: eps ≥ 0.1, far from the eps ~ 10⁻⁴⁹ of Goal 4a.

Usage:  python scripts/goal5_smeared4d.py sim LAM [LAM...] | pred | fit | plot
        (plot remakes the figure from figures/goal5_results.json, without data/)
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
    action_coefficients,
    pair_weight,
    smeared_pair_mecke_slope,
    smeared_pair_moment,
)
from causets.geometry import MinkowskiDiamondGeometry  # noqa: E402
from causets.intervals import interval_cardinalities  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data" / "goal5"
PRED = ROOT / "figures" / "goal5_pred.json"
FIT = ROOT / "figures" / "goal5_results.json"
PLAN = {400: 2000, 800: 2000, 1600: 1000, 3200: 1000, 6400: 500}
EPS = (1.0, 0.5, 0.2, 0.1)
CHUNK = 50
STEP2 = (0.317, 0.014)  # rest/J¹ of Step 2 (README, "Step 2")
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {1.0: "#0b0b0b", 0.5: "#2a78d6", 0.2: "#1baf7a", 0.1: "#eb6834"}


def histogram(C: np.ndarray) -> np.ndarray:
    """hist[n] = number of related pairs with n elements between them."""
    return np.bincount(interval_cardinalities(C)[C])


def sim(lams):
    g = MinkowskiDiamondGeometry(4)
    DATA.mkdir(parents=True, exist_ok=True)
    for lam in lams:
        for c in range(PLAN[lam] // CHUNK):
            path = DATA / f"hist_{lam}_{c:03d}.npz"
            if path.exists():
                continue
            rng = np.random.default_rng([91_000, lam, c])
            t0 = time.time()
            N, H = [], []
            for _ in range(CHUNK):
                C = minkowski_causal_matrix(sprinkle(g.region, lam / g.volume, rng))
                N.append(C.shape[0])
                H.append(histogram(C))
            hist = np.zeros((CHUNK, max(len(h) for h in H)), dtype=np.int64)
            for i, h in enumerate(H):
                hist[i, : len(h)] = h
            np.savez_compressed(path, N=np.array(N), hist=hist)
            print(f"ρV={lam} chunk {c} {time.time() - t0:.0f}s", flush=True)


def load(lam):
    N, H = [], []
    for p in sorted(DATA.glob(f"hist_{lam}_*.npz")):
        d = np.load(p)
        N.append(d["N"])
        H.append(d["hist"])
    width = max(h.shape[1] for h in H)
    return np.concatenate(N).astype(float), np.vstack([np.pad(h, ((0, 0), (0, width - h.shape[1]))) for h in H])


def pred():
    out = json.loads(PRED.read_text()) if PRED.exists() else {}
    for eps in EPS:
        for lam in PLAN:
            key = f"{eps}_{lam}"
            if key in out:
                continue
            t0 = time.time()
            out[key] = dict(eps=eps, rhoV=lam, J1=smeared_pair_moment(4, lam, eps, 2),
                            EF=smeared_pair_moment(4, lam, eps, 1), g=smeared_pair_mecke_slope(4, lam, eps))
            PRED.write_text(json.dumps(out, indent=1))
            print(key, out[key], f"{time.time() - t0:.0f}s", flush=True)


def _var_se(x):
    d = x - x.mean()
    return float(np.var(x, ddof=1)), float(np.std(d * d, ddof=1) / np.sqrt(len(x)))


def _slope(x, y, se):
    """Weighted linear fit y = a + b x: (b, error of b, chi², dof)."""
    w = 1 / se**2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a, b = cov @ (A.T @ (w * y))
    return float(b), float(np.sqrt(cov[1, 1])), float(np.sum(w * (y - a - b * x) ** 2)), len(x) - 2


def fit():
    P = json.loads(PRED.read_text())
    rows = []
    for lam in PLAN:
        N, hist = load(lam)
        for eps in EPS:
            p = P[f"{eps}_{lam}"]
            F = hist @ pair_weight(4, np.arange(hist.shape[1]), eps)
            M = len(F)
            dN, dF = N - N.mean(), F - F.mean()
            cov = float(np.sum(dN * dF) / (M - 1))
            cov_se = float(np.std(dN * dF, ddof=1) / np.sqrt(M))
            varF, varF_se = _var_se(F)
            varP, varP_se = _var_se(F - p["g"] * N)
            S = action_coefficients(4, eps)[0] * N + F
            rows.append(dict(
                eps=eps, rhoV=lam, M=M, J1=p["J1"],
                z_mean=float((F.mean() - p["EF"]) / (F.std(ddof=1) / np.sqrt(M))),
                z_cov=(cov - lam * p["g"]) / cov_se,
                varS=_var_se(S)[0], varS_se=_var_se(S)[1],
                rest_over_J1=(varF - p["J1"]) / p["J1"], rest_over_J1_se=varF_se / p["J1"],
                restP_over_J1=(varP - p["J1"]) / p["J1"], restP_over_J1_se=varP_se / p["J1"],
                N_part_over_J1=(lam * p["g"] ** 2) / p["J1"],
            ))
    res = dict(points=rows)
    # F0
    z = [abs(r[k]) for r in rows for k in ("z_mean", "z_cov")]
    res["F0"] = dict(max_abs_z=max(z), n=len(z), passed=bool(max(z) < 4))
    # F1
    one = [r for r in rows if r["eps"] == 1.0]
    w = np.array([1 / r["rest_over_J1_se"] ** 2 for r in one])
    m = float(np.sum(w * [r["rest_over_J1"] for r in one]) / w.sum())
    m_se = float(1 / np.sqrt(w.sum()))
    zF1 = (m - STEP2[0]) / np.hypot(m_se, STEP2[1])
    res["F1"] = dict(mean=m, se=m_se, step2=STEP2, z=float(zF1), passed=bool(abs(zF1) < 4))
    # F2 and F3
    res["F2"], res["F3"] = {}, {}
    for eps in EPS:
        sel = [r for r in rows if r["eps"] == eps]
        x = np.log([r["rhoV"] for r in sel])
        y = np.array([r["restP_over_J1"] for r in sel])
        se = np.array([r["restP_over_J1_se"] for r in sel])
        b, b_se, chi2, dof = _slope(x, y, se)
        wm = float(np.sum(y / se**2) / np.sum(1 / se**2))
        res["F3"][str(eps)] = dict(mean=wm, se=float(1 / np.sqrt(np.sum(1 / se**2))))
        entry = dict(slope=b, slope_se=b_se, z=b / b_se, chi2=chi2, dof=dof)
        if eps != 1.0:
            entry["passed"] = bool(abs(b / b_se) < 4)
        res["F2"][str(eps)] = entry
    FIT.write_text(json.dumps(res, indent=1))

    for r in rows:
        print(f"ε={r['eps']:<4} ρV={r['rhoV']:<5} z_mean={r['z_mean']:+.2f} z_cov={r['z_cov']:+.2f} "
              f"rest/J1={r['rest_over_J1']:+.3f}±{r['rest_over_J1_se']:.3f} "
              f"rest⊥/J1={r['restP_over_J1']:+.3f}±{r['restP_over_J1_se']:.3f} ρVg²/J1={r['N_part_over_J1']:.3f}")
    print("F0", res["F0"])
    print("F1", res["F1"])
    for eps in EPS:
        print(f"F2 ε={eps}", res["F2"][str(eps)], "F3", res["F3"][str(eps)])
    plot(rows)


def plot(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for eps in EPS:
        sel = [r for r in rows if r["eps"] == eps]
        lam = np.array([r["rhoV"] for r in sel], float)
        J1 = np.array([r["J1"] for r in sel])
        ax1.plot(lam, J1 / J1[0], "-", color=COLORS[eps], linewidth=1.5)
        ax1.errorbar(lam, [r["varS"] / J1[0] for r in sel], yerr=[r["varS_se"] / J1[0] for r in sel], fmt="o",
                     color=COLORS[eps], markerfacecolor="white", capsize=3, label=f"ε = {eps}")
        ax2.errorbar(lam * (1 + 0.03 * EPS.index(eps)), [r["restP_over_J1"] for r in sel],
                     yerr=[r["restP_over_J1_se"] for r in sel], fmt="o-", color=COLORS[eps],
                     markerfacecolor="white", capsize=3, label=f"ε = {eps}")
    ref = np.array([400, 6400.0])
    ax1.plot(ref, 0.5 * (ref / 400) ** 1.5, ":", color=INK_MUTED, label="∝ (ρV)^1.5 (p = 0.75)")
    ax1.set_yscale("log")
    ax1.set_ylabel("(ΔS)² / J¹(400)   (lines: exact J¹)", color=INK)
    ax1.set_title("Total variance versus the same-pair term J¹", color=INK, fontsize=10)
    ax2.axhline(STEP2[0], color=INK_MUTED, linestyle=":", linewidth=1)
    ax2.text(420, STEP2[0] + 0.01, "Step 2 (ε = 1)", color=INK_MUTED, fontsize=8)
    ax2.set_ylabel("rest⊥ / J¹", color=INK)
    ax2.set_title("Correlations without the part linear in N", color=INK, fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("ρV", color=INK)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    path = ROOT / "figures" / "goal5_smeared4d.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figure", path)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sim":
        sim([int(x) for x in sys.argv[2:]])
    else:
        {"pred": pred, "fit": fit, "plot": lambda: plot(json.loads(FIT.read_text())["points"])}[cmd]()
