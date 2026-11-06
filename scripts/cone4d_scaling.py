"""Point 1c: scaling of (Delta S)^2 of the local BD action in the truncated
past light cone J^-(q) ∩ {t >= 0} of 4D Minkowski.

Prediction (derived before computing, see README "Point 1c"): the almost null
pairs of proper time tau from x have rapidity up to chi_max with
tau e^chi_max / 2 = l(x, n), the length (in t) of the past null ray from x in
the direction n inside the region. With sinh^2 chi ≈ e^(2 chi)/4,
    <N_k> ≈ (rho V)^1.5 · (G / V^1.5) / (8 sqrt(c_4)) · Γ(k+½)/k!,
    G = ∫_M dV_x ∫_{S²} l(x, n)² dOmega,    c_4 = pi/24.
In the cone, l = t_x and G = 4π ∫ t² dV (closed form); in the diamond G is
computed by Monte Carlo. Same exponent (p -> 0.75) and prefactor
proportional to G: G_cone / G_diamond = 0.566 at equal volume.

Criteria (fixed before computing in the cone; the asymptotic formula was
calibrated beforehand in the diamond: exact/asymptotic ratio 0.995-0.987 at
rho V = 1e8):
  H1  exact <N_k> in the cone (``continuum.expected_abundance_cone_d``) versus
      sprinklings: |z| < 4.
  H2  ratio exact <N_k> / asymptotic in the cone, k = 0..3: |1 - r| decreases
      monotonically between rho V = 1e6, 1e7 and 1e8, and is < 3 % at 1e8.
  H3  rest/J¹ (correlations, = total - J¹ - NN - cross term): slope versus
      ln(rho V) compatible with 0 at 4σ (same rule as Step 2).

Phases:
  sim   sprinklings -> figures/cone4d_sim.json (incremental)
  fit   criteria, comparison with the diamond, figure
        figures/cone4d_scaling.png and summary figures/cone4d_results.json

Usage:  python scripts/cone4d_scaling.py sim | fit
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
from scipy.special import gamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causets import sprinkle  # noqa: E402
from causets.causal import minkowski_causal_matrix  # noqa: E402
from causets.continuum import expected_abundance, expected_abundance_cone_d  # noqa: E402
from causets.fluctuations import EnsembleFluctuations, abundance_vector, local_action_coefficients  # noqa: E402
from causets.geometry import MinkowskiConeGeometry  # noqa: E402
from causets.regions import CausalDiamond, TruncatedPastCone  # noqa: E402
from causets.sprinkling import sample_uniform  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from goal3d_log_origin import fit_log  # noqa: E402

warnings.filterwarnings("ignore")
SIM = ROOT / "figures" / "cone4d_sim.json"
FIT = ROOT / "figures" / "cone4d_results.json"
PLAN = {25: 4000, 50: 4000, 100: 4000, 200: 2000, 400: 2000, 800: 2000, 1600: 1000, 3200: 1000, 6400: 500}
ASYMP_LAMS = (1e6, 1e7, 1e8)
C4 = np.pi / 24
K_SIGMA = 4.0
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def g_cone() -> float:
    """G / V^1.5 of the cone: G = 4π ∫ t² dV = 4π (4π/3) h^6 B(3, 4)."""
    cone = TruncatedPastCone(4, 1.0)
    return 4 * np.pi * (4 * np.pi / 3) / 60 / cone.volume**1.5


def g_diamond(n_batches: int = 20, m: int = 500_000, seed: int = 0) -> tuple[float, float]:
    """G / V^1.5 of the diamond (half-height 1) by Monte Carlo: the past null
    ray x - λ(1, n) leaves through the future cone of p = (-1, 0) at
    λ = ((t+1)² - r²) / (2 (t + 1 - x·n))."""
    rng = np.random.default_rng(seed)
    D = CausalDiamond(4, 1.0)
    means = []
    for _ in range(n_batches):
        x = sample_uniform(D, m, rng)
        n = rng.normal(size=(m, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        t, r2, xn = x[:, 0], np.sum(x[:, 1:] ** 2, 1), np.sum(x[:, 1:] * n, 1)
        means.append(np.mean((((t + 1) ** 2 - r2) / (2 * (t + 1 - xn))) ** 2))
    k = 4 * np.pi / D.volume**0.5
    return float(k * np.mean(means)), float(k * np.std(means, ddof=1) / np.sqrt(n_batches))


def g_flrw(alpha: float, nodes: int = 64) -> float:
    """G / V^1.5 of the past cone down to the big bang of 4D FLRW, a = eta^alpha
    (point 1d). Generalises G = ∫dV ∫dOmega l² to a scale factor:

        G = ∫ dV_x ∫ dOmega ∫_0^{eta_x} 2 lambda a(eta_x - lambda)^4 / sqrt(w) d lambda,
        w(eta_y, lambda) = (6 / lambda³) ∫_0^lambda a(eta_y + s)^4 s (lambda - s) ds,

    where w is the mean of a^4 over an almost null interval of extent lambda
    (profile s(lambda - s) of the section of a thin diamond in 4D): its physical
    volume is c_4 tau^4 w with tau the conformal proper time. In the cone the
    past null ray reaches eta = 0, so l = eta_x and ∫dOmega = 4π. With a = 1
    it reduces to ``g_cone``. Gauss-Legendre quadrature (smooth integrands)."""
    from causets.geometry import FLRWConeGeometry

    gx, gw = np.polynomial.legendre.leggauss(nodes)
    u, wu = (gx + 1) / 2, gw / 2  # nodes in [0, 1]

    def a4(eta):
        return np.asarray(eta, float) ** (4 * alpha)

    def inner(ex):  # ∫_0^{ex} 2 lam a(ex - lam)^4 / sqrt(w) dlam
        lam = ex * u
        ey = ex - lam
        s = lam[:, None] * u[None, :]
        w = 6 / lam**3 * np.sum(wu * lam[:, None] * a4(ey[:, None] + s) * s * (lam[:, None] - s), axis=1)
        return ex * np.sum(wu * 2 * lam * a4(ey) / np.sqrt(w))

    G = 4 * np.pi * sum(wk * a4(e) * (4 * np.pi / 3) * (1 - e) ** 3 * inner(e) for e, wk in zip(u, wu))
    return float(G / FLRWConeGeometry(alpha, 1.0, 4).volume ** 1.5)


def asymptotic_abundance(g: float, lam: float, k: int) -> float:
    return lam**1.5 * g / (8 * np.sqrt(C4)) * gamma(k + 0.5) / gamma(k + 1)


def exact_parts(lam: float, means=expected_abundance_cone_d) -> dict:
    a = local_action_coefficients(4)
    aN, ak = a[0], a[1:]
    L = np.array([means(4, lam, k) for k in range(5)])
    j = np.arange(4)
    MNk = (j + 2) * L[:4] - (j + 1) * L[1:]
    return dict(L=L.tolist(), NN=aN**2 * lam, Nk=2 * aN * float(ak @ MNk), J1=float(ak**2 @ L[:4]))


def sim():
    g = MinkowskiConeGeometry(4)
    a = local_action_coefficients(4)
    rows = json.loads(SIM.read_text()) if SIM.exists() else []
    done = {r["rhoV"] for r in rows}
    for lam, m in PLAN.items():
        if lam in done:
            continue
        rng = np.random.default_rng(101_000 + lam)
        t0 = time.time()
        ab = np.array([abundance_vector(minkowski_causal_matrix(sprinkle(g.region, lam / g.volume, rng)), 4)
                       for _ in range(m)])
        E = EnsembleFluctuations(ab, a)
        row = dict(rhoV=lam, sim=E.variance, sim_se=E.variance_se, n_sprinklings=m,
                   means=ab.mean(axis=0).tolist(), means_se=(ab.std(axis=0, ddof=1) / np.sqrt(m)).tolist(),
                   t=time.time() - t0)
        print(json.dumps(row), flush=True)
        rows.append(row)
        SIM.write_text(json.dumps(rows, indent=1))


def fit():
    rows = sorted(json.loads(SIM.read_text()), key=lambda r: r["rhoV"])
    gc = g_cone()
    gd, gd_se = g_diamond()
    out = dict(G_cone=gc, G_diamond=gd, G_diamond_se=gd_se, G_ratio=gc / gd, H1=[], points=[])
    for r in rows:
        ex = exact_parts(r["rhoV"])
        z = [(m - l) / se for m, l, se in zip(r["means"][1:], ex["L"][:4], r["means_se"][1:])]
        out["H1"].append(dict(rhoV=r["rhoV"], z=z))
        rest = r["sim"] - ex["NN"] - ex["Nk"] - ex["J1"]
        out["points"].append(dict(rhoV=r["rhoV"], sim=r["sim"], sim_se=r["sim_se"], J1=ex["J1"], NN=ex["NN"], Nk=ex["Nk"],
                                  rest_over_J1=rest / ex["J1"], rest_over_J1_se=r["sim_se"] / ex["J1"]))
    # H2: convergence to the asymptotic prefactor
    ratios = {str(lam): [expected_abundance_cone_d(4, lam, k) / asymptotic_abundance(gc, lam, k) for k in range(4)]
              for lam in ASYMP_LAMS}
    dev = np.array([[abs(1 - v) for v in ratios[str(lam)]] for lam in ASYMP_LAMS])
    out["H2"] = dict(ratios=ratios, monotonic=bool(np.all(np.diff(dev, axis=0) < 0)), max_dev_last=float(dev[-1].max()))
    # H3: slope of rest/J¹ versus ln rho V
    lam = np.array([p["rhoV"] for p in out["points"]], float)
    y = np.array([p["rest_over_J1"] for p in out["points"]])
    se = np.array([p["rest_over_J1_se"] for p in out["points"]])
    b, b_se, c, chi2 = fit_log(lam, y, se)
    w = 1 / se**2
    out["H3"] = dict(slope=b, slope_se=b_se, chi2dof=chi2, z=b / b_se,
                     mean=float(np.sum(w * y) / np.sum(w)), mean_se=float(1 / np.sqrt(np.sum(w))))
    # local exponents and comparison with the diamond (Step 2) at equal rho V
    out["p_local_sim"] = [dict(rhoV=(int(lam[i]), int(lam[i + 1])),
                               p=float(np.log(rows[i + 1]["sim"] / rows[i]["sim"]) / np.log(lam[i + 1] / lam[i]) / 2),
                               se=float(np.hypot(rows[i]["sim_se"] / rows[i]["sim"], rows[i + 1]["sim_se"] / rows[i + 1]["sim"])
                                        / np.log(lam[i + 1] / lam[i]) / 2)) for i in range(len(rows) - 1)]
    diamond = {p["rhoV"]: p for p in json.loads((ROOT / "figures" / "step2_4d_results.json").read_text())["points"]}
    out["vs_diamond"] = [dict(rhoV=p["rhoV"], J1_ratio=p["J1"] / diamond[p["rhoV"]]["J1"],
                              total_ratio=p["sim"] / diamond[p["rhoV"]]["sim"],
                              total_ratio_se=p["sim"] / diamond[p["rhoV"]]["sim"]
                              * np.hypot(p["sim_se"] / p["sim"], diamond[p["rhoV"]]["sim_se"] / diamond[p["rhoV"]]["sim"]))
                         for p in out["points"] if p["rhoV"] in diamond]
    lam_j = np.geomspace(25, 1e8, 22)
    j1 = np.array([exact_parts(x)["J1"] for x in lam_j])
    j1d = np.array([exact_parts(x, expected_abundance)["J1"] for x in lam_j])
    out["J1_curve"] = dict(rhoV=lam_j.tolist(), J1_cone=j1.tolist(), J1_diamond=j1d.tolist())
    FIT.write_text(json.dumps(out, indent=1))

    print(f"G/V^1.5: cone {gc:.4f}, diamond {gd:.4f} ± {gd_se:.4f}, ratio {gc / gd:.4f}")
    print("H1 max |z|:", max(max(abs(v) for v in r["z"]) for r in out["H1"]))
    print("H2:", {k: [round(v, 4) for v in vs] for k, vs in ratios.items()}, "monotonic:", out["H2"]["monotonic"])
    for p in out["points"]:
        print(f"ρV={p['rhoV']:<5} total/ρV={p['sim'] / p['rhoV']:9.1f}±{p['sim_se'] / p['rhoV']:<6.1f} "
              f"J1/ρV={p['J1'] / p['rhoV']:9.1f}  rest/J1={p['rest_over_J1']:+.3f}±{p['rest_over_J1_se']:.3f}")
    print(f"H3: slope {b:+.4f} ± {b_se:.4f} (z = {b / b_se:+.2f}, chi2/dof {chi2:.2f}); "
          f"mean rest/J1 = {out['H3']['mean']:.3f} ± {out['H3']['mean_se']:.3f}")
    for p in out["p_local_sim"]:
        print(f"local p {p['rhoV']}: {p['p']:.3f} ± {p['se']:.3f}")
    for v in out["vs_diamond"]:
        print(f"ρV={v['rhoV']}: J1 cone/diamond {v['J1_ratio']:.3f}, total {v['total_ratio']:.3f} ± {v['total_ratio_se']:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax1.plot(lam_j, j1 / lam_j, "-", color=BLUE, linewidth=2, label="exact J¹, cone")
    ax1.plot(lam_j, j1d / lam_j, "--", color=INK_MUTED, linewidth=1.4, label="exact J¹, diamond")
    ax1.errorbar(lam, [p["sim"] / p["rhoV"] for p in out["points"]], yerr=[p["sim_se"] / p["rhoV"] for p in out["points"]],
                 fmt="o", color=INK, markerfacecolor="white", capsize=3, label="total (ΔS)², simulation (cone)")
    ref = lam_j[lam_j >= 1e5]
    asym = sum(a**2 * asymptotic_abundance(gc, 1.0, k) for k, a in enumerate(local_action_coefficients(4)[1:]))
    ax1.plot(ref, asym * ref**0.5, ":", color=BLUE, label="asymptotic ∝ G (ρV)^1.5")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("ρV", color=INK)
    ax1.set_ylabel("(ΔS)² / ρV", color=INK)
    ax1.set_title("Truncated past cone, 4D Minkowski, local action", color=INK, fontsize=10)
    ax1.legend(frameon=False, fontsize=8)
    ax2.errorbar(lam, y, yerr=se, fmt="o", color=INK, markerfacecolor="white", capsize=3, label="rest / J¹ (cone)")
    ax2.axhline(out["H3"]["mean"], color=GREEN, linestyle=":", linewidth=1.2, label=f"mean {out['H3']['mean']:.3f}")
    ax2.axhline(0.317, color=INK_MUTED, linestyle="--", linewidth=1, label="diamond (Step 2): 0.317")
    ax2.set_xscale("log")
    ax2.set_xlabel("ρV", color=INK)
    ax2.set_ylabel("correlations / J¹", color=INK)
    ax2.set_title("Fraction of correlations", color=INK, fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    path = ROOT / "figures" / "cone4d_scaling.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figure", path)


if __name__ == "__main__":
    {"sim": sim, "fit": fit}[sys.argv[1]]()
