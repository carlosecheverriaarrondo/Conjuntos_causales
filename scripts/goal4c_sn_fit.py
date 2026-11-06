"""Goal 4c: Everpresent Λ (Model 1) versus the fluctuations of the BD action
with real supernova data (Pantheon+SH0ES), following the analysis of Das,
Nasiri & Yazdi, JCAP 10 (2024) 076, arXiv:2307.13743, sec. 4.

Method (as DNY sec. 4.1): each "seed" fixes the sequence of random numbers
of the process; for each seed the best fit in (omega_m, amplitude, M_B) is
searched for and its minimum chi² is stored. M_B is minimised analytically;
(omega_m, amplitude) with a grid and two local refinements (DNY use MCMC per
seed). Data: Pantheon+SH0ES (Brout et al., ApJ 938 (2022) 110,
arXiv:2202.04077; Riess et al., ApJL 934 (2022) L7), standard cut zHD > 0.01
or calibrator, STAT+SYS covariance.

CRITERIA (fixed on 2026-10-10, BEFORE computing any chi²):
  V1  minimum chi² of flat ΛCDM (omega_m, h, M_B) versus that of DNY, 1485.3:
      |difference| < 2 (same data and cut). If it fails, this is documented and
      the thresholds are taken relative to OUR chi²_ΛCDM.
  V2  compressed chi² (spline of mu at nodes) = direct chi², |Δ| < 0.01 (test).
  V3  Model 1 versus DNY: fraction of seeds with chi² < chi²_ΛCDM + 14.7
      ("good"; DNY: 1500 = 1485.3 + 14.7, fraction ≈ 0.003 read off their
      fig. 1a) within [0.001, 0.009] (factor 3: grid versus MCMC, continuous
      process versus 10 000 steps, value read off a figure).
      Fraction with chi² < chi²_ΛCDM ("better"; DNY 16/90 000) compatible at
      4σ (Poisson).
  Main comparison: BD action (q = 1/4, beta = 0.9, from Goal 4b-1)
  versus Model 1, with the same number of seeds, the same data and the
  same grid in omega_m. Fraction of "good" seeds: BD better / worse
  if the difference exceeds 4σ (pooled binomial); otherwise indistinguishable.
  The "better" fraction and the parameters of the good seeds
  (omega_m, h, amplitude -> l_k) are also reported.

Out of scope: BAO and CMB (DNY sec. 5 use the full Planck likelihood).

Usage:  python scripts/goal4c_sn_fit.py lcdm | seeds MODEL START STOP | fit
        MODEL = model1 | bd
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
from scipy.integrate import quad
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causets import stochastic_lambda as sl  # noqa: E402
from causets.supernovae import PantheonPlusSH0ES, mu_from_dl  # noqa: E402

DNY_CHI2_LCDM = 1485.3
GOOD_MARGIN = 14.7  # DNY: 1500 - 1485.3
DNY_GOOD_FRAC = 0.003
DNY_BETTER = (16, 90_000)
BETA = 0.9
A_START, N_STEPS = 1e-5, 2000
N_SEEDS = 10_000
BATCH = 50
SEED0 = 4_000_000
OMEGA_M_GRID = np.geomspace(0.02, 0.8, 10)  # DNY prior: [0.01, 1]
AMP_GRID = {"model1": np.geomspace(0.001, 0.025, 10),  # alpha, DNY prior
            "bd": np.geomspace(0.1, 10.0, 10)}  # lambda of the BD action
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREEN, BLUE = "#1baf7a", "#2a78d6"


def spec_for(model, amp):
    if model == "model1":
        return sl.ProcessSpec(8 * np.pi * amp, 0.5, 0.5, "Model 1")
    return sl.ProcessSpec(amp, 0.25, 0.75 * BETA, "BD")


def lcdm_mu_nodes(om, h, nodes):
    orad = sl.OMEGA_R / h**2
    ol = 1 - om / h**2 - orad

    def inv_E(z):
        return 1 / np.sqrt(orad * (1 + z) ** 4 + om / h**2 * (1 + z) ** 3 + ol)

    chi = np.array([quad(inv_E, 0, z, epsrel=1e-11)[0] for z in nodes]) / h
    return mu_from_dl(chi * (1 + nodes))


def lcdm():
    sn = PantheonPlusSH0ES()

    def f(p):
        om, h = p
        if not (0.01 < om < 1 and 0.4 < h < 1 and om / h**2 < 1):
            return 1e9
        return float(sn.chi2_nodes(lcdm_mu_nodes(om, h, sn.nodes))[0])

    best = min((minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6)) for x0 in
                [(0.14, 0.73), (0.17, 0.70), (0.12, 0.75)]), key=lambda r: r.fun)
    om, h = best.x
    chi2, MB = sn.chi2_nodes(lcdm_mu_nodes(om, h, sn.nodes))
    # same cosmology with evolve (constant Λ) to measure the integration error
    real = sl.evolve(sl.ProcessSpec(3 * (h**2 - om - sl.OMEGA_R), 0, 0), 1, None, a_start=A_START,
                     n_steps=N_STEPS, z_grid=sn.nodes, s_fixed=1.0, omega_m=om)
    chi2_ev = float(sn.chi2_nodes(mu_from_dl(real.dL[0]))[0])
    out = dict(n_sn=sn.n, omega_m=om, h=h, Omega_m=om / h**2, M_B=float(MB), chi2=float(chi2),
               chi2_evolve=chi2_ev, dny_chi2=DNY_CHI2_LCDM, V1=bool(abs(chi2 - DNY_CHI2_LCDM) < 2))
    print(json.dumps(out, indent=1))
    (ROOT / "figures" / "goal4c_lcdm.json").write_text(json.dumps(out, indent=1))


def _eval(sn, model, seeds_noise, om, amp):
    """chi² minimised over M_B for each (seed, om, amp); flat arrays of equal length."""
    R = len(om)
    real = sl.evolve(spec_for(model, amp), R, None, a_start=A_START, n_steps=N_STEPS, z_grid=sn.nodes,
                     omega_m=om, noise=seeds_noise)
    mu = mu_from_dl(real.dL)
    chi2, MB = sn.chi2_nodes(np.nan_to_num(mu))
    bad = real.collapsed | ~np.all(np.isfinite(mu), axis=1)
    chi2 = np.where(bad, np.inf, chi2)
    return chi2, MB, real.h, real.omega_lambda0


def fit_batch(sn, model, seeds, evaluator=None):
    """``evaluator(model, noise, om, amp) -> (chi2, M_B, h, Omega_L0)``; by default, SN only (Goal 4c)."""
    if evaluator is None:
        evaluator = lambda m, nz, om, amp: _eval(sn, m, nz, om, amp)  # noqa: E731
    B = len(seeds)
    noise = np.stack([np.random.default_rng(SEED0 + s).standard_normal(N_STEPS) for s in seeds], axis=1)
    lom, lamp = np.log(OMEGA_M_GRID), np.log(AMP_GRID[model])
    d_om, d_amp = lom[1] - lom[0], lamp[1] - lamp[0]
    # grid: centres per seed (B, n_om, n_amp)
    cand_om = np.broadcast_to(lom[None, :, None], (B, len(lom), len(lamp)))
    cand_amp = np.broadcast_to(lamp[None, None, :], (B, len(lom), len(lamp)))
    best = None
    for stage in range(3):
        if stage > 0:
            frac = 0.5 ** stage
            off = np.linspace(-1, 1, 5) * frac
            cand_om = np.clip(best["lom"][:, None, None] + off[None, :, None] * d_om, lom[0], lom[-1])
            cand_amp = np.clip(best["lamp"][:, None, None] + off[None, None, :] * d_amp, lamp[0], lamp[-1])
            cand_om, cand_amp = np.broadcast_arrays(cand_om, cand_amp)
        G = cand_om.shape[1] * cand_om.shape[2]
        idx = np.repeat(np.arange(B), G)
        chi2, MB, h, ol = evaluator(model, noise[:, idx], np.exp(cand_om.reshape(-1)), np.exp(cand_amp.reshape(-1)))
        chi2 = chi2.reshape(B, G)
        j = np.argmin(chi2, axis=1)
        sel = np.arange(B) * G + j
        new = dict(chi2=chi2[np.arange(B), j], lom=cand_om.reshape(B, G)[np.arange(B), j],
                   lamp=cand_amp.reshape(B, G)[np.arange(B), j], MB=MB[sel], h=h[sel], ol=ol[sel])
        if best is None:
            best = new
        else:
            better = new["chi2"] < best["chi2"]
            best = {k: np.where(better, new[k], best[k]) for k in best}
    return [dict(seed=int(s), chi2=float(best["chi2"][b]), omega_m=float(np.exp(best["lom"][b])),
                 amp=float(np.exp(best["lamp"][b])), M_B=float(best["MB"][b]), h=float(best["h"][b]),
                 Omega_L0=float(best["ol"][b])) for b, s in enumerate(seeds)]


def seeds_path(model):
    return ROOT / "figures" / f"goal4c_seeds_{model}.json"


def seeds(model):
    sn = PantheonPlusSH0ES()
    path = seeds_path(model)
    rows = json.loads(path.read_text())["rows"] if path.exists() else []
    t0 = time.time()
    while len(rows) < N_SEEDS:
        start = len(rows)
        rows += fit_batch(sn, model, list(range(start, min(start + BATCH, N_SEEDS))))
        path.write_text(json.dumps(dict(model=model, beta=BETA if model == "bd" else None, rows=rows)))
        print(f"{model}: {len(rows)} seeds, {time.time() - t0:.0f} s", flush=True)


def _frac(k, n):
    return k / n, np.sqrt(max(k, 1) / n * (1 - k / n) / n)


def fit():
    ref = json.loads((ROOT / "figures" / "goal4c_lcdm.json").read_text())
    chi2_ref = ref["chi2"]
    out = dict(chi2_lcdm=chi2_ref, V1=ref["V1"], models={})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for model, col, label in (("model1", GREEN, "Model 1 (DNY)"), ("bd", BLUE, f"BD action (β = {BETA:g})")):
        if not seeds_path(model).exists():
            continue
        rows = json.loads(seeds_path(model).read_text())["rows"]
        n = len(rows)
        chi2 = np.array([r["chi2"] for r in rows])
        good, better = chi2 < chi2_ref + GOOD_MARGIN, chi2 < chi2_ref
        g = [r for r, ok in zip(rows, good) if ok]
        par = {k: [float(np.percentile([r[k] for r in g], p)) for p in (16, 50, 84)] if g else None
               for k in ("omega_m", "h", "amp", "Omega_L0")}
        if g:
            par["Omega_m"] = [float(np.percentile([r["omega_m"] / r["h"] ** 2 for r in g], p)) for p in (16, 50, 84)]
            if model == "bd":
                par["l_k_m"] = [sl.bd_lambda_to_lk(a) for a in par["amp"]]
        m = dict(n=n, n_good=int(good.sum()), n_better=int(better.sum()), f_good=_frac(good.sum(), n),
                 f_better=_frac(better.sum(), n), chi2_min=float(chi2.min()),
                 frac_at_amp_floor=float(np.mean([r["amp"] <= AMP_GRID[model][0] * 1.0001 for r in rows])),
                 good_params=par, best=sorted(rows, key=lambda r: r["chi2"])[:5])
        out["models"][model] = m
        print(f"--- {label}: {n} seeds; chi2 min = {m['chi2_min']:.1f} (ΛCDM {chi2_ref:.1f})")
        print(f"   good (< ΛCDM + {GOOD_MARGIN}): {m['n_good']}  f = {m['f_good'][0]:.4f} ± {m['f_good'][1]:.4f}")
        print(f"   better than ΛCDM: {m['n_better']}  f = {m['f_better'][0]:.5f}")
        print(f"   parameters of the good seeds (16/50/84 %): {json.dumps(par)}")
        xs = np.sort(chi2 - chi2_ref)
        xs = xs[np.isfinite(xs)]
        ax1.step(xs, np.arange(1, len(xs) + 1) / n, where="post", color=col, label=label)
        if g:
            ax2.plot([r["omega_m"] / r["h"] ** 2 for r in g], [100 * r["h"] for r in g], "o", color=col,
                     markersize=4, alpha=0.7, label=label)
    if "model1" in out["models"]:
        m1 = out["models"]["model1"]
        k_dny, n_dny = DNY_BETTER
        p_pool = (m1["n_better"] + k_dny) / (m1["n"] + n_dny)
        se = np.sqrt(p_pool * (1 / m1["n"] + 1 / n_dny))
        out["V3"] = dict(f_good_in_range=bool(0.001 <= m1["f_good"][0] <= 0.009),
                         better_z=float((m1["f_better"][0] - k_dny / n_dny) / se))
        out["V3"]["ok"] = out["V3"]["f_good_in_range"] and abs(out["V3"]["better_z"]) < 4
        print(f"V3: f_good in [0.001, 0.009]: {out['V3']['f_good_in_range']}; "
              f"better versus DNY: z = {out['V3']['better_z']:+.2f}  -> {out['V3']['ok']}")
    if len(out["models"]) == 2:
        a, b = out["models"]["bd"], out["models"]["model1"]
        z = (a["f_good"][0] - b["f_good"][0]) / np.hypot(a["f_good"][1], b["f_good"][1])
        out["comparison"] = dict(z=float(z), verdict="BD mejor" if z > 4 else "BD peor" if z < -4 else "indistinguibles")
        print(f"Comparison (fraction of good seeds): z = {z:+.2f} -> {out['comparison']['verdict']}")
    ax1.axvline(0, color=INK_MUTED, linestyle="--", linewidth=1)
    ax1.axvline(GOOD_MARGIN, color=INK_MUTED, linestyle=":", linewidth=1)
    ax1.set_xlim(-30, 300)
    ax1.set_yscale("log")
    ax1.set_xlabel("χ²_min(seed) − χ²_ΛCDM  (Pantheon+SH0ES)", color=INK)
    ax1.set_ylabel("fraction of seeds below", color=INK)
    ax1.set_title("Distribution of the best fit per seed", color=INK, fontsize=10)
    ax2.axhline(100 * ref["h"], color=INK_MUTED, linestyle="--", linewidth=1)
    ax2.axvline(ref["Omega_m"], color=INK_MUTED, linestyle="--", linewidth=1, label="ΛCDM (best fit)")
    ax2.set_xlabel("Ω_m0", color=INK)
    ax2.set_ylabel("H0 (km/s/Mpc)", color=INK)
    ax2.set_title(f"Good seeds (χ² < χ²_ΛCDM + {GOOD_MARGIN})", color=INK, fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    (ROOT / "figures" / "goal4c_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    path = ROOT / "figures" / "goal4c_sn.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figure", path)


if __name__ == "__main__":
    if sys.argv[1] == "lcdm":
        lcdm()
    elif sys.argv[1] == "seeds":
        seeds(sys.argv[2])
    else:
        fit()
