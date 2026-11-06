"""Goal 4a: are the fluctuations of the BD action in the causal past of the
observer compatible with the observed Λ? (See README "Goal 4a" and the
docstring of ``causets/everpresent.py`` for the central assumption and the
formulae.)

Decision rules (fixed before computing):
  D1  local action: excluded if the discreteness scale l required for
      delta Λ = Λ_obs leaves rho V < 1e3 today (no continuum regime).
  D2  smeared action: open route if the required non-locality scale l_k
      is < 1e-19 m (LHC bound cited by Belenchia et al., Phys. Rev.
      Lett. 116, 161303 (2016), arXiv:1512.02083), for l = l_p and for
      l = l_cs of Model 1 (alpha = 0.01), with r in [0, 1].
  D3  evolution: envelope delta Λ / (3 H²) versus z and effective w of the
      envelope in the matter era, compared with Model 1.

Background cosmology: Planck 2018 (H0 = 67.4 km/s/Mpc, Omega_m = 0.315,
Omega_r = 9.1e-5, Omega_Λ = 1 - Omega_m - Omega_r), only to fix V(t) and H(t).

Usage:  python scripts/goal4_compatibility.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from causets import everpresent as ep  # noqa: E402
from cone4d_scaling import g_cone, g_flrw  # noqa: E402

OUT = ROOT / "figures" / "goal4a_results.json"
H0_KMS, OM, ORAD = 67.4, 0.315, 9.1e-5
OL = 1 - OM - ORAD
ALPHA_M1 = 0.01  # Model 1 (DNY 2023 explore 0.005-0.012)
L_CS = np.sqrt(1 / (2 * ALPHA_M1))  # l_cs / l_p (DNY eq. 3.9)
R_LOCAL = (0.317, 0.348)  # measured correlations/J¹ (points 1c-1d)
LHC_BOUND_M = 1e-19
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def main():
    H0 = ep.hubble_planck(H0_KMS)
    v0 = ep.past_cone_volume(1.0, OM, ORAD, OL) / H0**4  # V today in l_p^4
    lam_obs = 3 * OL * H0**2  # Λ_obs l_p²
    G = {"cono Minkowski": g_cone(), "FLRW radiación": g_flrw(1.0), "FLRW materia": g_flrw(2.0)}
    out = dict(V_today_planck=v0, Lambda_obs_planck=lam_obs, G=G, l_cs_over_lp=L_CS)

    # --- D1: local action -------------------------------------------------
    A_loc = {k: [ep.amplitude(g, 1.0, r) for r in R_LOCAL] for k, g in G.items()}
    A_lo, A_hi = min(min(v) for v in A_loc.values()), max(max(v) for v in A_loc.values())
    d1 = dict(A_range=(A_lo, A_hi))
    for name, l in (("l_p", 1.0), ("l_cs", L_CS)):
        d1[f"ratio_{name}"] = [ep.delta_lambda_planck(A, v0, l) / lam_obs for A in (A_lo, A_hi)]
    # required l: A l_p² / (l V^(1/4)) = Λ_obs  ->  l / l_p = A / (Λ_obs V^(1/4))
    l_req = [A / (lam_obs * v0**0.25) for A in (A_lo, A_hi)]
    d1["l_required_over_lp"] = l_req
    d1["l_required_m"] = [x * ep.L_PLANCK_M for x in l_req]
    d1["rhoV_today_at_l_required"] = [v0 / x**4 for x in l_req]
    d1["excluded"] = bool(max(d1["rhoV_today_at_l_required"]) < 1e3)
    out["D1"] = d1

    # --- D2: smeared action ----------------------------------------------
    c_asym = ep.smeared_asymptotic_constant()
    d2 = dict(asymptotic_constant=c_asym, cases=[])
    for name, l in (("l_p", 1.0), ("l_cs", L_CS)):
        for gname in ("FLRW radiación", "FLRW materia"):
            for r in (0.0, 0.33, 1.0):
                # required A(eps): delta Λ l_p² = A (V/l^4)^(-1/4) / l² = Λ_obs
                A_req = lam_obs * l**2 * (v0 / l**4) ** 0.25
                S_req = A_req**2 * 8 * np.sqrt(ep.C4) / ((1 + r) * G[gname])
                eps0 = (S_req / c_asym) ** 0.4
                # refinement with the exact sum (valid if eps0 >= 1e-6; otherwise the asymptote)
                if eps0 > 1e-5:
                    eps = brentq(lambda e: np.log(ep.same_pair_sum(e) / S_req), eps0 / 3, min(1.0, eps0 * 3))
                else:
                    eps = eps0
                l_k_m = l * ep.L_PLANCK_M * eps**-0.25
                d2["cases"].append(dict(l=name, region=gname, r=r, eps=eps, l_k_over_l=eps**-0.25, l_k_m=l_k_m,
                                        open=bool(l_k_m < LHC_BOUND_M)))
    lk = [c["l_k_m"] for c in d2["cases"]]
    d2["l_k_range_m"] = (min(lk), max(lk))
    d2["open"] = bool(all(c["open"] for c in d2["cases"]))
    out["D2"] = d2

    # --- D3: evolution of the envelope -----------------------------------
    z = np.geomspace(1e-2, 1e4, 60)
    a = 1 / (1 + z)
    Hz = H0 * ep._E(a, OM, ORAD, OL)
    V = np.array([ep.past_cone_volume(x, OM, ORAD, OL) for x in a]) / H0**4
    A_m = ep.amplitude(G["FLRW materia"], 1.0, 0.33)
    case = next(c for c in d2["cases"] if c["l"] == "l_p" and c["region"] == "FLRW materia" and c["r"] == 0.33)
    A_sm = ep.amplitude(G["FLRW materia"], case["eps"], 0.33)
    env = {
        "Model 1 (α = 0.01)": ep.model1_delta_lambda(V, ALPHA_M1) / (3 * Hz**2),
        "BD local (l = l_p)": np.array([ep.delta_lambda_planck(A_m, v, 1.0) for v in V]) / (3 * Hz**2),
        "BD suavizada (l_k ajustada hoy)": np.array([ep.delta_lambda_planck(A_sm, v, 1.0) for v in V]) / (3 * Hz**2),
    }
    # effective w of the envelope in the matter era (z = 10 .. 100): rho ∝ a^(-3(1+w))
    sel = (z > 10) & (z < 100)
    w_eff = {}
    for k, y in env.items():
        dl = np.log(y * 3 * Hz**2)
        slope = np.polyfit(np.log(a[sel]), dl[sel], 1)[0]
        w_eff[k] = float(-1 - slope / 3)
    out["D3"] = dict(z=z.tolist(), envelope={k: v.tolist() for k, v in env.items()}, w_eff_matter_era=w_eff,
                     omega_env_z1100={k: float(np.interp(np.log(1100), np.log(z), v)) for k, v in env.items()})
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # --- report -------------------------------------------------------------
    print(f"V today = {v0:.3e} l_p^4;  Λ_obs = {lam_obs:.3e} l_p^-2;  l_cs(α=0.01) = {L_CS:.2f} l_p")
    print("G/V^1.5:", {k: round(v, 4) for k, v in G.items()})
    print(f"D1: local A = {A_lo:.2f} .. {A_hi:.2f}")
    print(f"    δΛ/Λ_obs with l = l_p: {d1['ratio_l_p'][0]:.2e} .. {d1['ratio_l_p'][1]:.2e};"
          f" with l = l_cs: {d1['ratio_l_cs'][0]:.2e} .. {d1['ratio_l_cs'][1]:.2e}")
    print(f"    required l = {l_req[0]:.2e} .. {l_req[1]:.2e} l_p = {d1['l_required_m'][0]:.2e} .. {d1['l_required_m'][1]:.2e} m;"
          f" ρV today = {min(d1['rhoV_today_at_l_required']):.2f} .. {max(d1['rhoV_today_at_l_required']):.2f}"
          f"  -> excluded: {d1['excluded']}")
    print(f"D2: asymptotic constant = {c_asym:.4f}")
    for c in d2["cases"]:
        print(f"    l = {c['l']:<4} {c['region']:<15} r = {c['r']:<4}: ε = {c['eps']:.3e}, l_k/l = {c['l_k_over_l']:.3e},"
              f" l_k = {c['l_k_m']:.2e} m  ({'< ' if c['open'] else '>= '}1e-19 m)")
    print(f"    l_k = {d2['l_k_range_m'][0]:.2e} .. {d2['l_k_range_m'][1]:.2e} m;  open route: {d2['open']}")
    print("D3: effective w of the envelope (z = 10..100):", {k: round(v, 3) for k, v in w_eff.items()})
    print("    δΛ/(3H²) at z = 1100:", {k: f"{v:.2e}" for k, v in out["D3"]["omega_env_z1100"].items()})

    # The dict keys are stored in the JSON (and read by the tests); the legend
    # uses English labels.
    legend = {"Model 1 (α = 0.01)": "Model 1 (α = 0.01)",
              "BD local (l = l_p)": "BD local (l = l_p)",
              "BD suavizada (l_k ajustada hoy)": "BD smeared (l_k fitted today)"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for (k, y), col in zip(env.items(), (GREEN, ORANGE, BLUE)):
        ax.plot(z, y, color=col, linewidth=2, label=legend.get(k, k))
    ax.plot(z, OL / ep._E(a, OM, ORAD, OL) ** 2, "--", color=INK_MUTED, linewidth=1.2, label="ΛCDM: Ω_Λ(z)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("z", color=INK)
    ax.set_ylabel("δΛ / (3H²)   (envelope)", color=INK)
    ax.set_title("Amplitude of the Λ fluctuations relative to the critical density", color=INK, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    path = ROOT / "figures" / "goal4a_envelope.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figure", path)


if __name__ == "__main__":
    main()
