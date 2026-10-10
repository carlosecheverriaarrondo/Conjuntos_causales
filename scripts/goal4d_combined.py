"""Goal 4d: Model 1 y acción BD frente a supernovas + BAO + CMB.

Extiende el Goal 4c (ajuste por semilla, Das-Nasiri-Yazdi, JCAP 10 (2024)
076, sec. 4) con:
  - BAO de DESI DR2 (DESI Collaboration, arXiv:2503.14738, tabla IV;
    ficheros de github.com/CobayaSampler/bao_data, desi_bao_dr2):
    D_V/r_d, D_M/r_d, D_H/r_d en 7 bins, covarianza 13×13.
  - CMB con los priors de distancia de Planck 2018 TT,TE,EE+lowE (Chen,
    Huang & Wang, JCAP 02 (2019) 028, arXiv:1808.05724, tabla I, ΛCDM):
    R = 1.7502 ± 0.0046, l_A = 301.471 ± 0.0895, omega_b = 0.02236 ± 0.00015,
    correlaciones 0.46, −0.66, −0.33. omega_b se fija a 0.02236 y se usa la
    gaussiana condicionada de (R, l_A).
Física (``causets/cmb_bao.py``): r_s(z) = ∫ c_s d eta con
c_s = 1/sqrt(3(1 + 3 omega_b a/(4 omega_γ))), omega_γ = 2.4728e-5 (T = 2.7255 K),
sobre la propia historia de cada realización (Λ estocástico incluido desde
z = 1e4). z* y z_drag: ajustes de Hu & Sugiyama (ApJ 471 (1996) 542, ec. E1)
y Eisenstein & Hu (ApJ 496 (1998) 605, ec. 4), desplazados por una
constante para que den los de Planck 2018 (A&A 641 A6, tabla 2,
TT,TE,EE+lowE: z* = 1089.95, z_drag = 1059.93) en su mejor ajuste ΛCDM
(omega_b = 0.02236, omega_m = 0.14327, h = 0.6727).

LIMITACIÓN CONOCIDA (antes de calcular): los priors de distancia suponen física
temprana de tipo ΛCDM. Vale para la acción BD (Λ/3H² ~ 4e-5 en z = 1100,
Goal 4a) pero NO para Model 1 (~0.5): para Model 1 el χ²_CMB es sólo
orientativo. La referencia para Model 1 es el análisis de DNY con el
likelihood completo de Planck (sec. 5: Δχ² > 150 frente a ΛCDM).

CRITERIOS (fijados el 2026-10-10, ANTES de calcular ningún χ² de BAO o CMB):
  W1  Validación con el mejor ajuste ΛCDM de Planck 2018 (arriba):
      r_* a menos del 0.3 % de 144.39 Mpc, r_drag a menos del 0.3 % de
      147.05 Mpc; R y l_A a menos de 1σ de Chen et al.
  W2  Validación: ΛCDM plano ajustado sólo a DESI DR2 BAO da
      Omega_m = 0.2975 ± 0.0086 (DESI DR2, BAO sola): |Δ| < 1σ.
  Principal: χ² = SN + BAO + CMB. Referencia ΛCDM: mínimo en (omega_m, h, M_B).
  Por semilla (10 000, las mismas del Goal 4c): mínimo en (omega_m,
  amplitud, M_B) con la rejilla y refinamientos del Goal 4c.
  "Buenas": χ² < χ²_ΛCDM + 14.7; "mejores": χ² < χ²_ΛCDM.
  BD mejor / peor que Model 1 si la diferencia en la fracción de buenas
  supera 4σ (binomial combinada); si no, indistinguibles. Si una fracción es
  0, se da el límite superior al 95 % (3/N).
  Secundario: fracción de semillas con χ²_CMB − χ²_CMB,ΛCDM < 9 en su mejor
  ajuste combinado.

Uso:  python scripts/goal4d_combined.py validate | lcdm | seeds MODEL | fit
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causets import stochastic_lambda as sl  # noqa: E402
from causets.cmb_bao import PLANCK_BF, CMBPriors, DESIDR2, early_quantities  # noqa: E402
from causets.supernovae import DL_UNIT_MPC, PantheonPlusSH0ES, mu_from_dl  # noqa: E402

_spec = importlib.util.spec_from_file_location("goal4c", ROOT / "scripts" / "goal4c_sn_fit.py")
g4c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g4c)

GOOD_MARGIN = g4c.GOOD_MARGIN
DESI_BAO_OMEGA_M = (0.2975, 0.0086)
INK, INK_MUTED, GRID = g4c.INK, g4c.INK_MUTED, g4c.GRID
GREEN, BLUE = g4c.GREEN, g4c.BLUE


class Combined:
    """``calibrate``: comprobación de robustez decidida DESPUÉS de que W1 fallara
    (l_A 1.45σ por debajo): l_A y r_drag se multiplican por constantes que hacen
    que el mejor ajuste de Planck dé exactamente l_A = 301.471 y r_drag = 147.05 Mpc."""

    def __init__(self, calibrate: bool = False):
        self.sn, self.bao, self.cmb = PantheonPlusSH0ES(), DESIDR2(), CMBPriors()
        self.K = len(self.sn.nodes)
        self.z_grid = np.concatenate([self.sn.nodes, self.bao.z_unique])
        self.cal_lA = self.cal_rd = 1.0
        if calibrate:
            p = self.lcdm(PLANCK_BF["omega_m"], PLANCK_BF["h"])
            self.cal_lA = 301.471 / float(p["lA"][0])
            self.cal_rd = PLANCK_BF["r_drag"] / float(p["r_drag"][0])

    def run(self, spec, om, noise=None, s_fixed=None):
        R = len(np.atleast_1d(om)) if noise is None else noise.shape[1]
        return sl.evolve(spec, R, None, a_start=g4c.A_START, n_steps=g4c.N_STEPS, z_grid=self.z_grid,
                         omega_m=om, noise=noise, s_fixed=s_fixed, record_path=True, exact_start=True)

    def parts(self, real, om):
        """χ² por partes (SN con M_B perfilado, BAO, CMB) y cantidades derivadas."""
        K = self.K
        mu = mu_from_dl(real.dL[:, :K])
        chi_sn, MB = self.sn.chi2_nodes(np.nan_to_num(mu))
        zb = self.z_grid[K:]
        dm = real.dL[:, K:] / (1 + zb)
        dh = 1 / (real.E[:, K:] * real.h[:, None])
        r_star, r_drag, R, lA = early_quantities(real, om)
        r_drag, lA = r_drag * self.cal_rd, lA * self.cal_lA
        chi_bao = self.bao.chi2(dm, dh, r_drag)
        chi_cmb = self.cmb.chi2(R, lA)
        bad = real.collapsed | ~np.all(np.isfinite(mu), axis=1) | ~np.isfinite(chi_bao + chi_cmb)
        inf = lambda x: np.where(bad, np.inf, x)  # noqa: E731
        return dict(sn=inf(chi_sn), bao=inf(chi_bao), cmb=inf(chi_cmb), M_B=MB, r_star=r_star * DL_UNIT_MPC,
                    r_drag=r_drag * DL_UNIT_MPC, R=R, lA=lA)

    def evaluator(self, model, noise, om, amp):
        real = self.run(g4c.spec_for(model, amp), om, noise=noise)
        p = self.parts(real, om)
        return p["sn"] + p["bao"] + p["cmb"], p["M_B"], real.h, real.omega_lambda0

    def lcdm(self, om, h):
        real = self.run(sl.ProcessSpec(3 * (h**2 - om - sl.OMEGA_R), 0.0, 0.0), np.array([om]), s_fixed=1.0)
        return self.parts(real, om)


def validate():
    c = Combined()
    p = c.lcdm(PLANCK_BF["omega_m"], PLANCK_BF["h"])
    w1 = dict(r_star=float(p["r_star"][0]), r_drag=float(p["r_drag"][0]), R=float(p["R"][0]), lA=float(p["lA"][0]))
    w1["dev_r_star"] = w1["r_star"] / PLANCK_BF["r_star"] - 1
    w1["dev_r_drag"] = w1["r_drag"] / PLANCK_BF["r_drag"] - 1
    w1["pull_R"] = (w1["R"] - 1.7502) / 0.0046
    w1["pull_lA"] = (w1["lA"] - 301.471) / 0.0895
    w1["ok"] = bool(abs(w1["dev_r_star"]) < 3e-3 and abs(w1["dev_r_drag"]) < 3e-3
                    and abs(w1["pull_R"]) < 1 and abs(w1["pull_lA"]) < 1)

    def f(x):
        om, h = x
        if not (0.05 < om < 0.4 and 0.5 < h < 0.9):
            return 1e9
        return float(c.lcdm(om, h)["bao"][0])

    r = minimize(f, (0.14, 0.68), method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7))
    Om = r.x[0] / r.x[1] ** 2
    w2 = dict(omega_m=float(r.x[0]), h=float(r.x[1]), Omega_m=float(Om), chi2_bao=float(r.fun),
              pull=float((Om - DESI_BAO_OMEGA_M[0]) / DESI_BAO_OMEGA_M[1]))
    w2["ok"] = bool(abs(w2["pull"]) < 1)
    out = dict(W1=w1, W2=w2)
    print(json.dumps(out, indent=1))
    (ROOT / "figures" / "goal4d_validation.json").write_text(json.dumps(out, indent=1))


def suffix(cal):
    return "_cal" if cal else ""


def lcdm(cal=False):
    c = Combined(cal)

    def f(x):
        om, h = x
        if not (0.02 < om < 0.8 and 0.4 < h < 1 and om + sl.OMEGA_R < h**2):
            return 1e9
        p = c.lcdm(om, h)
        return float(p["sn"][0] + p["bao"][0] + p["cmb"][0])

    best = min((minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-6))
                for x0 in [(0.14, 0.68), (0.15, 0.70), (0.13, 0.72)]), key=lambda r: r.fun)
    om, h = best.x
    p = c.lcdm(om, h)
    out = dict(calibrated=cal, cal_lA=c.cal_lA, cal_rd=c.cal_rd, omega_m=float(om), h=float(h),
               Omega_m=float(om / h**2), chi2=float(best.fun),
               **{k: float(p[k][0]) for k in ("sn", "bao", "cmb", "M_B", "r_drag", "R", "lA")})
    print(json.dumps(out, indent=1))
    (ROOT / "figures" / f"goal4d_lcdm{suffix(cal)}.json").write_text(json.dumps(out, indent=1))


def seeds_path(model, cal):
    return ROOT / "figures" / f"goal4d_seeds_{model}{suffix(cal)}.json"


def seeds(model, cal=False):
    c = Combined(cal)
    path = seeds_path(model, cal)
    rows = json.loads(path.read_text())["rows"] if path.exists() else []
    t0 = time.time()
    while len(rows) < g4c.N_SEEDS:
        start = len(rows)
        batch = list(range(start, min(start + g4c.BATCH, g4c.N_SEEDS)))
        new = g4c.fit_batch(c.sn, model, batch, evaluator=c.evaluator)
        # χ² por partes en el mejor punto de cada semilla
        noise = np.stack([np.random.default_rng(g4c.SEED0 + s).standard_normal(g4c.N_STEPS) for s in batch], axis=1)
        om = np.array([r["omega_m"] for r in new])
        p = c.parts(c.run(g4c.spec_for(model, np.array([r["amp"] for r in new])), om, noise=noise), om)
        for k, r in enumerate(new):
            r.update(sn=float(p["sn"][k]), bao=float(p["bao"][k]), cmb=float(p["cmb"][k]))
        rows += new
        path.write_text(json.dumps(dict(model=model, calibrated=cal, rows=rows)))
        print(f"{model}{suffix(cal)}: {len(rows)} semillas, {time.time() - t0:.0f} s", flush=True)


def fit():
    out = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    sn4c = {m: {r["seed"]: r["chi2"] for r in json.loads(g4c.seeds_path(m).read_text())["rows"]}
            for m in ("model1", "bd") if g4c.seeds_path(m).exists()}
    chi2_4c_ref = json.loads((ROOT / "figures" / "goal4c_lcdm.json").read_text())["chi2"]
    for cal in (False, True):
        refp = ROOT / "figures" / f"goal4d_lcdm{suffix(cal)}.json"
        if not refp.exists():
            continue
        ref = json.loads(refp.read_text())
        res = dict(lcdm=ref, models={})
        for model, col, label in (("model1", GREEN, "Model 1"), ("bd", BLUE, f"acción BD (β = {g4c.BETA:g})")):
            if not seeds_path(model, cal).exists():
                continue
            rows = json.loads(seeds_path(model, cal).read_text())["rows"]
            n = len(rows)
            chi2 = np.array([r["chi2"] for r in rows])
            good, better = chi2 < ref["chi2"] + GOOD_MARGIN, chi2 < ref["chi2"]
            cmb_ok = np.array([r["cmb"] - ref["cmb"] < 9 for r in rows])
            g = [r for r, ok in zip(rows, good) if ok]
            par = {k: [float(np.percentile([r[k] for r in g], q)) for q in (16, 50, 84)]
                   for k in ("omega_m", "h", "amp", "sn", "bao", "cmb")} if g else {}
            sn_good = [r["seed"] for r in rows if sn4c.get(model, {}).get(r["seed"], np.inf) < chi2_4c_ref + GOOD_MARGIN]
            m = dict(n=n, n_good=int(good.sum()), n_better=int(better.sum()), f_good=g4c._frac(good.sum(), n),
                     upper95_if_zero=3 / n if good.sum() == 0 else None, chi2_min=float(chi2.min()),
                     f_cmb_ok=float(cmb_ok.mean()), good_params=par,
                     sn_good_seeds_4c=len(sn_good),
                     sn_good_seeds_still_good=int(sum(good[[r["seed"] for r in rows].index(s)] for s in sn_good)),
                     best=sorted(rows, key=lambda r: r["chi2"])[:5])
            res["models"][model] = m
            tag = " (calibrado)" if cal else ""
            print(f"--- {label}{tag}: {n} semillas; chi2 min = {m['chi2_min']:.1f} (ΛCDM {ref['chi2']:.1f})")
            print(f"   buenas: {m['n_good']}  f = {m['f_good'][0]:.4f} ± {m['f_good'][1]:.4f}; mejores: {m['n_better']}")
            print(f"   CMB aceptable (Δχ²_CMB < 9): {m['f_cmb_ok']:.4f}")
            print(f"   buenas del Goal 4c (sólo SN) que siguen buenas: {m['sn_good_seeds_still_good']} de {m['sn_good_seeds_4c']}")
            if par:
                print(f"   parámetros de las buenas: {json.dumps(par)}")
            xs = np.sort(chi2 - ref["chi2"])
            xs = xs[np.isfinite(xs)]
            ax1.step(xs, np.arange(1, len(xs) + 1) / n, where="post", color=col, linestyle="--" if cal else "-",
                     label=label + tag)
            if not cal:
                best = sorted(rows, key=lambda r: r["chi2"])[:300]
                ax2.plot([r["sn"] - ref["sn"] for r in best], [r["cmb"] - ref["cmb"] for r in best], "o", color=col,
                         markersize=3, alpha=0.6, label=label + " (300 mejores semillas)")
        if len(res["models"]) == 2:
            a, b = res["models"]["bd"], res["models"]["model1"]
            se = np.hypot(a["f_good"][1], b["f_good"][1])
            z = (a["f_good"][0] - b["f_good"][0]) / se
            res["comparison"] = dict(z=float(z), verdict="BD mejor" if z > 4 else "BD peor" if z < -4 else "indistinguibles")
            print(f"Comparación{' (calibrado)' if cal else ''}: z = {z:+.2f} -> {res['comparison']['verdict']}")
        out["calibrated" if cal else "primary"] = res
    ax1.axvline(0, color=INK_MUTED, linestyle="--", linewidth=1)
    ax1.axvline(GOOD_MARGIN, color=INK_MUTED, linestyle=":", linewidth=1)
    ax1.set_xlim(-30, 500)
    ax1.set_yscale("log")
    ax1.set_xlabel("χ²_min(semilla) − χ²_ΛCDM  (SN + BAO + CMB)", color=INK)
    ax1.set_ylabel("fracción de semillas por debajo", color=INK)
    ax1.set_title("Distribución del mejor ajuste por semilla", color=INK, fontsize=10)
    ax2.axhline(0, color=INK_MUTED, linestyle="--", linewidth=1)
    ax2.axvline(0, color=INK_MUTED, linestyle="--", linewidth=1)
    ax2.set_xscale("symlog", linthresh=10)
    ax2.set_yscale("symlog", linthresh=10)
    ax2.set_xlabel("χ²_SN − χ²_SN,ΛCDM", color=INK)
    ax2.set_ylabel("χ²_CMB − χ²_CMB,ΛCDM", color=INK)
    ax2.set_title("Dónde pierden las mejores semillas (análisis principal)", color=INK, fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    (ROOT / "figures" / "goal4d_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    path = ROOT / "figures" / "goal4d_combined.png"
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print("figura", path)


if __name__ == "__main__":
    cal = "--cal" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--cal"]
    if args[0] == "validate":
        validate()
    elif args[0] == "lcdm":
        lcdm(cal)
    elif args[0] == "seeds":
        seeds(args[1], cal)
    else:
        fit()
