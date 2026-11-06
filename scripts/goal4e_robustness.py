"""Goal 4e: is the exclusion of the BD action of Goal 4d robust?

Goal 4d excludes the BD action (0 of 10 000 good seeds with SN + BAO +
CMB, minimum Δχ² +124) for ONE dynamics: Λ = lam V^(-q) s, with s an
Ornstein-Uhlenbeck process in ln V with rate c (``causets/stochastic_lambda.py``),
q = 1/4 (ΔS ∝ V^(3/4), Points 1c-1d, and δΛ ∝ ΔS/V, Goal 4a) and c = 3β/4 with
β = 0.9 (memory measured with the local action, Goal 4b-1: β = 0.88-0.97).
Here those choices are varied with the same fit per seed and the same χ²
(SN + BAO + CMB, uncalibrated main analysis of Goal 4d).

Amplitude: lam = λ V_ref^(q - 1/4), λ on the grid of Goal 4d [0.1, 10] and
V_ref = 0.5976 (volume of the causal past today at the Planck ΛCDM best fit,
H_100 units). Thus Λ today covers the same range for every q, and with
q = 1/4 the grid is identical to that of Goal 4d.

CRITERIA (fixed on 2026-10-10, BEFORE computing any variant):
  E0  Reproduction: the variant (q = 1/4, c = 0.675) gives, seed by seed,
      the same χ² as Goal 4d (|Δ| < 1e-6; test).
  E1  Main. BD family (q = 1/4) with β ∈ {0.5, 0.88, 0.97, 1.0}
      (c = 3β/4), 10 000 seeds each (the same as in Goal 4c/4d).
      0.88-0.97 is the measured range; 1.0, random walk; 0.5, stress test
      without support from the measurements.
      The exclusion is ROBUST in β if none of the four has good seeds
      (χ² < χ²_ΛCDM + 14.7). If any has, it is NOT robust and its fraction
      is given.
  E2  Map of the class (q, c): q ∈ {0, 1/8, 1/4, 3/8, 1/2},
      c ∈ {0.25, 0.5, 0.75, 1.0}, 1000 seeds per cell (seeds 0-999).
      Descriptive: minimum Δχ² per cell. If any cell has a good seed, it is
      repeated with 10 000 seeds. (Model 1 is q = c = 1/2 with another
      amplitude; the BD action, q = 1/4, c ≈ 0.66-0.73.)
  Edges: if the best seed of a variant falls on an edge of the grid
  (ω_m or amplitude), that variant is repeated with the range extended ×10 on
  that side.

Usage:  python scripts/goal4e_robustness.py beta B | map WORKER NWORKERS | fit
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causets import stochastic_lambda as sl  # noqa: E402

_spec = importlib.util.spec_from_file_location("goal4d", ROOT / "scripts" / "goal4d_combined.py")
g4d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g4d)
g4c = g4d.g4c

V_REF = 0.5976
BETAS = (0.5, 0.88, 0.97, 1.0)
MAP_Q = (0.0, 0.125, 0.25, 0.375, 0.5)
MAP_C = (0.25, 0.5, 0.75, 1.0)
MAP_SEEDS = 1000
GOOD_MARGIN = g4c.GOOD_MARGIN
INK, INK_MUTED, GRID = g4c.INK, g4c.INK_MUTED, g4c.GRID


def key(q, c):
    return f"q{q:g}_c{c:g}"


def register(q, c):
    """Registers the variant in the amplitude grid of Goal 4c and returns its key."""
    k = key(q, c)
    g4c.AMP_GRID[k] = g4c.AMP_GRID["bd"] * V_REF ** (q - 0.25)
    return k


def parse(k):
    q, c = k[1:].split("_c")
    return float(q), float(c)


class Variant(g4d.Combined):
    def spec(self, k, amp):
        q, c = parse(k)
        return sl.ProcessSpec(amp, q, c, k)

    def evaluator(self, k, noise, om, amp):
        real = self.run(self.spec(k, amp), om, noise=noise)
        p = self.parts(real, om)
        return p["sn"] + p["bao"] + p["cmb"], p["M_B"], real.h, real.omega_lambda0


def fit_seeds(v, k, seeds):
    rows = g4c.fit_batch(v.sn, k, seeds, evaluator=v.evaluator)
    noise = np.stack([np.random.default_rng(g4c.SEED0 + s).standard_normal(g4c.N_STEPS) for s in seeds], axis=1)
    om = np.array([r["omega_m"] for r in rows])
    p = v.parts(v.run(v.spec(k, np.array([r["amp"] for r in rows])), om, noise=noise), om)
    for j, r in enumerate(rows):
        r.update(sn=float(p["sn"][j]), bao=float(p["bao"][j]), cmb=float(p["cmb"][j]))
    return rows


def path_for(k, n):
    return ROOT / "figures" / f"goal4e_{k}_n{n}.json"


def run_variant(v, k, n):
    path = path_for(k, n)
    rows = json.loads(path.read_text())["rows"] if path.exists() else []
    t0 = time.time()
    while len(rows) < n:
        start = len(rows)
        rows += fit_seeds(v, k, list(range(start, min(start + g4c.BATCH, n))))
        q, c = parse(k)
        path.write_text(json.dumps(dict(q=q, c=c, v_ref=V_REF, rows=rows)))
        print(f"{k}: {len(rows)}/{n} seeds, {time.time() - t0:.0f} s", flush=True)


def beta(b):
    v = Variant()
    run_variant(v, register(0.25, 0.75 * b), g4c.N_SEEDS)


def map_cells(worker, nworkers):
    v = Variant()
    cells = [(q, c) for q in MAP_Q for c in MAP_C]
    for q, c in cells[worker::nworkers]:
        run_variant(v, register(q, c), MAP_SEEDS)


def summary(rows, ref, k):
    chi2 = np.array([r["chi2"] for r in rows])
    good = chi2 < ref["chi2"] + GOOD_MARGIN
    b = min(rows, key=lambda r: r["chi2"])
    lom, lamp = np.log(g4c.OMEGA_M_GRID), np.log(g4c.AMP_GRID[k])
    edge = []
    if np.isclose(np.log(b["omega_m"]), lom[0]) or np.isclose(np.log(b["omega_m"]), lom[-1]):
        edge.append("omega_m")
    if np.isclose(np.log(b["amp"]), lamp[0]) or np.isclose(np.log(b["amp"]), lamp[-1]):
        edge.append("amp")
    return dict(n=len(rows), n_good=int(good.sum()), f_good=g4c._frac(good.sum(), len(rows)),
                upper95_if_zero=3 / len(rows) if good.sum() == 0 else None,
                dchi2_min=float(chi2.min() - ref["chi2"]),
                best={kk: b[kk] - ref[kk] if kk in ("sn", "bao", "cmb") else b[kk]
                      for kk in ("seed", "omega_m", "amp", "h", "Omega_L0", "sn", "bao", "cmb")},
                best_at_edge=edge, chi2=chi2)


def fit():
    ref = json.loads((ROOT / "figures" / "goal4d_lcdm.json").read_text())
    out = dict(lcdm_chi2=ref["chi2"], E1={}, E2={})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(BETAS) + 1))
    base = json.loads(g4d.seeds_path("bd", False).read_text())["rows"]
    curves = [(0.9, base, "Goal 4d")]
    for b in BETAS:
        k = register(0.25, 0.75 * b)
        if path_for(k, g4c.N_SEEDS).exists():
            curves.append((b, json.loads(path_for(k, g4c.N_SEEDS).read_text())["rows"], ""))
    curves.sort(key=lambda t: t[0])
    any_good = False
    for (b, rows, tag), col in zip(curves, cols):
        k = register(0.25, 0.75 * b)
        s = summary(rows, ref, k)
        chi2 = s.pop("chi2")
        out["E1"][f"beta={b:g}"] = s
        any_good |= s["n_good"] > 0
        print(f"β = {b:g} (c = {0.75 * b:g}){' [' + tag + ']' if tag else ''}: {s['n']} seeds, good {s['n_good']}, "
              f"Δχ² min = {s['dchi2_min']:+.1f}; best: ΔSN {s['best']['sn']:+.1f}, ΔBAO {s['best']['bao']:+.1f}, "
              f"ΔCMB {s['best']['cmb']:+.1f}; edge: {s['best_at_edge'] or 'none'}")
        xs = np.sort(chi2 - ref["chi2"])
        xs = xs[np.isfinite(xs)]
        ax1.step(xs, np.arange(1, len(xs) + 1) / len(rows), where="post", color=col,
                 label=f"β = {b:g}" + (" (Goal 4d)" if tag else ""))
    n_e1 = sum(1 for b in BETAS if f"beta={b:g}" in out["E1"])
    out["E1_verdict"] = ("incompleto" if n_e1 < len(BETAS) else
                         "NO robusta: hay semillas buenas" if any_good else "robusta en β")
    print(f"E1 ({n_e1}/{len(BETAS)} variants): {out['E1_verdict']}")
    grid = np.full((len(MAP_C), len(MAP_Q)), np.nan)
    for i, c in enumerate(MAP_C):
        for j, q in enumerate(MAP_Q):
            k = register(q, c)
            p = path_for(k, MAP_SEEDS)
            if not p.exists():
                continue
            rows = json.loads(p.read_text())["rows"]
            if len(rows) < MAP_SEEDS:
                continue
            s = summary(rows, ref, k)
            s.pop("chi2")
            out["E2"][k] = s
            grid[i, j] = s["dchi2_min"]
    if out["E2"]:
        print("E2: minimum Δχ² per cell (rows c, columns q =", MAP_Q, ")")
        for i, c in enumerate(MAP_C):
            print(f"   c = {c:<5g}" + "".join(f"{x:9.1f}" for x in grid[i]))
        edges = {k: s["best_at_edge"] for k, s in out["E2"].items() if s["best_at_edge"]}
        goods = [k for k, s in out["E2"].items() if s["n_good"] > 0]
        print(f"   cells with good seeds: {goods or 'none'}; best seed on an edge: {edges or 'none'}")
        out["E2_cells_with_good"] = goods
        im = ax2.imshow(np.log10(np.clip(grid, 1, None)), origin="lower", cmap="viridis_r", aspect="auto",
                        extent=(-0.0625, 0.5625, -0.5, len(MAP_C) - 0.5))
        for i in range(len(MAP_C)):
            for j, q in enumerate(MAP_Q):
                if np.isfinite(grid[i, j]):
                    ax2.text(q, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8,
                             color="white" if grid[i, j] > 300 else INK)
        ax2.set_xticks(MAP_Q)
        ax2.set_yticks(range(len(MAP_C)), [f"{c:g}" for c in MAP_C])
        ax2.set_xlabel("q  (Λ ∝ V^(−q))", color=INK)
        ax2.set_ylabel("c  (memory: corr = (V1/V2)^c)", color=INK)
        ax2.set_title(f"Minimum Δχ² relative to ΛCDM ({MAP_SEEDS} seeds per cell)", color=INK, fontsize=10)
        fig.colorbar(im, ax=ax2, label="log10 Δχ²_min")
    ax1.axvline(0, color=INK_MUTED, linestyle="--", linewidth=1)
    ax1.axvline(GOOD_MARGIN, color=INK_MUTED, linestyle=":", linewidth=1)
    ax1.set_xlim(-30, 500)
    ax1.set_yscale("log")
    ax1.set_xlabel("χ²_min(seed) − χ²_ΛCDM  (SN + BAO + CMB)", color=INK)
    ax1.set_ylabel("fraction of seeds below", color=INK)
    ax1.set_title("BD action (q = 1/4): memory β", color=INK, fontsize=10)
    ax1.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    ax1.legend(frameon=False, fontsize=8)
    (ROOT / "figures" / "goal4e_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    path = ROOT / "figures" / "goal4e_robustness.png"
    fig.savefig(path, dpi=150)
    print(f"figure {path}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "beta":
        beta(float(sys.argv[2]))
    elif cmd == "map":
        map_cells(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "fit":
        fit()
    else:
        raise SystemExit(__doc__)
