"""Goal 2: acción de Benincasa-Dowker en Minkowski 2D y curvatura de de Sitter.

Genera ``figures/goal2_bd_action.png``:
  (a) <S> de un diamante 2D frente a rho V: simulación y valor exacto (-> 1).
  (b) E[B 1] / (-R/2) en de Sitter frente a l H (valor exacto, 2D y 4D) -> 1.
Uso:  python scripts/goal2_bd_action.py [--samples M]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causets import CausalDiamond, minkowski_causal_matrix, sprinkle  # noqa: E402
from causets.action import bd_action  # noqa: E402
from causets.continuum import expected_bd_action_diamond, expected_bd_on_constant_de_sitter  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
COLORS = {2: "#2a78d6", 4: "#eb6834"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)


def action_2d(samples: int, rng):
    lams = [25, 50, 100, 200, 400, 800, 1600]
    region = CausalDiamond(2)
    rows = []
    for lam in lams:
        rho = lam / region.volume
        s = np.array([bd_action(minkowski_causal_matrix(sprinkle(region, rho, rng)), 2) for _ in range(samples)])
        exact = expected_bd_action_diamond(2, lam)
        rows.append((lam, s.mean(), s.std(ddof=1) / np.sqrt(samples), exact))
        print(f"2D diamante  rho V={lam:5d}  <S>={s.mean():8.3f} ± {rows[-1][2]:.3f}   exacto={exact:.5f}")
    return np.array(rows)


def de_sitter_scan():
    l_h = np.array([0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05])
    out = {}
    for d in (2, 4):
        r_half = d * (d - 1) / 2
        ratio = np.array([expected_bd_on_constant_de_sitter(d, 1.0, lh ** (-d), 1.5, 1.0, 2) / -r_half for lh in l_h])
        out[d] = ratio
        for lh, r in zip(l_h, ratio):
            print(f"de Sitter {d}D  l H={lh:.3f}  E[B1]/(-R/2)={r:.4f}")
    return l_h, out


def plot(rows, l_h, ratios, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)

    lam, mean, sem, exact = rows.T
    ax1.axhline(1.0, color=INK_MUTED, linestyle="--", linewidth=1)
    ax1.plot(lam, exact, color=COLORS[2], linewidth=2, label="valor esperado exacto")
    ax1.errorbar(lam, mean, yerr=sem, fmt="o", color=COLORS[2], markersize=5, capsize=3,
                 markerfacecolor="white", label="simulación (media ± SEM)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(lam, [str(int(v)) for v in lam])
    _style(ax1)
    ax1.set_xlabel("ρV (número medio de elementos)", color=INK)
    ax1.set_ylabel("⟨S⟩", color=INK)
    ax1.set_title("Acción BD local de un diamante en Minkowski 2D (⟨S⟩ → 1;\nla dispersión crece con ρV)", color=INK, fontsize=11)
    ax1.legend(frameon=False, labelcolor=INK)

    ax2.axhline(1.0, color=INK_MUTED, linestyle="--", linewidth=1)
    for d in (2, 4):
        ax2.plot(l_h, ratios[d], color=COLORS[d], linewidth=2, marker="o", markersize=5, label=f"{d}D")
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set_xticks(l_h, [f"{v:g}" for v in l_h])
    ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    _style(ax2)
    ax2.set_xlabel("ℓH (escala de discreción × H) → 0", color=INK)
    ax2.set_ylabel("E[B·1] / (−R/2)", color=INK)
    ax2.set_title("de Sitter: el operador BD recupera R = d(d−1)H²", color=INK, fontsize=11)
    ax2.legend(frameon=False, labelcolor=INK, loc="lower right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print(f"figura guardada en {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    rows = action_2d(args.samples, np.random.default_rng(args.seed))
    l_h, ratios = de_sitter_scan()
    plot(rows, l_h, ratios, ROOT / "figures" / "goal2_bd_action.png")


if __name__ == "__main__":
    main()
