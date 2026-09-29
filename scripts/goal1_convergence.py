"""Goal 1: convergencia de la dimensión de Myrheim-Meyer en Minkowski 2D y 4D.

Genera ``figures/goal1_mm_convergence.png`` y una tabla por consola.
Uso:  python scripts/goal1_convergence.py [--samples M] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causets import CausalDiamond, minkowski_causal_matrix, myrheim_meyer_dimension, sample_uniform  # noqa: E402

N_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]
DIMS = [2, 4]
COLORS = {2: "#2a78d6", 4: "#eb6834"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def run(samples: int, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    results = {}
    for d in DIMS:
        region = CausalDiamond(d)
        table = np.empty((len(N_VALUES), samples))
        for a, n in enumerate(N_VALUES):
            for b in range(samples):
                C = minkowski_causal_matrix(sample_uniform(region, n, rng))
                table[a, b] = myrheim_meyer_dimension(C)
            m, s = table[a].mean(), table[a].std(ddof=1)
            print(f"d={d}  N={n:5d}  <d_MM>={m:.4f}  std={s:.4f}  sem={s / np.sqrt(samples):.4f}")
        results[d] = table
    return results


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_VALUES, [str(n) for n in N_VALUES])


def plot(results: dict[int, np.ndarray], samples: int, path: Path) -> None:
    n = np.array(N_VALUES)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    for ax, d in zip(axes[:2], DIMS):
        table = results[d]
        mean, std = table.mean(axis=1), table.std(axis=1, ddof=1)
        ax.axhline(d, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.fill_between(n, mean - std, mean + std, color=COLORS[d], alpha=0.18, linewidth=0)
        ax.plot(n, mean, color=COLORS[d], linewidth=2, marker="o", markersize=5)
        _style(ax)
        ax.set_title(f"Minkowski {d}D: $d_{{MM}}$ (media ± 1σ, M={samples})", color=INK, fontsize=11)
        ax.set_xlabel("N (puntos en el diamante causal)", color=INK)
        ax.set_ylabel("$d_{MM}$", color=INK)
        ax.text(n[-1], d, f" d = {d}", color=INK_MUTED, va="bottom", ha="right", fontsize=9)

    ax = axes[2]
    for d in DIMS:
        std = results[d].std(axis=1, ddof=1)
        ax.plot(n, std, color=COLORS[d], linewidth=2, marker="o", markersize=5, label=f"{d}D")
        ax.text(n[-1] * 1.08, std[-1], f"{d}D", color=INK, va="center", fontsize=9)
        guide = std[0] * np.sqrt(n[0] / n)
        ax.plot(n, guide, color=COLORS[d], linewidth=1, linestyle=":", alpha=0.9)
    _style(ax)
    ax.set_yscale("log")
    ax.set_title("Dispersión de $d_{MM}$ (punteado: $\\propto N^{-1/2}$)", color=INK, fontsize=11)
    ax.set_xlabel("N", color=INK)
    ax.set_ylabel("std($d_{MM}$)", color=INK)
    ax.legend(frameon=False, labelcolor=INK)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print(f"figura guardada en {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    results = run(args.samples, args.seed)
    plot(results, args.samples, ROOT / "figures" / "goal1_mm_convergence.png")


if __name__ == "__main__":
    main()
