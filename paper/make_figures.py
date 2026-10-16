"""Figures of the paper (English labels), from the JSON results in figures/.

No computation is redone: every number comes from the saved results of
scripts/step2_4d_scaling.py, scripts/goal6_referee.py, scripts/goal5b_decomposition.py,
scripts/goal4d_combined.py and scripts/goal4e_robustness.py.

Usage:  python paper/make_figures.py   ->  paper/figures/fig_*.pdf
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "figures"
OUT = ROOT / "paper" / "figures"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load(name):
    return json.loads((RES / name).read_text())


def fig_diamond():
    d = load("step2_4d_results.json")
    pts = d["points"]
    lam = np.array([p["rhoV"] for p in pts], float)
    curve = d["J1_curve"]
    lj, j1 = np.array(curve["rhoV"]), np.array(curve["J1"])
    R1 = load("goal6_results.json")["R1"]  # all independent eps = 1 sprinklings, combined per density
    smax = max(p["rhoV"] for p in R1["points"])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    for ax in (a1, a2):
        ax.axvspan(lam.min(), smax, color=GRID, alpha=0.5, lw=0)
    a1.plot(lj, j1 / lj, "-", color=BLUE, label=r"$J^1$ (exact)")
    a1.errorbar(lam, [p["sim"] / p["rhoV"] for p in pts], yerr=[p["sim_se"] / p["rhoV"] for p in pts],
                fmt="o", color=INK, mfc="white", capsize=2, label=r"total (sprinklings)")
    ref = lj[lj >= 1e4]
    a1.plot(ref, j1[-1] / lj[-1] * (ref / lj[-1]) ** 0.5, ":", color=INK2, label=r"slope $p=3/4$")
    a1.set(xscale="log", yscale="log", xlabel=r"$\rho V$", ylabel=r"$(\Delta S)^2/\rho V$", title="(a)")
    a1.legend(frameon=False, loc="lower right")
    mid = np.sqrt(lj[1:] * lj[:-1])
    pj = np.log(j1[1:] / j1[:-1]) / np.log(lj[1:] / lj[:-1]) / 2
    a2.plot(mid, pj, "-", color=BLUE, label=r"$J^1$ (exact)")
    ps = d["p_local_sim"]
    a2.errorbar([np.sqrt(p["rhoV"][0] * p["rhoV"][1]) for p in ps], [p["p"] for p in ps],
                yerr=[p["se"] for p in ps], fmt="o", color=INK, mfc="white", capsize=2, label="total")
    a2.axhline(0.75, color=BLUE, ls=":", lw=1)
    a2.axhline(0.5, color=ORANGE, ls="--", lw=1)
    a2.text(1.5e5, 0.68, r"$3/4$", color=INK2)
    a2.text(1.5e3, 0.52, r"Poisson, $1/2$", color=INK2)
    a2.set(xscale="log", xlabel=r"$\rho V$", ylabel=r"local exponent $p$", ylim=(0.4, 1.35), title="(b)")
    a2.legend(frameon=False, loc="upper right")
    rows = R1["points"]
    x = np.array([p["rhoV"] for p in rows], float)
    m, ms = R1["mean"], R1["mean_se"]
    a3.axhspan(m - ms, m + ms, color=BLUE, alpha=0.15, lw=0)
    a3.axhline(m, color=BLUE, lw=1, label=rf"mean ${m:.3f}\pm{ms:.3f}$")
    a3.errorbar(x, [p["r"] for p in rows], yerr=[p["se"] for p in rows], fmt="o", color=INK, mfc="white",
                capsize=2, label="sprinklings")
    a3.set(xscale="log", xlabel=r"$\rho V$", ylabel=r"remainder $r$", ylim=(0, 0.6), title="(c)")
    a3.legend(frameon=False, loc="lower right")
    for ax in (a1, a2, a3):
        _style(ax)
    fig.savefig(OUT / "fig_diamond.pdf")


def fig_smeared():
    rows = load("goal5b_results.json")["points"]
    r_local = load("goal5_results.json")["F3"]["1.0"]["mean"]  # remainder of the local action (eps = 1)
    series = [("J2", r"$J^2$", BLUE, "s"), ("J3", r"$J^3$", AQUA, "D"), ("2J4", r"$2J^4$", YELLOW, "^"),
              ("J6c", r"$J^6_{\rm c}$", ORANGE, "v"), ("rest", r"remainder $r$ (sum)", INK, "o")]
    eps_list = (0.5, 0.2, 0.1)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    for ax, eps in zip(axes, eps_list):
        sel = [r for r in rows if r["eps"] == eps]
        lam = np.array([r["rhoV"] for r in sel], float)
        for i, (k, lab, col, mk) in enumerate(series):
            ax.errorbar(lam * (1 + 0.03 * (i - 2)), [r[k + "_over_J1"] for r in sel],
                        yerr=[r[k + "_over_J1_se"] for r in sel], fmt=mk + "-", color=col, mfc="white",
                        capsize=1.5, lw=1.4 if k == "rest" else 1.1, label=lab)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axhline(r_local, color=MUTED, ls=":", lw=1, label=r"$r$ for $\epsilon=1$")
        ax.set(xscale="log", xlabel=r"$\rho V$", title=rf"$\epsilon={eps}$")
        ax.set_xticks([400, 1600, 6400], ["400", "1600", "6400"])
        ax.minorticks_off()
        _style(ax)
    axes[0].set_ylabel(r"term $/\,J^1$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
    fig.savefig(OUT / "fig_smeared.pdf")


def fig_data():
    e = load("goal4e_results.json")
    bd09 = load("goal4d_results.json")["primary"]
    lcdm = bd09["lcdm"]["chi2"]
    betas = {0.9: bd09["models"]["bd"]["chi2_min"] - lcdm}
    for k, v in e["E1"].items():
        betas[float(k.split("=")[1])] = v["dchi2_min"]
    b = np.array(sorted(betas))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    a1.axvspan(0.88, 0.97, color=GRID, lw=0)
    a1.plot(b, [betas[x] for x in b], "o-", color=BLUE, mfc="white")
    a1.axhline(14.7, color=ORANGE, ls="--", lw=1)
    a1.text(0.51, 20, r"good seed: $\Delta\chi^2<14.7$", color=INK2)
    a1.text(0.885, 165, "measured\n" + r"$\beta$", color=INK2, fontsize=7)
    a1.set(xlabel=r"memory parameter $\beta$ ($c=3\beta/4$, $q=1/4$)", ylabel=r"$\Delta\chi^2_{\rm min}$ vs $\Lambda$CDM",
           ylim=(0, 190))
    _style(a1)
    qs, cs = (0, 0.125, 0.25, 0.375, 0.5), (0.25, 0.5, 0.75, 1.0)
    Z = np.array([[e["E2"][f"q{q:g}_c{c:g}"]["dchi2_min"] for q in qs] for c in cs])
    im = a2.imshow(np.log10(Z), origin="lower", cmap="Blues", aspect="auto", vmin=1, vmax=3.3)
    for i in range(len(cs)):
        for j in range(len(qs)):
            a2.text(j, i + (0.2 if (i, j) == (2, 2) else 0), f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if np.log10(Z[i, j]) > 2.4 else INK)
    a2.set_xticks(range(len(qs)), ["0", "1/8", "1/4", "3/8", "1/2"])
    a2.set_yticks(range(len(cs)), ["0.25", "0.5", "0.75", "1"])
    a2.set(xlabel=r"volume exponent $q$  ($\Lambda\propto V^{-q}$)", ylabel=r"decorrelation rate $c$")
    # BD action: q = 1/4, c = 3 beta / 4 with the measured beta = 0.88-0.97 (row index is linear in c)
    cidx = lambda c: (c - 0.25) / 0.25  # noqa: E731
    a2.plot([2, 2], [cidx(0.66), cidx(0.7275)], "-", color=ORANGE, lw=3, solid_capstyle="round")
    a2.plot(2, cidx(0.675), "D", color=ORANGE, mec="white", mew=1, ms=6)
    a2.text(2.38, cidx(0.675), "BD", va="center", color=ORANGE, fontsize=7.5, fontweight="bold")
    for side in ("top", "right"):
        a2.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=a2, shrink=0.9)
    cb.set_label(r"$\log_{10}\Delta\chi^2_{\rm min}$")
    cb.outline.set_visible(False)
    fig.savefig(OUT / "fig_data.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_diamond()
    fig_smeared()
    fig_data()
    print("figures in", OUT)
