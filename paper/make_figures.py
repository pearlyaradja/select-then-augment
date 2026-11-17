"""Figures for the paper (English labels). No training; reads Stage 5-6 outputs.

fig_examples.jpg : real train patches vs synthetic images selected by C2 and C4 (fold 1)
fig_results.pdf  : (a) test Macro-F1 with 95% CI, (b) paired differences with 95% CI

  python paper/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "outputs" / "eval"
OUT = Path(__file__).resolve().parent / "figs"
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8})


def examples(fold=1, n=4):
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv")
    sel = pd.read_csv(ROOT / "outputs" / "selection" / "selection_N200.csv")
    real = ps[(ps.fold == fold) & (ps.split == "train") & (ps.tissue_frac > 0.5)]
    s = sel[sel.fold == fold]
    groups = [("Real (training)", None), ("C2 (realism)", "C2"), ("C4 (class + realism)", "C4")]
    # the score(s) each condition ranked by are printed in bold above its images
    used = {"C2": {"d"}, "C4": {"p", "d"}}
    # single-column layout: each group shows its n images as a 2-wide grid, with thin
    # spacer columns between groups; ADC and SCC rows are told apart by their labels
    k = n // 2
    fig = plt.figure(figsize=(3.5, 3.35))
    gs = fig.add_gridspec(2 * k, 3 * 2 + 2, width_ratios=[1, 1, 0.1, 1, 1, 0.1, 1, 1],
                          left=0.06, right=0.995, top=0.885, bottom=0.005, wspace=0.05, hspace=0.5)
    for r, (c, name) in enumerate([("aca", "ADC"), ("scc", "SCC")]):
        for g, (title, cond) in enumerate(groups):
            if cond is None:
                rows = real[real.superclass == c].sample(n, random_state=3).rename(columns={"patch_path": "path"})
            else:
                other = "C4" if cond == "C2" else "C2"
                # images picked by both conditions are left out so the contrast is visible
                rows = s[s[cond] & ~s[other] & (s.superclass == c)].sample(n, random_state=0)
            for j, (_, row) in enumerate(rows.iterrows()):
                ax = fig.add_subplot(gs[r * k + j // 2, g * 3 + j % 2])
                ax.imshow(Image.open(ROOT / row.path).convert("RGB").resize((256, 256)))
                ax.set_xticks([]), ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0 and j == 0:
                    # group title centred over its two columns, above the score lines
                    ax.text(1.025, 1.5, title, transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=7, color=INK)
                if g == 0 and j == 0:
                    ax.text(-0.12, -0.15, name, transform=ax.transAxes, rotation=90,
                            ha="center", va="center", fontsize=7, color=INK)
                if cond is not None:
                    # two stacked lines so the scores fit the narrow tiles
                    for y, key, txt in [(1.25, "p", f"$p$ = {row.p_intended:.2f}"),
                                        (1.04, "d", f"$d$ = {row.knn_dist:.2f}")]:
                        ax.text(0.5, y, txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=5.5,
                                color=INK, fontweight="bold" if key in used[cond] else "normal")
    fig.savefig(OUT / "fig_examples.jpg", dpi=300, pil_kwargs={"quality": 90})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def results():
    m = pd.read_csv(E / "bootstrap_metrics.csv")
    d = pd.read_csv(E / "bootstrap_diffs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[1, 1.2]))

    ax = axes[0]
    conds = ["C0", "C1", "C2", "C3", "C4"]
    label = {"C0": "C0 real only", "C1": "C1 random", "C2": "C2 realism",
             "C3": "C3 class + memo.", "C4": "C4 class + realism rank"}
    sub = m[m.metric == "macro_f1"].set_index("cond").loc[conds]
    y = range(len(conds))[::-1]
    ax.hlines(y, sub.ci_low, sub.ci_high, color=BLUE, lw=1.5)
    ax.plot(sub.estimate, y, "o", ms=5, color=BLUE, mec="white", mew=1)
    for yi, (_, r) in zip(y, sub.iterrows()):
        ax.text(r.ci_high + 0.004, yi, f"{r.estimate:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(list(y), [label[c] for c in conds])
    ax.set_xlim(0.64, 0.85)
    ax.set_title("(a) Test macro-F1, 95% CI", loc="left", fontsize=8, color=INK)
    style(ax)

    ax = axes[1]
    comps = ["C1 - C0", "C2 - C0", "C3 - C0", "C4 - C0", "C2 - C1", "C3 - C1", "C4 - C1"]
    for off, metric, mk in [(0.17, "macro_f1", "o"), (-0.17, "auc", "s")]:
        s = d[d.metric == metric].set_index("comparison").loc[comps]
        yy = [len(comps) - 1 - i + off for i in range(len(comps))]
        ax.hlines(yy, s.ci_low, s.ci_high, color=BLUE, lw=1.2)
        excl = (s.ci_low > 0) | (s.ci_high < 0)
        for yi, (_, r), e in zip(yy, s.iterrows(), excl):
            ax.plot(r["diff"], yi, mk, ms=4.5, color=BLUE if e else "white", mec=BLUE, mew=1.1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axhline(2.5, color=GRID, lw=0.8, ls="--")
    ax.set_yticks(range(len(comps))[::-1], [c.replace(" - ", " − ") for c in comps])
    ax.plot([], [], "o", color=BLUE, ms=4.5, label="Macro-F1")
    ax.plot([], [], "s", color=BLUE, ms=4.5, label="AUC")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title("(b) Paired differences, 95% CI (filled: CI excludes 0)", loc="left", fontsize=8, color=INK)
    style(ax)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_results.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    examples()
    results()
    print("saved to", OUT)
