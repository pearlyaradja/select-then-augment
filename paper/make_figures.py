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


def examples(fold=1, n=3):
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv")
    sel = pd.read_csv(ROOT / "outputs" / "selection" / "selection_N200.csv")
    real = ps[(ps.fold == fold) & (ps.split == "train") & (ps.tissue_frac > 0.5)]
    s = sel[sel.fold == fold]
    groups = [("Real (training)", None), ("Selected by C2 (realism)", "C2"),
              ("Selected by C4 (all filters)", "C4")]
    fig, axes = plt.subplots(2, 3 * n, figsize=(7.16, 1.85))
    for r, (c, name) in enumerate([("aca", "ADC"), ("scc", "SCC")]):
        for g, (title, cond) in enumerate(groups):
            if cond is None:
                paths = real[real.superclass == c].patch_path.sample(n, random_state=3)
            else:
                other = "C4" if cond == "C2" else "C2"
                # images picked by both conditions are left out so the contrast is visible
                paths = s[s[cond] & ~s[other] & (s.superclass == c)].path.sample(n, random_state=0)
            for j, p in enumerate(paths):
                ax = axes[r, g * n + j]
                ax.imshow(Image.open(ROOT / p).convert("RGB").resize((256, 256)))
                ax.set_xticks([]), ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0 and j == 1:
                    ax.set_title(title, fontsize=8, color=INK)
        axes[r, 0].set_ylabel(name, fontsize=8, color=INK)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.88, bottom=0.01, wspace=0.04, hspace=0.06)
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
             "C3": "C3 class + memo.", "C4": "C4 all filters"}
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
