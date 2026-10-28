"""Tahap 6c: gambar ringkasan hasil (Macro-F1 + CI, selisih berpasangan, KID). Tanpa training.
Input : outputs/eval/bootstrap_metrics.csv, bootstrap_diffs.csv, kid_fid.csv
Output: outputs/eval/hasil_tahap5_6.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "outputs" / "eval"
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"C0": "C0 real saja", "C1": "C1 + sintetis acak", "C2": "C2 + filter realism",
         "C3": "C3 + filter kelas+memo", "C4": "C4 + semua filter"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    m = pd.read_csv(E / "bootstrap_metrics.csv")
    d = pd.read_csv(E / "bootstrap_diffs.csv")
    k = pd.read_csv(E / "kid_fid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))
    fig.patch.set_facecolor("#fcfcfb")

    # (a) Macro-F1 per kondisi + CI 95%
    ax = axes[0]
    conds = ["C0", "C1", "C2", "C3", "C4"]
    sub = m[m.metric == "macro_f1"].set_index("cond").loc[conds]
    y = range(len(conds))[::-1]
    ax.hlines(y, sub.ci_low, sub.ci_high, color=BLUE, lw=2)
    ax.plot(sub.estimate, y, "o", ms=8, color=BLUE, mec="#fcfcfb", mew=2)
    for yi, (c, r) in zip(y, sub.iterrows()):
        ax.text(r.ci_high + 0.004, yi, f"{r.estimate:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), [LABEL[c] for c in conds])
    ax.set_xlim(0.64, 0.85)
    ax.set_title("(a) Macro-F1 test, CI 95%", loc="left", fontsize=11, color=INK)
    style(ax)

    # (b) Selisih AUC & Macro-F1 berpasangan; isi penuh = CI tidak memotong 0
    ax = axes[1]
    comps = ["C1 - C0", "C2 - C0", "C3 - C0", "C4 - C0", "C2 - C1", "C3 - C1", "C4 - C1"]
    for off, metric, mk in [(0.17, "macro_f1", "o"), (-0.17, "auc", "s")]:
        s = d[d.metric == metric].set_index("comparison").loc[comps]
        yy = [len(comps) - 1 - i + off for i in range(len(comps))]
        ax.hlines(yy, s.ci_low, s.ci_high, color=BLUE, lw=1.6)
        excl = (s.ci_low > 0) | (s.ci_high < 0)
        for yi, (_, r), e in zip(yy, s.iterrows(), excl):
            ax.plot(r["diff"], yi, mk, ms=7, color=BLUE if e else "#fcfcfb", mec=BLUE, mew=1.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axhline(2.5, color=GRID, lw=1, ls="--")
    ax.set_yticks(range(len(comps))[::-1], comps)
    ax.plot([], [], "o", color=BLUE, label="Macro-F1")
    ax.plot([], [], "s", color=BLUE, label="AUC")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("(b) Selisih berpasangan, CI 95%\nisi penuh = CI tak memotong 0; semua p_Holm ≥ 0,05",
                 loc="left", fontsize=11, color=INK)
    style(ax)

    # (c) KID gabungan set sintetis terpilih vs patch train asli
    ax = axes[2]
    order = ["patokan", "semua", "C1", "C2", "C3", "C4"]
    names = {"patokan": "asli (val) — patokan", "semua": "sintetis tanpa seleksi", "C1": "C1 acak",
             "C2": "C2 realism", "C3": "C3 kelas+memo", "C4": "C4 semua"}
    g = k.groupby("set").kid_all.agg(["mean", "std"]).loc[order] * 1e3
    y = range(len(order))[::-1]
    ax.barh(list(y), g["mean"], height=0.55, color=[MUTED] + [BLUE] * 5)
    for yi, (n, r) in zip(y, g.iterrows()):
        ax.text(r["mean"] + 1.5, yi, f"{r['mean']:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), [names[o] for o in order])
    ax.set_xlim(0, 75)
    ax.set_title("(c) KID ×10⁻³ vs patch train asli\nmakin kecil = makin mirip (rerata 5 fold)",
                 loc="left", fontsize=11, color=INK)
    style(ax)

    fig.tight_layout()
    fig.savefig(E / "hasil_tahap5_6.png", dpi=160, facecolor=fig.get_facecolor())
    print("tersimpan:", E / "hasil_tahap5_6.png")


if __name__ == "__main__":
    main()
