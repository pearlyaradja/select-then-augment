"""Tahap 6h: forest plot selisih berpasangan (isi Table III paper) per metrik. Tanpa training.
Input : outputs/eval/bootstrap_diffs.csv
Output: outputs/eval/forest_plot.png (+ .pdf untuk LaTeX)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "outputs" / "eval"
# Slot 1-2 palet kategorikal tervalidasi: warna = kelompok pertanyaan, bentuk marker = cadangan
# supaya tetap terbaca saat dicetak hitam-putih atau oleh pembaca buta warna.
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
METRICS = [("macro_f1", "Macro-F1"), ("auc", "AUC"), ("recall_ADC", "Recall ADC"), ("recall_SCC", "Recall SCC")]
# Urutan sama dengan Table III: dua pertanyaan penelitian, dibaca dari atas.
ROWS = [("C1 - C0", "C1 random", 0), ("C2 - C0", "C2 realism", 0), ("C3 - C0", "C3 class + memo.", 0),
        ("C4 - C0", "C4 class + realism", 0), ("C2 - C1", "C2 realism", 1), ("C3 - C1", "C3 class + memo.", 1),
        ("C4 - C1", "C4 class + realism", 1)]
GROUPS = [("vs. C0 (real only)", BLUE, "o"), ("vs. C1 (random)", ORANGE, "s")]


def main():
    d = pd.read_csv(E / "bootstrap_diffs.csv").set_index(["comparison", "metric"])
    # Celah satu baris antar kelompok supaya dua pertanyaan terpisah jelas.
    ys = [-(i + (1 if g else 0)) for i, (_, _, g) in enumerate(ROWS)]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.16, 2.7), sharey=True)
    for ax, (key, title) in zip(axes, METRICS):
        ax.axvline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
        for y, (comp, _, g) in zip(ys, ROWS):
            r = d.loc[(comp, key)]
            col, mk = GROUPS[g][1], GROUPS[g][2]
            ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
            # Isi penuh = signifikan setelah Holm; kosong = tidak. CI di sini belum dikoreksi Holm,
            # jadi CI bisa tidak menyentuh 0 tetapi tetap tidak signifikan (mis. C2-C0 macro-F1).
            sig = r.p_holm < 0.05
            ax.plot(r["diff"], y, mk, ms=5.5, mec=col, mew=1.3, mfc=col if sig else "white", zorder=3)
        ax.set_title(title, fontsize=8.5, color=INK, pad=4)
        ax.tick_params(labelsize=7, colors=MUTED, length=2)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlabel("Difference", fontsize=7.5, color=MUTED)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab, _ in ROWS], fontsize=7.5, color=INK)
    axes[0].tick_params(axis="y", length=0)
    # Legenda: kelompok (warna+bentuk) dan arti isi marker.
    h = [plt.Line2D([], [], color=c, marker=m, ms=5, lw=1.4, mfc=c, label=lab) for lab, c, m in GROUPS]
    h += [plt.Line2D([], [], color=MUTED, marker="o", ms=5, lw=0, mfc=MUTED, label="significant (Holm $p<0.05$)"),
          plt.Line2D([], [], color=MUTED, marker="o", ms=5, lw=0, mfc="white", label="not significant")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(E / f"forest_plot.{ext}", dpi=300, bbox_inches="tight")
    print("saved", E / "forest_plot.png")


if __name__ == "__main__":
    main()
