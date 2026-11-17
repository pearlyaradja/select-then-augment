"""Revisi (Review 1): visualisasi warna sebelum vs sesudah normalisasi Reinhard. Tanpa training.

Kiri : contoh fold 0 (selisih warna sintetis terbesar): patch TRAIN asli, sintetis asli,
       sintetis ternormalisasi, untuk ADC dan SCC.
Kanan: selisih saturasi SCC-ADC per fold (asli / sintetis sebelum / sesudah).
Input : outputs/revision/color_norm_stats.csv (07b), selection_N200.csv, patch_splits.csv
Output: outputs/revision/color_norm_comparison.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "revision"
SRC, DST = "data/processed/synthetic", "data/processed/synthetic_colornorm"
# Asli = abu netral (acuan), sebelum = oranye, sesudah = biru (slot 1-2 palet tervalidasi).
GRAY, ORANGE, BLUE, INK, MUTED, GRID = "#8a8985", "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
FOLD, N_IMG = 0, 3


def img(p):
    return np.asarray(Image.open(ROOT / p).convert("RGB").resize((200, 200)))


def main():
    st = pd.read_csv(OUT / "color_norm_stats.csv")
    sel = pd.read_csv(ROOT / "outputs" / "selection" / "selection_N200.csv")
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv", keep_default_na=False)

    # Kolom: [kelas, label baris, gambar x N_IMG, spasi, plot]. Lebar kolom gambar disetel
    # mendekati tinggi baris supaya gambar persegi tidak dikelilingi ruang kosong.
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(6, 4 + N_IMG, width_ratios=[0.12, 0.45] + [0.5] * N_IMG + [0.3, 3.0],
                          wspace=0.05, hspace=0.08)
    rows = [("Real\n(training)", None), ("Synthetic\nbefore", SRC), ("Synthetic\nafter", DST)]
    for ci, (cls, name) in enumerate([("aca", "ADC"), ("scc", "SCC")]):
        real = ps[(ps.fold == FOLD) & (ps.split == "train") & (ps.superclass == cls)
                  & (ps.tissue_frac > 0.6)].patch_path.sample(N_IMG, random_state=0)
        syn = sel[(sel.fold == FOLD) & (sel.superclass == cls) & sel.C1].path.sample(N_IMG, random_state=0)
        for ri, (label, root) in enumerate(rows):
            r = ci * 3 + ri
            paths = real if root is None else syn.str.replace(SRC, root, n=1, regex=False)
            for j, p in enumerate(paths):
                ax = fig.add_subplot(gs[r, 2 + j])
                ax.imshow(img(p))
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            lab = fig.add_subplot(gs[r, 1])
            lab.axis("off")
            lab.text(0.95, 0.5, label, ha="right", va="center", fontsize=8, color=INK)
        mid = fig.add_subplot(gs[ci * 3:ci * 3 + 3, 0])
        mid.axis("off")
        mid.text(0.5, 0.5, name, rotation=90, ha="center", va="center", fontsize=10, color=INK, weight="bold")

    # Kanan: dot plot selisih saturasi per fold (satu sumbu, nilai langsung terbaca).
    ax = fig.add_subplot(gs[:, -1])
    gap = st.pivot(index="fold", columns="superclass", values=["sat_real", "sat_syn", "sat_norm"])
    series = [("sat_real", "Real (target)", GRAY, "D"), ("sat_syn", "Synthetic before", ORANGE, "o"),
              ("sat_norm", "Synthetic after", BLUE, "s")]
    folds = gap.index.values
    for f in folds:  # garis penghubung sebelum -> sesudah per fold
        ax.plot([gap.loc[f, ("sat_syn", "scc")] - gap.loc[f, ("sat_syn", "aca")],
                 gap.loc[f, ("sat_norm", "scc")] - gap.loc[f, ("sat_norm", "aca")]], [f, f],
                color=GRID, lw=2, zorder=1)
    for key, lab, col, mk in series:
        v = gap[key]["scc"] - gap[key]["aca"]
        ax.scatter(v, folds, s=70, color=col, marker=mk, edgecolor="white", lw=1.5, zorder=3, label=lab)
    means = {k: (gap[k]["scc"] - gap[k]["aca"]).mean() for k, *_ in series}
    ax.set_yticks(folds, [f"Fold {f}" for f in folds], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Saturation gap, SCC minus ADC (HSV, 0-255)", fontsize=9, color=MUTED)
    ax.set_title("Class colour gap per fold\n"
                 f"mean: real {means['sat_real']:.1f} | before {means['sat_syn']:.1f} | after {means['sat_norm']:.1f}",
                 fontsize=10, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Reinhard colour normalisation of synthetic images (examples: fold {FOLD}, random C1 images)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.savefig(OUT / "color_norm_comparison.png", dpi=200, bbox_inches="tight")
    print("saved", OUT / "color_norm_comparison.png")


if __name__ == "__main__":
    main()
