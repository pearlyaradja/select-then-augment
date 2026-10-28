"""Tahap 6e: contoh visual untuk Bab 4. Tanpa training.

(1) contoh_seleksi_C2_vs_C4.png: gambar sintetis yang DIPILIH C2 vs C4 di fold 1
    (fold tempat C4 jatuh), 6 ADC + 6 SCC acak per kondisi, dengan saturasi rata-rata.
    Tujuan: memperlihatkan dugaan bahwa filter kelas memilih warna lebih ekstrem.
    Gambar yang dipilih oleh kedua kondisi tidak diikutkan supaya perbedaannya terlihat.
(2) contoh_pasien_salah_C2.png: 4 patch test dari setiap pasien yang tetap salah
    didiagnosis di C2 (prob SCC patch = rata-rata 3 seed), ditambah 1 pasien ADC dan
    1 SCC yang benar sebagai pembanding. Pasien campuran (18, 21) tidak ditampilkan.
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "outputs" / "eval"
INK, MUTED = "#0b0b0b", "#52514e"


def load(p, size=192):
    return Image.open(ROOT / p).convert("RGB").resize((size, size))


def saturation(im):
    return np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1].mean()


def gallery_selection(fold=1, n=6):
    sel = pd.read_csv(ROOT / "outputs" / "selection" / "selection_N200.csv")
    s = sel[sel.fold == fold]
    fig, axes = plt.subplots(4, n, figsize=(n * 1.9, 4 * 2.15))
    fig.patch.set_facecolor("white")
    for r, (cond, other, c) in enumerate([("C2", "C4", "aca"), ("C4", "C2", "aca"),
                                         ("C2", "C4", "scc"), ("C4", "C2", "scc")]):
        pool = s[s[cond] & ~s[other] & (s.superclass == c)]
        allsat = np.mean([saturation(load(p, 64)) for p in s[s[cond] & (s.superclass == c)].path])
        for j, (_, row) in enumerate(pool.sample(n, random_state=0).iterrows()):
            im = load(row.path)
            ax = axes[r, j]
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"p={row.p_intended:.2f}", fontsize=8, color=MUTED)
        axes[r, 0].set_ylabel(f"{cond} — {'ADC' if c == 'aca' else 'SCC'}\nsaturasi set {allsat:.0f}",
                              fontsize=10, color=INK)
    fig.suptitle(f"Fold {fold}: sintetis yang dipilih C2 (realism) vs C4 (semua filter); "
                 "p = keyakinan classifier filter terhadap kelasnya", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(E / "contoh_seleksi_C2_vs_C4.png", dpi=130)


def gallery_errors(n=4):
    pp = pd.read_csv(E / "patient_predictions.csv")
    pp = pp[~pp.campuran]
    wrong = pp[(pp.C2 >= 0.5).astype(int) != pp.label]
    right = pp[(pp.C2 >= 0.5).astype(int) == pp.label]
    # pembanding: pasien benar yang paling yakin per kelas
    ref = [right[right.label == 0].sort_values("C2").iloc[0], right[right.label == 1].sort_values("C2").iloc[-1]]
    rows = [(r, "SALAH") for _, r in wrong.iterrows()] + [(r, "benar (pembanding)") for r in ref]

    preds = []
    for f in glob.glob(str(ROOT / "outputs" / "downstream" / "fold*_C2_s*" / "test_predictions.csv")):
        preds.append(pd.read_csv(f))
    patch = pd.concat(preds).groupby(["patch_path", "patient_id", "label"]).prob_scc.mean().reset_index()

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 2.0, len(rows) * 2.25))
    fig.patch.set_facecolor("white")
    for i, (r, status) in enumerate(rows):
        ps = patch[patch.patient_id == r.patient_id].sample(n, random_state=0)
        for j, (_, q) in enumerate(ps.iterrows()):
            ax = axes[i, j]
            ax.imshow(load(q.patch_path))
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"P(SCC)={q.prob_scc:.2f}", fontsize=8, color=MUTED)
        lab = "ADC" if r.label == 0 else "SCC"
        axes[i, 0].set_ylabel(f"Pasien {int(r.patient_id)} ({lab})\n{status}\nP(SCC) pasien {r.C2:.2f}",
                              fontsize=9, color=INK)
    fig.suptitle("Kondisi C2: pasien yang salah didiagnosis (ambang 0,5) + pembanding yang benar",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(E / "contoh_pasien_salah_C2.png", dpi=130)


if __name__ == "__main__":
    gallery_selection()
    gallery_errors()
    print("tersimpan:", E / "contoh_seleksi_C2_vs_C4.png", "|", E / "contoh_pasien_salah_C2.png")
