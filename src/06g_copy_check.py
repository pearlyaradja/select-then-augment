"""Tahap 6g: periksa gambar sintetis yang ditandai filter memorization (c). Tanpa training.

Kenapa: filter (c) memakai fitur classifier filter, yang dilatih pada patch training
yang sama. Pemeriksaan kedua ini independen dari classifier itu: korelasi piksel
(grayscale 64x64, dinormalisasi rata-rata) antara gambar yang ditandai dan SEMUA patch
training kelas dan fold yang sama, pada 8 orientasi (4 rotasi x flip), karena LoRA
dilatih dengan augmentasi dihedral sehingga salinan bisa saja terputar.
Pembanding: korelasi tertinggi patch VAL asli (pasien lain) ke patch training yang sama.
Gambar yang korelasinya di bawah median val tidak lebih mirip ke data training daripada
gambar asli dari pasien lain.

Keterbatasan: resolusi rendah dan grayscale; ini cek near-duplicate, bukan bukti
bahwa tidak ada memorisasi pada tingkat fitur.

Output: outputs/eval/copy_check.csv, outputs/eval/copy_check.png

  python src/06g_copy_check.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "eval"


def gray(p, size=64):
    return np.asarray(Image.open(ROOT / p).convert("L").resize((size, size)), np.float32)


def unit(v):
    v = v.ravel() - v.mean()
    return v / (np.linalg.norm(v) + 1e-8)


def variants(p):
    """8 orientasi gambar sebagai vektor satuan [8, 64*64]."""
    a = gray(p)
    return np.stack([unit(np.rot90(f, k)) for f in (a, a[:, ::-1]) for k in range(4)])


def main():
    sel = pd.read_csv(ROOT / "outputs" / "selection" / "selection_N200.csv")
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv")
    flagged = sel[~sel.pass_c]
    print(f"gambar yang ditandai filter (c): {len(flagged)} dari {len(sel)}")

    rows, pairs = [], []
    for (k, c), grp in flagged.groupby(["fold", "superclass"]):
        f = ps[(ps.fold == k) & (ps.superclass == c)]
        tr = f[f.split == "train"].patch_path.tolist()
        T = np.stack([unit(gray(p)) for p in tr])
        # Pembanding: seberapa mirip gambar ASLI dari pasien lain (val) ke patch training
        v_best = np.array([(variants(p) @ T.T).max() for p in f[f.split == "val"].patch_path])
        for _, r in grp.iterrows():
            c_all = variants(r.path) @ T.T
            j = int(c_all.max(0).argmax())
            rows.append(dict(path=r.path, fold=k, superclass=c, max_sim_train_feature=r.max_sim_train,
                             best_pixel_corr=round(float(c_all.max()), 3), nearest_train=tr[j],
                             val_median=round(float(np.median(v_best)), 3), val_max=round(float(v_best.max()), 3),
                             in_C1=r.C1, in_C2=r.C2, in_C3=r.C3, in_C4=r.C4))
            pairs.append((r.path, tr[j], float(c_all.max())))

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "copy_check.csv", index=False)
    print(df.drop(columns=["path", "nearest_train"]).to_string(index=False))

    fig, ax = plt.subplots(len(pairs), 2, figsize=(6, 3 * len(pairs)), squeeze=False)
    for i, (syn, near, r) in enumerate(pairs):
        ax[i, 0].imshow(Image.open(ROOT / syn).convert("RGB"))
        ax[i, 0].set_title(Path(syn).name, fontsize=8)
        ax[i, 1].imshow(Image.open(ROOT / near).convert("RGB"))
        ax[i, 1].set_title(f"patch training terdekat (r={r:.2f})", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "copy_check.png", dpi=110)
    print("tersimpan:", OUT / "copy_check.csv", "|", OUT / "copy_check.png")


if __name__ == "__main__":
    main()
