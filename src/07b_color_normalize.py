"""Revisi (Review 1): normalisasi warna Reinhard gambar sintetis per fold & kelas.

KENAPA: di gambar sintetis, selisih saturasi SCC-ADC 2-3x lebih besar dari data asli
(12-16 vs 5.3). Classifier bisa belajar warna, bukan morfologi. Reviewer minta warna
sintetis disamakan dengan statistik warna data asli, lalu evaluasi diulang.

Reinhard (2001) di ruang CIELAB: tiap gambar sintetis distandarkan dengan rata-rata &
SD-nya sendiri per kanal, lalu diberi rata-rata & SD REFERENSI. Referensi = rata-rata
statistik per gambar dari patch TRAIN asli kelas yang sama di fold yang sama (bukan
val/test, supaya tidak bocor). Hasilnya selisih warna antar kelas ikut data asli.

Hanya gambar yang dipakai C1-C4 (selection_N200.csv) yang dinormalisasi. Tidak ada
training. Output:
  data/processed/synthetic_colornorm/fold{k}/{aca|scc}/<nama sama>.png
  outputs/revision/color_norm_stats.csv   selisih saturasi SCC-ADC sebelum/sesudah
  outputs/revision/color_norm_examples.png contoh sebelum/sesudah

  python src/07b_color_normalize.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = "data/processed/synthetic", "data/processed/synthetic_colornorm"
OUT = ROOT / "outputs" / "revision"
# sRGB (D65) -> XYZ; tidak ada skimage/cv2 di venv, jadi rumus standar ditulis langsung.
M = np.array([[0.4124564, 0.3575761, 0.1804375],
              [0.2126729, 0.7151522, 0.0721750],
              [0.0193339, 0.1191920, 0.9503041]])
WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    t = (c @ M.T) / WHITE
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    t = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * WHITE
    c = np.clip(t @ np.linalg.inv(M).T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def load(path, size=None):
    im = Image.open(ROOT / path).convert("RGB")
    return np.asarray(im.resize((size, size)) if size else im, dtype=np.float64)


def lab_stats(rgb):
    lab = rgb2lab(rgb).reshape(-1, 3)
    return lab.mean(0), lab.std(0) + 1e-6  # +eps: patch hampir polos tidak membagi nol


def saturation(path):
    # Ukuran yang sama dengan 05c_color_check.py (HSV PIL, 128 px) supaya angka sebanding.
    im = Image.open(ROOT / path).convert("RGB").resize((128, 128))
    return np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1].mean()


def main():
    sel = pd.read_csv(ROOT / "outputs" / "selection" / "selection_N200.csv")
    sel = sel[sel[["C1", "C2", "C3", "C4"]].any(axis=1)]
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv", keep_default_na=False)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, examples = [], []
    for (k, cls), g in sel.groupby(["fold", "superclass"]):
        real = ps[(ps.fold == k) & (ps.split == "train") & (ps.superclass == cls)].patch_path
        st = [lab_stats(load(p, 256)) for p in real]  # 256 px cukup untuk statistik, lebih cepat
        ref_mu, ref_sd = np.mean([s[0] for s in st], 0), np.mean([s[1] for s in st], 0)
        for p in g.path:
            rgb = load(p)
            lab = rgb2lab(rgb)
            mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
            out = lab2rgb((lab - mu) / sd * ref_sd + ref_mu)
            dst = ROOT / p.replace(SRC, DST, 1)
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out).save(dst)
        rows.append({"fold": k, "superclass": cls, "n_real": len(real), "n_syn": len(g),
                     "sat_real": np.mean([saturation(p) for p in real]),
                     "sat_syn": np.mean([saturation(p) for p in g.path]),
                     "sat_norm": np.mean([saturation(p.replace(SRC, DST, 1)) for p in g.path])})
        examples.append((g.path.iloc[0], k, cls))
        print(f"fold {k} {cls}: {len(g)} sintetis dinormalisasi ke {len(real)} patch train asli")
    df = pd.DataFrame(rows)
    gap = df.pivot(index="fold", columns="superclass", values=["sat_real", "sat_syn", "sat_norm"])
    summ = pd.DataFrame({s: gap[s]["scc"] - gap[s]["aca"] for s in ("sat_real", "sat_syn", "sat_norm")})
    print("Selisih saturasi SCC-ADC per fold (asli / sintetis / sintetis ternormalisasi):")
    print(summ.round(1).to_string())
    df.to_csv(OUT / "color_norm_stats.csv", index=False)

    # Contoh sebelum/sesudah: satu gambar per fold & kelas, baris atas asli, bawah normal.
    tiles = [np.hstack([load(p, 160).astype(np.uint8) for p, _, _ in examples]),
             np.hstack([load(p.replace(SRC, DST, 1), 160).astype(np.uint8) for p, _, _ in examples])]
    Image.fromarray(np.vstack(tiles)).save(OUT / "color_norm_examples.png")
    print("saved", OUT / "color_norm_stats.csv", "dan color_norm_examples.png")


if __name__ == "__main__":
    main()
