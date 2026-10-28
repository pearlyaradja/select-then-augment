"""Tahap 5 (diagnosis): apakah seleksi C1-C4 memilih gambar dengan warna ekstrem?

Dugaan: filter kelas (C3/C4) memilih sintetis yang paling "yakin" menurut classifier
filter (p ~0,99). Kalau keyakinan itu datang dari WARNA (SCC lebih pink, ADC lebih
ungu; dicatat sejak Tahap 3), classifier downstream ikut belajar jalan pintas warna
dan gagal pada pasien yang pewarnaannya berbeda.

Ukuran per gambar (downsample 128 px supaya cepat): rata-rata saturasi & hue (HSV),
dan rasio merah/biru (R/B; makin besar = makin pink). Dibandingkan dengan patch TRAIN
asli fold yang sama (bukan val/test). Tidak ada training. CPU saja.
Output: outputs/selection/color_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def color_stats(path):
    im = Image.open(ROOT / path).convert("RGB").resize((128, 128))
    rgb = np.asarray(im, dtype=np.float32)
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    return hsv[..., 1].mean(), hsv[..., 0].mean(), rgb[..., 0].mean() / max(rgb[..., 2].mean(), 1.0)


def main():
    sel = pd.read_csv(ROOT / "outputs" / "selection" / "selection_N200.csv")
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv")
    rows = []
    for k in range(5):
        real = ps[(ps.fold == k) & (ps.split == "train")]
        groups = [("asli", c, real[real.superclass == c].patch_path) for c in ("aca", "scc")]
        s = sel[sel.fold == k]
        for cond in ["C1", "C2", "C3", "C4"]:
            groups += [(cond, c, s[s[cond] & (s.superclass == c)].path) for c in ("aca", "scc")]
        for name, c, paths in groups:
            st = np.array([color_stats(p) for p in paths])
            rows.append(dict(fold=k, set=name, kelas={"aca": "ADC", "scc": "SCC"}[c],
                             saturasi=st[:, 0].mean(), hue=st[:, 1].mean(), r_per_b=st[:, 2].mean()))
        print(f"fold {k} selesai", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / "selection" / "color_check.csv", index=False)

    # Inti: SELISIH warna SCC - ADC. Kalau seleksi memperbesar selisih ini jauh melebihi
    # data asli, gambar terpilih membedakan kelas lewat warna (jalan pintas).
    w = df.pivot_table(index=["fold", "set"], columns="kelas", values=["saturasi", "r_per_b"])
    gap = pd.DataFrame({"selisih saturasi SCC-ADC": w[("saturasi", "SCC")] - w[("saturasi", "ADC")],
                        "selisih R/B SCC-ADC": w[("r_per_b", "SCC")] - w[("r_per_b", "ADC")]}).round(3)
    order = ["asli", "C1", "C2", "C3", "C4"]
    gap = gap.reindex(pd.MultiIndex.from_product([range(5), order], names=["fold", "set"]))
    print("\nSelisih warna SCC - ADC (makin besar = kelas makin dibedakan oleh warna):")
    print(gap.to_string())
    print("\nRata-rata 5 fold:")
    print(gap.groupby(level="set").mean().reindex(order).round(3).to_string())


if __name__ == "__main__":
    main()
