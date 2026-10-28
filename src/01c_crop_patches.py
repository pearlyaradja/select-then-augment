"""
Tahap 1 (bagian C): crop gambar 1600x1200 jadi patch 512x512 non-overlapping.

Grid 2 baris x 3 kolom, di-anchor dari pojok kiri atas:
    x0 in {0, 512, 1024}, y0 in {0, 512}
Sisa 64 px kanan dan 176 px bawah dibuang (sesuai Bab 3: "non-overlapping
patches").

Crop dilakukan SEKALI per gambar (tidak per fold). Fold tiap patch diturunkan
dari gambar asalnya lewat join ke outputs/splits.csv, sehingga 6 patch dari
gambar yang sama selalu berada di split yang sama (tidak bocor train/test).

Input : outputs/splits.csv           (hasil src/01b_split.py, ADC+SCC saja)
Output: data/processed/patches/{superclass}_{subclass}/*.png
        outputs/patches.csv          (1 baris per patch)
        outputs/patch_splits.csv     (1 baris per (fold, patch))

Jalankan dari root project:
    python src/01c_crop_patches.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLITS_CSV = ROOT / "outputs" / "splits.csv"
PATCH_DIR = ROOT / "data" / "processed" / "patches"
PATCHES_CSV = ROOT / "outputs" / "patches.csv"
PATCH_SPLITS_CSV = ROOT / "outputs" / "patch_splits.csv"

IMG_W, IMG_H = 1600, 1200
PATCH = 512
N_ROWS, N_COLS = IMG_H // PATCH, IMG_W // PATCH  # 2, 3 -> 6 patch/gambar

# Ambang saturasi HSV untuk estimasi fraksi jaringan. Latar kaca pada H&E
# hampir putih/abu (saturasi rendah), jaringan ber-eosin/hematoxylin punya
# saturasi lebih tinggi. Hanya untuk DIAGNOSTIK, tidak dipakai membuang patch.
TISSUE_SAT_THRESH = 20  # skala 0-255


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def tissue_fraction(patch_img):
    sat = np.asarray(patch_img.convert("HSV"))[:, :, 1]
    return float((sat > TISSUE_SAT_THRESH).mean())


def main():
    splits = pd.read_csv(SPLITS_CSV, keep_default_na=False)
    # Satu baris per gambar: info gambar sama di semua fold, ambil fold 0.
    images = (splits[splits["fold"] == 0]
              .drop(columns=["fold", "split"]).reset_index(drop=True))

    section("1. Cek ukuran gambar sebelum crop")
    print(f"Jumlah gambar sumber (ADC+SCC): {len(images)}")
    bad = []
    for fp in images["filepath"]:
        with Image.open(ROOT / fp) as im:
            if im.size != (IMG_W, IMG_H) or im.mode != "RGB":
                bad.append((fp, im.size, im.mode))
    if bad:
        # Grid 2x3 hanya benar untuk 1600x1200; jangan crop gambar lain diam-diam.
        print("[ERROR] Gambar dengan ukuran/mode tidak sesuai:")
        for b in bad:
            print("   ", b)
        sys.exit(1)
    print(f"Semua gambar {IMG_W}x{IMG_H} RGB. Grid {N_ROWS} baris x {N_COLS} kolom "
          f"= {N_ROWS * N_COLS} patch/gambar")
    print(f"Area terbuang per gambar: kanan {IMG_W - N_COLS * PATCH}px, "
          f"bawah {IMG_H - N_ROWS * PATCH}px")

    section("2. Crop dan simpan patch (PNG, lossless)")
    rows = []
    for i, img in images.iterrows():
        cls_dir = f"{img['superclass']}_{img['subclass']}"
        out_dir = PATCH_DIR / cls_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img["filename"]).stem
        with Image.open(ROOT / img["filepath"]) as im:
            im = im.convert("RGB")
            for r in range(N_ROWS):
                for c in range(N_COLS):
                    x0, y0 = c * PATCH, r * PATCH
                    patch = im.crop((x0, y0, x0 + PATCH, y0 + PATCH))
                    # PNG (bukan JPEG) supaya tidak ada kompresi ulang yang
                    # menambah artefak di atas artefak JPEG asli; artefak itu
                    # bisa ikut dipelajari LoRA dan memengaruhi KID/FID.
                    out_path = out_dir / f"{stem}_r{r}c{c}.png"
                    patch.save(out_path)
                    rows.append({
                        "patch_path": out_path.relative_to(ROOT).as_posix(),
                        "patch_name": out_path.name,
                        "source_filename": img["filename"],
                        "row": r, "col": c, "x0": x0, "y0": y0,
                        "superclass": img["superclass"],
                        "subclass": img["subclass"],
                        "magnification": img["magnification"],
                        "image_id": img["image_id"],
                        "patient_id": img["patient_id"],
                        "label": img["label"],
                        "tissue_frac": round(tissue_fraction(patch), 4),
                    })
        if (i + 1) % 100 == 0 or i + 1 == len(images):
            print(f"  {i + 1}/{len(images)} gambar diproses")
    patches = pd.DataFrame(rows)

    section("3. Verifikasi hasil crop")
    expected = len(images) * N_ROWS * N_COLS
    print(f"Jumlah patch: {len(patches)} (harapan {len(images)} x 6 = {expected})")
    assert len(patches) == expected, "Jumlah patch tidak sesuai"
    assert patches["patch_name"].is_unique, "Nama patch dobel"
    assert (patches.groupby("source_filename").size() == N_ROWS * N_COLS).all(), \
        "Ada gambar yang patch-nya tidak 6"
    # Cek ulang di disk pada sampel acak: file ada dan benar 512x512.
    for p in patches.sample(20, random_state=0)["patch_path"]:
        with Image.open(ROOT / p) as im:
            assert im.size == (PATCH, PATCH), f"{p} berukuran {im.size}"
    print("Semua assertion lulus: 6 patch/gambar, nama unik, sampel 512x512.")
    print("\nJumlah patch per kelas:")
    print(patches.groupby(["superclass", "subclass"]).size().to_string())

    section("4. Diagnostik fraksi jaringan (TIDAK dipakai untuk membuang patch)")
    tf = patches["tissue_frac"]
    print(tf.describe().round(3).to_string())
    for t in (0.1, 0.25, 0.5):
        print(f"  patch dengan jaringan < {t:.0%}: {(tf < t).sum()}")

    section("5. Turunkan fold patch dari gambar asal")
    patch_splits = splits[["fold", "split", "filename"]].merge(
        patches, left_on="filename", right_on="source_filename",
        how="inner", validate="many_to_many").drop(columns="filename")
    assert len(patch_splits) == 5 * len(patches), "Join patch ke fold tidak lengkap"

    # Assertion wajib: tidak ada pasien yang sama lintas split, dicek ulang di level patch.
    for fold in sorted(patch_splits["fold"].unique()):
        f = patch_splits[patch_splits["fold"] == fold]
        p = {s: set(f.loc[f["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
        src = {s: set(f.loc[f["split"] == s, "source_filename"]) for s in ("train", "val", "test")}
        tr_te, tr_va, va_te = p["train"] & p["test"], p["train"] & p["val"], p["val"] & p["test"]
        print(f"Fold {fold}: overlap patient train-test={sorted(tr_te)}  "
              f"train-val={sorted(tr_va)}  val-test={sorted(va_te)}  | "
              f"patch train/val/test = "
              f"{(f['split'] == 'train').sum()}/{(f['split'] == 'val').sum()}/"
              f"{(f['split'] == 'test').sum()}")
        assert not (tr_te or tr_va or va_te), f"Fold {fold}: overlap patient_id"
        # Gambar sumber yang sama tidak boleh muncul di dua split.
        assert not (src["train"] & src["test"] or src["train"] & src["val"]
                    or src["val"] & src["test"]), f"Fold {fold}: gambar sumber terbelah"
    print("\nSEMUA ASSERTION LULUS di level patch.")

    patches.to_csv(PATCHES_CSV, index=False)
    patch_splits.to_csv(PATCH_SPLITS_CSV, index=False)
    print(f"\nTersimpan: {PATCHES_CSV.relative_to(ROOT).as_posix()} ({len(patches)} baris)")
    print(f"Tersimpan: {PATCH_SPLITS_CSV.relative_to(ROOT).as_posix()} "
          f"({len(patch_splits)} baris)")
    print(f"Patch PNG di: {PATCH_DIR.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
