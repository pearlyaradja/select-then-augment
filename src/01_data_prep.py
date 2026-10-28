"""
Tahap 1 (bagian A): scan gambar, parse nama file, join ke metadata.csv.

Scope script ini SENGAJA berhenti di join. Patient-level GroupKFold split
dan cropping patch dikerjakan setelah hasil join ini diverifikasi.

Jalankan dari root project:
    python src/01_data_prep.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

# Path relatif ke root project (bukan ke cwd) supaya script tetap benar
# walau dipanggil dari folder lain.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
METADATA_CSV = ROOT / "data" / "metadata.csv"
OUT_DIR = ROOT / "outputs"
OUT_CSV = OUT_DIR / "metadata_joined.csv"

EXPECTED_FOLDERS = ["aca_bd", "aca_md", "aca_pd", "scc_bd", "scc_md", "scc_pd", "nor"]

# Kelas "nor" tidak punya subclass di nama file (nor_20x_1.jpg), jadi bagian
# subclass dibuat opsional. Regex ketat (bukan split "_") supaya file dengan
# format aneh langsung ketahuan, bukan diam-diam ter-parse salah.
FILENAME_RE = re.compile(
    r"^(?P<superclass>aca|scc|nor)"
    r"(?:_(?P<subclass>bd|md|pd))?"
    r"_(?P<magnification>\d+x)"
    r"_(?P<index>\d+)\.jpg$"
)

# resolution WAJIB ikut kunci: image_id yang sama di 20x dan 40x bisa milik
# pasien berbeda, jadi join tanpa resolution akan memberi patient_id salah.
JOIN_KEYS_LEFT = ["superclass", "subclass", "magnification", "image_id"]
JOIN_KEYS_RIGHT = ["superclass", "subclass", "resolution", "image_id"]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def scan_images():
    """Scan data/raw/, parse tiap nama file. Gagal keras kalau ada yang aneh."""
    found = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
    if sorted(EXPECTED_FOLDERS) != found:
        sys.exit(f"[ERROR] Folder di {RAW_DIR} tidak sesuai. Ditemukan: {found}")

    rows, bad = [], []
    for folder in EXPECTED_FOLDERS:
        for path in sorted((RAW_DIR / folder).iterdir()):
            if not path.is_file():
                continue
            m = FILENAME_RE.match(path.name)
            if m is None:
                bad.append((folder, path.name, "format nama tidak cocok"))
                continue
            sup = m["superclass"]
            sub = m["subclass"] or ""  # "" untuk nor, konsisten dengan sisi CSV
            # Nama file harus konsisten dengan foldernya; kalau tidak, ada
            # file yang salah taruh dan labelnya tidak bisa dipercaya.
            if folder != (f"{sup}_{sub}" if sub else sup):
                bad.append((folder, path.name, "tidak cocok dengan folder"))
                continue
            rows.append({
                "filepath": path.relative_to(ROOT).as_posix(),
                "filename": path.name,
                "superclass": sup,
                "subclass": sub,
                "magnification": m["magnification"],
                "image_id": int(m["index"]),
            })

    if bad:
        print("[ERROR] File yang gagal di-parse:")
        for b in bad:
            print("   ", b)
        sys.exit(1)
    return pd.DataFrame(rows)


def load_metadata():
    # keep_default_na=False supaya subclass kosong milik "nor" terbaca sebagai
    # "" (bukan NaN); NaN tidak pernah sama dengan NaN, jadi join akan gagal.
    df = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    print("Kolom metadata.csv:", df.columns.tolist())
    expected_cols = {"superclass", "subclass", "resolution", "image_id", "patient_id"}
    missing = expected_cols - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Kolom berikut tidak ada di metadata.csv: {missing}")

    for c in df.columns:
        df[c] = df[c].str.strip()  # jaga-jaga spasi / sisa \r dari CRLF
    # image_id jadi int supaya cocok dengan index hasil parse (int).
    df["image_id"] = df["image_id"].astype(int)
    # patient_id kosong dianggap gagal, bukan diam-diam di-cast.
    empty_pid = (df["patient_id"] == "").sum()
    if empty_pid:
        sys.exit(f"[ERROR] {empty_pid} baris metadata.csv punya patient_id kosong")
    df["patient_id"] = df["patient_id"].astype(int)

    dup = df.duplicated(JOIN_KEYS_RIGHT, keep=False)
    if dup.any():
        print("[ERROR] Kunci join duplikat di metadata.csv:")
        print(df[dup].sort_values(JOIN_KEYS_RIGHT).to_string())
        sys.exit(1)
    return df


def main():
    section("1. Scan folder gambar")
    files = scan_images()
    print(f"Total file gambar ter-parse: {len(files)}")

    section("2. Load metadata.csv")
    meta = load_metadata()
    print(f"Total baris metadata.csv: {len(meta)}")
    print(f"Jumlah patient_id unik: {meta['patient_id'].nunique()}")

    section("3. Cek format kolom kunci join (sebelum join)")
    for name, df, cols in [("Hasil parse nama file", files, JOIN_KEYS_LEFT),
                           ("metadata.csv", meta, JOIN_KEYS_RIGHT)]:
        print(f"\n[{name}]")
        for c in cols:
            vals = df[c].drop_duplicates().head(5).tolist()
            py_types = sorted({type(v).__name__ for v in df[c]})
            print(f"  {c:<14} dtype={str(df[c].dtype):<7} "
                  f"tipe python={py_types}  contoh={vals!r}")
    print("\nNilai unik magnification (parse):", sorted(files["magnification"].unique()))
    print("Nilai unik resolution (CSV)     :", sorted(meta["resolution"].unique()))

    section("4. Jumlah gambar per kelompok (superclass + subclass + magnification)")
    cnt_files = files.groupby(["superclass", "subclass", "magnification"]).size()
    cnt_meta = meta.groupby(["superclass", "subclass", "resolution"]).size()
    cnt_meta.index.names = cnt_files.index.names
    cmp = pd.concat({"folder": cnt_files, "csv": cnt_meta}, axis=1).fillna(0).astype(int)
    cmp["selisih"] = cmp["folder"] - cmp["csv"]
    cmp.loc[("TOTAL", "", "")] = cmp.sum()
    print(cmp.to_string())
    n_mismatch = (cmp["selisih"].iloc[:-1] != 0).sum()
    print(f"\nKelompok dengan jumlah berbeda: {n_mismatch}")

    section("5. Join (superclass, subclass, magnification=resolution, image_id)")
    meta_r = meta.rename(columns={"resolution": "magnification"})
    # outer + indicator supaya dua arah kegagalan terlihat:
    # file tanpa baris CSV (left_only) dan baris CSV tanpa file (right_only).
    merged = files.merge(meta_r, on=JOIN_KEYS_LEFT, how="outer",
                         indicator=True, validate="one_to_one")
    ok = merged[merged["_merge"] == "both"].drop(columns="_merge")
    no_pid = merged[merged["_merge"] == "left_only"]
    no_file = merged[merged["_merge"] == "right_only"]

    print(f"Baris berhasil join          : {len(ok)}")
    print(f"File GAGAL dapat patient_id  : {len(no_pid)}")
    if len(no_pid):
        print(no_pid[["filepath"] + JOIN_KEYS_LEFT].to_string())
    print(f"Baris CSV tanpa file gambar  : {len(no_file)}")
    if len(no_file):
        print(no_file[JOIN_KEYS_LEFT + ["patient_id"]].to_string())

    # Validasi akhir: setiap file harus dapat patient_id yang tidak kosong.
    assert len(no_pid) == 0, "Ada file yang tidak dapat patient_id"
    assert ok["patient_id"].notna().all(), "Ada patient_id kosong setelah join"
    assert len(ok) == len(files), "Jumlah hasil join != jumlah file"

    section("6. Ringkasan hasil join")
    ok["patient_id"] = ok["patient_id"].astype(int)
    ok = ok.sort_values(["superclass", "subclass", "magnification", "image_id"])
    ok = ok[["filepath", "filename", "superclass", "subclass",
             "magnification", "image_id", "patient_id"]].reset_index(drop=True)
    print(f"Total gambar: {len(ok)}, pasien unik: {ok['patient_id'].nunique()}")
    print("\nJumlah gambar & pasien per superclass:")
    print(ok.groupby("superclass").agg(n_gambar=("filename", "size"),
                                       n_pasien=("patient_id", "nunique")).to_string())
    # Info untuk tahap split: pasien yang punya >1 superclass penting
    # diketahui karena memengaruhi stratifikasi GroupKFold nanti.
    multi = ok.groupby("patient_id")["superclass"].nunique()
    print(f"\nPasien dengan >1 superclass: {(multi > 1).sum()}"
          + (f" -> {sorted(multi[multi > 1].index.tolist())}" if (multi > 1).any() else ""))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok.to_csv(OUT_CSV, index=False)
    print(f"\nTersimpan: {OUT_CSV.relative_to(ROOT).as_posix()} ({len(ok)} baris)")


if __name__ == "__main__":
    main()
