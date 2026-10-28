"""
Tahap 1 (bagian B): patient-level split K=5 untuk ADC vs SCC.

- Outer split: StratifiedGroupKFold (group = patient_id) -> train / test.
- Inner split: ~15% pasien dari bagian train tiap fold -> validation,
  juga StratifiedGroupKFold per pasien, supaya test fold tidak pernah
  dipakai untuk memilih model.

Input : outputs/metadata_joined.csv  (hasil src/01_data_prep.py)
Output: outputs/splits.csv  (format long: satu baris per (fold, gambar))

Jalankan dari root project:
    python src/01b_split.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "outputs" / "metadata_joined.csv"
OUT_CSV = ROOT / "outputs" / "splits.csv"

N_FOLDS = 5
VAL_FRAC = 0.15
VAL_N_SPLITS = round(1 / VAL_FRAC)  # 7 -> val ~14% pasien train+val
SEED = 42  # seed tetap supaya split bisa direproduksi persis di semua tahap
CLASSES = {"aca": 0, "scc": 1}  # nor dikeluarkan: eksperimen utama ADC vs SCC


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def describe(df):
    """Ringkasan satu subset: jumlah pasien, gambar, dan proporsi ADC:SCC."""
    n_aca = int((df["superclass"] == "aca").sum())
    n_scc = int((df["superclass"] == "scc").sum())
    n = n_aca + n_scc
    return {
        "pasien": df["patient_id"].nunique(),
        "gambar": n,
        "ADC": n_aca,
        "SCC": n_scc,
        "ADC:SCC": f"{n_aca / n:.2f}:{n_scc / n:.2f}",
    }


def main():
    # keep_default_na=False: subclass kosong tetap "" (konsisten dengan 01_data_prep)
    df = pd.read_csv(IN_CSV, keep_default_na=False)

    section("1. Filter ADC vs SCC (kelas nor dikeluarkan)")
    print("Sebelum filter:", df["superclass"].value_counts().to_dict())
    df = df[df["superclass"].isin(CLASSES)].reset_index(drop=True)
    # label per GAMBAR: ini yang dipakai classifier, tidak diubah oleh
    # label mayoritas pasien di bawah.
    df["label"] = df["superclass"].map(CLASSES)
    print("Setelah filter :", df["superclass"].value_counts().to_dict())
    print(f"Total: {len(df)} gambar, {df['patient_id'].nunique()} pasien")

    section("2. Label mayoritas per pasien (hanya untuk stratifikasi)")
    counts = df.groupby(["patient_id", "superclass"]).size().unstack(fill_value=0)
    ties = counts[counts["aca"] == counts["scc"]]
    if len(ties):
        # Label mayoritas tidak terdefinisi kalau seri; berhenti, jangan menebak.
        sys.exit(f"[ERROR] Pasien dengan jumlah ADC == SCC: {ties.index.tolist()}")
    patient_label = (counts["scc"] > counts["aca"]).astype(int)  # 0=ADC, 1=SCC
    mixed = counts[(counts["aca"] > 0) & (counts["scc"] > 0)]
    print("Pasien dengan ADC dan SCC sekaligus:")
    for pid, row in mixed.iterrows():
        lab = "SCC" if patient_label[pid] else "ADC"
        print(f"  patient {pid}: ADC={row['aca']}, SCC={row['scc']} -> label stratifikasi {lab}")
    print("Jumlah pasien per label stratifikasi:",
          {"ADC": int((patient_label == 0).sum()), "SCC": int((patient_label == 1).sum())})
    df["strat_label"] = df["patient_id"].map(patient_label)

    section("3. Outer split: StratifiedGroupKFold (K=5)")
    # y = label mayoritas pasien, bukan label gambar: dengan begitu satu pasien
    # punya satu y, dan SGKF menyeimbangkan jumlah gambar per kelas per fold
    # tanpa pernah memecah gambar satu pasien ke fold berbeda.
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (trval_idx, test_idx) in enumerate(
            sgkf.split(df, y=df["strat_label"], groups=df["patient_id"])):
        trval = df.iloc[trval_idx]
        test = df.iloc[test_idx]

        # Inner split juga pakai StratifiedGroupKFold (bukan stratify per
        # pasien biasa): jumlah gambar per pasien sangat bervariasi (2 s/d
        # 30-an), jadi seimbang di level pasien belum tentu seimbang di level
        # gambar. n_splits=round(1/0.15)=7 -> tiap kandidat val ~1/7 pasien.
        # Dari 7 kandidat, pilih yang proporsi SCC-nya paling dekat dengan
        # trval. Pemilihan hanya melihat label trval, tidak menyentuh test.
        inner = StratifiedGroupKFold(n_splits=VAL_N_SPLITS, shuffle=True,
                                     random_state=SEED + fold)
        target = (trval["superclass"] == "scc").mean()
        best = None
        for _, v_idx in inner.split(trval, y=trval["strat_label"],
                                    groups=trval["patient_id"]):
            v = trval.iloc[v_idx]
            if v["superclass"].nunique() < 2:  # val wajib berisi kedua kelas
                continue
            gap = abs((v["superclass"] == "scc").mean() - target)
            if best is None or gap < best[0]:
                best = (gap, set(v["patient_id"]))
        assert best is not None, f"Fold {fold}: tidak ada kandidat val berisi 2 kelas"
        val_pids = best[1]
        train_pids = set(trval["patient_id"]) - val_pids

        for split_name, part in [("train", trval[trval["patient_id"].isin(train_pids)]),
                                 ("val", trval[trval["patient_id"].isin(val_pids)]),
                                 ("test", test)]:
            rows.append(part.assign(fold=fold, split=split_name))

    splits = pd.concat(rows, ignore_index=True)

    section("4. ASSERTION: tidak ada overlap patient_id antar train / val / test")
    all_test_pids = []
    for fold in range(N_FOLDS):
        f = splits[splits["fold"] == fold]
        p = {s: set(f.loc[f["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
        tr_te, tr_va, va_te = p["train"] & p["test"], p["train"] & p["val"], p["val"] & p["test"]
        # Teks ASCII (bukan simbol irisan) karena konsol Windows cp1252.
        print(f"Fold {fold}: overlap train-test={sorted(tr_te)}  train-val={sorted(tr_va)}  "
              f"val-test={sorted(va_te)}")
        assert not tr_te, f"Fold {fold}: patient_id overlap train-test {tr_te}"
        assert not tr_va, f"Fold {fold}: patient_id overlap train-val {tr_va}"
        assert not va_te, f"Fold {fold}: patient_id overlap val-test {va_te}"
        # Setiap gambar harus muncul tepat sekali per fold (tidak hilang/dobel).
        assert f["filename"].is_unique and len(f) == len(df), \
            f"Fold {fold}: gambar hilang atau dobel"
        all_test_pids.extend(p["test"])
    # Setiap pasien harus jadi test tepat di satu fold (definisi K-fold).
    assert sorted(all_test_pids) == sorted(df["patient_id"].unique()), \
        "Ada pasien yang tidak pernah / lebih dari sekali masuk test"
    print("\nSEMUA ASSERTION LULUS: tidak ada overlap patient_id, "
          "setiap pasien masuk test tepat 1 kali.")

    section("5. Ringkasan per fold")
    summary = []
    for fold in range(N_FOLDS):
        f = splits[splits["fold"] == fold]
        for s in ("train", "val", "test"):
            summary.append({"fold": fold, "split": s, **describe(f[f["split"] == s])})
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False))

    overall = describe(df)
    print(f"\nProporsi keseluruhan ADC:SCC = {overall['ADC:SCC']}")
    test_scc = summary[summary["split"] == "test"]["SCC"] / summary[summary["split"] == "test"]["gambar"]
    print(f"Proporsi SCC di test fold: min={test_scc.min():.2f}, max={test_scc.max():.2f} "
          f"(keseluruhan {overall['SCC'] / overall['gambar']:.2f})")

    print("\nDaftar pasien test per fold:")
    for fold in range(N_FOLDS):
        f = splits[(splits["fold"] == fold) & (splits["split"] == "test")]
        print(f"  fold {fold}: {sorted(f['patient_id'].unique().tolist())}")

    cols = ["fold", "split", "filepath", "filename", "superclass", "subclass",
            "magnification", "image_id", "patient_id", "label", "strat_label"]
    splits[cols].to_csv(OUT_CSV, index=False)
    print(f"\nTersimpan: {OUT_CSV.relative_to(ROOT).as_posix()} "
          f"({len(splits)} baris = {N_FOLDS} fold x {len(df)} gambar)")


if __name__ == "__main__":
    main()
