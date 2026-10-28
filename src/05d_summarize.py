"""Tahap 5: ringkasan hasil test downstream C0-C4 dari semua run yang sudah selesai.

Agregasi dua tingkat (supaya setiap fold berbobot sama walau jumlah seed-nya belum
lengkap): rata-rata antar SEED di tiap fold -> lalu rata-rata & simpangan antar FOLD.
Ini hanya ringkasan deskriptif; uji signifikansi (cluster bootstrap per pasien) ada
di Tahap 6. Tidak ada training. Output: outputs/downstream/summary.csv
"""
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["test_macro_f1", "test_auc", "test_recall_ADC", "test_recall_SCC"]
NAMES = {"test_macro_f1": "Macro-F1", "test_auc": "AUC", "test_recall_ADC": "Recall ADC",
         "test_recall_SCC": "Recall SCC"}


def main():
    files = glob.glob(str(ROOT / "outputs" / "downstream" / "fold*_C*_s*" / "test_metrics.json"))
    d = pd.DataFrame([json.load(open(f)) for f in files])
    d = d[~d.cond.isna()]
    n = d.groupby("cond").size()
    print(f"Run selesai: {len(d)}/75 | per kondisi: {n.to_dict()} | seed: {sorted(d.seed.unique())}")

    per_fold = d.groupby(["cond", "fold"])[METRICS].mean()
    agg = per_fold.groupby("cond").agg(["mean", "std"])
    out = pd.DataFrame({NAMES[m]: [f"{agg.loc[c, (m, 'mean')]:.3f} ± {agg.loc[c, (m, 'std')]:.3f}"
                                   for c in agg.index] for m in METRICS}, index=agg.index)
    print("\nTEST (rata-rata antar fold ± simpangan antar fold; tiap fold = rata-rata seed):")
    print(out.to_string())

    f1 = per_fold["test_macro_f1"].unstack("cond")
    diff = f1.sub(f1["C0"], axis=0)
    print("\nSelisih Macro-F1 terhadap C0 per fold (+ = lebih baik dari real saja):")
    print(diff.round(3).to_string())
    print("Jumlah fold lebih baik dari C0:", {c: int((diff[c] > 0).sum()) for c in diff.columns if c != "C0"})

    # Variasi antar seed: kalau sebesar selisih antar kondisi, selisih itu belum bisa dipercaya.
    sd_seed = d.groupby(["cond", "fold"]).test_macro_f1.std().groupby("cond").mean()
    print("\nRata-rata simpangan Macro-F1 antar SEED (dalam satu fold):", sd_seed.round(3).to_dict())
    agg.to_csv(ROOT / "outputs" / "downstream" / "summary.csv")


if __name__ == "__main__":
    main()
