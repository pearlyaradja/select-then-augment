"""Revisi camera-ready: ringkasan deskriptif hasil 105 run tambahan. Tanpa training.

(1) Review 1 (warna): C1-C4 dengan sintetis ternormalisasi (_cn) vs sintetis asli, C0 sebagai
    acuan. Metrik test per run -> rata-rata 3 seed -> rata-rata 5 fold (sama dengan Table II).
(2) Review 2 (undian acak): 10 undian acak seed 0 (C1 + R1..R9). Untuk tiap undian: rata-rata
    5 fold. C2/C3/C4 seed 0 dibandingkan dengan sebaran 10 undian itu (peringkat = berapa
    undian yang dikalahkan). Seed 0 saja karena undian baru hanya dilatih dengan seed 0.
Input : outputs/downstream/*/test_metrics.json
Output: outputs/revision/colornorm_summary.csv, outputs/revision/random_draws_summary.csv
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "outputs" / "downstream"
OUT = ROOT / "outputs" / "revision"
METRICS = ["test_macro_f1", "test_auc", "test_recall_ADC", "test_recall_SCC"]


def load():
    rows = []
    for f in DS.glob("fold*_s*/test_metrics.json"):
        m = json.loads(f.read_text())
        m["colornorm"] = f.parent.name.endswith("_cn")
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    df = load()
    OUT.mkdir(parents=True, exist_ok=True)

    # (1) warna: rata-rata seed dulu, lalu fold (urutan sama dengan hasil utama).
    main_ = df[df.cond.isin(["C0", "C1", "C2", "C3", "C4"])]
    by_fold = main_.groupby(["cond", "colornorm", "fold"])[METRICS].mean()
    n_runs = main_.groupby(["cond", "colornorm"]).size()
    assert (n_runs == 15).all(), f"jumlah run tidak 15 per kondisi:\n{n_runs}"
    col = by_fold.groupby(["cond", "colornorm"]).agg(["mean", "std"])
    col.columns = [f"{m.replace('test_', '')}_{s}" for m, s in col.columns]
    col = col.reset_index()
    col.to_csv(OUT / "colornorm_summary.csv", index=False)
    print("=== (1) Warna asli vs ternormalisasi: rata-rata 5 fold x 3 seed ===")
    show = col[["cond", "colornorm"] + [c for c in col if c.endswith("_mean")]]
    print(show.round(3).to_string(index=False))

    # (2) undian acak, seed 0.
    s0 = df[(df.seed == 0) & ~df.colornorm]
    draws = [c for c in ["C1", *(f"R{i}" for i in range(1, 10))]]
    per = s0.groupby(["cond", "fold"])[METRICS].mean().groupby("cond").mean()
    assert all((s0.cond == c).sum() == 5 for c in draws + ["C2", "C3", "C4"]), "run seed 0 kurang"
    rnd = per.loc[draws]
    rows = []
    for c in ["C0", "C2", "C3", "C4"]:
        r = {"cond": c}
        for m in METRICS:
            r[m] = per.loc[c, m]
            r[f"{m}_beats"] = int((per.loc[c, m] > rnd[m]).sum())  # dari 10 undian
        rows.append(r)
    for m in METRICS:
        rows.append({"cond": f"random min/mean/max", m: f"{rnd[m].min():.3f}/{rnd[m].mean():.3f}/{rnd[m].max():.3f}"})
    out = pd.DataFrame(rows)
    rnd.reset_index().to_csv(OUT / "random_draws_per_draw.csv", index=False)
    out.to_csv(OUT / "random_draws_summary.csv", index=False)
    print("\n=== (2) Seed 0: 10 undian acak (C1 + R1..R9), rata-rata 5 fold per undian ===")
    print(rnd.round(3).to_string())
    print("\nKondisi vs sebaran undian (beats = jumlah undian yang dikalahkan, dari 10):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
