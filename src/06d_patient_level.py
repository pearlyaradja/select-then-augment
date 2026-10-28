"""Tahap 6d: evaluasi TINGKAT PASIEN (diagnosis per pasien, bukan per patch). Tanpa training.

Kenapa: dalam praktik, diagnosis ADC/SCC dibuat per pasien. Prediksi pasien =
rata-rata prob_scc seluruh patch test pasien itu (setiap pasien hanya muncul di
test pada SATU fold, jadi 5 fold bersama-sama memberi tepat 1 prediksi untuk
setiap 40 pasien -> metrik dihitung atas 40 pasien gabungan).
Label pasien = label mayoritas patch-nya; pasien campuran (18 & 21) dilaporkan
terpisah karena label tunggalnya tidak sepenuhnya benar.
Per seed dihitung akurasi, balanced accuracy, AUC; lalu dirata-rata antar seed.
Signifikansi: bootstrap pasien berstrata kelas (B=5000), berpasangan, vs C0 dan C1.
Output: outputs/eval/patient_level.csv, patient_level_diffs.csv, patient_predictions.csv
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "outputs" / "eval"
CONDS = ["C0", "C1", "C2", "C3", "C4"]
COMPARISONS = [("C1", "C0"), ("C2", "C0"), ("C3", "C0"), ("C4", "C0"),
               ("C2", "C1"), ("C3", "C1"), ("C4", "C1")]


def auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    rec = [np.mean(pred[y == c] == c) for c in (0, 1)]
    return dict(acc=np.mean(pred == y), bal_acc=np.mean(rec), auc=auc(y, p),
                recall_ADC=rec[0], recall_SCC=rec[1])


def main():
    rows = []
    for f in glob.glob(str(ROOT / "outputs" / "downstream" / "fold*_C*_s*" / "test_predictions.csv")):
        k, c, s = Path(f).parent.name.split("_")
        d = pd.read_csv(f)
        g = d.groupby("patient_id").agg(prob=("prob_scc", "mean"), frac_scc=("label", "mean"),
                                        n_patch=("label", "size")).reset_index()
        g["fold"], g["cond"], g["seed"] = int(k[4:]), c, int(s[1:])
        rows.append(g)
    P = pd.concat(rows)
    assert P.groupby(["cond", "seed"]).patient_id.nunique().eq(40).all(), "harus 40 pasien per (kondisi, seed)"
    assert P.groupby(["cond", "seed", "patient_id"]).fold.nunique().eq(1).all(), "pasien muncul di >1 fold test"
    P["label"] = (P.frac_scc >= 0.5).astype(int)
    P["campuran"] = (P.frac_scc > 0) & (P.frac_scc < 1)
    print("pasien campuran:", sorted(P.loc[P.campuran, "patient_id"].unique()),
          "| ADC", (P.groupby("patient_id").label.first() == 0).sum(),
          "SCC", (P.groupby("patient_id").label.first() == 1).sum())

    # Matriks prob [kondisi, seed] -> array 40 pasien berurutan sama
    pids = np.sort(P.patient_id.unique())
    y = P.drop_duplicates("patient_id").set_index("patient_id").loc[pids, "label"].to_numpy()
    prob = {(c, s): P[(P.cond == c) & (P.seed == s)].set_index("patient_id").loc[pids, "prob"].to_numpy()
            for c in CONDS for s in (0, 1, 2)}

    def stat(idx):
        return {c: pd.DataFrame([metrics(y[idx], prob[(c, s)][idx]) for s in (0, 1, 2)]).mean().to_dict()
                for c in CONDS}

    point = stat(np.arange(len(y)))
    rng = np.random.default_rng(42)
    strata = [np.flatnonzero(y == c) for c in (0, 1)]
    boots = [stat(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in strata])) for _ in range(5000)]

    met = []
    for c in CONDS:
        for m in point[c]:
            v = [b[c][m] for b in boots]
            met.append(dict(cond=c, metric=m, estimate=point[c][m],
                            ci_low=np.percentile(v, 2.5), ci_high=np.percentile(v, 97.5)))
    met = pd.DataFrame(met)
    met.to_csv(E / "patient_level.csv", index=False)

    dif = []
    for x, r in COMPARISONS:
        for m in ["bal_acc", "auc"]:
            dv = np.array([b[x][m] - b[r][m] for b in boots])
            p = min(1.0, 2 * (min((dv <= 0).sum(), (dv >= 0).sum()) + 1) / (len(dv) + 1))
            dif.append(dict(comparison=f"{x} - {r}", metric=m, diff=point[x][m] - point[r][m],
                            ci_low=np.percentile(dv, 2.5), ci_high=np.percentile(dv, 97.5), p=p))
    dif = pd.DataFrame(dif)
    dif["p_holm"] = np.nan
    for m in dif.metric.unique():
        sub = dif[dif.metric == m].sort_values("p")
        adj = np.maximum.accumulate([(len(sub) - i) * p for i, p in enumerate(sub.p)])
        dif.loc[sub.index, "p_holm"] = np.minimum(adj, 1.0)
    dif.to_csv(E / "patient_level_diffs.csv", index=False)

    # Prediksi per pasien (rata-rata 3 seed) untuk melihat pasien mana yang sulit
    pp = P.groupby(["patient_id", "fold", "label", "campuran", "cond"]).prob.mean().unstack("cond").reset_index()
    pp.to_csv(E / "patient_predictions.csv", index=False)

    print("\n=== Metrik TINGKAT PASIEN (40 pasien, rata-rata 3 seed, CI 95% bootstrap pasien) ===")
    t = met.pivot(index="cond", columns="metric", values="estimate")
    lo = met.pivot(index="cond", columns="metric", values="ci_low")
    hi = met.pivot(index="cond", columns="metric", values="ci_high")
    for c in CONDS:
        print(f"{c}: " + "  ".join(f"{m} {t.loc[c, m]:.3f} [{lo.loc[c, m]:.3f}, {hi.loc[c, m]:.3f}]"
                                   for m in ["acc", "bal_acc", "auc", "recall_ADC", "recall_SCC"]))
    print("\nJumlah pasien benar (dari 40, rata-rata seed):",
          {c: round(t.loc[c, "acc"] * 40, 1) for c in CONDS})
    print("\n=== Selisih berpasangan ===")
    for _, r in dif.iterrows():
        print(f"  {r.comparison} {r.metric}: {r['diff']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] "
              f"p={r.p:.3f} p_Holm={r.p_holm:.3f}")
    print("\nPasien SALAH di C0 (prob SCC rata-rata seed per kondisi):")
    print(pp[(pp.C0 >= 0.5).astype(int) != pp.label].round(2).to_string(index=False))
    print("\nPasien SALAH di C2:")
    print(pp[(pp.C2 >= 0.5).astype(int) != pp.label].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
