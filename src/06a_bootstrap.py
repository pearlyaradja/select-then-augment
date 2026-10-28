"""Tahap 6a: uji signifikansi selisih C0-C4 dengan CLUSTER BOOTSTRAP per pasien.

Kenapa cluster bootstrap (bukan Wilcoxon / bootstrap per patch):
- Hanya 5 fold -> Wilcoxon antar fold praktis tidak bisa signifikan (n=5).
- Patch dari pasien yang sama sangat berkorelasi; me-resample patch satu-satu
  membuat CI terlalu sempit. Unit yang di-resample = PASIEN (cluster).

Desain (sama dengan ringkasan Tahap 5 supaya angkanya konsisten):
- Per replikasi: di tiap fold, pasien TEST di-resample dengan pengembalian,
  BERSTRATA per kelas pasien (supaya tiap resample tetap punya ADC & SCC; AUC
  tidak terdefinisi kalau satu kelas hilang). Pasien campuran (18 & 21) memakai
  label mayoritas patch-nya, seperti di Tahap 1.
- Metrik dihitung per (fold, kondisi, seed) pada patch pasien terpilih ->
  rata-rata antar seed -> rata-rata antar fold.
- Resample pasien yang SAMA dipakai untuk semua kondisi (berpasangan), sehingga
  selisih antar kondisi tidak tercampur variasi pemilihan pasien.
- p-value dua sisi dari distribusi selisih bootstrap; koreksi Holm untuk 7
  perbandingan (C1-C4 vs C0, C2-C4 vs C1).
Ambang prediksi 0,5 (sama dengan Tahap 5). Tidak ada training.
Output: outputs/eval/bootstrap_metrics.csv, outputs/eval/bootstrap_diffs.csv
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "eval"
CONDS = ["C0", "C1", "C2", "C3", "C4"]
METRICS = ["macro_f1", "auc", "recall_ADC", "recall_SCC"]
COMPARISONS = [("C1", "C0"), ("C2", "C0"), ("C3", "C0"), ("C4", "C0"),
               ("C2", "C1"), ("C3", "C1"), ("C4", "C1")]


def metrics(y, p):
    """Macro-F1, AUC (Mann-Whitney via rank), recall per kelas. Ditulis numpy langsung
    karena dipanggil ~150 ribu kali; sklearn terlalu lambat untuk itu."""
    pred = p >= 0.5
    out = {}
    f1 = []
    for c, name in [(0, "ADC"), (1, "SCC")]:
        t, q = (y == c), (pred == c)
        tp = np.sum(t & q)
        rec = tp / max(t.sum(), 1)
        prec = tp / max(q.sum(), 1)
        f1.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
        out[f"recall_{name}"] = rec
    out["macro_f1"] = float(np.mean(f1))
    n1 = y.sum()
    n0 = len(y) - n1
    r = rankdata(p)
    out["auc"] = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def load():
    rows = []
    for f in glob.glob(str(ROOT / "outputs" / "downstream" / "fold*_C*_s*" / "test_predictions.csv")):
        if not Path(f).with_name("test_metrics.json").exists():
            continue  # run belum selesai
        name = Path(f).parent.name  # fold{k}_{C}_s{seed}
        k, c, s = name.split("_")
        d = pd.read_csv(f).sort_values("patch_path").reset_index(drop=True)
        rows.append((int(k[4:]), c, int(s[1:]), d))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--allow_incomplete", action="store_true",
                    help="izinkan jalan walau belum 75 run (untuk uji skrip saja)")
    # Analisis sensitivitas: apakah kesimpulan bertahan tanpa fold tertentu (mis. fold 0,
    # tempat hampir seluruh kenaikan terjadi)? Hasil disimpan dengan akhiran _tanpa_fold{..}.
    ap.add_argument("--exclude_folds", type=int, nargs="*", default=[])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_tanpa_fold" + "".join(map(str, a.exclude_folds)) if a.exclude_folds else ""

    runs = load()
    print(f"run dimuat: {len(runs)}/75")
    if len(runs) < 75 and not a.allow_incomplete:
        raise SystemExit("Belum 75 run. Pakai --allow_incomplete hanya untuk uji skrip.")
    runs = [r for r in runs if r[0] not in a.exclude_folds]
    if a.exclude_folds:
        print(f"SENSITIVITAS: fold {a.exclude_folds} dikeluarkan")

    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv")
    folds = {}
    for k in sorted({r[0] for r in runs}):
        rs = [r for r in runs if r[0] == k]
        ref = rs[0][3]
        test = ps[(ps.fold == k) & (ps.split == "test")].sort_values("patch_path").reset_index(drop=True)
        # Prediksi semua kondisi/seed harus untuk patch test yang sama persis dan berurutan sama.
        assert test.patch_path.tolist() == ref.patch_path.tolist(), f"fold {k}: patch test tidak cocok"
        for _, _, _, d in rs:
            assert d.patch_path.tolist() == ref.patch_path.tolist()
        # Patient-level: pasien test fold ini tidak boleh ada di train/val fold ini.
        tr = set(ps[(ps.fold == k) & (ps.split != "test")].patient_id)
        assert not tr & set(test.patient_id), f"fold {k}: pasien test bocor ke train/val"
        y = ref.label.to_numpy()
        pid = ref.patient_id.to_numpy()
        strata = ref.groupby("patient_id").label.mean().round().astype(int)  # label mayoritas
        groups = {c: [np.flatnonzero(pid == p) for p in strata.index[strata == c]] for c in (0, 1)}
        probs = {(c, s): d.prob_scc.to_numpy() for kk, c, s, d in rs}
        folds[k] = dict(y=y, groups=groups, probs=probs)
        print(f"fold {k}: {len(y)} patch test, pasien ADC {len(groups[0])}, SCC {len(groups[1])}, "
              f"run {len(probs)}; cek overlap pasien OK")

    def stat(idx_per_fold):
        """Metrik per kondisi: rata-rata seed di tiap fold -> rata-rata fold."""
        res = {c: {m: [] for m in METRICS} for c in CONDS}
        for k, F in folds.items():
            idx = idx_per_fold[k]
            y = F["y"][idx]
            for c in CONDS:
                ms = [metrics(y, F["probs"][(c, s)][idx]) for (cc, s) in F["probs"] if cc == c]
                for m in METRICS:
                    res[c][m].append(np.mean([x[m] for x in ms]))
        return {c: {m: float(np.mean(v)) for m, v in res[c].items()} for c in res}

    rng = np.random.default_rng(42)
    point = stat({k: np.arange(len(F["y"])) for k, F in folds.items()})
    boots = []
    for b in range(a.B):
        idx = {}
        for k, F in folds.items():
            parts = []
            for c in (0, 1):
                g = F["groups"][c]
                parts += [g[i] for i in rng.integers(0, len(g), len(g))]
            idx[k] = np.concatenate(parts)
        boots.append(stat(idx))
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{a.B}", flush=True)

    rows = []
    for c in CONDS:
        for m in METRICS:
            v = np.array([bb[c][m] for bb in boots])
            rows.append(dict(cond=c, metric=m, estimate=point[c][m],
                             ci_low=np.percentile(v, 2.5), ci_high=np.percentile(v, 97.5)))
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"bootstrap_metrics{suffix}.csv", index=False)

    rows = []
    for x, ref in COMPARISONS:
        for m in METRICS:
            dv = np.array([bb[x][m] - bb[ref][m] for bb in boots])
            # +1 di pembilang & penyebut: p tidak pernah tepat 0 (resolusi terbatas oleh B).
            p = min(1.0, 2 * (min((dv <= 0).sum(), (dv >= 0).sum()) + 1) / (len(dv) + 1))
            rows.append(dict(comparison=f"{x} - {ref}", metric=m, diff=point[x][m] - point[ref][m],
                             ci_low=np.percentile(dv, 2.5), ci_high=np.percentile(dv, 97.5), p=p))
    dif = pd.DataFrame(rows)
    # Holm per metrik atas 7 perbandingan: makin kecil p, makin ketat pengalinya.
    dif["p_holm"] = np.nan
    for m in METRICS:
        sub = dif[dif.metric == m].sort_values("p")
        adj = np.maximum.accumulate([(len(sub) - i) * p for i, p in enumerate(sub.p)])
        dif.loc[sub.index, "p_holm"] = np.minimum(adj, 1.0)
    dif.to_csv(OUT / f"bootstrap_diffs{suffix}.csv", index=False)

    show = met.pivot(index="cond", columns="metric", values="estimate")
    lo = met.pivot(index="cond", columns="metric", values="ci_low")
    hi = met.pivot(index="cond", columns="metric", values="ci_high")
    print(f"\n=== Metrik TEST (CI 95% cluster bootstrap per pasien, B={a.B}) ===")
    print(pd.DataFrame({m: [f"{show.loc[c, m]:.3f} [{lo.loc[c, m]:.3f}, {hi.loc[c, m]:.3f}]" for c in CONDS]
                        for m in METRICS}, index=CONDS).to_string())
    print("\n=== Selisih berpasangan (+ = kondisi pertama lebih baik) ===")
    for m in ["macro_f1", "auc"]:
        print(f"\n{m}:")
        d = dif[dif.metric == m]
        for _, r in d.iterrows():
            sig = "SIGNIFIKAN" if r.p_holm < 0.05 else "tidak signifikan"
            print(f"  {r.comparison}: {r['diff']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  "
                  f"p={r.p:.3f}  p_Holm={r.p_holm:.3f}  -> {sig}")


if __name__ == "__main__":
    main()
