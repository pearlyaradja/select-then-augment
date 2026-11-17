"""Revisi camera-ready: cluster bootstrap per pasien untuk 105 run tambahan. Tanpa training.

Desain sama dengan 06a_bootstrap.py (fungsi metrik dipakai ulang dari sana): tiap replikasi
me-resample pasien TEST per fold, berstrata kelas; resample yang SAMA untuk semua kondisi
(berpasangan); metrik per run -> rata-rata anggota kondisi di tiap fold -> rata-rata fold.

Kondisi (anggota = daftar (cond, seed, cn)):
  C0, C1..C4        sintetis asli, seed 0-2 (hasil utama, Table II/III)
  C1n..C4n          sintetis ternormalisasi warna (07b), seed 0-2           -> Review 1
  C0s0, C2s0..C4s0  seed 0 saja
  RAND              10 undian acak seed 0 (C1 + R1..R9), rata-rata undian   -> Review 2

Tiga keluarga perbandingan, Holm per metrik DI DALAM tiap keluarga:
  colornorm  C1n..C4n - C0, C2n..C4n - C1n   (7, cermin Table III)
  colour     Cxn - Cx untuk x = 1..4         (4, berapa yang hilang karena warna disamakan)
  random     C2s0/C3s0/C4s0 - RAND           (3, filter vs rata-rata seleksi acak)
Output: outputs/revision/bootstrap_revision_metrics.csv, bootstrap_revision_diffs.csv

  python src/07f_bootstrap_revision.py --B 5000
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "revision"
_spec = importlib.util.spec_from_file_location("boot6a", ROOT / "src" / "06a_bootstrap.py")
boot6a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boot6a)
METRICS = boot6a.METRICS
DRAWS = ["C1", *(f"R{i}" for i in range(1, 10))]

GROUPS = {"C0": [("C0", s, False) for s in range(3)]}
for c in ("C1", "C2", "C3", "C4"):
    GROUPS[c] = [(c, s, False) for s in range(3)]
    GROUPS[c + "n"] = [(c, s, True) for s in range(3)]
for c in ("C0", "C2", "C3", "C4"):
    GROUPS[c + "s0"] = [(c, 0, False)]
GROUPS["RAND"] = [(d, 0, False) for d in DRAWS]
FAMILIES = {
    "colornorm": [("C1n", "C0"), ("C2n", "C0"), ("C3n", "C0"), ("C4n", "C0"),
                  ("C2n", "C1n"), ("C3n", "C1n"), ("C4n", "C1n")],
    "colour": [(f"C{i}n", f"C{i}") for i in range(1, 5)],
    "random": [("C2s0", "RAND"), ("C3s0", "RAND"), ("C4s0", "RAND")],
}


def load():
    """{(fold, cond, seed, cn): DataFrame prediksi test}, hanya run yang selesai."""
    runs = {}
    for f in (ROOT / "outputs" / "downstream").glob("fold*_s*/test_predictions.csv"):
        if not f.with_name("test_metrics.json").exists():
            continue
        parts = f.parent.name.split("_")  # fold{k}_{C}_s{seed}[_cn]
        if len(parts) > 3 and parts[3] != "cn":
            continue  # tag lain (mis. smoke) bukan bagian analisis
        key = (int(parts[0][4:]), parts[1], int(parts[2][1:]), len(parts) > 3)
        runs[key] = pd.read_csv(f).sort_values("patch_path").reset_index(drop=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    runs = load()
    needed = {m for g in GROUPS.values() for m in g}
    folds = {}
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv")
    for k in range(5):
        missing = [m for m in needed if (k, *m) not in runs]
        assert not missing, f"fold {k}: run belum ada {missing[:5]}"
        rs = {m: runs[(k, *m)] for m in needed}
        ref = next(iter(rs.values()))
        test = ps[(ps.fold == k) & (ps.split == "test")].sort_values("patch_path").reset_index(drop=True)
        # Semua run harus memprediksi patch test yang sama dengan urutan sama.
        assert test.patch_path.tolist() == ref.patch_path.tolist(), f"fold {k}: patch test tidak cocok"
        assert all(d.patch_path.tolist() == ref.patch_path.tolist() for d in rs.values())
        tr = set(ps[(ps.fold == k) & (ps.split != "test")].patient_id)
        assert not tr & set(test.patient_id), f"fold {k}: pasien test bocor ke train/val"
        pid = ref.patient_id.to_numpy()
        strata = ref.groupby("patient_id").label.mean().round().astype(int)  # label mayoritas
        folds[k] = dict(y=ref.label.to_numpy(),
                        groups={c: [np.flatnonzero(pid == p) for p in strata.index[strata == c]] for c in (0, 1)},
                        probs={m: d.prob_scc.to_numpy() for m, d in rs.items()})
    print(f"run dipakai: {len(needed)} per fold x 5 fold; cek patch test & overlap pasien OK")

    def stat(idx_per_fold):
        res = {g: {m: [] for m in METRICS} for g in GROUPS}
        for k, F in folds.items():
            idx = idx_per_fold[k]
            y = F["y"][idx]
            # Satu run bisa jadi anggota beberapa kondisi (mis. C2 seed 0) -> hitung sekali.
            cache = {m: boot6a.metrics(y, p[idx]) for m, p in F["probs"].items()}
            for g, members in GROUPS.items():
                for m in METRICS:
                    res[g][m].append(np.mean([cache[x][m] for x in members]))
        return {g: {m: float(np.mean(v)) for m, v in res[g].items()} for g in res}

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
        if (b + 1) % 500 == 0:
            print(f"  bootstrap {b + 1}/{a.B}", flush=True)

    rows = []
    for g in GROUPS:
        for m in METRICS:
            v = np.array([bb[g][m] for bb in boots])
            rows.append(dict(cond=g, metric=m, estimate=point[g][m],
                             ci_low=np.percentile(v, 2.5), ci_high=np.percentile(v, 97.5)))
    pd.DataFrame(rows).to_csv(OUT / "bootstrap_revision_metrics.csv", index=False)

    rows = []
    for fam, comps in FAMILIES.items():
        for x, r in comps:
            for m in METRICS:
                dv = np.array([bb[x][m] - bb[r][m] for bb in boots])
                p = min(1.0, 2 * (min((dv <= 0).sum(), (dv >= 0).sum()) + 1) / (len(dv) + 1))
                rows.append(dict(family=fam, comparison=f"{x} - {r}", metric=m,
                                 diff=point[x][m] - point[r][m], ci_low=np.percentile(dv, 2.5),
                                 ci_high=np.percentile(dv, 97.5), p=p))
    dif = pd.DataFrame(rows)
    dif["p_holm"] = np.nan
    for (fam, m), sub in dif.groupby(["family", "metric"]):
        sub = sub.sort_values("p")
        adj = np.maximum.accumulate([(len(sub) - i) * p for i, p in enumerate(sub.p)])
        dif.loc[sub.index, "p_holm"] = np.minimum(adj, 1.0)
    dif.to_csv(OUT / "bootstrap_revision_diffs.csv", index=False)

    print(f"\n=== Selisih berpasangan (B={a.B}); + = kondisi pertama lebih baik ===")
    for fam in FAMILIES:
        print(f"\n[{fam}]")
        for _, r in dif[dif.family == fam].iterrows():
            flag = "SIGNIFIKAN" if r.p_holm < 0.05 else ""
            print(f"  {r.comparison:12s} {r.metric:10s} {r['diff']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
                  f"  p={r.p:.3f}  p_Holm={r.p_holm:.3f} {flag}")


if __name__ == "__main__":
    main()
