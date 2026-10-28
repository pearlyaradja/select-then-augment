"""Tahap 6b: KID (utama) dan FID (pendukung) untuk set sintetis C1-C4 yang DIPAKAI di Tahap 5.

Pertanyaan: apakah filter benar-benar memilih gambar yang lebih mirip data asli?
Per fold k, semua dibandingkan dengan patch TRAIN asli fold k (patch test tidak dipakai):
  - per kelas: 200 sintetis terpilih kondisi C vs patch train asli kelas itu (KID)
  - gabungan  : 400 sintetis terpilih (ADC+SCC) vs semua patch train asli (KID & FID)
  - "semua"   : 1000/kelas sintetis tanpa seleksi (pembanding; di-subsample 200/kelas
                supaya ukuran sampel sama dengan C1-C4)
  - "patokan" : patch VAL asli vs patch train asli (batas wajar, pasien berbeda)
Catatan: FID bias ke atas untuk sampel kecil (400); hanya sah untuk MEMBANDINGKAN
kondisi dengan ukuran sampel sama, bukan dibandingkan dengan angka FID paper lain.
Fitur Inception-v3 standar (alat ukur, TIDAK dilatih). Tidak ada training.
Output: outputs/eval/kid_fid.csv
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("kid4f", ROOT / "src" / "04f_kid_check.py")
kid4f = importlib.util.module_from_spec(spec)
sys.modules["kid4f"] = kid4f
spec.loader.exec_module(kid4f)  # memakai features() & kid() yang sudah divalidasi di Tahap 4


def fid(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)  # scipy baru: argumen disp sudah dihapus
    covmean = covmean.real  # bagian imajiner kecil = galat numerik
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def main():
    device = "cuda"
    model = kid4f.NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv")
    sel = pd.read_csv(ROOT / "outputs" / "selection" / "selection_N200.csv")

    real_paths = sorted(ps.loc[ps.split != "test", "patch_path"].unique())
    print(f"ekstrak fitur: {len(real_paths)} patch asli + {len(sel)} sintetis ...", flush=True)
    feat = dict(zip(real_paths, kid4f.features(model, real_paths, device)))
    for k in range(5):
        p = sel.loc[sel.fold == k, "path"].tolist()
        feat.update(zip(p, kid4f.features(model, p, device)))
        print(f"  sintetis fold {k} selesai", flush=True)
    F = lambda paths: np.stack([feat[p] for p in paths])  # noqa: E731

    rng = np.random.default_rng(0)
    rows = []
    for k in range(5):
        f = ps[ps.fold == k]
        assert not set(f.loc[f.split == "test", "patient_id"]) & set(f.loc[f.split != "test", "patient_id"])
        s = sel[sel.fold == k]
        sets = {}
        for cond in ["C1", "C2", "C3", "C4"]:
            sets[cond] = {c: s[s[cond] & (s.superclass == c)].path.tolist() for c in ("aca", "scc")}
        # Tanpa seleksi: subsample acak 200/kelas dari 1000 (ukuran sama dengan C1-C4).
        sets["semua"] = {c: s[s.superclass == c].path.sample(200, random_state=k).tolist() for c in ("aca", "scc")}
        val = {c: f[(f.split == "val") & (f.superclass == c)].patch_path.tolist() for c in ("aca", "scc")}
        tr = {c: F(f[(f.split == "train") & (f.superclass == c)].patch_path) for c in ("aca", "scc")}
        tr_all = np.concatenate([tr["aca"], tr["scc"]])
        for name, paths in list(sets.items()) + [("patokan", val)]:
            assert name == "patokan" or all(len(v) == 200 for v in paths.values()), (k, name)
            r = dict(fold=k, set=name)
            for c in ("aca", "scc"):
                r[f"kid_{c}"], _ = kid4f.kid(F(paths[c]), tr[c], rng)
            both = F(paths["aca"] + paths["scc"])
            r["kid_all"], _ = kid4f.kid(both, tr_all, rng)
            r["fid_all"] = fid(both, tr_all)
            rows.append(r)
            print(f"fold {k} {name:8s} KIDx1e3 ADC {r['kid_aca']*1e3:6.1f} SCC {r['kid_scc']*1e3:6.1f} "
                  f"gab {r['kid_all']*1e3:6.1f} | FID {r['fid_all']:6.1f}", flush=True)

    df = pd.DataFrame(rows)
    (ROOT / "outputs" / "eval").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "outputs" / "eval" / "kid_fid.csv", index=False)
    order = ["patokan", "semua", "C1", "C2", "C3", "C4"]
    t = df.groupby("set")[["kid_aca", "kid_scc", "kid_all", "fid_all"]].agg(["mean", "std"]).reindex(order)
    print("\n=== Rata-rata 5 fold (KID x1e-3; makin KECIL = makin mirip data asli) ===")
    for name in order:
        print(f"{name:8s} " + "  ".join(
            f"{m}: {t.loc[name, (m, 'mean')] * (1e3 if m.startswith('kid') else 1):6.1f}"
            f"±{t.loc[name, (m, 'std')] * (1e3 if m.startswith('kid') else 1):5.1f}"
            for m in ["kid_aca", "kid_scc", "kid_all", "fid_all"]))


if __name__ == "__main__":
    main()
