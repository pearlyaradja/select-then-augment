"""Tahap 4 (diagnosis): KID gambar sintetis vs asli, per fold per kelas.

Pertanyaan: apakah SCC sintetis fold 0 memang berkualitas lebih buruk (masalah
LoRA), atau hanya dinilai buruk oleh classifier filter fold 0 (masalah classifier)?
KID memakai fitur Inception-v3 standar (alat ukur, TIDAK dilatih), sehingga
penilaiannya lepas dari classifier filter kita.

Per fold k, per kelas c (semua dibandingkan dengan patch TRAIN asli fold k):
  syn_same : sintetis c       vs asli c        -> kualitas/kemiripan ke kelasnya
  syn_other: sintetis c       vs asli kelas lain -> harus LEBIH BESAR dari syn_same
  real_ref : patch VAL asli c vs asli c        -> patokan "wajar" (pasien berbeda)
Patch test tidak dipakai sama sekali.
Output: outputs/filter/kid_check.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
# Bobot Inception diunduh sekali dan disimpan di dalam project (bukan cache global).
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_hub"))
from torchmetrics.image.fid import NoTrainInceptionV3  # noqa: E402

SUBSET, N_SUBSETS = 150, 100  # 150 <= ukuran set terkecil (patch val per kelas ~174)


@torch.no_grad()
def features(model, paths, device, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(ROOT / p).convert("RGB")).copy()).permute(2, 0, 1)
                         for p in paths[i:i + bs]]).to(device)  # uint8; model me-resize sendiri ke 299
        out.append(model(x).double().cpu())
    return torch.cat(out).numpy()


def kid(a, b, rng):
    """KID = MMD^2 tak-bias dengan kernel polinomial (d=3), dirata-rata atas subset acak.
    Rumus sama dengan torchmetrics/torch-fidelity; ditulis ulang supaya fitur cukup
    diekstrak sekali lalu dipakai untuk banyak pasangan."""
    d = a.shape[1]
    vals = []
    for _ in range(N_SUBSETS):
        x = a[rng.choice(len(a), SUBSET, replace=False)]
        y = b[rng.choice(len(b), SUBSET, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        m = SUBSET
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def main():
    device = "cuda"
    model = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv")
    man = pd.read_csv(ROOT / "outputs" / "synthetic_manifest.csv")

    # Ekstrak fitur sekali per gambar unik (patch asli dipakai ulang lintas fold).
    real_paths = sorted(ps.loc[ps.split != "test", "patch_path"].unique())
    print(f"ekstrak fitur: {len(real_paths)} patch asli + {len(man)} sintetis ...", flush=True)
    feat = dict(zip(real_paths, features(model, real_paths, device)))
    for k in range(5):
        p = man.loc[man.fold == k, "path"].tolist()
        feat.update(zip(p, features(model, p, device)))
        print(f"  sintetis fold {k} selesai", flush=True)

    F = lambda paths: np.stack([feat[p] for p in paths])  # noqa: E731
    rng = np.random.default_rng(0)
    rows = []
    for k in range(5):
        f = ps[ps.fold == k]
        assert not set(f.loc[f.split == "test", "patient_id"]) & set(f.loc[f.split != "test", "patient_id"])
        for c, other in [("aca", "scc"), ("scc", "aca")]:
            tr_c = F(f[(f.split == "train") & (f.superclass == c)].patch_path)
            tr_o = F(f[(f.split == "train") & (f.superclass == other)].patch_path)
            va_c = F(f[(f.split == "val") & (f.superclass == c)].patch_path)
            sy_c = F(man[(man.fold == k) & (man.superclass == c)].path)
            for name, a, b in [("syn_same", sy_c, tr_c), ("syn_other", sy_c, tr_o), ("real_ref", va_c, tr_c)]:
                m, s = kid(a, b, rng)
                rows.append(dict(fold=k, cls=c, pair=name, kid=m, kid_std=s))
            print(f"fold {k} {c.upper()}: " + ", ".join(f"{r['pair']} {r['kid']*1e3:.1f}" for r in rows[-3:]) + "  (x1e-3)",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    report(df)


def report(df):
    """Tabel ringkas dengan nama kolom yang bisa dibaca tanpa membuka kode."""
    t = df.pivot_table(index=["fold", "cls"], columns="pair", values="kid") * 1e3
    s = df.pivot_table(index=["fold", "cls"], columns="pair", values="kid_std") * 1e3
    # Selisih lebih kecil dari gabungan simpangan = tidak bisa dibedakan -> "seimbang".
    beda = (t.syn_same - t.syn_other).abs() > (s.syn_same + s.syn_other)
    out = pd.DataFrame({
        "Fold": t.index.get_level_values("fold"),
        "Kelas": t.index.get_level_values("cls").map({"aca": "ADC", "scc": "SCC"}),
        "Sintetis vs kelas SENDIRI": t.syn_same.round(1).values,
        "Sintetis vs kelas LAWAN": t.syn_other.round(1).values,
        "Lebih mirip ke": np.where(~beda.values, "seimbang",
                                   np.where(t.syn_same.values < t.syn_other.values,
                                            "kelas sendiri (OK)", "kelas LAWAN (MASALAH)")),
        "Patokan asli-vs-asli": t.real_ref.round(1).values,
        "Jarak sintetis / patokan": [f"{v:.1f}x" for v in t.syn_same.values / t.real_ref.values],
    }).set_index("Fold")
    print("\n=== Cek KID: seberapa mirip gambar sintetis dengan gambar asli ===")
    print("Angka KID x1e-3: makin KECIL = makin MIRIP.\n")
    print(out.to_string())
    print("\nCara membaca:")
    print("- 'Sintetis vs kelas SENDIRI' harus lebih kecil dari 'vs kelas LAWAN';")
    print("  kalau tidak, gambar sintetis kelas itu lebih mirip kelas lawannya.")
    print("- 'seimbang' = selisih keduanya lebih kecil dari simpangan (tidak bisa dibedakan).")
    print("- 'Patokan asli-vs-asli' = KID antar gambar ASLI (pasien val vs train), batas terbaik")
    print("  yang realistis. 'Jarak sintetis / patokan' = berapa kali sintetis lebih jauh dari itu.")


CSV_OUT = ROOT / "outputs" / "filter" / "kid_check.csv"

if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:  # cetak ulang tabel dari hasil yang sudah ada, tanpa hitung ulang
        report(pd.read_csv(CSV_OUT))
    else:
        main()
