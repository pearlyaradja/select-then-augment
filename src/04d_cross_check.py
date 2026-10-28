"""Tahap 4 (diagnosis): uji silang classifier filter x gambar sintetis.

Pertanyaan: kenapa SCC sintetis fold 0 hampir semua gagal filter (a)?
- Kalau classifier fold LAIN juga menilainya "mirip ADC"  -> masalah di gambar
  (LoRA SCC fold 0 lemah).
- Kalau hanya classifier fold 0 yang menilai begitu -> masalah di classifier.

Matriks: classifier fold i (ensemble v3+v3s1+v3s2, TTA 8 orientasi, sama persis
dengan skor filter resmi) x 200 gambar sintetis kelas --cls dari fold j.
Tidak ada training dan tidak menyentuh patch test: murni penilaian ulang.
Output: outputs/filter/cross_check_{cls}.csv
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
# Pakai ulang fungsi embed() dari 04b supaya preprocessing & TTA identik dengan
# skor resmi (nama file diawali angka, jadi tidak bisa di-import biasa).
_spec = importlib.util.spec_from_file_location("fs", ROOT / "src" / "04b_filter_scores.py")
fs = importlib.util.module_from_spec(_spec)
# Didaftarkan ke sys.modules: worker DataLoader di Windows (spawn) perlu bisa
# menemukan modul "fs" lagi saat membongkar (unpickle) ImgDataset.
sys.modules["fs"] = fs
_spec.loader.exec_module(fs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls", default="scc", choices=["aca", "scc"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--clf", default="v3+v3s1+v3s2")
    args = p.parse_args()
    device = "cuda"

    man = pd.read_csv(fs.SYN_MANIFEST)
    # Sampel tetap (seed 0) per fold, sama untuk semua classifier -> sel matriks sebanding.
    samples = {j: man[(man.fold == j) & (man.superclass == args.cls)].sample(args.n, random_state=0)
               for j in range(5)}
    target = 1 if args.cls == "scc" else 0

    rows = []
    for i in range(5):
        models = []
        for tag in args.clf.split("+"):
            m = fs.timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
            m.load_state_dict(fs.load_file(ROOT / "outputs" / "filter_clf" / f"fold{i}_{tag}" / "best.safetensors"))
            models.append(m.to(device).eval())
        for j in range(5):
            p_scc, _ = fs.embed(models, samples[j].path, device, tta=True)
            p_int = p_scc if target == 1 else 1 - p_scc
            rows.append(dict(clf_fold=i, syn_fold=j, mean_p=p_int.mean(), pct_correct=(p_int >= 0.5).mean() * 100))
            print(f"classifier fold {i} x sintetis {args.cls.upper()} fold {j}: "
                  f"P(benar) rata-rata {p_int.mean():.2f}, diprediksi benar {(p_int >= 0.5).mean() * 100:.0f}%")
        del models
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / "filter" / f"cross_check_{args.cls}.csv", index=False)
    print(f"\n% diprediksi {args.cls.upper()} (baris = classifier fold, kolom = sintetis fold):")
    print(df.pivot(index="clf_fold", columns="syn_fold", values="pct_correct").round(0).to_string())


if __name__ == "__main__":
    main()
