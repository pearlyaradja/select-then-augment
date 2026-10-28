"""
Tahap 4c: bandingkan versi classifier filter di patch VAL asli (5 fold).

Aturan pemilihan (ditetapkan SEBELUM melihat hasil v3):
  versi baru dipakai hanya jika rata-rata val AUC 5 fold > baseline (0.863);
  val macro-F1 sebagai pembanding kedua.
Gambar sintetis TIDAK dipakai untuk memilih classifier (melingkar), dan patch
TEST tidak disentuh.

Opsi:
  --tta       : nilai tiap patch dalam 8 orientasi (flip/rotasi 90), rata-ratakan
                probabilitas -> penilaian lebih stabil, tanpa training tambahan
  --versions  : daftar versi; "" = model utama, "v3" = outputs/filter_clf/fold{k}_v3,
                "v3+v3s1+v3s2" = ensemble (rata-rata probabilitas beberapa model)

Contoh:
  python src/04c_eval_filter_clf.py --versions "" v3
  python src/04c_eval_filter_clf.py --versions "" v3 --tta
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from safetensors.torch import load_file
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
PATCH_SPLITS_CSV = ROOT / "outputs" / "patch_splits.csv"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ImgDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        return (x - MEAN) / STD


def dihedral_views(x):
    """8 orientasi: rotasi 0/90/180/270, masing-masing dengan & tanpa flip."""
    views = []
    for flip in (False, True):
        xf = torch.flip(x, dims=[3]) if flip else x
        for k in range(4):
            views.append(torch.rot90(xf, k=k, dims=[2, 3]))
    return views


@torch.no_grad()
def predict(model, paths, tta, bs=16):
    out = []
    for x in DataLoader(ImgDataset(paths), batch_size=bs, num_workers=2):
        x = x.cuda()
        views = dihedral_views(x) if tta else [x]
        ps = []
        for v in views:
            with torch.autocast("cuda", dtype=torch.float16):
                ps.append(torch.softmax(model(v).float(), 1)[:, 1])
        out.append(torch.stack(ps).mean(0).cpu())
    return torch.cat(out).numpy()


def load(fold, tag):
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    d = ROOT / "outputs" / "filter_clf" / (f"fold{fold}" + (f"_{tag}" if tag else ""))
    m.load_state_dict(load_file(d / "best.safetensors"))
    return m.cuda().eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--versions", nargs="+", default=["", "v3"])
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()

    ps = pd.read_csv(PATCH_SPLITS_CSV, keep_default_na=False)
    rows = []
    for ver in args.versions:
        tags = ver.split("+")
        for fold in range(5):
            va = ps[(ps["fold"] == fold) & (ps["split"] == "val")]
            y = va["label"].values
            probs = []
            for t in tags:
                m = load(fold, t)
                probs.append(predict(m, va["patch_path"], args.tta))
                del m
                torch.cuda.empty_cache()
            pr = np.mean(probs, axis=0)
            pred = (pr >= 0.5).astype(int)
            rows.append({"versi": ver or "baseline", "fold": fold,
                         "AUC": roc_auc_score(y, pr), "F1": f1_score(y, pred, average="macro"),
                         "recall_ADC": float((pred[y == 0] == 0).mean()),
                         "recall_SCC": float((pred[y == 1] == 1).mean())})
    df = pd.DataFrame(rows)
    print(f"TTA: {'ya (8 orientasi)' if args.tta else 'tidak'}\n")
    print(df.pivot(index="fold", columns="versi", values=["AUC", "F1"]).round(3).to_string())
    print("\nRata-rata 5 fold:")
    print(df.groupby("versi")[["AUC", "F1", "recall_ADC", "recall_SCC"]].mean().round(3).to_string())
    out = ROOT / "outputs" / "filter_clf" / f"compare{'_tta' if args.tta else ''}.csv"
    df.to_csv(out, index=False)
    print(f"\nTersimpan: {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
