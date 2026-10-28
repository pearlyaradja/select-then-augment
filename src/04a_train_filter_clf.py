"""
Tahap 4a: TRAINING classifier untuk filter class consistency (dan sumber fitur
untuk filter realism & memorization), satu model per fold.

Ini TRAINING sungguhan (forward -> loss -> backward -> optimizer.step).
EfficientNet-B0 diinisialisasi dari bobot ImageNet lokal
(models/efficientnet_b0_ra_in1k.safetensors), lalu SELURUH bobotnya di-fine-tune
dengan patch ASLI fold ini. Classifier ini KHUSUS untuk filtering; classifier
Tahap 5 adalah model baru yang terpisah.

Data   : outputs/patch_splits.csv, fold k
         train = patch asli split "train"  (untuk belajar)
         val   = patch asli split "val"    (untuk memilih epoch terbaik)
         test  = TIDAK dipakai sama sekali
Output : outputs/filter_clf/fold{k}/best.safetensors, train_log.csv, train_config.json

Contoh:
  # uji coba kecil (WAJIB dulu)
  python src/04a_train_filter_clf.py --fold 0 --subset 32 --epochs 1 --tag smoke
  # penuh
  python src/04a_train_filter_clf.py --fold 0
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
PATCH_SPLITS_CSV = ROOT / "outputs" / "patch_splits.csv"
IMAGENET_WEIGHTS = ROOT / "models" / "efficientnet_b0_ra_in1k.safetensors"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Matriks stain H&E (Ruifrok & Johnston 2001), sama dengan skimage.color.
RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29],
                             [0.07, 0.99, 0.11],
                             [0.27, 0.57, 0.78]])
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)
LOG_ADJUST = float(np.log(1e-6))


def hed_jitter(x, alpha_sigma, beta_sigma):
    """Augmentasi stain H&E (Tellez dkk. 2018): ubah intensitas Hematoxylin,
    Eosin, DAB secara acak. Tujuan: classifier belajar MORFOLOGI, bukan warna
    pewarnaan (SCC asli & sintetis cenderung lebih pink dari ADC).
    x: tensor (3,H,W) di [0,1]."""
    rgb = x.permute(1, 2, 0).clamp(min=1e-6)
    stains = (torch.log(rgb) / LOG_ADJUST) @ HED_FROM_RGB
    # Sengaja TIDAK di-clamp ke >=0 (beda dengan skimage.rgb2hed): clamp
    # membuang informasi warna sehingga konversi balik tidak kembali ke gambar
    # asli (diuji: selisih maks 0.60 dengan clamp, ~0 tanpa clamp).
    # Kekuatan dipilih visual (outputs/preview_hed_aug.png): beta 0.05 membuat
    # warna kuning/cyan/putih tak realistis; alpha 0.05 + beta 0.01 memberi
    # variasi pink-ungu & terang-gelap yang masih wajar untuk H&E.
    alpha = torch.empty(3).uniform_(1 - alpha_sigma, 1 + alpha_sigma)
    beta = torch.empty(3).uniform_(-beta_sigma, beta_sigma)
    stains = stains * alpha + beta
    out = torch.exp(-(stains * -LOG_ADJUST) @ RGB_FROM_HED)
    return out.clamp(0, 1).permute(2, 0, 1)


# Augmentasi "strong" (--aug strong), ditambahkan setelah diagnosis Tahap 4:
# classifier baseline overfit dan menilai gambar sintetis tidak stabil.
# - RandomResizedCrop (zoom 60-100%): variasi skala, melawan hafalan tata letak
# - GaussianBlur: gambar sintetis lebih halus dari asli; classifier dibuat tidak
#   bergantung pada tekstur halus/ketajaman, tapi pada bentuk & susunan sel
# - ColorJitter kecerahan/kontras: variasi pencahayaan mikroskop
STRONG_AUG = v2.Compose([
    v2.RandomResizedCrop(512, scale=(0.6, 1.0), ratio=(1.0, 1.0), antialias=True),
    v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.5),
    v2.ColorJitter(brightness=0.15, contrast=0.15),
])


class PatchDataset(Dataset):
    def __init__(self, df, img_size, train, hed, strong=False):
        self.paths = df["patch_path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.img_size, self.train, self.hed = img_size, train, hed  # hed = (alpha, beta)
        self.strong = strong

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        if self.train:
            # Dihedral: jaringan tidak punya orientasi "atas".
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1])
            x = torch.rot90(x, k=random.randint(0, 3), dims=[1, 2])
            x = hed_jitter(x, *self.hed)
            if self.strong:
                x = STRONG_AUG(x)
        return (x - MEAN) / STD, self.labels[i]


def build_model(drop_path=0.0):
    # drop_path (stochastic depth): regularisasi tambahan melawan overfitting;
    # data kita kecil (~2.200 patch, 13-16 pasien per kelas).
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2, drop_rate=0.2,
                              drop_path_rate=drop_path)
    sd = load_file(IMAGENET_WEIGHTS)
    # Head ImageNet (1000 kelas) dibuang; head baru 2 kelas diinisialisasi acak.
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
    res = model.load_state_dict(sd, strict=False)
    assert set(res.missing_keys) == {"classifier.weight", "classifier.bias"}, res.missing_keys
    assert not res.unexpected_keys, res.unexpected_keys
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        labels.append(y.cpu())
    p, t = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    pred = (p >= 0.5).astype(int)
    return {"loss": loss_sum / len(t), "acc": float((pred == t).mean()),
            "macro_f1": float(f1_score(t, pred, average="macro")),
            "auc": float(roc_auc_score(t, p)) if len(set(t)) == 2 else float("nan")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--aug", choices=["base", "strong"], default="base")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--drop_path", type=float, default=0.0)
    p.add_argument("--hed_alpha", type=float, default=0.05)
    p.add_argument("--hed_beta", type=float, default=0.01)
    p.add_argument("--subset", type=int, default=0, help=">0: N patch train & val saja (uji coba)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    run = f"fold{args.fold}" + (f"_{args.tag}" if args.tag else "")
    out_dir = ROOT / "outputs" / "filter_clf" / run
    if (out_dir / "best.safetensors").exists():
        raise SystemExit(f"[ERROR] {out_dir.relative_to(ROOT)} sudah berisi model. Pindahkan dulu.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    ps = pd.read_csv(PATCH_SPLITS_CSV, keep_default_na=False)
    f = ps[ps["fold"] == args.fold]
    tr, va = f[f["split"] == "train"], f[f["split"] == "val"]
    te_p = set(f.loc[f["split"] == "test", "patient_id"])
    tr_p, va_p = set(tr["patient_id"]), set(va["patient_id"])
    # Tidak boleh ada pasien yang sama di train/val/test.
    print(f"[CHECK] fold {args.fold}: overlap train-val={sorted(tr_p & va_p)} "
          f"train-test={sorted(tr_p & te_p)} val-test={sorted(va_p & te_p)}")
    assert not (tr_p & va_p or tr_p & te_p or va_p & te_p), "patient_id bocor antar split"
    if args.subset:
        tr = tr.groupby("label").sample(args.subset // 2, random_state=args.seed)
        va = va.groupby("label").sample(args.subset // 2, random_state=args.seed)
    print(f"Train: {len(tr)} patch {tr['superclass'].value_counts().to_dict()} | "
          f"Val: {len(va)} patch {va['superclass'].value_counts().to_dict()}")

    hed = (args.hed_alpha, args.hed_beta)
    tr_loader = DataLoader(PatchDataset(tr, args.img_size, True, hed, args.aug == "strong"),
                           batch_size=args.batch_size, shuffle=True, drop_last=True,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=args.num_workers > 0)
    va_loader = DataLoader(PatchDataset(va, args.img_size, False, hed),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=args.num_workers > 0)

    # ------------------------------------------------------------ model & optim
    model = build_model(args.drop_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(tr_loader)
    warm = len(tr_loader)  # 1 epoch warmup: head baru acak, jangan rusak backbone di awal
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda")

    with open(out_dir / "train_config.json", "w") as fh:
        json.dump({**vars(args), "n_train": len(tr), "n_val": len(va),
                   "train_patient_ids": sorted(tr_p), "val_patient_ids": sorted(va_p),
                   "init": IMAGENET_WEIGHTS.relative_to(ROOT).as_posix()}, fh, indent=2)
    log_f = open(out_dir / "train_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "val_auc",
                  "lr", "sec", "peak_vram_gb"])

    best = (-1.0, -1.0)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            # ===== LOSS: cross-entropy antara prediksi dan label kelas asli =====
            # label_smoothing>0: target tidak 100% yakin -> mengurangi overconfidence.
            loss = F.cross_entropy(logits.float(), y, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            # ===== BACKWARD =====
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            # ===== OPTIMIZER STEP: update SELURUH bobot EfficientNet-B0 =====
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)

        m = evaluate(model, va_loader, device)
        sec = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 2**30
        log.writerow([epoch, f"{run_loss / n:.4f}", f"{m['loss']:.4f}", f"{m['acc']:.4f}",
                      f"{m['macro_f1']:.4f}", f"{m['auc']:.4f}",
                      f"{sched.get_last_lr()[0]:.2e}", f"{sec:.0f}", f"{vram:.2f}"])
        log_f.flush()
        # Pilih epoch terbaik berdasar macro-F1 VAL (bukan test); seri -> AUC.
        score = (m["macro_f1"], m["auc"])
        mark = ""
        if score > best:
            best = score
            save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                      out_dir / "best.safetensors")
            mark = "  <- terbaik, disimpan"
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss={run_loss / n:.4f}  "
              f"val_loss={m['loss']:.4f}  val_acc={m['acc']:.3f}  val_F1={m['macro_f1']:.3f}  "
              f"val_AUC={m['auc']:.3f}  {sec:.0f}s  VRAM={vram:.2f}GB{mark}")
    log_f.close()
    print(f"\nSelesai. Val macro-F1 terbaik = {best[0]:.3f} (AUC {best[1]:.3f}). "
          f"Model: {(out_dir / 'best.safetensors').relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
