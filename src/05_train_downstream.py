"""
Tahap 5: TRAINING downstream classifier ADC vs SCC untuk kondisi C0-C4.

Ini TRAINING sungguhan (forward -> loss -> backward -> optimizer.step).
EfficientNet-B0 BARU per (fold, kondisi, seed): init bobot ImageNet lokal, TIDAK
mewarisi apa pun dari classifier filter Tahap 4 (keputusan 2026-10-19).

Data fold k:
  train = patch ASLI split "train"  +  sintetis terpilih kondisi itu (C1-C4)
  val   = patch ASLI split "val"    -> memilih model terbaik (sintetis tidak pernah masuk)
  test  = patch ASLI split "test"   -> dievaluasi SEKALI di akhir, sama untuk C0-C4
Sintetis: outputs/selection/selection_N{N}.csv (Tahap 4e), hanya baris fold k.
  C0 = tanpa sintetis | C1 acak | C2 realism | C3 kelas+jiplakan | C4 gabungan

KENAPA jumlah STEP tetap (bukan epoch): C1-C4 punya ~18% data lebih banyak dari C0.
Dengan epoch tetap, C1-C4 mendapat lebih banyak update bobot -> perbedaan hasil bisa
karena "latihan lebih lama", bukan karena gambar sintetisnya. Step tetap menyamakan itu.

Output: outputs/downstream/fold{k}_{C}_s{seed}/
  best.safetensors, train_log.csv, train_config.json,
  test_predictions.csv (per patch: patient_id, label, prob_scc) -> bahan Tahap 6,
  test_metrics.json (penanda run selesai)

Contoh:
  # uji coba kecil (WAJIB dulu)
  python src/05_train_downstream.py --fold 0 --cond C4 --seed 0 --subset 32 --steps 4 --eval_every 2 --tag smoke
  # penuh
  python src/05_train_downstream.py --fold 0 --cond C4 --seed 0
"""
import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
# Pakai ulang dataset, augmentasi (dihedral + stain HED + strong) dan build_model dari
# 04a yang sudah diverifikasi, supaya pra-proses identik dan tidak ada salinan kode.
_spec = importlib.util.spec_from_file_location("clf4a", ROOT / "src" / "04a_train_filter_clf.py")
clf4a = importlib.util.module_from_spec(_spec)
sys.modules["clf4a"] = clf4a  # perlu untuk worker DataLoader di Windows (spawn)
_spec.loader.exec_module(clf4a)

CONDS = ["C0", "C1", "C2", "C3", "C4"]


def infinite(loader):
    """Loader tanpa batas: training dihitung dalam STEP, bukan epoch."""
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            probs.append(torch.softmax(model(x.to(device)).float(), 1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    rec = recall_score(y, pred, average=None, labels=[0, 1])
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
            "recall_ADC": float(rec[0]), "recall_SCC": float(rec[1]),
            "acc": float((pred == y).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--cond", choices=CONDS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_syn", type=int, default=200, help="N per kelas dari seleksi top-N (Tahap 4e)")
    # 2000 step x batch 16 ~ 14,5 epoch data real C0 (~2.200 patch): setara 15 epoch classifier filter.
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=140, help="~1 epoch data real")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=140)
    ap.add_argument("--subset", type=int, default=0, help=">0: N patch real train/val/test saja (uji coba)")
    ap.add_argument("--num_workers", type=int, default=6, help="worker data TRAIN saja")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    run = f"fold{args.fold}_{args.cond}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out_dir = ROOT / "outputs" / "downstream" / run
    if (out_dir / "test_metrics.json").exists():
        raise SystemExit(f"[SKIP] {out_dir.relative_to(ROOT)} sudah selesai.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv", keep_default_na=False)
    f = ps[ps.fold == args.fold]
    tr, va, te = (f[f.split == s] for s in ("train", "val", "test"))
    tr_p, va_p, te_p = set(tr.patient_id), set(va.patient_id), set(te.patient_id)
    # Tidak boleh ada pasien yang sama di train/val/test.
    print(f"[CHECK] fold {args.fold}: overlap pasien train-val={sorted(tr_p & va_p)} "
          f"train-test={sorted(tr_p & te_p)} val-test={sorted(va_p & te_p)}")
    assert not (tr_p & va_p or tr_p & te_p or va_p & te_p), "patient_id bocor antar split"

    syn = pd.DataFrame(columns=["patch_path", "label"])
    if args.cond != "C0":
        sel = pd.read_csv(ROOT / "outputs" / "selection" / f"selection_N{args.n_syn}.csv")
        s = sel[sel[args.cond]]
        # Sintetis fold lain dibuat LoRA yang dilatih dari pasien test fold ini -> bocor.
        # Jadi hanya fold ini, dan dicek ulang lewat path file (bukan hanya kolom fold).
        s = s[s.fold == args.fold]
        cnt = s.groupby("superclass").size().to_dict()
        assert cnt == {"aca": args.n_syn, "scc": args.n_syn}, f"jumlah sintetis salah: {cnt}"
        assert s.path.str.contains(f"/fold{args.fold}/").all(), "ada sintetis dari fold lain"
        syn = s.rename(columns={"path": "patch_path"})[["patch_path", "label"]]
    print(f"[CHECK] {args.cond}: sintetis {len(syn)} "
          f"(ADC {int((syn.label == 0).sum())}, SCC {int((syn.label == 1).sum())}), semua dari fold {args.fold}")

    if args.subset:
        tr, va, te = (d.groupby("label").sample(args.subset // 2, random_state=args.seed) for d in (tr, va, te))
        syn = syn.groupby("label").sample(min(len(syn) // 2, args.subset // 2), random_state=args.seed) \
            if len(syn) else syn
    train_df = pd.concat([tr[["patch_path", "label"]], syn], ignore_index=True)
    print(f"Train: {len(tr)} real + {len(syn)} sintetis = {len(train_df)} | Val: {len(va)} | Test: {len(te)}")

    hed = (0.05, 0.01)  # sama dengan classifier filter (dipilih visual, Tahap 4a)
    # Worker HANYA untuk train (augmentasi strong berat di CPU). Val/test dibaca di proses
    # utama: setiap worker di Windows memakan ~0,5 GB commit memory, dan 3 loader x 4 worker
    # sempat membuat pembuatan worker macet (commit habis, uji 2026-10-21).
    tr_loader = DataLoader(clf4a.PatchDataset(train_df, 512, True, hed, strong=True),
                           batch_size=args.batch_size, shuffle=True, drop_last=True,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=args.num_workers > 0)
    va_loader, te_loader = (DataLoader(clf4a.PatchDataset(d, 512, False, hed), batch_size=args.batch_size,
                                       shuffle=False, num_workers=0) for d in (va, te))

    # ------------------------------------------------------------ model & optim
    model = clf4a.build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Warmup lalu cosine, dihitung per STEP (sama untuk semua kondisi).
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / args.warmup if s < args.warmup
        else 0.5 * (1 + np.cos(np.pi * (s - args.warmup) / max(1, args.steps - args.warmup))))
    scaler = torch.amp.GradScaler("cuda")

    with open(out_dir / "train_config.json", "w") as fh:
        json.dump({**vars(args), "n_real_train": len(tr), "n_syn": len(syn), "n_val": len(va),
                   "n_test": len(te), "aug": "dihedral+HED+strong", "init": "ImageNet (models/)",
                   "train_patient_ids": sorted(tr_p), "val_patient_ids": sorted(va_p),
                   "test_patient_ids": sorted(te_p)}, fh, indent=2)
    log_f = open(out_dir / "train_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "train_loss", "val_macro_f1", "val_auc", "lr", "sec", "peak_vram_gb"])

    best, t0, run_loss, n = (-1.0, -1.0), time.time(), 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    batches = infinite(tr_loader)
    model.train()
    for step in range(1, args.steps + 1):
        x, y = next(batches)
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)                                   # ===== FORWARD =====
        loss = F.cross_entropy(logits.float(), y)              # ===== LOSS: cross-entropy vs label =====
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()                          # ===== BACKWARD =====
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)                                 # ===== OPTIMIZER STEP: update seluruh bobot =====
        scaler.update()
        sched.step()
        run_loss += loss.item() * len(y)
        n += len(y)

        if step % args.eval_every == 0 or step == args.steps:
            m = metrics(va.label.values.astype(int), predict(model, va_loader, device))
            model.train()
            vram = torch.cuda.max_memory_allocated() / 2**30
            log.writerow([step, f"{run_loss / n:.4f}", f"{m['macro_f1']:.4f}", f"{m['auc']:.4f}",
                          f"{sched.get_last_lr()[0]:.2e}", f"{time.time() - t0:.0f}", f"{vram:.2f}"])
            log_f.flush()
            # Model terbaik dipilih dari VAL asli (bukan test); seri -> AUC.
            mark = ""
            if (m["macro_f1"], m["auc"]) > best:
                best = (m["macro_f1"], m["auc"])
                save_file({k: v.contiguous() for k, v in model.state_dict().items()}, out_dir / "best.safetensors")
                mark = "  <- terbaik"
            print(f"step {step:4d}/{args.steps}  train_loss={run_loss / n:.4f}  val_F1={m['macro_f1']:.3f}  "
                  f"val_AUC={m['auc']:.3f}  {time.time() - t0:.0f}s  VRAM={vram:.2f}GB{mark}", flush=True)
            run_loss, n = 0.0, 0
    log_f.close()
    del batches, tr_loader  # matikan worker train sebelum test -> bebaskan memori

    # ----------------------------------------------------- test (sekali, model terbaik val)
    model.load_state_dict(load_file(out_dir / "best.safetensors"))
    p_te = predict(model, te_loader, device)
    pd.DataFrame({"patch_path": te.patch_path.values, "patient_id": te.patient_id.values,
                  "label": te.label.values.astype(int), "prob_scc": p_te}).to_csv(
        out_dir / "test_predictions.csv", index=False)
    mt = metrics(te.label.values.astype(int), p_te)
    with open(out_dir / "test_metrics.json", "w") as fh:
        json.dump({"fold": args.fold, "cond": args.cond, "seed": args.seed,
                   "val_best_macro_f1": best[0], "val_best_auc": best[1], **{f"test_{k}": v for k, v in mt.items()}},
                  fh, indent=2)
    print(f"\nSelesai {run}: val F1 terbaik {best[0]:.3f} | TEST F1 {mt['macro_f1']:.3f}  "
          f"AUC {mt['auc']:.3f}  recall ADC {mt['recall_ADC']:.3f}  SCC {mt['recall_SCC']:.3f}  "
          f"({(time.time() - t0) / 60:.1f} menit)")


if __name__ == "__main__":
    main()
