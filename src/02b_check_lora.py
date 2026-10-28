"""
Tahap 2 (verifikasi): CEK VISUAL hasil LoRA per kelas. BUKAN generation untuk eksperimen.

Tujuan:
  (1) LoRA menghasilkan gambar mirip H&E paru.
  (2) LoRA ADC dan LoRA SCC menghasilkan gambar BERBEDA (masalah LoRA gabungan:
      prompt kelas diabaikan).
  (3) Gambar sintetis BERVARIASI antar seed (bukan satu pola saja).
  (4) Gambar sintetis TIDAK menjiplak patch training (cek memorization awal).
Gambar di sini TIDAK dipakai di tahap mana pun (filtering/classifier).
Cek memorization resmi tetap di Tahap 4c; ini hanya pemeriksaan dini.

Output:
  outputs/lora_check/fold{k}_class_grid.png : ASLI vs LoRA ADC/SCC per checkpoint
  outputs/lora_check/fold{k}_nn_grid.png    : gambar sintetis (checkpoint terakhir)
                                              vs patch training paling mirip

Contoh:
  python src/02b_check_lora.py --fold 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sd-v1-5"
PATCH_SPLITS_CSV = ROOT / "outputs" / "patch_splits.csv"
CLASS_NAMES = {"aca": "adenocarcinoma", "scc": "squamous cell carcinoma"}
CLASS_LABEL = {"aca": "ADC", "scc": "SCC"}
MAGS = ["20x", "40x"]
THUMB = 256
NN_SIZE = 64  # resolusi untuk pencarian kembaran terdekat (cukup untuk tata letak jaringan)


def make_prompt(superclass, magnification):
    # HARUS identik dengan make_prompt di 02_train_lora.py
    return (f"H&E stained lung histopathology, {CLASS_NAMES[superclass]}, "
            f"{magnification} magnification")


def build_pipeline(device):
    # Dirakit manual dari komponen lokal: model_index.json menyebut
    # safety_checker/feature_extractor yang foldernya tidak ada, dan
    # from_pretrained bisa mencoba mengunduhnya dari Hub (model harus selalu dimuat dari folder lokal).
    kw = dict(local_files_only=True)
    pipe = StableDiffusionPipeline(
        vae=AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae",
                                          torch_dtype=torch.float16, **kw),
        text_encoder=CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder",
                                                   torch_dtype=torch.float16, **kw),
        tokenizer=CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer", **kw),
        unet=UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet",
                                                  torch_dtype=torch.float16, **kw),
        scheduler=DPMSolverMultistepScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler",
                                                              **kw),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def generate_row(pipe, superclass, n, seed, steps, cfg):
    imgs = []
    for m, mag in enumerate(MAGS):
        for j in range(n):
            # Seed unik per kolom (8 seed berbeda), tapi SAMA untuk ADC & SCC
            # di kolom yang sama -> perbandingan antar kelas tetap adil.
            g = torch.Generator("cuda").manual_seed(seed + m * n + j)
            imgs.append(pipe(make_prompt(superclass, mag), num_inference_steps=steps,
                             guidance_scale=cfg, generator=g).images[0])
    return imgs


def mean_abs_diff(a_imgs, b_imgs):
    return float(np.mean([np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean()
                          for a, b in zip(a_imgs, b_imgs)]))


# ------------------------------------------------------ kembaran terdekat (NN)
def dihedral(arr, v):
    """8 varian flip/rotasi; sama dengan augmentasi saat training LoRA."""
    out = np.rot90(arr, k=v % 4, axes=(0, 1))
    return np.flip(out, axis=1) if v >= 4 else out


def feat(img):
    """Korelasi piksel: dikurangi rata-rata & dinormalisasi -> tahan beda
    kecerahan, tapi sensitif ke tata letak sel. Tanpa model pretrained."""
    a = np.asarray(img.convert("RGB").resize((NN_SIZE, NN_SIZE), Image.BILINEAR),
                   np.float32).reshape(-1)
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-8)


def build_train_bank(paths):
    """Fitur semua patch training x 8 varian dihedral."""
    feats = []
    for p in paths:
        small = Image.open(ROOT / p).convert("RGB").resize((NN_SIZE, NN_SIZE), Image.BILINEAR)
        arr = np.asarray(small)
        for v in range(8):
            feats.append(feat(Image.fromarray(np.ascontiguousarray(dihedral(arr, v)))))
    return np.stack(feats)  # (n_patch*8, D)


def nearest(bank, imgs):
    q = np.stack([feat(im) for im in imgs])
    sims = q @ bank.T
    idx = sims.argmax(axis=1)
    return sims.max(axis=1), idx // 8, idx % 8


def pairwise_mean(imgs):
    f = np.stack([feat(im) for im in imgs])
    s = f @ f.T
    return float(s[~np.eye(len(f), dtype=bool)].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n_per_mag", type=int, default=4, help="4 x 2 magnification = 8 seed")
    p.add_argument("--checkpoints", nargs="+",
                   default=["checkpoint-250", "checkpoint-500", "checkpoint-1000"])
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    device = torch.device("cuda")
    out_dir = ROOT / "outputs" / "lora_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = pd.read_csv(PATCH_SPLITS_CSV, keep_default_na=False)
    fold = ps[ps["fold"] == args.fold]

    rows = []  # (label, [PIL images])
    real, train_paths, val_paths = {}, {}, {}
    for sup in CLASS_NAMES:
        tr = fold[(fold["split"] == "train") & (fold["superclass"] == sup)]
        train_paths[sup] = tr["patch_path"].tolist()
        val_paths[sup] = fold[(fold["split"] == "val") &
                              (fold["superclass"] == sup)]["patch_path"].tolist()
        imgs = []
        for mag in MAGS:
            sub = tr[tr["magnification"] == mag]
            for path in sub.sample(args.n_per_mag, random_state=args.seed)["patch_path"]:
                imgs.append(Image.open(ROOT / path).convert("RGB"))
        real[sup] = imgs
        rows.append((f"ASLI {CLASS_LABEL[sup]}", imgs))

    # ------------------------------------------------------------ generate
    pipe = build_pipeline(device)
    gen_all = {}  # (ck, sup) -> imgs
    for ck in args.checkpoints:
        for sup in CLASS_NAMES:
            ck_dir = ROOT / "outputs" / "lora" / f"fold{args.fold}_{sup}" / ck
            assert (ck_dir / "pytorch_lora_weights.safetensors").exists(), f"{ck_dir} tidak ada"
            pipe.unload_lora_weights()  # pastikan LoRA sebelumnya tidak menumpuk
            pipe.load_lora_weights(ck_dir)
            print(f"Generate: LoRA {CLASS_LABEL[sup]} {ck} ...")
            gen_all[(ck, sup)] = generate_row(pipe, sup, args.n_per_mag, args.seed,
                                              args.steps, args.cfg)
            rows.append((f"LoRA {CLASS_LABEL[sup]}\n{ck.replace('checkpoint-', 'step ')}",
                         gen_all[(ck, sup)]))
    del pipe
    torch.cuda.empty_cache()

    # ------------------------------------------- (2) beda ADC vs SCC
    print("\n=== (2) Apakah LoRA ADC dan LoRA SCC menghasilkan gambar berbeda? ===")
    print("Selisih piksel rata-rata (0-255) pada seed yang sama. ~0 = identik (masalah lama).")
    print(f"  Referensi ASLI ADC vs ASLI SCC (gambar berbeda): "
          f"{mean_abs_diff(real['aca'], real['scc']):.1f}")
    for ck in args.checkpoints:
        print(f"  {ck:16s} LoRA ADC vs LoRA SCC: "
              f"{mean_abs_diff(gen_all[(ck, 'aca')], gen_all[(ck, 'scc')]):.1f}")

    # ------------------------------------------- (3) variasi antar seed
    print("\n=== (3) Variasi antar seed (korelasi rata-rata antar gambar; 1 = identik) ===")
    for sup in CLASS_NAMES:
        print(f"  {CLASS_LABEL[sup]} ASLI: {pairwise_mean(real[sup]):.3f}  |  " +
              "  ".join(f"{ck.replace('checkpoint-', 'step ')}: "
                        f"{pairwise_mean(gen_all[(ck, sup)]):.3f}" for ck in args.checkpoints))

    # ------------------------------------------- (4) kembaran terdekat
    print("\n=== (4) Cek menjiplak: korelasi ke patch TRAINING paling mirip "
          "(termasuk flip/rotasi) ===")
    print("Pembanding = patch VAL asli (tidak pernah dilihat LoRA, jelas bukan jiplakan).")
    last = args.checkpoints[-1]
    nn_rows = []
    for sup in CLASS_NAMES:
        bank = build_train_bank(train_paths[sup])
        val_imgs = [Image.open(ROOT / q).convert("RGB") for q in val_paths[sup]]
        val_sim, _, _ = nearest(bank, val_imgs)
        print(f"  {CLASS_LABEL[sup]}: {len(train_paths[sup])} patch train x 8 varian | "
              f"VAL asli ({len(val_imgs)} patch): median={np.median(val_sim):.3f}, "
              f"p95={np.percentile(val_sim, 95):.3f}, max={val_sim.max():.3f}")
        for ck in args.checkpoints:
            s, _, _ = nearest(bank, gen_all[(ck, sup)])
            n_above = int((s > val_sim.max()).sum())
            print(f"     {ck:16s} sintetis: median={np.median(s):.3f}, max={s.max():.3f}, "
                  f"di atas max VAL: {n_above}/{len(s)}")
        s, pidx, var = nearest(bank, gen_all[(last, sup)])
        nn_imgs = []
        for i, v in zip(pidx, var):
            arr = np.asarray(Image.open(ROOT / train_paths[sup][i]).convert("RGB"))
            nn_imgs.append(Image.fromarray(np.ascontiguousarray(dihedral(arr, v))))
        nn_rows.append((f"Sintetis {CLASS_LABEL[sup]}\n{last.replace('checkpoint-', 'step ')}",
                        gen_all[(last, sup)], None))
        nn_rows.append((f"Train paling\nmirip ({CLASS_LABEL[sup]})", nn_imgs, s))

    # ---------------------------------------------------------- simpan grid
    def save_grid(grid_rows, path, scores_row=False):
        label_w, head_h, gap = 130, 40, 4
        n_cols = len(MAGS) * args.n_per_mag
        W = label_w + n_cols * (THUMB + gap)
        H = head_h + len(grid_rows) * (THUMB + gap)
        grid = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(grid)
        for m, mag in enumerate(MAGS):
            d.text((label_w + m * args.n_per_mag * (THUMB + gap) + 5, 12), mag, fill="black")
        for r, row in enumerate(grid_rows):
            label, imgs = row[0], row[1]
            scores = row[2] if len(row) > 2 else None
            y = head_h + r * (THUMB + gap)
            d.multiline_text((8, y + THUMB // 2 - 10), label, fill="black")
            for c, im in enumerate(imgs):
                x = label_w + c * (THUMB + gap)
                grid.paste(im.resize((THUMB, THUMB)), (x, y))
                if scores is not None:
                    d.rectangle([x, y, x + 92, y + 16], fill="white")
                    d.text((x + 3, y + 2), f"korelasi {scores[c]:.3f}", fill="black")
        grid.save(path)
        print(f"Tersimpan: {path.relative_to(ROOT).as_posix()}")

    print()
    save_grid(rows, out_dir / f"fold{args.fold}_class_grid.png")
    save_grid(nn_rows, out_dir / f"fold{args.fold}_nn_grid.png")


if __name__ == "__main__":
    main()
