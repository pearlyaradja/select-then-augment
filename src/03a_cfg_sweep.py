"""
Tahap 3a: pilih guidance scale (CFG) untuk generation. BUKAN generation eksperimen.

Masalah dari cek LoRA: dengan CFG 7.5 gambar sintetis lebih jenuh/magenta
daripada data asli. CFG tinggi dikenal memperkuat kontras & saturasi, jadi
beberapa nilai dicoba dan dibandingkan dengan statistik warna patch TRAIN asli
(bukan val/test, supaya data evaluasi tidak dipakai untuk menyetel apa pun).

Metrik per gambar: rata-rata R, G, B, dan saturasi HSV. Untuk tiap CFG dihitung
standardized mean difference (SMD) sintetis vs asli per metrik:
    SMD = |mean_sintetis - mean_asli| / std_asli
lalu dirata-rata. Makin kecil = warna makin mirip data asli.
Seed sama di semua CFG -> beda antar baris murni karena CFG.

Output: outputs/cfg_sweep/fold{k}_{aca|scc}_grid.png, outputs/cfg_sweep/fold{k}_summary.csv

Contoh:
  python src/03a_cfg_sweep.py --fold 0
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
METRICS = ["R", "G", "B", "S"]
THUMB = 224


def make_prompt(superclass, magnification):
    # HARUS identik dengan make_prompt di 02_train_lora.py
    return (f"H&E stained lung histopathology, {CLASS_NAMES[superclass]}, "
            f"{magnification} magnification")


def build_pipeline(device):
    # Dirakit manual dari komponen lokal (lihat 02b_check_lora.py): mencegah
    # diffusers mencoba mengunduh safety_checker dari Hub (model harus selalu dimuat dari folder lokal).
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


def color_stats(img):
    arr = np.asarray(img.convert("RGB"), np.float32)
    sat = np.asarray(img.convert("HSV"), np.float32)[:, :, 1]
    return {"R": arr[:, :, 0].mean(), "G": arr[:, :, 1].mean(),
            "B": arr[:, :, 2].mean(), "S": sat.mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--cfgs", type=float, nargs="+", default=[2.0, 3.0, 4.5, 6.0, 7.5])
    p.add_argument("--n", type=int, default=16, help="gambar per kelas per CFG (genap)")
    p.add_argument("--n_real", type=int, default=400, help="sampel patch train asli per kelas")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=5000)
    args = p.parse_args()

    device = torch.device("cuda")
    out_dir = ROOT / "outputs" / "cfg_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = pd.read_csv(PATCH_SPLITS_CSV, keep_default_na=False)
    tr = ps[(ps["fold"] == args.fold) & (ps["split"] == "train")]

    pipe = build_pipeline(device)
    rows = []
    for sup in CLASS_NAMES:
        real_df = tr[tr["superclass"] == sup].sample(args.n_real, random_state=args.seed)
        real_imgs = [Image.open(ROOT / q).convert("RGB") for q in real_df["patch_path"]]
        real_stats = pd.DataFrame([color_stats(im) for im in real_imgs])
        mu_r, sd_r = real_stats.mean(), real_stats.std()
        rows.append({"kelas": CLASS_LABEL[sup], "cfg": "ASLI", **mu_r.round(1).to_dict(),
                     "SMD_rata": 0.0})

        pipe.unload_lora_weights()
        pipe.load_lora_weights(ROOT / "outputs" / "lora" / f"fold{args.fold}_{sup}")
        grid_rows = [("ASLI", real_imgs[:8])]
        for cfg in args.cfgs:
            imgs = []
            for j in range(args.n):
                mag = "20x" if j % 2 == 0 else "40x"  # setengah-setengah, ~ proporsi train
                g = torch.Generator("cuda").manual_seed(args.seed + j)
                imgs.append(pipe(make_prompt(sup, mag), num_inference_steps=args.steps,
                                 guidance_scale=cfg, generator=g).images[0])
            st = pd.DataFrame([color_stats(im) for im in imgs])
            smd = ((st.mean() - mu_r).abs() / sd_r)
            rows.append({"kelas": CLASS_LABEL[sup], "cfg": cfg, **st.mean().round(1).to_dict(),
                         **{f"SMD_{m}": round(smd[m], 2) for m in METRICS},
                         "SMD_rata": round(smd.mean(), 3)})
            grid_rows.append((f"CFG {cfg:g}", imgs[:8]))
            print(f"{CLASS_LABEL[sup]} CFG {cfg:>4g}: SMD rata = {smd.mean():.3f}")

        # grid: baris = ASLI + tiap CFG, kolom = 8 gambar (seed sama antar baris)
        label_w, gap = 90, 4
        W = label_w + 8 * (THUMB + gap)
        H = len(grid_rows) * (THUMB + gap)
        grid = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(grid)
        for r, (label, imgs) in enumerate(grid_rows):
            y = r * (THUMB + gap)
            d.text((8, y + THUMB // 2), label, fill="black")
            for c, im in enumerate(imgs):
                grid.paste(im.resize((THUMB, THUMB)), (label_w + c * (THUMB + gap), y))
        gpath = out_dir / f"fold{args.fold}_{sup}_grid.png"
        grid.save(gpath)
        print(f"Tersimpan: {gpath.relative_to(ROOT).as_posix()}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"fold{args.fold}_summary.csv", index=False)
    print("\nRingkasan (rata-rata warna; SMD makin kecil = makin mirip patch train asli):")
    print(summary.to_string(index=False))
    best = (summary[summary["cfg"] != "ASLI"].groupby("cfg")["SMD_rata"].mean().sort_values())
    print("\nSMD rata-rata ADC+SCC per CFG (urut terbaik):")
    print(best.round(3).to_string())


if __name__ == "__main__":
    main()
