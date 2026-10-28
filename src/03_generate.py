"""
Tahap 3: generation gambar sintetis per kelas per fold.

Untuk tiap fold k dan kelas c: SD 1.5 (base, lokal) + LoRA fold{k}_{c} hasil
Tahap 2 -> N gambar 512x512. LoRA ADC hanya untuk gambar ADC, LoRA SCC hanya
untuk gambar SCC, jadi label kelas gambar sintetis dijamin oleh LoRA-nya.

Pengaturan:
  - guidance scale 4.5: hasil src/03a_cfg_sweep.py (warna paling mirip patch
    train asli; 7.5 terlalu jenuh/magenta)
  - DPM-Solver++ 25 step
  - prompt magnification 20x/40x mengikuti proporsi train asli fold & kelas itu
    (catatan: LoRA cenderung mengabaikan magnification; dicatat sebagai
    keterbatasan, kolom 'mag_prompt' hanya merekam prompt yang dipakai)
  - seed unik & deterministik per gambar -> setiap gambar bisa dibuat ulang persis

Bisa dilanjutkan kalau terputus: file yang sudah ada dilewati.

Output:
  data/processed/synthetic/fold{k}/{aca|scc}/syn_f{k}_{c}_{i:04d}.png
  outputs/synthetic/manifest_fold{k}_{c}.csv   (per bagian)
  outputs/synthetic_manifest.csv               (gabungan semua bagian yang ada)

Contoh:
  # uji coba kecil
  python src/03_generate.py --folds 0 --n_per_class 4 --tag smoke
  # penuh
  python src/03_generate.py --folds 0 1 2 3 4 --n_per_class 1000
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sd-v1-5"
PATCH_SPLITS_CSV = ROOT / "outputs" / "patch_splits.csv"
CLASS_NAMES = {"aca": "adenocarcinoma", "scc": "squamous cell carcinoma"}
CLASS_IDX = {"aca": 0, "scc": 1}  # sama dengan kolom 'label' di Tahap 1


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
    # Decode batch satu per satu: hemat VRAM 6 GB. (diffusers 0.41: method
    # ini ada di komponen VAE, bukan lagi di pipeline.)
    pipe.vae.enable_slicing()
    return pipe.to(device)


def plan(fold, sup, n, ps, seed_base):
    """Daftar gambar yang harus ada untuk (fold, kelas): magnification & seed."""
    tr = ps[(ps["fold"] == fold) & (ps["split"] == "train") & (ps["superclass"] == sup)]
    p20 = (tr["magnification"] == "20x").mean()
    n20 = int(round(n * p20))
    # Selang-seling 20x/40x (bukan blok) supaya kalau generation berhenti di
    # tengah, gambar yang sudah ada tetap berproporsi mirip.
    mags, c20 = [], 0
    for i in range(n):
        want20 = round((i + 1) * n20 / n) > c20
        mags.append("20x" if want20 else "40x")
        c20 += want20
    assert mags.count("20x") == n20
    return [{
        "fold": fold, "superclass": sup, "label": CLASS_IDX[sup], "index": i,
        "mag_prompt": mags[i], "prompt": make_prompt(sup, mags[i]),
        # Seed unik global: fold & kelas berbeda tidak pernah berbagi noise awal.
        "seed": seed_base + fold * 100_000 + CLASS_IDX[sup] * 10_000 + i,
    } for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--classes", nargs="+", default=["aca", "scc"])
    p.add_argument("--n_per_class", type=int, default=1000)
    p.add_argument("--cfg", type=float, default=4.5)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed_base", type=int, default=1_000_000)
    p.add_argument("--tag", type=str, default="",
                   help="mis. 'smoke' -> folder terpisah supaya tidak tercampur hasil penuh")
    p.add_argument("--lora_tag", type=str, default="",
                   help="pakai LoRA outputs/lora/fold{k}_{c}_{lora_tag} (mis. LoRA latih ulang 's2')")
    args = p.parse_args()

    device = torch.device("cuda")
    syn_root = ROOT / "data" / "processed" / ("synthetic" + (f"_{args.tag}" if args.tag else ""))
    man_dir = ROOT / "outputs" / ("synthetic" + (f"_{args.tag}" if args.tag else ""))
    man_dir.mkdir(parents=True, exist_ok=True)
    ps = pd.read_csv(PATCH_SPLITS_CSV, keep_default_na=False)

    pipe = build_pipeline(device)
    for fold in args.folds:
        for sup in args.classes:
            lora_dir = ROOT / "outputs" / "lora" / (f"fold{fold}_{sup}" + (f"_{args.lora_tag}" if args.lora_tag else ""))
            assert (lora_dir / "pytorch_lora_weights.safetensors").exists(), f"{lora_dir} tidak ada"
            out_dir = syn_root / f"fold{fold}" / sup
            out_dir.mkdir(parents=True, exist_ok=True)

            items = plan(fold, sup, args.n_per_class, ps, args.seed_base)
            for it in items:
                it["path"] = (out_dir / f"syn_f{fold}_{sup}_{it['index']:04d}.png"
                              ).relative_to(ROOT).as_posix()
            todo = [it for it in items if not (ROOT / it["path"]).exists()]
            print(f"\n=== fold {fold} {sup}: {len(items) - len(todo)} sudah ada, "
                  f"{len(todo)} akan dibuat (CFG {args.cfg}, {args.steps} step) ===")

            pipe.unload_lora_weights()  # wajib: jangan sampai LoRA kelas lain ikut aktif
            pipe.load_lora_weights(lora_dir)

            t0, done = time.time(), 0
            for b in range(0, len(todo), args.batch_size):
                batch = todo[b:b + args.batch_size]
                gens = [torch.Generator("cuda").manual_seed(it["seed"]) for it in batch]
                imgs = pipe([it["prompt"] for it in batch], num_inference_steps=args.steps,
                            guidance_scale=args.cfg, generator=gens).images
                for it, im in zip(batch, imgs):
                    im.save(ROOT / it["path"])
                done += len(batch)
                if done % 40 < args.batch_size or done == len(todo):
                    sec = (time.time() - t0) / done
                    print(f"  {done}/{len(todo)}  {sec:.2f} s/gambar  "
                          f"ETA bagian ini {sec * (len(todo) - done) / 60:.0f} mnt  "
                          f"peakVRAM={torch.cuda.max_memory_allocated() / 2**30:.2f}GB")

            man = pd.DataFrame(items).assign(cfg=args.cfg, steps=args.steps,
                                             lora=lora_dir.relative_to(ROOT).as_posix())
            missing = [q for q in man["path"] if not (ROOT / q).exists()]
            assert not missing, f"{len(missing)} file tidak tersimpan, mis. {missing[:2]}"
            man.to_csv(man_dir / f"manifest_fold{fold}_{sup}.csv", index=False)
            print(f"  manifest: {(man_dir / f'manifest_fold{fold}_{sup}.csv').relative_to(ROOT)}"
                  f" ({len(man)} gambar, 20x={int((man.mag_prompt == '20x').sum())})")

    # Gabungan dari SEMUA bagian yang sudah selesai (termasuk dari run sebelumnya).
    parts = sorted(man_dir.glob("manifest_fold*_*.csv"))
    full = pd.concat([pd.read_csv(q) for q in parts], ignore_index=True)
    full.to_csv(man_dir.parent / f"{man_dir.name}_manifest.csv", index=False)
    print(f"\nManifest gabungan: {len(full)} gambar dari {len(parts)} bagian")
    print(full.groupby(["fold", "superclass"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
