"""
Tahap 2: LoRA fine-tuning Stable Diffusion v1.5, satu LoRA per KELAS per fold.

Ini TRAINING sungguhan (forward -> loss -> backward -> optimizer.step),
bukan inference. Base model dibekukan; yang dilatih hanya adapter LoRA
di layer attention UNet.

Kenapa per kelas: percobaan LoRA gabungan (ADC+SCC dalam satu LoRA) di
fold 0 menghasilkan gambar yang hampir identik untuk prompt ADC dan SCC
(prompt kelas diabaikan). Dengan LoRA terpisah, kelas gambar sintetis
dijamin oleh data training-nya.

Data   : outputs/patch_splits.csv, HANYA split == "train" pada fold terpilih
         dan superclass terpilih (val & test tidak pernah dilihat LoRA).
Prompt : superclass + magnification, mis.
         "H&E stained lung histopathology, adenocarcinoma, 20x magnification"
Output : outputs/lora/fold{k}_{aca|scc}/pytorch_lora_weights.safetensors
         outputs/lora/fold{k}_{aca|scc}/train_log.csv, train_config.json
         outputs/lora/fold{k}_{aca|scc}/checkpoint-*/ (+ training_state.pt untuk resume)

Contoh:
  # uji coba kecil (WAJIB dulu): 10 patch, 5 optimizer step
  python src/02_train_lora.py --fold 0 --superclass aca --subset 10 --max_train_steps 5 --tag smoke
  # training penuh
  python src/02_train_lora.py --fold 0 --superclass aca --max_train_steps 1000
  # lanjutkan dari checkpoint terakhir kalau terputus (laptop mati, dsb.)
  python src/02_train_lora.py --fold 0 --superclass aca --max_train_steps 1000 --resume
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sd-v1-5"  # lokal, JANGAN download dari Hub
PATCH_SPLITS_CSV = ROOT / "outputs" / "patch_splits.csv"
CLASS_NAMES = {"aca": "adenocarcinoma", "scc": "squamous cell carcinoma"}


def make_prompt(superclass, magnification):
    # Magnification ikut prompt karena 20x dan 40x tampil sangat berbeda;
    # dengan begitu proporsi 20x:40x bisa dikontrol saat generation (Tahap 3).
    return (f"H&E stained lung histopathology, {CLASS_NAMES[superclass]}, "
            f"{magnification} magnification")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--superclass", choices=sorted(CLASS_NAMES), required=True,
                   help="kelas yang dipelajari LoRA ini (satu LoRA per kelas per fold)")
    p.add_argument("--resume", action="store_true",
                   help="lanjutkan dari checkpoint-* terakhir yang punya training_state.pt")
    p.add_argument("--max_train_steps", type=int, default=1000,
                   help="jumlah OPTIMIZER step (bukan micro-batch)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--subset", type=int, default=0,
                   help=">0: pakai N patch saja (uji coba kecil)")
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--tag", type=str, default="",
                   help="sufiks folder output, mis. 'smoke' supaya tidak menimpa hasil penuh")
    return p.parse_args()


class PatchDataset(Dataset):
    def __init__(self, df, prompt_to_idx):
        self.paths = df["patch_path"].tolist()
        self.prompt_idx = [prompt_to_idx[make_prompt(s, m)]
                           for s, m in zip(df["superclass"], df["magnification"])]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float()
        x = x / 127.5 - 1.0  # [-1, 1], rentang input VAE SD
        # Augmentasi dihedral (flip + rotasi 90 derajat): histopatologi tidak
        # punya orientasi "atas", jadi ini menambah variasi tanpa mengubah
        # makna morfologi. Dilakukan di ruang piksel (exact), bukan latent.
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
        x = torch.rot90(x, k=random.randint(0, 3), dims=[1, 2])
        return x, self.prompt_idx[i]


def save_lora(unet, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    # Format standar diffusers -> Tahap 3 cukup pipe.load_lora_weights(out_dir)
    StableDiffusionPipeline.save_lora_weights(out_dir, unet_lora_layers=lora_state,
                                              safe_serialization=True)


def save_checkpoint(unet, ckpt_dir, optimizer, scaler, lr_sched, global_step):
    save_lora(unet, ckpt_dir)
    # State lengkap untuk resume: tanpa optimizer/scaler/lr state, melanjutkan
    # training = Adam mulai dari nol lagi (momentum hilang), hasilnya beda.
    torch.save({
        "global_step": global_step,
        "lora": get_peft_model_state_dict(unet),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "lr_sched": lr_sched.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
    }, ckpt_dir / "training_state.pt")


def latest_checkpoint(out_dir):
    ckpts = [d for d in out_dir.glob("checkpoint-*") if (d / "training_state.pt").exists()]
    return max(ckpts, key=lambda d: int(d.name.split("-")[1]), default=None)


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA tidak tersedia"
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = f"fold{args.fold}_{args.superclass}" + (f"_{args.tag}" if args.tag else "")
    out_dir = ROOT / "outputs" / "lora" / run_name
    if not args.resume and (out_dir / "train_log.csv").exists():
        # Jangan menimpa hasil training lama diam-diam.
        raise SystemExit(f"[ERROR] {out_dir.relative_to(ROOT)} sudah berisi hasil training. "
                         f"Pakai --resume untuk melanjutkan, atau pindahkan folder itu dulu.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    ps = pd.read_csv(PATCH_SPLITS_CSV, keep_default_na=False)
    fold_df = ps[ps["fold"] == args.fold]
    train_df = fold_df[fold_df["split"] == "train"].reset_index(drop=True)
    held_pids = set(fold_df.loc[fold_df["split"] != "train", "patient_id"])
    # Pasien val/test tidak boleh ikut melatih LoRA, kalau tidak
    # gambar sintetis bisa membawa informasi pasien test (leakage).
    overlap = set(train_df["patient_id"]) & held_pids
    print(f"[CHECK] fold {args.fold}: overlap patient_id train vs val/test = {sorted(overlap)}")
    assert not overlap, "patient_id train bocor ke val/test"

    # Filter kelas SETELAH cek overlap: cek patient_id berlaku untuk seluruh fold.
    train_df = train_df[train_df["superclass"] == args.superclass].reset_index(drop=True)
    assert train_df["superclass"].nunique() == 1, "Data training tercampur kelas lain"

    if args.subset > 0:
        train_df = train_df.sample(min(len(train_df), args.subset),
                                   random_state=args.seed).reset_index(drop=True)
    print(f"Patch training: {len(train_df)} "
          f"({train_df['superclass'].value_counts().to_dict()}), "
          f"pasien: {train_df['patient_id'].nunique()}")

    prompts = sorted({make_prompt(s, m) for s, m in
                      zip(train_df["superclass"], train_df["magnification"])})
    prompt_to_idx = {p: i for i, p in enumerate(prompts)}
    print("Prompt yang dipakai:")
    for p in prompts:
        print("   ", p)

    # ------------------------------------------------- load komponen SD 1.5
    # Dimuat satu per satu dari path lokal (bukan pipeline utuh) supaya tiap
    # komponen bisa diatur dtype/device-nya sendiri demi hemat VRAM 6 GB.
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder",
                                                 torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae",
                                        torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet",
                                                torch_dtype=torch.float16).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    # Bekukan base model: hanya LoRA yang boleh menerima gradient.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Prompt hanya beberapa macam (<=4) -> encode sekali, lalu text encoder
    # dikeluarkan dari GPU (~250 MB VRAM dihemat untuk UNet).
    with torch.no_grad():
        tok = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        prompt_embeds = text_encoder(tok.input_ids.to(device))[0]  # (P, 77, 768) fp16
    del text_encoder
    torch.cuda.empty_cache()

    # ---------------------------------------------------- pasang LoRA ke UNet
    lora_config = LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                             target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet.add_adapter(lora_config)
    # Base UNet fp16, tapi parameter LoRA fp32: update kecil dari Adam bisa
    # hilang (underflow) kalau bobotnya fp16.
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    for p in lora_params:
        p.data = p.data.float()
    n_train = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"Parameter dilatih (LoRA): {n_train:,} dari {n_total:,} "
          f"({100 * n_train / n_total:.2f}%)")
    assert n_train > 0, "Tidak ada parameter yang dilatih"

    unet.enable_gradient_checkpointing()  # tukar waktu komputasi dengan VRAM
    unet.train()

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
    scaler = torch.amp.GradScaler("cuda")  # wajib untuk fp16 agar gradient tidak underflow

    loader = DataLoader(PatchDataset(train_df, prompt_to_idx), batch_size=1, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0)

    # ---------------------------------------------------------------- resume
    global_step = 0
    log_path = out_dir / "train_log.csv"
    if args.resume:
        ckpt = latest_checkpoint(out_dir)
        if ckpt is None:
            print("[RESUME] tidak ada checkpoint dengan training_state.pt -> mulai dari step 0")
        else:
            st = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
            res = set_peft_model_state_dict(unet, st["lora"])
            assert not res.unexpected_keys, f"Key LoRA tidak cocok: {res.unexpected_keys[:3]}"
            optimizer.load_state_dict(st["optimizer"])
            scaler.load_state_dict(st["scaler"])
            lr_sched.load_state_dict(st["lr_sched"])
            random.setstate(st["rng_python"])
            np.random.set_state(st["rng_numpy"])
            torch.set_rng_state(st["rng_torch"])
            torch.cuda.set_rng_state(st["rng_cuda"])
            global_step = st["global_step"]
            print(f"[RESUME] lanjut dari {ckpt.name} (step {global_step})")
    if global_step >= args.max_train_steps:
        # Sudah selesai sebelumnya: keluar normal (exit 0) SEBELUM menyentuh
        # log/config, supaya perintah berantai untuk banyak fold bisa diulang.
        print(f"[SKIP] {run_name} sudah selesai sampai step {global_step}, dilewati.")
        return
    if global_step > 0 and log_path.exists():
        # Buang baris log setelah checkpoint (step yang akan diulang).
        old = pd.read_csv(log_path)
        old[old["step"] <= global_step].to_csv(log_path, index=False)
        log_f = open(log_path, "a", newline="")
        log = csv.writer(log_f)
    else:
        log_f = open(log_path, "w", newline="")
        log = csv.writer(log_f)
        log.writerow(["step", "loss", "lr", "sec_per_step", "peak_vram_gb"])

    with open(out_dir / "train_config.json", "w") as f:
        json.dump({**vars(args), "model_path": str(MODEL_PATH.relative_to(ROOT)),
                   "n_patches": len(train_df), "prompts": prompts,
                   "train_patient_ids": sorted(train_df["patient_id"].unique().tolist()),
                   "effective_batch": args.grad_accum,
                   "resumed_from_step": global_step}, f, indent=2)

    # ----------------------------------------------------------- training loop
    torch.cuda.reset_peak_memory_stats()
    start_step = global_step
    micro, running_loss, n_nonfinite = 0, 0.0, 0
    t0 = time.time()
    done = False
    while not done:
        for pixels, pidx in loader:
            pixels = pixels.to(device, dtype=torch.float16)

            with torch.no_grad():
                # Gambar -> latent 4x64x64. scaling_factor (0.18215) dari config
                # VAE supaya varians latent ~1, sesuai saat SD dilatih.
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor

            # Forward diffusion: tambah noise Gaussian di timestep acak.
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = prompt_embeds[pidx.to(device)]

            with torch.autocast("cuda", dtype=torch.float16):
                # UNet (+LoRA) memprediksi noise yang ditambahkan.
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # ===== LOSS: MSE antara noise prediksi dan noise asli =====
            # (SD 1.5 prediction_type = epsilon). Dihitung di fp32 supaya stabil.
            loss = F.mse_loss(model_pred.float(), noise.float())

            if not torch.isfinite(loss):
                n_nonfinite += 1
                print(f"[WARN] loss tidak finite di micro-step {micro}, dilewati")
                assert n_nonfinite <= 10, "Terlalu banyak loss NaN/inf, hentikan"
                continue

            # ===== BACKWARD: gradient hanya mengalir ke parameter LoRA =====
            # Dibagi grad_accum supaya akumulasi 4 micro-batch = rata-rata.
            scaler.scale(loss / args.grad_accum).backward()
            running_loss += loss.item()
            micro += 1

            if micro % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                # ===== OPTIMIZER STEP: update bobot LoRA =====
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_sched.step()
                global_step += 1

                avg_loss = running_loss / args.grad_accum
                running_loss = 0.0
                # Step 1 ikut menanggung startup (spawn worker DataLoader,
                # warmup CUDA), jadi kecepatan diukur mulai setelah step 1.
                n_done = global_step - start_step  # step di sesi ini (resume-aware)
                if n_done == 1:
                    t_first = time.time()
                    sec = t_first - t0
                else:
                    sec = (time.time() - t_first) / (n_done - 1)
                vram = torch.cuda.max_memory_allocated() / 2**30
                log.writerow([global_step, f"{avg_loss:.5f}",
                              f"{lr_sched.get_last_lr()[0]:.2e}", f"{sec:.2f}", f"{vram:.2f}"])
                log_f.flush()
                if global_step % 10 == 0 or n_done <= 5:
                    eta = sec * (args.max_train_steps - global_step) / 60
                    print(f"step {global_step}/{args.max_train_steps}  loss={avg_loss:.4f}  "
                          f"{sec:.2f}s/step  peakVRAM={vram:.2f}GB  ETA={eta:.0f} mnt")
                if global_step % args.save_every == 0:
                    save_checkpoint(unet, out_dir / f"checkpoint-{global_step}",
                                    optimizer, scaler, lr_sched, global_step)
                if global_step >= args.max_train_steps:
                    done = True
                    break
    log_f.close()

    save_lora(unet, out_dir)
    total_min = (time.time() - t0) / 60
    print(f"\nSelesai: step {global_step}/{args.max_train_steps} "
          f"({global_step - start_step} step di sesi ini, {total_min:.1f} menit), "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"loss non-finite dilewati: {n_nonfinite}")
    print(f"LoRA tersimpan di: {out_dir.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
