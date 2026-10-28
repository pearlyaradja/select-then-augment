"""
Tahap 4b: hitung skor 3 filter untuk setiap gambar sintetis, per fold.

Memakai classifier filter fold k (hasil TRAINING di 04a_train_filter_clf.py):
  - probabilitas kelas -> filter (a) class consistency
  - fitur 1280-d (sebelum head) -> filter (b) realism & (c) memorization
Tidak ada training di script ini; ini penilaian (scoring) memakai model yang
sudah dilatih dari data kita.

Data per fold k: patch ASLI train (referensi), patch ASLI val (kalibrasi ambang),
gambar sintetis fold k. Patch TEST tidak dipakai sama sekali.

Ambang dikalibrasi dari patch VAL asli, sehingga artinya:
  (a) class consistency : p(kelas yang dimaksud) >= 0.5 (tidak salah kelas) DAN
                          >= persentil ke-10 confidence patch val asli yang
                          diklasifikasi benar di kelas itu
                          -> "setidaknya seyakin 90% gambar asli yang benar"
  (b) realism           : jarak rata-rata ke 5 patch train asli terdekat (kelas sama)
                          <= persentil ke-95 jarak yang sama untuk patch val asli
                          -> "tidak lebih asing daripada 95% gambar asli baru"
  (c) memorization      : kemiripan maksimum ke patch train asli mana pun
                          <= kemiripan maksimum patch val asli mana pun
                          -> "tidak lebih mirip ke data training daripada gambar
                             asli dari pasien lain yang paling mirip"

Kondisi Tahap 5: C1 semua | C2 lolos (b) | C3 lolos (a)&(c) | C4 lolos (a)&(b)&(c)

Output:
  outputs/filter/fold{k}_scores.csv      skor & status lolos per gambar sintetis
  outputs/filter/fold{k}_thresholds.json ambang hasil kalibrasi + cek patch val
  outputs/filter/summary.csv             retensi per fold/kelas/kondisi
  outputs/filter/gallery_fold{k}.png     galeri berskor

Contoh:
  python src/04b_filter_scores.py --folds 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
PATCH_SPLITS_CSV = ROOT / "outputs" / "patch_splits.csv"
SYN_MANIFEST = ROOT / "outputs" / "synthetic_manifest.csv"
OUT_DIR = ROOT / "outputs" / "filter"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLASS_LABEL = {0: "ADC", 1: "SCC"}


class ImgDataset(Dataset):
    """Tanpa augmentasi: penilaian harus deterministik."""
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        return (x - MEAN) / STD


def dihedral_views(x):
    """8 orientasi (rotasi 0/90/180/270 x flip) untuk test-time augmentation."""
    views = []
    for flip in (False, True):
        xf = torch.flip(x, dims=[3]) if flip else x
        views += [torch.rot90(xf, k=k, dims=[2, 3]) for k in range(4)]
    return views


@torch.no_grad()
def embed(models, paths, device, tta=False, bs=16):
    """-> (prob SCC [N], fitur L2-normalized [N,1280]).
    Rata-rata atas semua model (ensemble) dan semua orientasi (TTA): menilai
    satu gambar dari banyak 'sudut pandang' membuat skor lebih stabil, karena
    diagnosis menunjukkan satu classifier saja tidak andal."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    probs, feats = [], []
    for x in DataLoader(ImgDataset(paths), batch_size=bs, num_workers=2):
        x = x.to(device)
        p_sum, f_sum, n = 0, 0, 0
        for model in models:
            for v in (dihedral_views(x) if tta else [x]):
                with torch.autocast("cuda", dtype=torch.float16):
                    f = model.forward_head(model.forward_features(v), pre_logits=True)
                    logits = model.get_classifier()(f)
                p_sum = p_sum + torch.softmax(logits.float(), 1)[:, 1]
                f_sum = f_sum + F.normalize(f.float(), dim=1)
                n += 1
        probs.append((p_sum / n).cpu())
        feats.append(F.normalize(f_sum / n, dim=1).cpu())
    return torch.cat(probs).numpy(), torch.cat(feats)


def knn_dist(q, ref, k):
    """Jarak kosinus rata-rata ke k tetangga terdekat (makin kecil = makin mirip data asli)."""
    sim = q @ ref.T
    return (1 - sim.topk(k, dim=1).values).mean(1).numpy()


def max_sim(q, ref):
    s = q @ ref.T
    v, i = s.max(1)
    return v.numpy(), i.numpy()


def lpo_stats(feats, pids, k):
    """Leave-patient-out di patch TRAIN: setiap patch dibandingkan hanya dengan
    patch train dari PASIEN LAIN (meniru 'gambar asli dari pasien yang belum
    dilihat'). Dipakai sebagai kalibrasi pembanding karena val hanya berisi
    2 pasien per kelas per fold, sedangkan train 13-16 pasien."""
    s = feats @ feats.T
    same = torch.from_numpy(pids[:, None] == pids[None, :])
    s[same] = -2.0  # singkirkan pasangan dari pasien yang sama (termasuk dirinya sendiri)
    knn = (1 - s.topk(k, dim=1).values).mean(1).numpy()
    return knn, s.max(1).values.numpy()


def score_fold(fold, args, device, ps, man):
    f = ps[ps["fold"] == fold]
    tr = f[f["split"] == "train"].reset_index(drop=True)
    va = f[f["split"] == "val"].reset_index(drop=True)
    syn = man[man["fold"] == fold].reset_index(drop=True)
    # Pastikan tidak ada patch test yang ikut sebagai referensi/kalibrasi.
    assert not set(f.loc[f["split"] == "test", "patch_path"]) & (set(tr.patch_path) | set(va.patch_path))

    models = []
    for tag in args.clf.split("+"):
        mdl = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        d = ROOT / "outputs" / "filter_clf" / (f"fold{fold}" + (f"_{tag}" if tag else ""))
        mdl.load_state_dict(load_file(d / "best.safetensors"))
        models.append(mdl.to(device).eval())

    print(f"\n=== fold {fold}: train {len(tr)}, val {len(va)}, sintetis {len(syn)} | "
          f"classifier {args.clf or 'baseline'} x{len(models)}, TTA={args.tta} ===")
    p_tr, f_tr = embed(models, tr.patch_path, device, args.tta)
    p_va, f_va = embed(models, va.patch_path, device, args.tta)
    p_sy, f_sy = embed(models, syn.path, device, args.tta)
    del models
    torch.cuda.empty_cache()

    y_tr, y_va, y_sy = tr.label.values, va.label.values, syn.label.values
    pt_va = np.where(y_va == 1, p_va, 1 - p_va)  # confidence kelas BENAR (val asli)
    pt_sy = np.where(y_sy == 1, p_sy, 1 - p_sy)  # confidence kelas YANG DIMAKSUD (sintetis)

    thr = {"fold": fold}
    syn_a = np.zeros(len(syn), bool)
    syn_b = np.zeros(len(syn), bool)
    knn_sy = np.zeros(len(syn))
    val_pass = {"a": [], "b": []}
    for c in (0, 1):
        # (a) ambang confidence dari patch val asli yang benar diklasifikasi
        ok = (y_va == c) & (pt_va >= 0.5)
        t_a = max(0.5, float(np.percentile(pt_va[ok], args.conf_pct)))
        thr[f"a_conf_{CLASS_LABEL[c]}"] = t_a
        syn_a |= (y_sy == c) & (pt_sy >= t_a)
        val_pass["a"].append(((y_va == c) & (pt_va >= t_a)).sum())

        # (b) realism: jarak kNN ke train asli KELAS SAMA
        ref = f_tr[torch.from_numpy(y_tr == c)]
        d_va = knn_dist(f_va[torch.from_numpy(y_va == c)], ref, args.k)
        t_b = float(np.percentile(d_va, args.real_pct))
        thr[f"b_knn_{CLASS_LABEL[c]}"] = t_b
        m = y_sy == c
        knn_sy[m] = knn_dist(f_sy[torch.from_numpy(m)], ref, args.k)
        syn_b |= m & (knn_sy <= t_b)
        val_pass["b"].append((d_va <= t_b).sum())

    # (c) memorization: kemiripan maksimum ke train asli mana pun
    ms_va, _ = max_sim(f_va, f_tr)
    t_c = float(ms_va.max())
    thr["c_maxsim"] = t_c
    ms_sy, nn_idx = max_sim(f_sy, f_tr)
    syn_c = ms_sy <= t_c

    # ---- Kalibrasi PEMBANDING (analisis sensitivitas, bukan hasil utama) ----
    # (a') hanya buang yang salah kelas (p<0.5), tanpa ambang confidence val
    # (b') & (c') ambang dari train leave-patient-out, bukan dari val
    syn_b_lpo = np.zeros(len(syn), bool)
    pid_tr = tr.patient_id.values
    for c in (0, 1):
        m_tr = y_tr == c
        knn_lpo, _ = lpo_stats(f_tr[torch.from_numpy(m_tr)], pid_tr[m_tr], args.k)
        t = float(np.percentile(knn_lpo, args.real_pct))
        thr[f"lpo_b_knn_{CLASS_LABEL[c]}"] = t
        syn_b_lpo |= (y_sy == c) & (knn_sy <= t)
    _, ms_lpo = lpo_stats(f_tr, pid_tr, args.k)
    thr["lpo_c_maxsim"] = float(ms_lpo.max())
    syn_a_lpo = pt_sy >= 0.5
    syn_c_lpo = ms_sy <= thr["lpo_c_maxsim"]

    # Cek kewajaran: berapa persen patch VAL asli yang lolos filter (a)/(b)?
    thr["val_real_pass_a"] = float(sum(val_pass["a"]) / len(va))
    thr["val_real_pass_b"] = float(sum(val_pass["b"]) / len(va))
    thr["val_real_acc"] = float(((p_va >= 0.5).astype(int) == y_va).mean())
    print("Ambang:", {k: round(v, 4) if isinstance(v, float) else v for k, v in thr.items()})

    out = syn[["path", "fold", "superclass", "label", "seed"]].copy()
    out["p_intended"] = pt_sy.round(4)
    out["pred_correct"] = pt_sy >= 0.5
    out["knn_dist"] = knn_sy.round(4)
    out["max_sim_train"] = ms_sy.round(4)
    out["nn_train_path"] = tr.patch_path.values[nn_idx]
    out["pass_a"], out["pass_b"], out["pass_c"] = syn_a, syn_b, syn_c
    out["C1"] = True
    out["C2"] = out.pass_b
    out["C3"] = out.pass_a & out.pass_c
    out["C4"] = out.pass_a & out.pass_b & out.pass_c
    out["pass_a_lpo"], out["pass_b_lpo"], out["pass_c_lpo"] = syn_a_lpo, syn_b_lpo, syn_c_lpo
    out["C2_lpo"] = out.pass_b_lpo
    out["C3_lpo"] = out.pass_a_lpo & out.pass_c_lpo
    out["C4_lpo"] = out.pass_a_lpo & out.pass_b_lpo & out.pass_c_lpo
    return out, thr


def gallery(df, fold, path, n=6):
    """Galeri berskor: tiap gambar diberi skor 3 filter & status lolos/gagal."""
    T, G, L, H_TXT = 210, 4, 150, 46
    rows = []
    for c, lab in CLASS_LABEL.items():
        d = df[df.label == c]
        rows += [
            (f"{lab}\nlolos semua (C4)", d[d.C4].sample(min(n, d.C4.sum()), random_state=0)),
            (f"{lab}\ngagal (a) kelas", d[~d.pass_a].nsmallest(n, "p_intended")),
            (f"{lab}\ngagal (b) realism", d[~d.pass_b].nlargest(n, "knn_dist")),
            (f"{lab}\nskor tertinggi (c)\nkemiripan train", d.nlargest(n, "max_sim_train")),
        ]
    W = L + n * (T + G)
    img = Image.new("RGB", (W, len(rows) * (T + H_TXT + G)), "white")
    dr = ImageDraw.Draw(img)
    for r, (label, sub) in enumerate(rows):
        y = r * (T + H_TXT + G)
        dr.multiline_text((6, y + T // 2 - 14), label, fill="black")
        for j, (_, it) in enumerate(sub.iterrows()):
            x = L + j * (T + G)
            img.paste(Image.open(ROOT / it.path).convert("RGB").resize((T, T)), (x, y))
            ok = "LOLOS C4" if it.C4 else "GAGAL: " + ",".join(
                k for k, v in (("a", it.pass_a), ("b", it.pass_b), ("c", it.pass_c)) if not v)
            dr.text((x + 2, y + T + 2),
                    f"kelas {it.p_intended:.2f}  realism {it.knn_dist:.3f}\n"
                    f"mirip-train {it.max_sim_train:.3f}  {ok}",
                    fill=(0, 120, 0) if it.C4 else (190, 0, 0))
    img.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--conf_pct", type=float, default=10, help="persentil confidence val (a)")
    p.add_argument("--real_pct", type=float, default=95, help="persentil jarak kNN val (b)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--clf", type=str, default="",
                   help='versi classifier; "" = baseline, "v3+v3s1+v3s2" = ensemble')
    p.add_argument("--tta", action="store_true", help="rata-rata 8 orientasi per gambar")
    p.add_argument("--out", type=str, default="filter", help="nama folder di outputs/")
    args = p.parse_args()

    global OUT_DIR
    OUT_DIR = ROOT / "outputs" / args.out
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ps = pd.read_csv(PATCH_SPLITS_CSV, keep_default_na=False)
    man = pd.read_csv(SYN_MANIFEST)

    for fold in args.folds:
        df, thr = score_fold(fold, args, device, ps, man)
        df.to_csv(OUT_DIR / f"fold{fold}_scores.csv", index=False)
        with open(OUT_DIR / f"fold{fold}_thresholds.json", "w") as fh:
            json.dump({**thr, **vars(args)}, fh, indent=2)
        gallery(df, fold, OUT_DIR / f"gallery_fold{fold}.png")
        s = df.groupby("superclass")[["pass_a", "pass_b", "pass_c", "C2", "C3", "C4"]].sum()
        print(f"Lolos per kelas (dari 1000):\n{s.to_string()}")

    # Ringkasan semua fold yang sudah punya skor.
    allf = [pd.read_csv(q) for q in sorted(OUT_DIR.glob("fold*_scores.csv"))]
    cols = ["C1", "C2", "C3", "C4", "C2_lpo", "C3_lpo", "C4_lpo"]
    summ = pd.concat(allf).groupby(["fold", "superclass"])[cols].sum().reset_index()
    summ.to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nRingkasan jumlah lolos per kondisi (utama = kalibrasi val; _lpo = pembanding):\n"
          + summ.to_string(index=False))
    print(f"Minimum lintas fold/kelas, kalibrasi val : {int(summ[['C2', 'C3', 'C4']].min().min())}")
    print(f"Minimum lintas fold/kelas, kalibrasi lpo : "
          f"{int(summ[['C2_lpo', 'C3_lpo', 'C4_lpo']].min().min())}")


if __name__ == "__main__":
    main()
