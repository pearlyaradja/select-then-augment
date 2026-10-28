"""Bandingkan SCC sintetis fold 0: LoRA lama (seed 42) vs LoRA latih ulang (s2, seed 1234).

Ukuran yang SAMA dengan diagnosis sebelumnya, supaya lama vs baru adil:
 1. KID (fitur Inception, lepas dari classifier kita): vs SCC asli & vs ADC asli
    (patch TRAIN fold 0). Target: lebih dekat ke SCC daripada ke ADC.
 2. Tiga filter fold 0 dengan ambang RESMI dari outputs/filter/fold0_thresholds.json
    (classifier ensemble v3 + TTA): jumlah lolos (a), (b), (c), dan ketiganya.
 3. Uji silang: classifier fold 0-4 menilai 200 gambar sampel yang sama (seed acak 0).
Gambar lama vs baru memakai noise awal (seed generation) yang sama persis.
Tidak ada training; patch test tidak dipakai. Output: outputs/filter/retrain_fold0_scc.csv
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # perlu untuk worker DataLoader di Windows (spawn)
    spec.loader.exec_module(mod)
    return mod


fs = load("fs", "04b_filter_scores.py")
kc = load("kc", "04f_kid_check.py")


def load_clf(fold, device, tags=("v3", "v3s1", "v3s2")):
    models = []
    for t in tags:
        m = fs.timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        m.load_state_dict(fs.load_file(ROOT / "outputs" / "filter_clf" / f"fold{fold}_{t}" / "best.safetensors"))
        models.append(m.to(device).eval())
    return models


def main():
    device = "cuda"
    ps = pd.read_csv(ROOT / "outputs" / "patch_splits.csv")
    f0 = ps[ps.fold == 0]
    tr = f0[f0.split == "train"].reset_index(drop=True)
    sets = {
        "lama": pd.read_csv(ROOT / "outputs" / "synthetic_manifest.csv").query("fold == 0 and superclass == 'scc'"),
        "baru": pd.read_csv(ROOT / "outputs" / "synthetic_s2_manifest.csv").query("fold == 0 and superclass == 'scc'"),
    }
    for s in sets.values():
        assert len(s) == 1000
    assert (sets["lama"].seed.values == sets["baru"].seed.values).all(), "seed generation harus sama"
    thr = json.load(open(ROOT / "outputs" / "filter" / "fold0_thresholds.json"))
    res = {k: {} for k in sets}

    # ---- 1. KID ----
    inc = kc.NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    real_scc = kc.features(inc, tr[tr.superclass == "scc"].patch_path.tolist(), device)
    real_adc = kc.features(inc, tr[tr.superclass == "aca"].patch_path.tolist(), device)
    for k, s in sets.items():
        fsyn = kc.features(inc, s.path.tolist(), device)
        rng = np.random.default_rng(0)
        res[k]["KID vs SCC asli"] = kc.kid(fsyn, real_scc, rng)[0] * 1e3
        res[k]["KID vs ADC asli"] = kc.kid(fsyn, real_adc, rng)[0] * 1e3
    del inc
    torch.cuda.empty_cache()
    print("KID selesai", flush=True)

    # ---- 2. Filter fold 0 dengan ambang resmi ----
    models = load_clf(0, device)
    _, f_tr = fs.embed(models, tr.patch_path, device, tta=True)
    ref_scc = f_tr[torch.from_numpy(tr.label.values == 1)]
    for k, s in sets.items():
        p_scc, f_sy = fs.embed(models, s.path, device, tta=True)
        pass_a = p_scc >= thr["a_conf_SCC"]
        pass_b = fs.knn_dist(f_sy, ref_scc, thr["k"]) <= thr["b_knn_SCC"]
        pass_c = fs.max_sim(f_sy, f_tr)[0] <= thr["c_maxsim"]
        res[k].update({"P(SCC) rata-rata, clf fold 0": float(p_scc.mean()),
                       "lolos (a) kelas": int(pass_a.sum()), "lolos (b) realism": int(pass_b.sum()),
                       "lolos (c) jiplakan": int(pass_c.sum()), "lolos ketiganya (C4)": int((pass_a & pass_b & pass_c).sum())})
    del models
    torch.cuda.empty_cache()
    print("filter fold 0 selesai", flush=True)

    # ---- 3. Uji silang: sampel 200 yang sama dengan 04d (random_state=0) ----
    idx = sets["lama"].sample(200, random_state=0).index
    pos = sets["lama"].index.get_indexer(idx)
    for i in range(5):
        models = load_clf(i, device)
        for k, s in sets.items():
            p, _ = fs.embed(models, s.path.iloc[pos], device, tta=True)
            res[k][f"% dinilai SCC, clf fold {i}"] = float((p >= 0.5).mean() * 100)
        del models
        torch.cuda.empty_cache()
        print(f"uji silang classifier fold {i} selesai", flush=True)

    out = pd.DataFrame(res)
    out.to_csv(ROOT / "outputs" / "filter" / "retrain_fold0_scc.csv")
    print("\n=== SCC sintetis fold 0: LoRA lama vs LoRA latih ulang ===")
    print(out.round(2).to_string())
    print("\nTarget: 'KID vs SCC asli' < 'KID vs ADC asli', dan jumlah lolos filter naik.")


if __name__ == "__main__":
    main()
