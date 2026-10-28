"""Tahap 4e: seleksi top-N gambar sintetis untuk kondisi C1-C4.

KENAPA top-N, bukan ambang tetap: diagnosis Tahap 4 menunjukkan jumlah gambar yang
lolos ambang tetap sangat bergantung pada classifier filter mana yang dipakai
(10-985 per fold/kelas), sehingga syarat "jumlah sintetis sama di C1-C4" tidak
bisa dipenuhi. Dengan top-N, jumlah selalu N per kelas per fold; yang dibandingkan
antar kondisi adalah GAMBAR MANA yang dipilih (inti "Select, Then Augment").

Tidak ada training di sini: hanya mengurutkan skor dari 04b_filter_scores.py
(outputs/filter/fold{k}_scores.csv).

Aturan per fold k, per kelas c (1000 kandidat):
  C1  acak N (seed tetap)                         -> pembanding "tanpa seleksi"
  C2  N terbaik menurut realism (knn_dist terkecil)
  C3  buang gagal jiplakan (pass_c), lalu N terbaik menurut kelas (p_intended terbesar)
  C4  buang gagal jiplakan, lalu N terbaik menurut RATA-RATA PERINGKAT kelas & realism
      (peringkat menyamakan skala dua skor yang berbeda tanpa bobot tambahan)

Output (outputs/selection/):
  selection_N{N}.csv   satu baris per gambar sintetis, kolom C1..C4 (True = dipakai)
  summary_N{N}.csv     per fold/kelas: berapa yang terpilih juga lolos ambang (a)/(b),
                       dan tumpang tindih antar kondisi
  sensitivity.csv      ringkasan untuk beberapa N (dasar memilih N)
  gallery_N{N}_fold{k}.png  contoh terpilih vs terbuang per kondisi

  python src/04e_select_topn.py --n 200 300 500 --main_n 200
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "outputs" / "filter"
OUT = ROOT / "outputs" / "selection"
CONDS = ["C1", "C2", "C3", "C4"]
CLS_NAME = {"aca": "ADC", "scc": "SCC"}


def select(g, n, seed):
    """g = 1000 kandidat satu fold & kelas -> dict kondisi -> index terpilih."""
    out = {"C1": g.sample(n, random_state=seed).index}
    out["C2"] = g.nsmallest(n, "knn_dist").index
    ok = g[g.pass_c]
    # Kalau kandidat lolos jiplakan < N, seleksi tidak bisa adil -> berhenti, jangan diam-diam.
    assert len(ok) >= n, f"hanya {len(ok)} lolos jiplakan < N={n}"
    out["C3"] = ok.nlargest(n, "p_intended").index
    rank = ok.p_intended.rank(ascending=False) + ok.knn_dist.rank(ascending=True)
    out["C4"] = rank.nsmallest(n).index
    return out


def run(scores, n, seed):
    sel = scores[["path", "fold", "superclass", "label", "p_intended", "knn_dist",
                  "max_sim_train", "pass_a", "pass_b", "pass_c"]].copy()
    for c in CONDS:
        sel[c] = False
    rows = []
    for (k, cls), g in scores.groupby(["fold", "superclass"]):
        picks = select(g, n, seed)
        for c, idx in picks.items():
            sel.loc[idx, c] = True
        r = {"N": n, "fold": k, "kelas": CLS_NAME[cls]}
        for c, idx in picks.items():
            s = scores.loc[idx]
            r[f"{c}_lolos_a_%"] = round(s.pass_a.mean() * 100, 1)
            r[f"{c}_lolos_b_%"] = round(s.pass_b.mean() * 100, 1)
            r[f"{c}_p_kelas"] = round(s.p_intended.mean(), 3)
        for a, b in [("C2", "C3"), ("C2", "C4"), ("C3", "C4"), ("C1", "C4")]:
            r[f"overlap_{a}_{b}_%"] = round(len(set(picks[a]) & set(picks[b])) / n * 100, 1)
        rows.append(r)
    return sel, pd.DataFrame(rows)


def check(sel, n):
    """Pengecekan wajib: jumlah tepat N, tidak ada gambar gagal jiplakan di C3/C4."""
    cnt = sel.groupby(["fold", "superclass"])[CONDS].sum()
    assert (cnt == n).all().all(), f"jumlah per kondisi tidak sama dengan N:\n{cnt}"
    for c in ("C3", "C4"):
        assert sel.loc[sel[c], "pass_c"].all(), f"{c} berisi gambar yang gagal cek jiplakan"
    assert sel.path.is_unique
    print(f"[CEK N={n}] OK: setiap fold x kelas x kondisi tepat {n} gambar; "
          f"C3/C4 bebas gambar gagal jiplakan; path unik.")


def gallery(sel, fold, n, path, per=6, size=128):
    """Baris = kondisi x kelas: 'terpilih' (kiri) vs 'terbuang' (kanan) acak."""
    rng = np.random.default_rng(0)
    pad, lab = 4, 150
    W = lab + 2 * per * (size + pad) + 20
    rows = [(c, cls) for cls in ("aca", "scc") for c in CONDS[1:]]
    canvas = Image.new("RGB", (W, 30 + len(rows) * (size + pad)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((lab, 8), "TERPILIH", fill="black")
    d.text((lab + per * (size + pad) + 20, 8), "TERBUANG", fill="black")
    for r, (c, cls) in enumerate(rows):
        g = sel[(sel.fold == fold) & (sel.superclass == cls)]
        y = 30 + r * (size + pad)
        d.text((6, y + size // 2 - 6), f"{c} {CLS_NAME[cls]}", fill="black")
        for j, part in enumerate([g[g[c]], g[~g[c]]]):
            pick = part.iloc[rng.choice(len(part), per, replace=False)]
            for i, p in enumerate(pick.path):
                im = Image.open(ROOT / p).convert("RGB").resize((size, size))
                canvas.paste(im, (lab + j * (per * (size + pad) + 20) + i * (size + pad), y))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[200, 300, 500])
    ap.add_argument("--main_n", type=int, default=200, help="N yang dipakai untuk galeri (N final di paper = 200)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    scores = pd.concat([pd.read_csv(SCORES / f"fold{k}_scores.csv") for k in range(5)], ignore_index=True)
    assert len(scores) == 10_000 and scores.path.is_unique

    sens = []
    for n in args.n:
        sel, summ = run(scores, n, args.seed)
        check(sel, n)
        sel.to_csv(OUT / f"selection_N{n}.csv", index=False)
        summ.to_csv(OUT / f"summary_N{n}.csv", index=False)
        sens.append(summ)
        if n == args.main_n:
            for k in range(5):
                gallery(sel, k, n, OUT / f"gallery_N{n}_fold{k}.png")
    sens = pd.concat(sens)
    sens.to_csv(OUT / "sensitivity.csv", index=False)

    pd.set_option("display.width", 250)
    for n in args.n:
        s = sens[sens.N == n]
        print(f"\n=== N = {n} ===")
        print(s[["fold", "kelas", "C1_lolos_a_%", "C3_lolos_a_%", "C4_lolos_a_%", "C2_lolos_b_%",
                 "C4_lolos_b_%", "overlap_C2_C4_%", "overlap_C3_C4_%", "overlap_C2_C3_%"]].to_string(index=False))


if __name__ == "__main__":
    main()
