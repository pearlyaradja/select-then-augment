"""Revisi (Review 2): 9 undian acak tambahan untuk pembanding C1.

KENAPA: C1 hanya satu undian (seed 42). Selisih C2-C1 (+0.013 macro-F1) lebih kecil dari
SD C1 antar fold (0.043), jadi C2 belum bisa dibedakan dari undian acak yang kebetulan
buruk. Dengan 10 undian (C1 + R1..R9) kita lihat apakah C2 jatuh di luar sebarannya.

Aturan undian sama persis dengan C1 di 04e_select_topn.py (g.sample(N) per fold & kelas),
hanya seed berbeda. Tidak ada training di sini.
Output: outputs/selection/selection_N200_random.csv (kolom R1..R9, True = dipakai)

  python src/07a_random_draws.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEL = ROOT / "outputs" / "selection"
N = 200
SEEDS = {f"R{i}": i for i in range(1, 10)}  # 1..9: semuanya berbeda dari seed C1 (42)


def main():
    sel = pd.read_csv(SEL / f"selection_N{N}.csv")
    out = sel[["path", "fold", "superclass", "label", "C1"]].copy()
    for name, seed in SEEDS.items():
        out[name] = False
        for _, g in sel.groupby(["fold", "superclass"]):
            out.loc[g.sample(N, random_state=seed).index, name] = True
        cnt = out[out[name]].groupby(["fold", "superclass"]).size()
        assert (cnt == N).all() and len(cnt) == 10, f"{name}: jumlah per fold/kelas salah"
        # Undian kembar dengan C1 tidak menambah informasi -> pastikan berbeda.
        assert (out[name] != out.C1).any(), f"{name} identik dengan C1"
    cols = ["C1", *SEEDS]
    ov = {c: round(float((out[c] & out.C1).sum() / out.C1.sum()), 3) for c in SEEDS}
    print("Tumpang tindih dengan C1 (harapan acak 0.20):", ov)
    out.drop(columns="C1").to_csv(SEL / f"selection_N{N}_random.csv", index=False)
    print("saved", SEL / f"selection_N{N}_random.csv", "| kolom:", cols[1:])


if __name__ == "__main__":
    main()
