# Select, Then Augment

Code for **"Select, Then Augment: Filtering LoRA-Generated Synthetic Images for Lung Cancer Histopathology Classification"** (M. H. Pearlyaradja and R. Wijaya, Telkom University).

Class-specific LoRA adapters for Stable Diffusion v1.5 are trained on adenocarcinoma (ADC) and squamous cell carcinoma (SCC) patches of [LungHist700](https://doi.org/10.6084/m9.figshare.25459174) within five patient-level folds. Three filters (class consistency, feature-space realism, memorization) score the generated images, and EfficientNet-B0 classifiers trained with each selection are compared against real-only training with a patient-cluster bootstrap.

## Pipeline

| Stage | Scripts | Output |
|---|---|---|
| 1. Data preparation | `01_data_prep.py`, `01b_split.py`, `01c_crop_patches.py` | patient-level folds, 512×512 patches |
| 2. LoRA fine-tuning | `02_train_lora.py`, `02b_check_lora.py` | one adapter per class and fold |
| 3. Generation | `03a_cfg_sweep.py`, `03_generate.py` | 1,000 images per class and fold |
| 4. Filtering | `04a_train_filter_clf.py`, `04b_filter_scores.py`, `04e_select_topn.py` (+ diagnostics `04c`, `04d`, `04f`, `04g`) | filter scores, top-200 selections C1–C4 |
| 5. Downstream training | `05_train_downstream.py`, `05c_color_check.py`, `05d_summarize.py` | 75 runs (5 folds × 5 conditions × 3 seeds) |
| 6. Evaluation | `06a_bootstrap.py`, `06b_kid_fid.py`, `06c_plot.py`, `06d_patient_level.py`, `06e_visual_examples.py` | bootstrap CIs, KID/FID, patient-level results |

Conditions: C0 real only, C1 random synthetic, C2 realism, C3 class consistency + memorization, C4 all filters (200 synthetic images per class and fold in C1–C4).

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

1. Download LungHist700 and place the class folders in `data/raw/` (`aca_bd`, `aca_md`, `aca_pd`, `scc_bd`, `scc_md`, `scc_pd`) and the metadata file at `data/metadata.csv`.
2. Download Stable Diffusion v1.5 (diffusers format) to `models/sd-v1-5/`.
3. Run the stages in order from the repository root. Each script has `--help`; the settings used in the paper are:

```bash
python src/01_data_prep.py
python src/01b_split.py
python src/01c_crop_patches.py
# for each fold k in 0..4 and class c in aca, scc
python src/02_train_lora.py --fold k --superclass c --max_train_steps 1000
python src/03_generate.py --folds k --n_per_class 1000 --cfg 4.5
# filter classifier: three seeds per fold, strong augmentation
python src/04a_train_filter_clf.py --fold k --epochs 15 --tag v3 --aug strong
python src/04a_train_filter_clf.py --fold k --epochs 15 --tag v3s1 --aug strong --seed 1
python src/04a_train_filter_clf.py --fold k --epochs 15 --tag v3s2 --aug strong --seed 2
python src/04b_filter_scores.py --folds k --clf "v3+v3s1+v3s2" --tta --out filter
python src/04e_select_topn.py --n 200 --main_n 200
# for each fold k, condition C in C0..C4 and seed s in 0..2
python src/05_train_downstream.py --fold k --cond C --seed s
python src/05d_summarize.py
python src/06a_bootstrap.py
python src/06b_kid_fid.py
python src/06d_patient_level.py
```

The SCC adapter of fold 0 was retrained with `--seed 1234` (see the paper).

The experiments ran on a single 6 GB laptop GPU (fp16, gradient checkpointing, batch size 1 with gradient accumulation for LoRA).

## Results

`results/` holds the outputs reported in the paper, so the analysis can be checked without retraining:

- `splits.csv`, `patch_splits.csv`: patient-level folds and patch assignments
- `synthetic_manifest.csv`: prompt, seed, and settings of every generated image
- `filter/`: per-image filter scores and validation-calibrated thresholds per fold
- `selection/selection_N200.csv`: which images were selected for C1–C4
- `downstream/`: test predictions and metrics of all 75 runs
- `eval/`: bootstrap CIs and paired differences (also leave-one-fold-out), KID/FID, patient-level results

Generated images, adapters, and classifier weights are not included because of their size.

## License and data

LungHist700 is released by its authors under CC BY 4.0; please cite their paper when using it.
