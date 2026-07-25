# P-SPGA Net

P-SPGA Net is a PyTorch research codebase for binary plant disease image
classification. The main proposal adds a compact P-SPGA block to an
EfficientNetV2-style backbone and evaluates it against the baseline and
ablation variants.

This upload package contains source code only. Datasets, trained weights,
logs, result CSVs, and generated figures are intentionally excluded.

## What Is Included

- `main_proposal.py`: original proposal comparison loop for field-like classes.
- `main_ablation.py`: explicit ablation runner for baseline, P-SPGA variants,
  M3 attention pooling, and domain-transform variants.
- `models/p_spga_block.py`: standalone P-SPGA block implementation.
- `models/efficientnetv2_P1.py`: EfficientNetV2 baseline and P-SPGA model.
- `models/efficientnetv2_m3.py`: M3 attention-pooling variant.
- `models/`: legacy comparison model files kept for reproducibility.
- `utils/`: binary dataloader, metrics, plots, and training loop.
- `run.sh`: four field-like classes with the proposal loop.
- `run_a.sh`: explicit ablation set over the selected class list.
- `scripts/verify_integrity.py`: upload-package integrity checks.
- `scripts/make_checksums.py`: SHA256 checksum generation.

## Dataset Layout

Place data under `./data` by default. Each disease class must contain
`normal` and `abnormal` folders:

```text
data/
  downy_mildew/
    normal/
    abnormal/
  powdery_mildew/
    normal/
    abnormal/
```

The code expects actual image files, not broken symlinks. Supported image
extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff`.

You can override the data location:

```bash
DATA_ROOT=/path/to/data bash run_a.sh
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the default proposal loop:

```bash
bash run.sh
```

Run the default ablation set:

```bash
bash run_a.sh
```

Run a single class manually:

```bash
python -u main_ablation.py \
  --class_name downy_mildew \
  --data_root ./data \
  --ablation B0,A0,A1,A2,A3,A5
```

## Ablation Codes

| Code | Model name | Description |
|---|---|---|
| `B0` | `efficientnetv2_baseline` | EfficientNetV2 baseline |
| `A0` | `A0_prior_only` | spatial prior only |
| `A1` | `A1_channel_only` | statistical channel gate only |
| `A2` | `A2_stat_only` | channel gate + spatial prior |
| `A3` | `A3_chunk_multiscale_s01` | chunked multiscale branch without learnable attention |
| `A4` | `A4_stat_learnable` | statistical prior + learnable attention |
| `A5` | `A5_chunk_full_s01` | final P-SPGA variant |
| `M3` | `M3_pspga_attnpool` | A5 plus latent attention pooling |
| `DT` | `DT_pspga_domain_blur` | A5 with stronger Y-channel domain blur |
| `A5DT` | `A5_DTL_chunk_full_s01_domain_blur_k3_s04` | A5 with light Y-channel domain blur |

`run_a.sh` defaults to `B0,A0,A1,A2,A3,A5`. Override it with:

```bash
ABLATIONS=A5,DT,A5DT bash run_a.sh
```

## Reproducibility Notes

- The training scripts use seed `1007`.
- `main_ablation.py` uses 2-fold stratified cross-validation.
- Each fold splits the training side again into train/validation with a
  stratified validation split.
- Models are selected through the existing validation/early-stopping loop.
- Metrics and plots are saved under `results/`; weights are saved under
  `model_save/`. Both are ignored by Git.

## Reported Local Trend

On the local 14-class experiment set, A5 was the main proposal variant. It was
especially useful on field-like downy/powdery mildew classes, while controlled
PlantVillage-like classes were already near ceiling. Domain transform variants
were useful as auxiliary analysis, but A5 remained the cleaner main model.

These numbers are local experimental results and are not bundled with this
upload package. Re-run the scripts with the same dataset layout to reproduce.

## Integrity Check

Before uploading or committing:

```bash
python scripts/verify_integrity.py
python scripts/make_checksums.py
```

The integrity check verifies that no data folders, checkpoint files, media
files, logs, results, symlinks, or oversized files are present in the upload
package. It also syntax-checks all Python files with `py_compile`.

