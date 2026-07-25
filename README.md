# P-SPGA Net

P-SPGA Net (Plant-aware Statistical Prior-Guided Attention Network) is a
PyTorch research codebase for binary plant disease image classification. The
main proposal adds a compact plant-aware statistical prior-guided attention
block to an EfficientNetV2-style backbone and evaluates it against the baseline
and ablation variants.

This upload package contains source code only. Datasets, trained weights,
logs, result CSVs, and generated figures are intentionally excluded.

## What Is Included

- `main_proposal.py`: original proposal comparison loop for field-like classes.
- `main_ablation.py`: explicit ablation runner for the EfficientNetV2
  baseline, P-SPGA variants, and domain-transform variants.
- `models/p_spga_block.py`: standalone P-SPGA block implementation.
- `models/efficientnetv2_P1.py`: EfficientNetV2 baseline and P-SPGA model.
- `models/`: EfficientNetV2 baseline/proposal code and the standalone P-SPGA
  block.
- `utils/`: binary dataloader, metrics, plots, and training loop.
- `run.sh`: four field-like classes with the proposal loop.
- `run_a.sh`: explicit ablation set over the selected class list.

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
DATA_ROOT=./custom_data bash run_a.sh
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
| `DT` | `DT_pspga_domain_blur` | A5 with stronger Y-channel domain blur |
| `A5DT` | `A5_DTL_chunk_full_s01_domain_blur_k3_s04` | A5 with light Y-channel domain blur |

In the reported experiments, the statistical prior and learnable spatial
attention are fused with a fixed alpha value when the model is called from the
training loop. Epoch-aware dynamic alpha scheduling is intentionally left as
follow-up work so the released code stays aligned with the reported local
results.

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

## Reported Local Results

The local experiment set covered 14 binary disease tasks: four field-like
mildew tasks (`downy_mildew`, `powdery_mildew`, `sim_downy_mildew`,
`sim_powdery_mildew`) and ten PlantVillage-derived tasks across cherry, apple,
corn, grape, potato, and tomato. A5 was the main proposal variant.

The raw result CSV files are not bundled with this source-code release. The
summary below is included only to document the local experimental trend; re-run
the scripts with the same dataset layout to reproduce the numbers.

### Overall Mean, 14 Tasks

| Model | Accuracy | Loss | F1 score | AUROC |
|---|---:|---:|---:|---:|
| EfficientNetV2 baseline | 0.9158 | 0.2418 | 0.9114 | 0.9508 |
| A5 P-SPGA | 0.9407 | 0.1554 | 0.9407 | 0.9714 |
| DT domain transform | 0.9398 | 0.1430 | 0.9394 | 0.9703 |

### Overall Mean, Excluding Cherry Baseline Collapse

One cherry baseline run collapsed in the local logs, so the table below reports
the same aggregation with the cherry task excluded.

| Model | Accuracy | Loss | F1 score | AUROC |
|---|---:|---:|---:|---:|
| EfficientNetV2 baseline | 0.9288 | 0.1688 | 0.9296 | 0.9661 |
| A5 P-SPGA | 0.9369 | 0.1649 | 0.9369 | 0.9693 |
| DT domain transform | 0.9358 | 0.1520 | 0.9353 | 0.9681 |

### Domain Split, Excluding Cherry

| Domain | Model | Accuracy | Loss | F1 score | AUROC |
|---|---|---:|---:|---:|---:|
| Field-like mildew | EfficientNetV2 baseline | 0.8260 | 0.4312 | 0.8249 | 0.8976 |
| Field-like mildew | A5 P-SPGA | 0.8399 | 0.4372 | 0.8391 | 0.9065 |
| Field-like mildew | DT domain transform | 0.8356 | 0.3922 | 0.8346 | 0.9039 |
| PlantVillage-like | EfficientNetV2 baseline | 0.9744 | 0.0521 | 0.9761 | 0.9966 |
| PlantVillage-like | A5 P-SPGA | 0.9801 | 0.0439 | 0.9803 | 0.9972 |
| PlantVillage-like | DT domain transform | 0.9803 | 0.0452 | 0.9801 | 0.9966 |

In these local experiments, A5 showed the clearest value on the field-like
mildew tasks, while the PlantVillage-like tasks were often already near
ceiling. Domain-transform variants were useful as auxiliary analysis,
especially for loss/confidence behavior, but A5 remained the cleaner main
model.
