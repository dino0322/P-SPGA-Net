# Plant Disease / Pest Dataset Candidates

Created: 2026-06-23
Purpose: candidate datasets for follow-up disease/pest experiments. Do not use symlinks when importing. Copy actual image files into `./data/{class_name}/normal|abnormal` or a separate relative raw dataset root.

## Priority A: useful for current binary/class-wise disease protocol

### 1. PlantDoc
- Type: in-the-wild / internet-scraped plant disease images.
- Scale: about 2,598 images, 13 plant species, up to 17 disease classes.
- Why useful: much less controlled than PlantVillage; good domain-shift / field-like comparison.
- Fit to current project: good. Can convert disease classes to per-crop normal/abnormal if healthy samples exist; otherwise use multi-class or abnormal-only auxiliary.
- Source: https://arxiv.org/abs/1911.10317

### 2. Plant Pathology 2020 / 2021 FGVC apple leaf datasets
- Type: apple leaf disease, real orchard-style images.
- Scale: 2020 paper reports 3,651 high-quality real-life apple foliar images; 2021 extends multi-label disease setting.
- Classes: healthy, apple scab, cedar apple rust, multiple diseases and related apple foliar labels depending on year.
- Why useful: stronger field relevance than PlantVillage apple subset.
- Fit to current project: excellent for apple disease binary sets and multi-label/multi-class sanity checks.
- Sources:
  - Paper: https://arxiv.org/abs/2004.11958
  - Kaggle 2020: https://www.kaggle.com/c/plant-pathology-2020-fgvc7
  - Kaggle 2021: https://www.kaggle.com/c/plant-pathology-2021-fgvc8

### 3. Cassava disease / iCassava
- Type: cassava leaf disease images collected from farmer/field-like settings.
- Classes: cassava bacterial blight, brown streak disease, green mite, mosaic disease, healthy, etc. depending on release.
- Why useful: field/mobile-photo domain; visually harder than lab leaf images.
- Fit to current project: good as a non-PlantVillage disease generalization set.
- Sources:
  - iCassava paper: https://arxiv.org/abs/1908.02900
  - Kaggle competition: https://www.kaggle.com/c/cassava-leaf-disease-classification

### 4. Paddy Doctor
- Type: paddy/rice disease image dataset.
- Scale: paper reports 16,225 annotated paddy leaf images across 13 classes, including normal.
- Why useful: has normal class and many disease classes, so binary per-disease conversion is straightforward.
- Fit to current project: excellent for per-disease binary experiments.
- Source: https://arxiv.org/abs/2205.11108

### 5. Dhan-Shomadhan / rice leaf disease datasets
- Type: rice disease dataset with field and white-background variants.
- Scale: Dhan-Shomadhan paper reports 1,106 images across five rice diseases with background variation.
- Why useful: useful to test whether model behaves differently on controlled vs field background inside one crop.
- Fit to current project: medium-good; check healthy/normal availability before binary conversion.
- Source: https://arxiv.org/abs/2309.07515

### 6. MangoLeafBD
- Type: mango leaf disease dataset from orchards.
- Scale: paper reports 4,000 images covering seven diseases.
- Why useful: field-collected crop-specific disease set, not PlantVillage.
- Fit to current project: good if healthy class is available in released data; otherwise multi-class disease-only comparison.
- Source: https://arxiv.org/abs/2209.02377

## Priority B: useful but protocol may need adaptation

### 7. PlantSeg
- Type: in-the-wild plant disease segmentation dataset.
- Scale: paper reports 11,400 disease images with segmentation masks plus 8,000 healthy images categorized by plant type.
- Why useful: has lesion masks, so could support attention/lesion-localization claims later.
- Fit to current project: classification labels can be derived, but segmentation masks are extra. Good for later visual analysis.
- Source: https://arxiv.org/abs/2409.04038

### 8. IP102 insect pest dataset
- Type: large insect pest image classification benchmark.
- Scale/classes: 102 pest categories in the benchmark.
- Why useful: adds pest/insect side, not only leaf disease; hard fine-grained recognition.
- Fit to current project: not normal/abnormal by default. Better as pest multi-class or pest-vs-non-pest auxiliary experiment.
- Sources:
  - Related benchmark usage paper: https://arxiv.org/abs/2107.12189
  - DeWi paper/code mention: https://arxiv.org/abs/2409.10445

### 9. Coffee leaf rust / coffee leaf miner datasets
- Type: coffee leaf disease/pest imagery, often smartphone/field-style.
- Why useful: disease + pest symptoms in one crop; good for domain discussion.
- Fit to current project: need to verify public data availability and labels before use.
- Sources:
  - Coffee rust low-resolution CNN paper: https://arxiv.org/abs/2407.14737
  - Coffee leaf miner/rust smartphone app paper: https://arxiv.org/abs/1904.00742

## Priority C: current PlantVillage-related variants

### 10. Hugging Face PlantVillage mirrors
- Type: re-hosted PlantVillage variants.
- Why useful: easier scripted download, but mostly controlled/lab-like and may duplicate current data.
- Fit to current project: low priority except for reproducible download tests.
- Sources:
  - https://hf.co/datasets/mohanty/PlantVillage
  - https://hf.co/datasets/geraldmc/plantvillage-full

### 11. PV-ALE apple leaf extension
- Type: PlantVillage-derived/extended apple leaf disease classes.
- Why useful: apple-specific extended class coverage.
- Fit to current project: lower priority than Plant Pathology 2020/2021 because field realism needs checking.
- Source: https://arxiv.org/abs/2410.22490

## Import rules for future use

1. Never create symlink-only dataset roots unless explicitly requested.
2. Copy real image files into the target dataset directory.
3. Before training, verify for each class:
   - `normal` and `abnormal` directories exist.
   - both sides have nonzero image counts.
   - there are zero symlinks unless intentionally allowed.
   - at least one sample per split passes `cv2.imread`.
4. Keep raw downloads separate from experiment-ready data:
   - raw: `./raw_datasets/{dataset_name}`
   - prepared: `./data/{class_name}/normal|abnormal`
5. Kaggle sources may require API token/login; do not attempt download without user confirmation.

## Suggested next import order

1. Plant Pathology 2020/2021 for apple field-style validation.
2. Paddy Doctor for larger normal/disease binary conversion.
3. PlantDoc for noisy in-the-wild cross-domain test.
4. Cassava for mobile/field disease generalization.
5. IP102 only if pest-specific experiment becomes a separate section.
