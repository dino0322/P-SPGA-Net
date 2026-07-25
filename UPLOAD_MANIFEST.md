# Upload Manifest

Package: `p_spga_net_upload_20260725`

Created for source-code upload. This package is a clean copy of the research
code and excludes local datasets, checkpoints, logs, result tables, generated
figures, and Python cache files.

## Included

- Core training entry points: `main_proposal.py`, `main_ablation.py`
- Run scripts: `run.sh`, `run_a.sh`
- Model source files: `models/`
- Binary training utilities: `utils/data_utils.py`, `utils/train_loop.py`
- Dataset candidate notes: `dataset_candidates.md`
- Upload hygiene files: `.gitignore`, `README.md`, requirements files
- Integrity tooling: `scripts/verify_integrity.py`, `scripts/make_checksums.py`

## Excluded

- `data/`
- `color/`
- `model_save/`
- `results/`
- `logs/`
- `outputs/`
- `__pycache__/`
- `main_multiclass.py`
- `run_m.sh`
- `utils/multiclass_data_utils.py`
- `utils/multiclass_train_loop.py`
- `models/module_backup.py`
- `models/create_model.py`
- `models/create_safe_model.py`
- local scratch file `1`

## Notes

The default upload behavior is conservative. The release package keeps the
binary classification protocol and explicit ablation runner, while excluding
the disabled multiclass experiment files that were not part of the final
binary protocol.
