from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


FORBIDDEN_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "data",
    "color",
    "raw_datasets",
    "model_save",
    "checkpoints",
    "weights",
    "results",
    "logs",
    "outputs",
}

FORBIDDEN_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".csv",
    ".xlsx",
    ".xls",
}

REQUIRED_FILES = {
    "README.md",
    ".gitignore",
    "requirements.txt",
    "requirements-optional.txt",
    "main_ablation.py",
    "main_proposal.py",
    "run.sh",
    "run_a.sh",
    "models/p_spga_block.py",
    "models/efficientnetv2_P1.py",
    "models/efficientnetv2_m3.py",
    "utils/data_utils.py",
    "utils/train_loop.py",
    "scripts/verify_integrity.py",
    "scripts/make_checksums.py",
}

EXCLUDED_FINAL_PROTOCOL_FILES = {
    "main_multiclass.py",
    "run_m.sh",
    "models/module_backup.py",
    "models/create_model.py",
    "models/create_safe_model.py",
    "utils/multiclass_data_utils.py",
    "utils/multiclass_train_loop.py",
    "1",
}


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part == ".git" for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def check_package(root: Path, max_file_mb: float) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for required in sorted(REQUIRED_FILES):
        if not (root / required).is_file():
            errors.append(f"missing required file: {required}")

    for excluded in sorted(EXCLUDED_FINAL_PROTOCOL_FILES):
        if (root / excluded).exists():
            errors.append(f"excluded final-protocol file is present: {excluded}")

    for path in root.rglob("*"):
        if any(part == ".git" for part in path.relative_to(root).parts):
            continue
        relative = rel(path, root)
        if path.is_symlink():
            errors.append(f"symlink found: {relative}")
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            errors.append(f"forbidden directory found: {relative}")

    max_bytes = int(max_file_mb * 1024 * 1024)
    for path in iter_files(root):
        relative = rel(path, root)
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_EXTENSIONS:
            errors.append(f"forbidden file extension found: {relative}")
        if path.stat().st_size > max_bytes:
            errors.append(
                f"oversized file found: {relative} "
                f"({path.stat().st_size / (1024 * 1024):.2f} MB)"
            )

    py_files = sorted(path for path in iter_files(root) if path.suffix == ".py")
    for path in py_files:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"syntax check failed for {rel(path, root)}: {exc}")
        except UnicodeDecodeError as exc:
            errors.append(f"UTF-8 decode failed for {rel(path, root)}: {exc}")

    if not py_files:
        errors.append("no Python files found")

    return errors, warnings


def build_report(root: Path, errors: list[str], warnings: list[str], max_file_mb: float) -> str:
    file_count = sum(1 for _ in iter_files(root))
    py_count = sum(1 for path in iter_files(root) if path.suffix == ".py")
    total_bytes = sum(path.stat().st_size for path in iter_files(root))
    status = "PASS" if not errors else "FAIL"
    lines = [
        "# Upload Verification",
        "",
        f"Status: {status}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Package root: `{root}`",
        f"File count: {file_count}",
        f"Python file count: {py_count}",
        f"Total size: {total_bytes / 1024:.1f} KB",
        f"Max file size rule: {max_file_mb:.1f} MB",
        "",
        "## Checks",
        "",
        "- Required files present",
        "- Final binary protocol files only",
        "- Forbidden data/checkpoint/result directories absent",
        "- Forbidden media/checkpoint/result file extensions absent",
        "- No symlinks",
        "- Python syntax check without writing bytecode",
        "",
    ]

    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")

    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    if not errors and not warnings:
        lines.extend(["## Result", "", "No integrity issues found.", ""])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the upload package is clean.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-file-mb", type=float, default=5.0)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    errors, warnings = check_package(root, args.max_file_mb)
    report = build_report(root, errors, warnings, args.max_file_mb)
    print(report)

    if args.write_report:
        (root / "UPLOAD_VERIFICATION.md").write_text(report + "\n", encoding="utf-8")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
