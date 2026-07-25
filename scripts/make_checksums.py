from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_NAMES = {
    ".git",
    "__pycache__",
    "checksums.sha256",
}


def should_skip(path: Path, root: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part in EXCLUDED_NAMES for part in relative_parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create package SHA256 checksums.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default="checksums.sha256")
    args = parser.parse_args()

    root = args.root.resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not should_skip(path, root):
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")

    output_path = root / args.output
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {output_path} ({len(rows)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

