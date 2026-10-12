"""Package a small or full reproducibility bundle. See src/repro_package.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.repro_package import write_artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assemble a reproducibility artifact.")
    parser.add_argument("--mode", choices=("small_artifact", "full_artifact"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    manifest = write_artifact(args.repo, args.output, args.mode)
    copied = sum(1 for entry in manifest["files"] if entry["copied"])
    hashed = len(manifest["files"])
    print(f"mode {args.mode} files {hashed} copied {copied}")
    print(args.output / "MANIFEST.json")


if __name__ == "__main__":
    main()
