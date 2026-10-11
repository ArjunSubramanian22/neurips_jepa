#!/usr/bin/env python3
"""Regenerate the cascade smoke figure from saved pairs.json.

    python3.13 scripts/plot_cascade_layers.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.plots import plot_cascade_pairs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "experiments" / "runs" / "cascade_protocol_smoke_v1" / "pairs.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "runs" / "cascade_protocol_smoke_v1" / "cascade_layers.png",
    )
    args = parser.parse_args(argv)
    payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    plot_cascade_pairs(payload["pairs"], args.out)
    print(args.out)


if __name__ == "__main__":
    main()
