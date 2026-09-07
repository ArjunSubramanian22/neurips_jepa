#!/usr/bin/env python3
"""Evaluate a reviewer-strengthening checkpoint with the same last.pt protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.corrected_seed43_audit import eval_checkpoint  # noqa: E402
from src.config import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expect-action", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-path", default="outputs/physics_mpm")
    args = parser.parse_args()
    spec = {
        "key": Path(args.ckpt).parent.name,
        "label": args.label,
        "seed": args.seed,
        "ckpt": Path(args.ckpt),
        "expect_action": bool(args.expect_action),
    }
    device = resolve_device(args.device)
    payload = eval_checkpoint(
        spec,
        args.dataset_path,
        device,
        probe_epochs=80,
        batch_size=8,
        run_counterfactual=True,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # strip absolute paths
    ckpt_path = Path(args.ckpt)
    try:
        ckpt_rel = str(ckpt_path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        ckpt_rel = ckpt_path.name
    payload["checkpoint"] = ckpt_rel
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload.get("summary", payload), indent=2), flush=True)


if __name__ == "__main__":
    main()
