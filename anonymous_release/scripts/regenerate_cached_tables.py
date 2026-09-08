#!/usr/bin/env python3
"""Rebuild paper-facing tables from cached CSVs/JSON. Does not train or load checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "regenerated"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def heldout_table() -> Path:
    rows = _load_csv(ROOT / "results" / "final_checkpoint_results.csv")
    cols = [
        "model",
        "seed",
        "MSE@1",
        "MSE@5",
        "MSE@10",
        "VN-MSE@1",
        "VN-MSE@5",
        "VN-MSE@10",
        "pred_rank@1",
        "pred_rank@10",
        "target_rank@1",
        "geometry_probe",
        "stiffness_probe",
        "contact_probe",
    ]
    lines = [
        "# Held-out checkpoint metrics (cached)",
        "",
        "Source: `results/final_checkpoint_results.csv`.",
        "Evaluator: `scripts/evaluate_final.py` (validation episodes 1800:2000).",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells = []
        for key in cols:
            val = row[key]
            if key in {"model", "seed"}:
                cells.append(val)
            else:
                cells.append(f"{float(val):.6g}")
        lines.append("| " + " | ".join(cells) + " |")
    dest = OUT / "heldout_eval_table.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def phase1_summary() -> Path:
    payload = json.loads(
        (ROOT / "results" / "audit_phase1" / "phase1_summary.json").read_text(encoding="utf-8")
    )
    dest = OUT / "phase1_summary_excerpt.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    table = heldout_table()
    excerpt = phase1_summary()
    print("wrote", table.relative_to(ROOT))
    print("wrote", excerpt.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
