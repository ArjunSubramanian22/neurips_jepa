#!/usr/bin/env python3
"""Phase 11 artifact sizes and regeneration notes. No dump overwrite."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_strengthening" / "11_release_plan"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    targets = [
        ROOT / "outputs" / "physics_mpm" / "observations.npy",
        ROOT / "outputs" / "physics_mpm" / "actions.npy",
        ROOT / "outputs" / "physics_mpm" / "stiffness_labels.npy",
        ROOT / "outputs" / "physics_mpm" / "stiffness_values.npy",
        ROOT / "outputs" / "physics_mpm" / "geometry_labels.npy",
        ROOT / "outputs" / "vicreg_nocrop" / "last.pt",
        ROOT / "outputs" / "vicreg_nocrop_seed43" / "last.pt",
        ROOT / "outputs" / "no_action" / "last.pt",
        ROOT / "outputs" / "vicreg_nocrop_noaction_seed43" / "last.pt",
        ROOT / "outputs" / "corrected_seed43_audit.json",
    ]
    for path in targets:
        rows.append(
            {
                "path": str(path.relative_to(ROOT)) if path.exists() else str(path.name),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    import csv

    with (OUT / "artifact_sizes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "exists", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "minimum_reproduction_bundle.md").write_text(
        "# Minimum reproduction bundle\n\n"
        "## Without giant tensors\n"
        "- `results/reviewer_strengthening/` analysis CSVs/JSON/figures\n"
        "- `outputs/corrected_seed43_audit.json` and other metric JSON\n"
        "- code + `configs/workshop_vicreg.yaml` + tests\n\n"
        "## Requires checkpoints (~12 MB each last.pt)\n"
        "- rollout / probes / shuffle / Jacobian / latent counterfactuals\n\n"
        "## Requires observations.npy (~375 MB)\n"
        "- any encoding of real dump frames, probes, identifiability, eval\n\n"
        "## Dataset regeneration\n"
        "`python scripts/generate_trajectories.py --n-episodes 2000 --seed 0`\n"
        "Deterministic regeneration of the **full** dump is not claimed until a "
        "bitwise/hash comparison is executed (Phase 11 verification).\n",
        encoding="utf-8",
    )
    (OUT / "regeneration_manifest.json").write_text(
        json.dumps(
            {
                "dump_command": "python scripts/generate_trajectories.py --n-episodes 2000 --episode-length 16 --img-size 64 --seed 0 --output outputs/physics_mpm",
                "dump_seed": 0,
                "n_episodes": 2000,
                "full_bitwise_regen_tested": False,
                "subset_replay_tested": True,
                "subset_replay_test": "tests/test_counterfactual_fork.py dump bitwise match",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("release plan written", flush=True)


if __name__ == "__main__":
    main()
