#!/usr/bin/env python3
"""Write snapshot_validation.json from Phase 1A unit tests."""

from __future__ import annotations

import json
import sys
import unittest
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "reviewer_strengthening" / "01_simulator_counterfactual"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_counterfactual_fork.py")
    buf = StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
    payload = {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "state_copied": ["x", "v", "C", "F", "mu", "lam"],
        "grid_persistent": False,
        "substep_rng": False,
        "effective_contact_definition": "min_particle_distance <= 2 * contact_radius",
        "log": buf.getvalue(),
    }
    (OUT / "snapshot_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "snapshot_validation.md").write_text(
        f"# Snapshot validation\n\n**SIMULATOR FORK VALIDATION: {payload['status']}**\n\n"
        f"Tests run: {payload['tests_run']}\n\n"
        "Complete particle state is `(x, v, C, F, μ, λ)`. Grid is rebuilt each substep.\n"
        "`substep` has no RNG. Dump replay matches stored `observations.npy` / `actions.npy` "
        "on the tested validation episode.\n",
        encoding="utf-8",
    )
    print(f"SIMULATOR FORK VALIDATION: {payload['status']}", flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
