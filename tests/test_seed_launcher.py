"""The headline seed registry keeps every seed, including ones that have not run."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.launch_seed_sweep import (
    FAILED,
    NOT_RUN,
    launch_not_run,
    load_seed_registry,
    plan_sweep,
    write_sweep_status,
)


class SeedLauncherTests(unittest.TestCase):
    def test_registry_contains_five_fixed_seeds_for_every_configuration(self) -> None:
        registry = load_seed_registry(ROOT / "configs" / "seeds" / "headline_v1.yaml")
        self.assertEqual(registry["seeds"], [42, 43, 44, 45, 46])
        rows = plan_sweep(registry, ROOT)
        self.assertEqual(len(rows), 15)
        for configuration in ("vicreg_action", "vicreg_noaction", "mse_action"):
            seeds = [row["seed"] for row in rows if row["configuration"] == configuration]
            self.assertEqual(seeds, [42, 43, 44, 45, 46])
        missing = [row for row in rows if row["status"] == NOT_RUN]
        self.assertIn(46, [row["seed"] for row in missing])

    def test_failed_launch_stays_in_the_plan_and_the_log(self) -> None:
        registry = {
            "seed_registry_id": "headline_v1",
            "seeds": [42, 43, 44, 45, 46],
            "minimum_complete_seeds": 5,
            "split_version": "legacy_v1",
            "is_test_set": False,
            "configurations": [
                {
                    "id": "vicreg_action",
                    "objective": "vicreg",
                    "condition_on_action": True,
                    "checkpoints": {seed: None for seed in (42, 43, 44, 45, 46)},
                }
            ],
        }
        planned = plan_sweep(registry, ROOT)
        self.assertTrue(all(row["status"] == NOT_RUN for row in planned))

        def runner(row: dict) -> None:
            if row["seed"] == 46:
                raise RuntimeError("training crashed")

        launched = launch_not_run(planned, runner)
        failed = [row for row in launched if row["seed"] == 46]
        self.assertEqual(len(launched), 5)
        self.assertEqual(failed[0]["status"], FAILED)
        self.assertIn("training crashed", failed[0]["failure_reason"])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_sweep_status(directory, launched, registry)
            write_sweep_status(directory, launched, registry)
            lines = (directory / "registry.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["n_failed"], 1)
            self.assertEqual([slot["seed"] for slot in first["slots"]], [42, 43, 44, 45, 46])


if __name__ == "__main__":
    unittest.main()
