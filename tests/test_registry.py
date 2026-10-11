"""Registry records keep failures and refuse to overwrite a finished run."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry import RunRecorder


GIT = {"commit": "abc123", "dirty": False, "dirty_count": 0, "status": "measured"}


def _recorder(root: Path, experiment_id: str) -> RunRecorder:
    return RunRecorder(
        experiment_id,
        repo_root=root,
        config={"protocol": "unit"},
        command="python -m unittest",
        seed=0,
        model={"name": "none"},
        environment={"name": "synthetic"},
        dataset_split={"protocol_id": "unit", "test_defined": False},
        checkpoint={"path": None, "note": "no checkpoint"},
        git_state=GIT,
    )


class RegistryTests(unittest.TestCase):
    def test_complete_appends_one_registry_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with _recorder(root, "unit_complete") as run:
                run.complete({"value": 1})
            record = json.loads((root / "experiments" / "runs" / "unit_complete" / "record.json").read_text())
            lines = (root / "experiments" / "registry.jsonl").read_text().splitlines()
            self.assertEqual(record["status"], "COMPLETE")
            self.assertEqual(record["metrics"], {"value": 1})
            self.assertEqual(record["git_commit"], "abc123")
            self.assertIsNone(record["failure_reason"])
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["status"], "COMPLETE")
            self.assertTrue((root / "experiments" / "runs" / "unit_complete" / "config.json").is_file())

    def test_exception_is_stored_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                with _recorder(root, "unit_fail") as run:
                    raise RuntimeError("boom")
                    _ = run
            record = json.loads((root / "experiments" / "runs" / "unit_fail" / "record.json").read_text())
            self.assertEqual(record["status"], "FAILED")
            self.assertIn("boom", record["failure_reason"])
            self.assertIsNone(record["metrics"])
            lines = (root / "experiments" / "registry.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(lines[0])["status"], "FAILED")

    def test_existing_id_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _recorder(root, "unit_once")
            run.complete({"value": 2})
            with self.assertRaises(FileExistsError):
                _recorder(root, "unit_once")
            record = json.loads((root / "experiments" / "runs" / "unit_once" / "record.json").read_text())
            self.assertEqual(record["metrics"], {"value": 2})


if __name__ == "__main__":
    unittest.main()
