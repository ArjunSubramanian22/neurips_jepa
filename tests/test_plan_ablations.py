"""The dry-run prints commands and will not replace an existing plan file."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_main():
    spec = importlib.util.spec_from_file_location("plan_ablations", ROOT / "scripts" / "plan_ablations.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.main


class PlanAblationCommandTests(unittest.TestCase):
    def test_dry_run_appends_a_ledger_and_refuses_an_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            output = Path(tmp) / "dry_run.json"
            main = _load_main()
            text = main(["--ledger", str(ledger), "--output", str(output), "--tier", "1"])
            self.assertIn("jobs 1", text)
            self.assertIn("COMPLETE=1", text)
            second = main(["--ledger", str(ledger), "--tier", "1"])
            self.assertIn("COMPLETE", second)
            lines = ledger.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["summary"]["n_jobs"], 1)
            output.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(["--ledger", str(ledger), "--output", str(output), "--tier", "1"])


if __name__ == "__main__":
    unittest.main()
