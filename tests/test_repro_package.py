"""The small bundle hashes large files and the full bundle copies each one once."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.repro_package import FULL_COPY_MAX_BYTES, reproduce_markdown, sha256_file, write_artifact


def _mini_repo(root: Path) -> None:
    (root / "configs/seeds").mkdir(parents=True)
    (root / "configs/seeds/headline_v1.yaml").write_text(
        "seeds: [1]\nconfigurations:\n- id: toy\n  checkpoints:\n    1: weights/last.pt\n",
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text("numpy==0\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src/eval.py").write_text("print('eval')\n", encoding="utf-8")
    weights = root / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"checkpoint-bytes")
    huge = root / "outputs/physics_mpm"
    huge.mkdir(parents=True)
    (huge / "observations.npy").write_bytes(b"x" * (FULL_COPY_MAX_BYTES + 10))
    (root / "anonymous_release").mkdir()
    (root / "anonymous_release/LICENSE_AUDIT.md").write_text("no root license\n", encoding="utf-8")


class PackageTests(unittest.TestCase):
    def test_small_artifact_hashes_and_does_not_copy_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            dest = Path(tmp) / "small"
            _mini_repo(repo)
            manifest = write_artifact(repo, dest, "small_artifact")
            by_path = {entry["path"]: entry for entry in manifest["files"]}
            self.assertIn("weights/last.pt", by_path)
            self.assertFalse(by_path["weights/last.pt"]["copied"])
            self.assertFalse((dest / "weights/last.pt").exists())
            self.assertTrue((dest / "src/eval.py").is_file())
            self.assertEqual(by_path["weights/last.pt"]["sha256"], sha256_file(repo / "weights/last.pt"))
            self.assertEqual(by_path["src/eval.py"]["sha256"], hashlib.sha256(b"print('eval')\n").hexdigest())
            text = (dest / "REPRODUCE.md").read_text(encoding="utf-8")
            for heading in (
                "## 1. Environment setup",
                "## 2. Baseline reproduction",
                "## 3. One lightweight smoke run",
                "## 4. Evaluator reproduction from supplied checkpoints",
                "## 5. Figure and table regeneration",
            ):
                self.assertIn(heading, text)
            self.assertTrue((dest / "licenses/NOTICES.md").is_file())
            self.assertFalse(any(entry["copied"] and entry["bytes"] > FULL_COPY_MAX_BYTES for entry in manifest["files"]))

    def test_full_artifact_copies_a_small_checkpoint_once_and_skips_the_large_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            dest = Path(tmp) / "full"
            _mini_repo(repo)
            manifest = write_artifact(repo, dest, "full_artifact")
            by_path = {entry["path"]: entry for entry in manifest["files"]}
            self.assertTrue(by_path["weights/last.pt"]["copied"])
            self.assertTrue((dest / "weights/last.pt").is_file())
            self.assertEqual((dest / "weights/last.pt").read_bytes(), b"checkpoint-bytes")
            self.assertFalse(by_path["outputs/physics_mpm/observations.npy"]["copied"])
            self.assertEqual(by_path["outputs/physics_mpm/observations.npy"]["reason"], "larger_than_copy_limit")
            self.assertFalse((dest / "outputs/physics_mpm/observations.npy").exists())
            loaded = json.loads((dest / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(len(loaded["files"]), len({entry["path"] for entry in loaded["files"]}))

    def test_markdown_names_the_commit(self) -> None:
        text = reproduce_markdown("abc123")
        self.assertIn("`abc123`", text)
        self.assertNotIn("statistically significant", text)


if __name__ == "__main__":
    unittest.main()
