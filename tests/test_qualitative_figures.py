"""A synthetic fork writes PDF, SVG, PNG, and panel metadata."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qualitative_figures import PANEL_FIELDS, fork_sequence_figure, panel_meta, save_figure


class FigureTests(unittest.TestCase):
    def test_saved_files_name_the_panel_source(self) -> None:
        current = np.zeros((3, 8, 8), dtype=np.uint8)
        recorded = current.copy()
        recorded[0] = 40
        alternative = current.copy()
        alternative[1] = 80
        figure = fork_sequence_figure([("raster", current, recorded, alternative)])
        meta = {
            "figure": "fork_sequence",
            "panels": [
                panel_meta(
                    environment="mpm_legacy",
                    episode=1801,
                    time=0,
                    seed=42,
                    checkpoint="vicreg_action_s42",
                    intervention="force_reversal",
                    row="raster",
                    column="recorded",
                )
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "fork"
            written = save_figure(figure, stem, meta)
            for suffix in (".pdf", ".svg", ".png", ".json"):
                self.assertTrue(Path(written["pdf"]).with_suffix(suffix).is_file())
            payload = json.loads(Path(written["metadata"]).read_text(encoding="utf-8"))
        for field in PANEL_FIELDS:
            self.assertIn(field, payload["panels"][0])


if __name__ == "__main__":
    unittest.main()
