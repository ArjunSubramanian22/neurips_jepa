#!/usr/bin/env python3
"""Run the scientific regression suite.

These tests check simulator, rendering, data, metric, model, planning, and
canonical-evaluator invariants. They do not require a training run to match
a stored loss.

    python3.13 -m unittest discover -s tests/scientific -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tests" / "scientific"),
        pattern="test_*.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
