"""Write reports/latest from the saved experiment files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scientific_report import write_report


def main() -> None:
    report = write_report(ROOT, ROOT / "reports" / "latest")
    print(f"rows {report['n_rows']} flags {report['n_flags']}")
    print(ROOT / "reports" / "latest" / "RESULTS.md")


if __name__ == "__main__":
    main()
