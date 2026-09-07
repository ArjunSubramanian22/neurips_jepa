#!/usr/bin/env python3
"""Scan submission-facing files for identity leaks. Never prints secret values."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_strengthening" / "12_anonymity"

SKIP_DIRS = {".git", ".mplconfig", "__pycache__", ".pytest_cache"}
SCAN_ROOTS = [
    ROOT / "results" / "reviewer_strengthening",
    ROOT / "results" / "PAPER_INTEGRATION_PHYSWORLD",
    ROOT / "REPRODUCIBILITY.md",
    ROOT / "reproduce_paper_results.py",
]
PATTERNS = [
    ("home_path", re.compile(r"/Users/[A-Za-z0-9._-]+")),
    ("unix_home", re.compile(r"/home/[A-Za-z0-9._-]+")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("github_url", re.compile(r"github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")),
    ("git_remote", re.compile(r"git@github\.com:")),
    ("api_key_assign", re.compile(r"(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]+['\"]", re.I)),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hits = []
    files = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            for path in root.rglob("*"):
                if any(p in SKIP_DIRS for p in path.parts):
                    continue
                if path.is_file() and path.suffix.lower() in {
                    ".md", ".txt", ".json", ".csv", ".tex", ".py", ".yml", ".yaml", ".sh",
                }:
                    files.append(path)
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for kind, pat in PATTERNS:
                if pat.search(line):
                    hits.append(
                        {
                            "file": str(path.relative_to(ROOT)),
                            "line": i,
                            "credential_type": kind,
                            "value": "REDACTED",
                        }
                    )
    (OUT / "anonymity_audit.md").write_text(
        "# Anonymity audit\n\n"
        f"Scanned {len(files)} submission-facing text files under results/reviewer_strengthening "
        "and the reproducibility entrypoint.\n\n"
        + (
            "## Hits\n\n"
            + "\n".join(
                f"- `{h['file']}` line {h['line']}: {h['credential_type']} REDACTED"
                for h in hits
            )
            if hits
            else "## Hits\n\n_none_\n"
        )
        + "\n\nInternal git commit hashes are allowed in reviewer_strengthening audit JSON. "
        "They must not appear in PAPER_INTEGRATION files if those files are uploaded to OpenReview.\n",
        encoding="utf-8",
    )
    print(f"ANONYMITY: {'FAIL' if hits else 'PASS'}  hits={len(hits)}", flush=True)
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
