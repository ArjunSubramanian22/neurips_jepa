#!/usr/bin/env python3
"""Enumerate attached ZIP archives for the Med-Reasoner audit. Read-only."""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

ZIPS = [
    Path("/Users/Arjun/Downloads/HealthyCircle (1).zip"),
    Path("/Users/Arjun/Downloads/HealthySpiral (1).zip"),
    Path("/Users/Arjun/Downloads/PatientCircle (1).zip"),
    Path("/Users/Arjun/Downloads/PatientSpiral.zip"),
    Path("/Users/Arjun/Downloads/PatientSignal.zip"),
    Path("/Users/Arjun/Downloads/gait-in-neurodegenerative-disease-database-1.0.0 (1).zip"),
    Path("/Users/Arjun/Downloads/gait-in-parkinsons-disease-1.0.0.zip"),
]


def top_level(name: str) -> str:
    return name.split("/")[0] if name else ""


def enumerate_zip(path: Path) -> dict:
    rec: dict = {
        "filename": path.name,
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
    }
    if not path.is_file():
        return rec
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        files = [i for i in infos if not i.is_dir()]
        nested = [i.filename for i in files if Path(i.filename).suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".7z"}]
        exts = Counter(Path(i.filename).suffix.lower() or "(none)" for i in files)
        tops = Counter(top_level(n.rstrip("/")) for n in names)
        rec.update(
            {
                "n_members": len(infos),
                "n_files": len(files),
                "n_nested_archives": len(nested),
                "nested_archives": nested[:50],
                "top_level": dict(tops),
                "ext_counts": dict(exts.most_common(30)),
                "sample_paths": names[:80],
                "comment": (zf.comment or b"").decode("utf-8", "replace"),
            }
        )
        # README-like files
        readmes = [i.filename for i in files if Path(i.filename).name.lower() in {
            "readme", "readme.txt", "readme.md", "annotation.txt", "records", "sha256sums.txt",
            "subject-description.txt", "rems", "sha1sums.txt",
        } or "readme" in Path(i.filename).name.lower()]
        rec["readme_like"] = readmes[:40]
        rec["largest_files"] = [
            {"name": i.filename, "bytes": i.file_size}
            for i in sorted(files, key=lambda x: x.file_size, reverse=True)[:15]
        ]
    return rec


def main() -> None:
    out = [enumerate_zip(p) for p in ZIPS]
    dest = Path("docs/_tmp_zip_enum.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps([{k: r.get(k) for k in ("filename", "bytes", "n_files", "top_level", "ext_counts", "n_nested_archives")} for r in out], indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
