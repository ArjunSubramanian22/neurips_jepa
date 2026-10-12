"""Assemble a reproducibility bundle. Huge arrays are hashed, not copied twice.

    python3.13 scripts/package_reproduction.py --mode small_artifact --output /tmp/repro_small
    python3.13 scripts/package_reproduction.py --mode full_artifact --output /tmp/repro_full
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml

# Files larger than this are hashed and listed, not copied, in either mode.
# Checkpoints under the limit are copied only by full_artifact, once.
FULL_COPY_MAX_BYTES = 50 * 1024 * 1024

COPY_GLOBS = (
    "requirements.txt",
    "configs/**/*.yaml",
    "configs/**/*.yml",
    "configs/**/*.json",
    "src/eval.py",
    "src/config.py",
    "src/registry.py",
    "src/metrics/**/*.py",
    "src/envs/**/*.py",
    "benchmark/**/*.py",
    "benchmark/README.md",
    "scripts/evaluate_final.py",
    "scripts/generate_trajectories.py",
    "scripts/generate_scientific_report.py",
    "scripts/render_qualitative_figures.py",
    "scripts/reproduce_all.sh",
    "scripts/train_reviewer_seed.sh",
    "scripts/plan_ablations.py",
    "scripts/package_reproduction.py",
    "src/repro_package.py",
    "reports/latest/**/*.csv",
    "reports/latest/**/*.tex",
    "reports/latest/RESULTS.md",
    "anonymous_release/LICENSE_AUDIT.md",
)

HASH_ONLY_GLOBS = (
    "outputs/physics_mpm/*.npy",
    "outputs/physics_mpm/*.json",
    "outputs/mujoco_arm/*",
    "experiments/runs/horizon_audit_v1/queries.csv",
    "experiments/runs/horizon_audit_v1/cache/frames_b.npy",
    "experiments/runs/horizon_audit_v1/cache/frames_a.npy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> dict[str, Any]:
    from src.registry import capture_git

    return capture_git(repo)


def _matches(repo: Path, pattern: str) -> list[Path]:
    return sorted(path for path in repo.glob(pattern) if path.is_file())


def checkpoint_paths(repo: Path) -> list[str]:
    registry = yaml.safe_load((repo / "configs/seeds/headline_v1.yaml").read_text(encoding="utf-8"))
    found = []
    for configuration in registry.get("configurations", []):
        for raw in configuration.get("checkpoints", {}).values():
            if raw not in (None, ""):
                found.append(str(raw))
    for extra in (
        "outputs/mujoco_arm_runs/jepa/last.pt",
        "outputs/mujoco_arm_runs/rssm/last.pt",
        "outputs/mujoco_arm_runs/latent_tf/last.pt",
    ):
        found.append(extra)
    return found


def reproduce_markdown(commit: str | None) -> str:
    shown = commit or "see MANIFEST.json"
    return f"""# Reproduce

Source commit: `{shown}`.

The project tree has no root LICENSE. `anonymous_release/LICENSE_AUDIT.md` records third-party status and says redistribution beyond confidential review is not authorized by that note. This bundle does not change that. MuJoCo, if installed, is Apache-2.0.

Hashes are SHA-256 in `MANIFEST.json`. `small_artifact` stores hashes of checkpoints and large arrays. `full_artifact` copies a file once when it is under {FULL_COPY_MAX_BYTES} bytes. Larger arrays stay hashed, with the command that regenerates them.

## 1. Environment setup

```bash
python3.13 -m pip install -r requirements.txt
python3.13 -m pip install 'mujoco==3.3.7'
```

The second install is optional and is only needed for the MuJoCo box and arm. Device selection is CUDA, then MPS, then CPU.

## 2. Baseline reproduction

```bash
SCALE=workshop DEVICE=auto OUTPUT_ROOT=outputs_repro bash scripts/reproduce_all.sh
```

This writes a new directory. It does not replace `outputs/`.

## 3. One lightweight smoke run

```bash
SCALE=smoke DEVICE=cpu OUTPUT_ROOT=outputs_repro bash scripts/reproduce_all.sh
```

The smoke scale builds a 48-episode dump and a short train. It is not a headline result.

## 4. Evaluator reproduction from supplied checkpoints

```bash
python3.13 scripts/evaluate_final.py --data outputs/physics_mpm --repo .
python3.13 -m benchmark.audit --config configs/benchmark/arm_jepa_audit.yaml
```

`evaluate_final.py` scores the legacy validation episodes 1800-1999. That split is not a test set. The audit command scores the arm checkpoint already named in the config.

## 5. Figure and table regeneration

```bash
MPLCONFIGDIR=outputs/.mplconfig python3.13 scripts/render_qualitative_figures.py
python3.13 scripts/generate_scientific_report.py
python3.13 scripts/plan_ablations.py
```

Figures read saved caches and tables. The ablation command is a dry run and does not train.
"""


def _entry(repo: Path, path: Path, *, copied: bool, reason: str) -> dict[str, Any]:
    relative = str(path.relative_to(repo))
    size = path.stat().st_size
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": size,
        "copied": copied,
        "reason": reason,
    }


def iter_package_files(repo: Path) -> list[Path]:
    found: list[Path] = []
    seen = set()
    for pattern in COPY_GLOBS:
        for path in _matches(repo, pattern):
            relative = str(path.relative_to(repo))
            if relative not in seen:
                seen.add(relative)
                found.append(path)
    return found


def collect_manifest(repo: Path, mode: str) -> dict[str, Any]:
    if mode not in {"small_artifact", "full_artifact"}:
        raise ValueError("mode must be small_artifact or full_artifact.")
    commit = git_commit(repo)
    files = []
    seen = set()

    def add(path: Path, *, copy_allowed: bool, role: str) -> None:
        relative = str(path.relative_to(repo))
        if relative in seen or not path.is_file():
            return
        seen.add(relative)
        size = path.stat().st_size
        if size > FULL_COPY_MAX_BYTES:
            copied = False
            reason = "larger_than_copy_limit"
        elif role == "source":
            copied = True
            reason = "copied"
        elif mode == "full_artifact" and copy_allowed:
            copied = True
            reason = "copied_once"
        else:
            copied = False
            reason = "hashed_only"
        entry = _entry(repo, path, copied=copied, reason=reason)
        entry["role"] = role
        files.append(entry)

    for path in iter_package_files(repo):
        add(path, copy_allowed=True, role="source")
    for pattern in HASH_ONLY_GLOBS:
        for path in _matches(repo, pattern):
            add(path, copy_allowed=True, role="data")
    for relative in checkpoint_paths(repo):
        add(repo / relative, copy_allowed=True, role="checkpoint")
    return {
        "mode": mode,
        "git": commit,
        "full_copy_max_bytes": FULL_COPY_MAX_BYTES,
        "files": files,
    }


def write_artifact(repo: Path, destination: Path, mode: str) -> dict[str, Any]:
    """Write one bundle. Existing destination files are not reused as a second copy."""
    repo = Path(repo)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {destination}.")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = collect_manifest(repo, mode)
    for entry in manifest["files"]:
        if not entry["copied"]:
            continue
        target = destination / entry["path"]
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / entry["path"], target)
    commit = (manifest.get("git") or {}).get("commit")
    (destination / "REPRODUCE.md").write_text(reproduce_markdown(commit), encoding="utf-8")
    notices = destination / "licenses"
    notices.mkdir(exist_ok=True)
    audit = repo / "anonymous_release" / "LICENSE_AUDIT.md"
    if audit.is_file():
        shutil.copy2(audit, notices / "LICENSE_AUDIT.md")
    (notices / "NOTICES.md").write_text(
        "Project code has no root LICENSE. See LICENSE_AUDIT.md. MuJoCo 3.3.7 is Apache-2.0 when installed.\n",
        encoding="utf-8",
    )
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
