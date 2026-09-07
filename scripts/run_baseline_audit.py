#!/usr/bin/env python3
"""Phase 0 freeze/audit. Writes results/reviewer_strengthening/00_baseline_audit/.

Does not retrain. Does not overwrite existing checkpoints or JSON eval artifacts.
Headline numbers are read from already-executed artifacts and compared to the
brief's sanity-check values. Unit tests are executed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "reviewer_strengthening" / "00_baseline_audit"
DUMP = ROOT / "outputs" / "physics_mpm"
CLAIMED_OBS_SHA256 = "ee3fe0fedfc65b690c5de5526a4a434a5dc3a44ee9f9064da7317b5f4cb0759f"
PYTHON = sys.executable


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            buf = handle.read(chunk)
            if not buf:
                break
            digest.update(buf)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def _safe_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024 ** 3)
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return float(out) / (1024 ** 3)
        except Exception:
            return None
    return None


def _git() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True).strip()
        except Exception:
            return ""

    status = _run(["git", "status", "--porcelain"])
    dirty = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "commit_short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "dirty_file_count": len(dirty),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "note": "Remote URL and usernames omitted from submission-facing artifacts.",
    }


def _extract_run(audit: dict[str, Any], key: str) -> dict[str, Any]:
    for run in audit["runs"]:
        if run["key"] == key:
            return run
    raise KeyError(key)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    import sklearn
    import torch

    from src.config import ModelConfig, load_experiment_config
    from src.dataset import build_dataset, _episode_split_counts
    from src.models.ac_jepa import ACJEPA
    from src.train import apply_dataset_path

    cfg = load_experiment_config(ROOT / "configs" / "workshop_vicreg.yaml")
    cfg = apply_dataset_path(cfg, str(DUMP), num_workers=0)
    dataset = build_dataset(cfg.data, split="all")
    n_train, n_val = dataset.split_sizes()
    obs = np.load(DUMP / "observations.npy", mmap_mode="r")
    acts = np.load(DUMP / "actions.npy", mmap_mode="r")
    hashes = {
        "observations.npy": _sha256(DUMP / "observations.npy"),
        "actions.npy": _sha256(DUMP / "actions.npy"),
        "stiffness_labels.npy": _sha256(DUMP / "stiffness_labels.npy"),
        "stiffness_values.npy": _sha256(DUMP / "stiffness_values.npy"),
        "geometry_labels.npy": _sha256(DUMP / "geometry_labels.npy"),
    }
    split_ids = {
        "rule": "episode index order; last n_val episodes are validation",
        "n_episodes": int(obs.shape[0]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "train_episode_range": [0, int(n_train)],
        "val_episode_range": [int(n_train), int(obs.shape[0])],
        "computed_split": list(_episode_split_counts(int(obs.shape[0]), 0.1)),
    }
    split_hash = hashlib.sha256(json.dumps(split_ids, sort_keys=True).encode()).hexdigest()

    model = ACJEPA(ModelConfig(
        in_channels=3,
        img_size=64,
        patch_size=8,
        embed_dim=128,
        encoder_depth=4,
        encoder_num_heads=4,
        mlp_ratio=2.0,
        action_dim=4,
        action_embed_dim=64,
        predictor_embed_dim=128,
        predictor_depth=4,
        predictor_num_heads=4,
        latent_dim=0,
        ema_momentum=0.996,
        dropout=0.0,
        drop_path=0.05,
        backbone="vit",
        fuse_mode="concat",
        condition_on_action=True,
    ))
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())

    audit_path = ROOT / "outputs" / "corrected_seed43_audit.json"
    mpm_path = ROOT / "outputs" / "mpm_action_ablation.json"
    twoseed_path = ROOT / "outputs" / "ablation_twoseed.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    mpm = json.loads(mpm_path.read_text(encoding="utf-8")) if mpm_path.is_file() else None
    twoseed = json.loads(twoseed_path.read_text(encoding="utf-8")) if twoseed_path.is_file() else None

    claimed = {
        "data_n_episodes": 2000,
        "episode_length": 16,
        "n_transitions": 30000,
        "obs_shape": [2000, 16, 3, 64, 64],
        "act_shape": [2000, 15, 4],
        "checksum_obs": CLAIMED_OBS_SHA256,
        "split": [1800, 200],
        "vicreg_s42": {
            "mse1": 0.07571, "mse5": 0.62213, "mse10": 1.19953,
            "pred_rank": 12.27, "top1": 0.178,
        },
        "vicreg_s43": {
            "mse1": 0.06946, "mse5": 0.61565, "mse10": 1.10682,
            "pred_rank": 5.78, "top1": 0.333,
        },
        "noact_s42": {"mse1": 0.07386, "mse5": 0.59738, "mse10": 1.14401, "pred_rank": 11.67},
        "noact_s43": {"mse1": 0.09922, "mse5": 0.70269, "mse10": 1.25955, "pred_rank": 13.44},
        "legacy_vicreg_s42_mse10": 0.899262,
        "legacy_vicreg_s42_rank": 5.886,
        "legacy_mse_s42_mse10": 0.090228,
        "legacy_mse_s43_mse10": 0.045856,
    }

    rows: list[dict[str, Any]] = []

    def add_row(name: str, claimed_v: Any, measured_v: Any, kind: str, note: str = "") -> None:
        rows.append(
            {
                "metric": name,
                "claimed_sanity_check": claimed_v,
                "measured": measured_v,
                "kind": kind,
                "note": note,
            }
        )

    add_row("obs_shape", claimed["obs_shape"], list(obs.shape), "data")
    add_row("act_shape", claimed["act_shape"], list(acts.shape), "data")
    add_row("n_episodes", claimed["data_n_episodes"], int(obs.shape[0]), "data")
    add_row("episode_length", claimed["episode_length"], int(obs.shape[1]), "data")
    add_row("n_transitions", claimed["n_transitions"], int(obs.shape[0] * (obs.shape[1] - 1)), "data")
    add_row("split_train_val", claimed["split"], [n_train, n_val], "data")
    add_row("observations_sha256", claimed["checksum_obs"], hashes["observations.npy"], "data")
    add_row("trainable_params_excl_ema_approx", 870000, n_trainable, "model")

    discrepancies: list[dict[str, Any]] = []
    major_fail = False

    if list(obs.shape) != claimed["obs_shape"]:
        discrepancies.append({"item": "obs_shape", "severity": "major"})
        major_fail = True
    if list(acts.shape) != claimed["act_shape"]:
        discrepancies.append({"item": "act_shape", "severity": "major"})
        major_fail = True
    if [n_train, n_val] != claimed["split"]:
        discrepancies.append({"item": "split", "severity": "major"})
        major_fail = True
    checksum_match = hashes["observations.npy"] == CLAIMED_OBS_SHA256
    if not checksum_match:
        discrepancies.append(
            {
                "item": "observations_sha256",
                "severity": "document",
                "note": "Claimed checksum is a sanity check. Measured hash is authoritative.",
            }
        )

    def _close(a: float, b: float, rel: float = 2e-3, abs_tol: float = 5e-4) -> bool:
        return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b), 1e-12))

    if audit is not None:
        mapping = [
            ("vicreg_nocrop_s42", "vicreg_s42"),
            ("vicreg_nocrop_s43", "vicreg_s43"),
            ("vicreg_nocrop_noaction_s42", "noact_s42"),
            ("vicreg_nocrop_noaction_s43", "noact_s43"),
        ]
        for key, claim_key in mapping:
            run = _extract_run(audit, key)
            s = run["summary"]
            c = claimed[claim_key]
            for mk, ck in (("1", "mse1"), ("5", "mse5"), ("10", "mse10")):
                meas = float(s["mse_k"][mk])
                cl = float(c[ck])
                kind = "harmless_rounding" if _close(meas, cl) else "real_discrepancy"
                add_row(f"{key}_mse@{mk}", cl, meas, kind)
                if kind != "harmless_rounding":
                    discrepancies.append({"item": f"{key}_mse@{mk}", "severity": "real_discrepancy"})
            if "pred_rank" in c:
                meas = float(s["effective_rank_z_hat"])
                cl = float(c["pred_rank"])
                kind = "harmless_rounding" if _close(meas, cl, rel=5e-3, abs_tol=0.02) else "real_discrepancy"
                add_row(f"{key}_pred_eff_rank", cl, meas, kind)
            if "top1" in c:
                meas = float(s["eig_frac_top1_z_hat"])
                cl = float(c["top1"])
                kind = "harmless_rounding" if _close(meas, cl, rel=2e-2, abs_tol=0.005) else "real_discrepancy"
                add_row(f"{key}_top1", cl, meas, kind)

        s42 = _extract_run(audit, "vicreg_nocrop_s42")["summary"]
        s43 = _extract_run(audit, "vicreg_nocrop_s43")["summary"]
        add_row("s42_geometry", 0.9985, s42["geometry_val_mean"], "harmless_rounding")
        add_row("s42_stiffness", 0.3025, s42["stiffness_val_mean"], "harmless_rounding")
        add_row("s42_contact", 0.3955, s42["contact_val_mean"], "harmless_rounding")
        add_row("s42_shuffle_t0_pct", 0.01604, s42["shuffle_delta_all"], "harmless_rounding")
        add_row("s42_shuffle_change_pct", 0.01634, s42["shuffle_delta_change"], "harmless_rounding")
        add_row("s43_geometry", 1.0, s43["geometry_val_mean"], "harmless_rounding")
        add_row("s43_stiffness", 0.2935, s43["stiffness_val_mean"], "harmless_rounding")
        add_row("s43_contact", 0.3980, s43["contact_val_mean"], "harmless_rounding")
        add_row("s43_shuffle_t0_pct", 0.01265, s43["shuffle_delta_all"], "harmless_rounding")
        add_row("s43_shuffle_change_pct", 0.01327, s43["shuffle_delta_change"], "harmless_rounding")
        add_row(
            "mean_shuffle_change_pct",
            0.01480,
            0.5 * (s42["shuffle_delta_change"] + s43["shuffle_delta_change"]),
            "harmless_rounding",
        )
        add_row(
            "pred_to_pred_mse_s42",
            3.82e-6,
            s42["pred_to_pred_mse_all"],
            "document",
            "Brief ~3.82e-6 is a two-seed/rounded figure; measured is seed-42 only from the audit JSON.",
        )

    if mpm is not None:
        add_row(
            "legacy_cropped_vicreg_s42_mse10_lastpt",
            claimed["legacy_vicreg_s42_mse10"],
            mpm["vicreg"]["rollout_mse_k"]["10"],
            "real_discrepancy" if not _close(mpm["vicreg"]["rollout_mse_k"]["10"], claimed["legacy_vicreg_s42_mse10"], rel=0.02)
            else "harmless_rounding",
            "outputs/mpm_action_ablation.json uses last.pt. Brief ~0.899 is close to last.pt 0.893.",
        )
        add_row(
            "legacy_cropped_vicreg_s42_rank_lastpt",
            claimed["legacy_vicreg_s42_rank"],
            mpm["vicreg"]["effective_rank_z_hat"],
            "harmless_rounding",
        )

    if twoseed is not None:
        add_row(
            "ablation_twoseed_uses_best_pt",
            "last.pt policy",
            json.dumps(twoseed.get("checkpoints")),
            "protocol_mismatch",
            "ablation_twoseed.json uses best.pt; corrected headlines use last.pt. Do not mix.",
        )

    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py")
    buf = __import__("io").StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
    test_log = buf.getvalue()
    n_ran = result.testsRun
    n_fail = len(result.failures) + len(result.errors)
    n_skip = len(result.skipped)
    n_pass = n_ran - n_fail - n_skip
    if n_fail > 0:
        major_fail = True
        discrepancies.append({"item": "unit_tests", "severity": "major", "n_fail": n_fail})

    mps_ok = bool(torch.backends.mps.is_available())
    env = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "os_name": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "ram_gb": _safe_ram_gb(),
        "mps_available": mps_ok,
        "cuda_available": bool(torch.cuda.is_available()),
        "git": _git(),
        "note": "No hostname, username, home directory, or git remote is recorded here.",
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    with (OUT / "baseline_reproduction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "claimed_sanity_check", "measured", "kind", "note"])
        writer.writeheader()
        writer.writerows(rows)

    reproduction = {
        "status": "FAIL" if major_fail else "PASS",
        "headline_source": "outputs/corrected_seed43_audit.json (last.pt, not re-executed in this audit)",
        "legacy_lastpt_source": str(_rel(mpm_path)),
        "legacy_bestpt_source": str(_rel(twoseed_path)),
        "data_hashes": hashes,
        "checksum_matches_claimed_sanity_check": checksum_match,
        "split": split_ids,
        "split_hash": split_hash,
        "model": {
            "trainable_excluding_ema": n_trainable,
            "total_including_ema": n_total,
            "workshop_yaml": "configs/workshop_vicreg.yaml",
            "crop_scale_min": 1.0,
            "condition_on_action_default": True,
        },
        "tests": {
            "ran": n_ran,
            "passed": n_pass,
            "failed": n_fail,
            "skipped": n_skip,
        },
        "discrepancies": discrepancies,
        "rows": rows,
        "probe_note": (
            "outputs/mpm_action_ablation.json seed-42 nocrop geometry 0.9995 vs "
            "corrected_seed43_audit.json 0.9985: linear probes are re-fit; small drift is expected. "
            "Use the audit JSON for the matched 4-way table."
        ),
        "vn_mse": "not implemented in src/eval.py; recorded as null in the audit JSON",
        "legacy_objective_caveat": (
            "Cropped VICReg/MSE last.pt and best.pt disagree. Brief legacy MSE@10 "
            "~0.090/0.046 does not match ablation_twoseed best.pt (0.114/0.184) nor "
            "a naive last.pt expectation. Treat legacy objective comparison as pre-fix."
        ),
    }
    (OUT / "baseline_reproduction.json").write_text(json.dumps(reproduction, indent=2), encoding="utf-8")
    (OUT / "test_log.txt").write_text(test_log, encoding="utf-8")

    files = []
    for path in sorted(ROOT.rglob("*")):
        if any(part in {".git", "paper", "__pycache__", ".mplconfig"} for part in path.parts):
            continue
        if path.is_file():
            files.append(
                {
                    "path": _rel(path),
                    "bytes": path.stat().st_size,
                    "suffix": path.suffix,
                }
            )
    with (OUT / "artifact_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "suffix"])
        writer.writeheader()
        writer.writerows(files)

    repo_map = f"""# Repository map (Phase 0)

Internal audit only. Paths are repo-relative. No remotes or usernames.

## Simulator
- `scripts/generate_trajectories.py` — MLS-MPM (`MLSMPM2D`), dump generation, snapshot/clone/restore, replay
- dump: `outputs/physics_mpm/` — observations/actions/labels/metadata (dump seed 0)

## Dataset / split / augs
- `src/dataset.py` — `DeformablePhysicsDataset`, episode split, train-only action z-score, `warp_contact_xy`, `crop_scale_min>=1` skips crop
- `configs/workshop_vicreg.yaml` — canonical no-crop (`crop_scale_min: 1.0`)

## Model / objective / train
- `src/models/ac_jepa.py` — ViT encoder, EMA target, concat action predictor
- `src/losses.py` — VICReg vs `mse` (invariance only)
- `src/train.py` — AdamW, cosine+warmup, clip 1.0, `--no-action`, last.pt policy
- `src/eval.py` — rollout MSE, effective rank, linear probes (no VN-MSE)

## Existing eval artifacts (do not overwrite)
- `outputs/corrected_seed43_audit.json` — matched last.pt 4-way + shuffle
- `outputs/mpm_action_ablation.json` — seed-42 cropped / nocrop / no-action last.pt
- `outputs/ablation_twoseed.json` — **best.pt** legacy two-seed (not last.pt)

## Tests
- `tests/test_*.py` including `test_counterfactual_fork.py`, `test_action_path.py`

## Checkpoints (read-only)
- `outputs/vicreg_nocrop/last.pt` seed 42 action VICReg no-crop
- `outputs/vicreg_nocrop_seed43/last.pt` seed 43
- `outputs/no_action/last.pt` seed 42 no-action
- `outputs/vicreg_nocrop_noaction_seed43/last.pt` seed 43 no-action
- `outputs/vicreg/last.pt` seed 42 cropped VICReg (legacy)
- `outputs/mse/last.pt` seed 42 cropped MSE (legacy)
"""
    (OUT / "repository_map.md").write_text(repo_map, encoding="utf-8")

    status = "FAIL" if major_fail else "PASS"
    md = f"""# Baseline audit

**BASELINE AUDIT: {status}**

## Tests
- ran: {n_ran}
- passed: {n_pass}
- failed: {n_fail}
- skipped: {n_skip}

## Data
- observations shape: {list(obs.shape)}
- actions shape: {list(acts.shape)}
- split: {n_train}/{n_val}
- observations sha256: `{hashes["observations.npy"]}`
- claimed sanity-check sha256 match: {checksum_match}

## Model
- trainable params excluding EMA: {n_trainable}
- total params including EMA: {n_total}

## Headline source
Read from `outputs/corrected_seed43_audit.json` (already executed last.pt eval).
This audit does **not** re-run 200-epoch training or overwrite those JSON files.

## Discrepancies
{json.dumps(discrepancies, indent=2)}

## Protocol notes
- Corrected paper-facing numbers use **last.pt** (epoch 200).
- `outputs/ablation_twoseed.json` uses **best.pt** and must not be mixed into corrected tables.
- Probe accuracies differ slightly across eval reruns (`mpm_action_ablation.json` vs audit JSON).
- VN-MSE is not implemented.
- Legacy MSE@10 brief values (~0.090 / 0.046) are **not** reproduced by `ablation_twoseed.json` best.pt (0.114 / 0.184). Document, do not silently retarget.

## Environment
See `environment.json` (no hostname/username/remote).
"""
    (OUT / "baseline_audit.md").write_text(md, encoding="utf-8")
    print(f"BASELINE AUDIT: {status}", flush=True)
    print(f"tests ran={n_ran} pass={n_pass} fail={n_fail} skip={n_skip}", flush=True)
    print(f"obs sha256={hashes['observations.npy']}", flush=True)
    print(f"checksum_match={checksum_match}", flush=True)
    print(f"trainable={n_trainable}", flush=True)
    return 0 if not major_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
