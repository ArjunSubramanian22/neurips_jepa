"""Probe diagnosis runner (smoke vs physics dump).

Default paths match the audited ``outputs/`` tree. Pass ``--output-dir`` from
``scripts/reproduce_all.sh`` so a fresh tree is not mixed with that audit.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.dataset import build_dataset
from src.eval import evaluate_model, write_probe_diagnostics
from src.train import apply_dataset_path, load_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
DEVICE = torch.device("cpu")


def _eval_ckpt(ckpt: Path, *, dataset_path: str | None, name: str, max_horizon: int, probe_epochs: int) -> dict:
    model, cfg = load_model_from_checkpoint(ckpt, DEVICE)
    if dataset_path:
        cfg = apply_dataset_path(cfg, dataset_path, num_workers=0)
        cfg.training.device = "cpu"
    dataset = build_dataset(cfg.data, split="all")
    n_train, n_val = dataset.split_sizes()
    print(f"== {name}  ckpt={ckpt}  n_train={n_train} n_val={n_val}  path={cfg.data.dataset_path}")
    payload = evaluate_model(
        model,
        dataset,
        DEVICE,
        max_horizon=max_horizon,
        probe_epochs=probe_epochs,
        batch_size=8,
    )
    payload["checkpoint"] = str(ckpt)
    payload["dataset_path"] = cfg.data.dataset_path
    return payload


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write probe_diagnostics.json for available checkpoints.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Physics dump for the scaled probe. Defaults to <output-dir>/physics_mpm if present.",
    )
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument(
        "--audit-notes",
        action="store_true",
        help="Attach the frozen 2026-09-04 paper_limitation text (only valid for the audited outputs/ tree).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    dump = args.dataset_path
    if dump is None:
        candidate = output_dir / "physics_mpm"
        dump = str(candidate) if candidate.is_dir() else None

    results: dict = {}
    mse_ckpt = output_dir / "mse" / "last.pt"
    if mse_ckpt.is_file():
        results["mse_smoke"] = _eval_ckpt(
            mse_ckpt, dataset_path=None, name="mse_smoke", max_horizon=args.max_horizon, probe_epochs=args.probe_epochs
        )
    abl = output_dir / "ablation" / "full_vicreg" / "last.pt"
    if abl.is_file():
        results["vicreg_smoke"] = _eval_ckpt(
            abl, dataset_path=None, name="vicreg_smoke", max_horizon=args.max_horizon, probe_epochs=args.probe_epochs
        )

    src = output_dir / "vicreg" / "best.pt"
    if not src.is_file():
        src = output_dir / "vicreg" / "last.pt"
    if src.is_file() and dump and Path(dump).exists():
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "vicreg_best.pt"
            shutil.copy2(src, copied)
            results["vicreg_physics_mpm"] = _eval_ckpt(
                copied,
                dataset_path=dump,
                name="vicreg_physics_mpm",
                max_horizon=args.max_horizon,
                probe_epochs=args.probe_epochs,
            )

    extra: dict = {
        "physics_dump": {
            "path": dump,
            "exists": bool(dump and Path(dump).exists()),
        },
        "checkpoint_note": {
            "evaluated": sorted(results),
        },
    }
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        metrics_snapshot = json.loads(metrics_path.read_text(encoding="utf-8"))
        extra["metrics_json"] = {
            "path": str(metrics_path),
            "n_train": metrics_snapshot.get("vicreg", {}).get("n_train"),
            "n_val": metrics_snapshot.get("vicreg", {}).get("n_val"),
            "chance_stiffness": metrics_snapshot.get("vicreg", {}).get("chance_stiffness"),
            "chance_contact": metrics_snapshot.get("vicreg", {}).get("chance_contact"),
        }

    if args.audit_notes:
        extra["paper_limitation"] = (
            "Frozen audit note (2026-09-04): smoke probes in outputs/metrics.json use n_val=12 "
            "and are underpowered. Physics probes in the audited outputs/probe_diagnostics.json "
            "used an in-progress workshop VICReg checkpoint. See MAPPING.md."
        )

    out = output_dir / "probe_diagnostics.json"
    write_probe_diagnostics(results, out, extra=extra)
    print("wrote", out)
    return out


if __name__ == "__main__":
    main()
