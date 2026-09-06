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
from src.train import apply_dataset_path, load_model_from_checkpoint, peek_checkpoint_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
DEVICE = torch.device("cpu")


def _eval_ckpt(
    ckpt: Path,
    *,
    dataset_path: str | None,
    name: str,
    max_horizon: int,
    probe_epochs: int,
    source_path: Path | None = None,
    metrics_jsonl_epochs: int = 0,
    require_final: bool = False,
    min_epochs: int = 200,
    start_frames: tuple[int, ...] = (0,),
) -> dict:
    source = Path(source_path) if source_path is not None else Path(ckpt)
    ckpt_epoch = peek_checkpoint_epoch(source)
    jsonl_epochs = int(metrics_jsonl_epochs)
    is_final = ckpt_epoch >= min_epochs and jsonl_epochs >= min_epochs
    if require_final and not is_final:
        raise SystemExit(
            f"refuse in-progress {name}: checkpoint={source} "
            f"checkpoint_epoch={ckpt_epoch} metrics.jsonl_train_epoch={jsonl_epochs} "
            f"(need both >= {min_epochs})"
        )
    model, cfg = load_model_from_checkpoint(ckpt, DEVICE)
    if dataset_path:
        cfg = apply_dataset_path(cfg, dataset_path, num_workers=0)
        cfg.training.device = "cpu"
    dataset = build_dataset(cfg.data, split="all")
    n_train, n_val = dataset.split_sizes()
    print(
        f"== {name}  source={source}  ckpt_epoch={ckpt_epoch}  "
        f"jsonl_epochs={jsonl_epochs}  final={is_final}  "
        f"n_train={n_train} n_val={n_val}  path={cfg.data.dataset_path}"
    )
    payload = evaluate_model(
        model,
        dataset,
        DEVICE,
        max_horizon=max_horizon,
        probe_epochs=probe_epochs,
        batch_size=8,
        start_frames=start_frames,
    )
    payload["checkpoint"] = str(ckpt)
    payload["checkpoint_source"] = str(source)
    payload["checkpoint_epoch"] = ckpt_epoch
    payload["metrics_jsonl_train_epochs"] = jsonl_epochs
    payload["final"] = is_final
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
        "--physics-only",
        action="store_true",
        help="Skip dummy-lattice smoke checkpoints; evaluate physics_mpm only.",
    )
    parser.add_argument(
        "--audit-notes",
        action="store_true",
        help="Attach the frozen 2026-09-04 paper_limitation text (only valid for the audited outputs/ tree).",
    )
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="Refuse to evaluate physics checkpoints unless last.pt epoch and metrics.jsonl are both >= 200.",
    )
    parser.add_argument(
        "--start-frames",
        default="0,3,5,7",
        help="Comma-separated rollout start indices. Frame 0 is the rest state; later "
        "starts encode after the object has already deformed.",
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
    dump_ok = bool(dump and Path(dump).exists())
    start_frames = tuple(int(part.strip()) for part in str(args.start_frames).split(",") if part.strip())
    if not start_frames:
        start_frames = (0,)
    if not args.physics_only:
        mse_ckpt = output_dir / "mse" / "last.pt"
        if mse_ckpt.is_file():
            results["mse_smoke"] = _eval_ckpt(
                mse_ckpt,
                dataset_path=None,
                name="mse_smoke",
                max_horizon=args.max_horizon,
                probe_epochs=args.probe_epochs,
            )
        abl = output_dir / "ablation" / "full_vicreg" / "last.pt"
        if abl.is_file():
            results["vicreg_smoke"] = _eval_ckpt(
                abl,
                dataset_path=None,
                name="vicreg_smoke",
                max_horizon=args.max_horizon,
                probe_epochs=args.probe_epochs,
            )

    def _best_or_last(run_dir: Path) -> Path | None:
        for name in ("best.pt", "last.pt"):
            path = run_dir / name
            if path.is_file():
                return path
        return None

    def _completed_epochs(metrics_path: Path) -> int:
        if not metrics_path.is_file():
            return 0
        best = 0
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("split") == "train_epoch":
                    best = max(best, int(rec["epoch"]))
        return best

    vic_ep = _completed_epochs(output_dir / "vicreg" / "metrics.jsonl")
    mse_ep = _completed_epochs(output_dir / "mse" / "metrics.jsonl")

    if dump_ok:
        for key, run_name, jsonl_ep in (
            ("vicreg_physics_mpm", "vicreg", vic_ep),
            ("mse_physics_mpm", "mse", mse_ep),
        ):
            run_dir = output_dir / run_name
            # Paper effective-rank uses last.pt (weights after the logged schedule),
            # not best.pt (which may be an earlier val snapshot).
            src = run_dir / "last.pt"
            if not src.is_file():
                src = _best_or_last(run_dir)
            if src is None:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                copied = Path(tmp) / f"{run_name}.pt"
                shutil.copy2(src, copied)
                results[key] = _eval_ckpt(
                    copied,
                    dataset_path=dump,
                    name=key,
                    max_horizon=args.max_horizon,
                    probe_epochs=args.probe_epochs,
                    source_path=src,
                    metrics_jsonl_epochs=jsonl_ep,
                    require_final=args.require_final,
                    start_frames=start_frames,
                )

    def _ckpt_note(name: str, jsonl_epochs: int) -> str:
        payload = results.get(name) or {}
        src = payload.get("checkpoint_source") or "missing"
        ckpt_ep = payload.get("checkpoint_epoch")
        is_final = payload.get("final")
        if name not in results:
            return f"Not evaluated ({name} checkpoint missing)."
        return (
            f"source={src} checkpoint_epoch={ckpt_ep} "
            f"metrics.jsonl_train_epoch={jsonl_epochs} final={is_final}"
        )

    extra: dict = {
        "physics_dump": {
            "path": dump,
            "exists": bool(dump and Path(dump).exists()),
        },
        "checkpoint_note": {
            "evaluated": sorted(results),
            "vicreg_epochs": vic_ep,
            "mse_epochs": mse_ep,
            "vicreg_physics_mpm": _ckpt_note("vicreg_physics_mpm", vic_ep),
            "mse_physics_mpm": _ckpt_note("mse_physics_mpm", mse_ep),
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
