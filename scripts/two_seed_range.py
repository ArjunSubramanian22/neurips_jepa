#!/usr/bin/env python3
"""Evaluate seed-42 and seed-43 VICReg / MSE checkpoints; write mean ± half-range.

Not a sample std. Output: outputs/ablation_twoseed.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.dataset import build_dataset
from src.eval import evaluate_model
from src.train import apply_dataset_path, load_model_from_checkpoint

LOGGER = logging.getLogger("ac_jepa.twoseed")
HORIZONS = (1, 5, 10)


def _ckpt(run_dir: Path) -> Path:
    for name in ("best.pt", "last.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No best.pt/last.pt in {run_dir}")


def _mse_at(payload: dict[str, Any], k: int) -> float:
    horizons = [int(h) for h in payload["horizons"]]
    values = [float(v) for v in payload["rollout_mse"]]
    return float(values[horizons.index(k)])


def _mean_half_range(a: float, b: float) -> tuple[float, float]:
    return 0.5 * (a + b), 0.5 * abs(a - b)


def _eval(ckpt: Path, dataset_path: str, device: torch.device, max_horizon: int) -> dict[str, Any]:
    model, cfg = load_model_from_checkpoint(ckpt, device)
    cfg = apply_dataset_path(cfg, dataset_path, num_workers=0)
    cfg.training.device = str(device)
    dataset = build_dataset(cfg.data, split="all")
    payload = evaluate_model(model, dataset, device, max_horizon=max_horizon, probe_epochs=80, batch_size=8)
    payload["checkpoint"] = str(ckpt)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--dataset-path", default="outputs/physics_mpm")
    parser.add_argument("--seed42-root", default="outputs")
    parser.add_argument("--seed43-root", default="outputs/seed43")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-horizon", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    pairs = {
        "full_vicreg": ("vicreg", "vicreg"),
        "mse": ("mse", "mse"),
    }
    seeds: dict[str, dict[int, dict[str, Any]]] = {key: {} for key in pairs}
    for key, (dir42, dir43) in pairs.items():
        seeds[key][42] = _eval(
            _ckpt(Path(args.seed42_root) / dir42),
            args.dataset_path,
            device,
            args.max_horizon,
        )
        seeds[key][43] = _eval(
            _ckpt(Path(args.seed43_root) / dir43),
            args.dataset_path,
            device,
            args.max_horizon,
        )

    variants: dict[str, Any] = {}
    for key, by_seed in seeds.items():
        by_horizon: dict[str, Any] = {}
        for k in HORIZONS:
            a = _mse_at(by_seed[42], k)
            b = _mse_at(by_seed[43], k)
            mean, half = _mean_half_range(a, b)
            by_horizon[str(k)] = {
                "rollout_mse_mean": mean,
                "rollout_mse_range": half,
                "seed42": a,
                "seed43": b,
            }
        variants[key] = {"by_horizon": by_horizon}

    report = {
        "n_seeds": 2,
        "seeds": [42, 43],
        "stat": "mean_plus_half_range",
        "note": "Entries are mean ± (max-min)/2 over seeds 42 and 43. This is a 2-seed range, not a sample standard deviation.",
        "dataset_path": args.dataset_path,
        "variants": variants,
        "checkpoints": {
            "seed42_vicreg": str(_ckpt(Path(args.seed42_root) / "vicreg")),
            "seed42_mse": str(_ckpt(Path(args.seed42_root) / "mse")),
            "seed43_vicreg": str(_ckpt(Path(args.seed43_root) / "vicreg")),
            "seed43_mse": str(_ckpt(Path(args.seed43_root) / "mse")),
        },
    }
    out = output_dir / "ablation_twoseed.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return report


if __name__ == "__main__":
    main()
