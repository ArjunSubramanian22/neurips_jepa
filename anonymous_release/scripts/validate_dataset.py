"""Validate a physics dump through :func:`src.dataset.build_dataset`.

Checks the AC-JEPA data contract used by :class:`DeformablePhysicsDataset`:

* Frame / action shapes match ModelConfig defaults (64×64 RGB, action_dim=4)
* ``stiffness_labels`` are a balanced ``n_stiffness_classes``-way codebook
* Observations and actions contain no NaNs or Infs
* Optional ``geometry_labels.npy`` covers at least two object shapes

Run::

    python scripts/validate_dataset.py --dataset-path outputs/physics_mpm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DataConfig, ModelConfig
from src.dataset import build_dataset, inspect_physics_source


def _disk_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_physics_dataset(
    dataset_path: str | Path,
    *,
    img_size: int | None = None,
    in_channels: int | None = None,
    action_dim: int | None = None,
    n_stiffness_classes: int = 3,
    min_episodes: int = 1,
    min_geometries: int = 2,
    balance_min_frac: float = 0.18,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load ``dataset_path`` via ``build_dataset`` and assert the training contract.

    Returns a JSON-serializable stats dict (also written to ``report_path``).
    """
    dataset_path = Path(dataset_path)
    _assert(dataset_path.exists(), f"Dataset path does not exist: {dataset_path}")
    model_defaults = ModelConfig()
    img_size = int(img_size if img_size is not None else model_defaults.img_size)
    in_channels = int(in_channels if in_channels is not None else model_defaults.in_channels)
    action_dim = int(action_dim if action_dim is not None else model_defaults.action_dim)
    _assert(img_size == 64, f"Expected ModelConfig default img_size=64, got check value {img_size}.")
    _assert(in_channels == 3, f"Expected ModelConfig default in_channels=3, got {in_channels}.")
    _assert(action_dim == 4, f"Expected ModelConfig default action_dim=4, got {action_dim}.")

    meta = inspect_physics_source(dataset_path)
    native_t = int(meta["episode_length"])
    cfg = DataConfig(
        dataset_path=str(dataset_path),
        img_size=img_size,
        in_channels=in_channels,
        action_dim=action_dim,
        n_stiffness_classes=n_stiffness_classes,
        chunk_length=min(16, native_t),
        chunk_stride=1,
        augment=False,
        normalize_actions=False,
        val_fraction=0.1,
        pin_memory=False,
    )
    dataset = build_dataset(cfg, split="all")
    observations = dataset.observations
    actions = dataset.actions
    labels = dataset.stiffness_labels

    _assert(observations.ndim == 5, f"observations ndim={observations.ndim}, expected 5.")
    n_ep, t_len, c, h, w = (int(v) for v in observations.shape)
    _assert(n_ep >= min_episodes, f"n_episodes={n_ep} < min_episodes={min_episodes}.")
    _assert(t_len >= 2, f"episode_length={t_len} must be >= 2.")
    _assert(c == in_channels, f"in_channels={c} != {in_channels}.")
    _assert(h == img_size and w == img_size, f"frames are {h}x{w}, expected {img_size}x{img_size}.")
    _assert(tuple(actions.shape) == (n_ep, t_len - 1, action_dim), f"actions shape {tuple(actions.shape)}.")
    _assert(int(labels.shape[0]) == n_ep, f"stiffness_labels {tuple(labels.shape)} != N={n_ep}.")

    sample = dataset[0]
    _assert(tuple(sample[0].shape) == (in_channels, img_size, img_size), f"x_t {tuple(sample[0].shape)}")
    _assert(tuple(sample[1].shape) == (action_dim,), f"a_t {tuple(sample[1].shape)}")
    _assert(tuple(sample[2].shape) == (in_channels, img_size, img_size), f"x_tp1 {tuple(sample[2].shape)}")

    obs_f = observations.float()
    act_f = actions.float()
    _assert(bool(torch.isfinite(obs_f).all()), "NaN/Inf in observations.")
    _assert(bool(torch.isfinite(act_f).all()), "NaN/Inf in actions.")

    class_ids = labels.cpu().numpy().astype(np.int64).reshape(-1)
    n_present = int(class_ids.max()) + 1 if class_ids.size else 0
    _assert(n_present >= n_stiffness_classes, f"found {n_present} stiffness classes, need {n_stiffness_classes}.")
    counts = np.bincount(class_ids, minlength=n_stiffness_classes)[:n_stiffness_classes]
    fracs = counts / max(int(counts.sum()), 1)
    _assert(
        bool((fracs >= balance_min_frac).all()),
        f"stiffness class balance {fracs.tolist()} fell below {balance_min_frac}.",
    )

    geom_path = dataset_path / "geometry_labels.npy" if dataset_path.is_dir() else None
    n_geom = 0
    geom_counts: list[int] = []
    if geom_path is not None and geom_path.is_file():
        geom = np.load(geom_path)
        _assert(int(geom.shape[0]) == n_ep, "geometry_labels length mismatch.")
        n_geom = int(np.unique(geom).size)
        geom_counts = np.bincount(geom.astype(np.int64)).tolist()
        _assert(n_geom >= min_geometries, f"need >= {min_geometries} geometries, found {n_geom}.")

    contact = act_f[..., :2]
    _assert(bool((contact >= -0.05).all() and (contact <= 1.05).all()), "contact xy expected in ~[0, 1].")

    disk = _disk_bytes(dataset_path)
    report: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "loader": type(dataset).__name__,
        "n_episodes": n_ep,
        "episode_length": t_len,
        "img_size": img_size,
        "in_channels": in_channels,
        "action_dim": action_dim,
        "n_stiffness_classes": n_stiffness_classes,
        "stiffness_counts": counts.tolist(),
        "stiffness_fractions": [float(v) for v in fracs],
        "n_geometries": n_geom,
        "geometry_counts": geom_counts,
        "n_transitions": int(len(dataset)),
        "disk_bytes": disk,
        "disk_mb": round(disk / (1024 * 1024), 3),
        "finite_observations": True,
        "finite_actions": True,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {report_path}")
    print(
        f"OK  N={n_ep} T={t_len} C={in_channels} H=W={img_size} A={action_dim}  "
        f"stiffness={counts.tolist()}  disk={report['disk_mb']} MB"
    )
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an AC-JEPA physics dump.")
    parser.add_argument("--dataset-path", default="outputs/physics_mpm")
    parser.add_argument("--report", default="outputs/dataset_report.json")
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--n-stiffness-classes", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return validate_physics_dataset(
        args.dataset_path,
        n_stiffness_classes=args.n_stiffness_classes,
        min_episodes=args.min_episodes,
        report_path=args.report,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
