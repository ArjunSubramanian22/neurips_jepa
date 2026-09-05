"""Within-simulator external baselines vs AC-JEPA.

Trains every method on one frozen mass-spring split and scores them with
:func:`src.eval.evaluate_model` at horizons ``k = 1..10``.

Methods
-------
* AC-JEPA (VICReg / no-cov / MSE / pixel AE) — existing in-repo family.
* No-action predictor — ``g(z_t)`` with ``ModelConfig.condition_on_action=False``.
* Particle GNN — GNS/DPI-Net-style message passing on ``_edge_list``.

This is **not** a comparison against PhysWorld or PhysTwin on captured video.
Those systems use real observations and MPM-optimized digital twins; we only
compare world-model families inside the same synthetic lattice simulator.

Run::

    python src/baselines.py
    python src/baselines.py --epochs 8 --n-episodes 160
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .config import ExperimentConfig, seed_everything
    from .dataset import DeformableObjectDataset, _edge_list
    from .eval import evaluate_model
    from .train import ACJEPATrainer, build_smoke_config
else:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.config import ExperimentConfig, seed_everything
    from src.dataset import DeformableObjectDataset, _edge_list
    from src.eval import evaluate_model
    from src.train import ACJEPATrainer, build_smoke_config

LOGGER = logging.getLogger("ac_jepa.baselines")

COMPARISON_HORIZONS: tuple[int, ...] = tuple(range(1, 11))

README_TEXT = """# Within-simulator baseline comparison

This table compares world-model families **inside the same mass-spring
lattice simulator** (the in-process `DataConfig.n_grid` particles and
`_edge_list` spring connectivity in `src/dataset.py`).

## What is being compared

| method | family | dynamics |
| --- | --- | --- |
| `ac_jepa_vicreg` | AC-JEPA | action-conditioned latent predictor + VICReg |
| `ac_jepa_no_cov` | AC-JEPA ablation | VICReg without covariance |
| `ac_jepa_mse` | AC-JEPA ablation | latent MSE only |
| `pixel_ae` | pixel autoencoder | decode `Predictor(z_t, a_t)` in image space |
| `no_action` | AC-JEPA ablation | predictor sees `z_t` only (`condition_on_action=False`) |
| `particle_gnn` | GNS / DPI-Net family | message passing on `_edge_list`, next-position MSE |

All rows use the **same held-out episodes**, the **same** `evaluate_model`
metrics (`rollout_mse`, `stiffness_probe_acc`, `contact_probe_acc`), and
horizons `k = 1 … 10`.

The particle GNN is scored in its pooled graph-readout space (not the ViT
latent). Magnitudes of `rollout_mse` are therefore **not** interchangeable
across methods; ranking within a method over horizon, and probe accuracies
across methods, are the fair columns.

## Domain gap (limitation)

Do **not** read this table as state-of-the-art vs [PhysWorld](https://physworld.github.io/)
or PhysTwin. Those systems:

* train / evaluate on **real captured video** of deformable objects, and
* couple a learned model to an **MPM-optimized digital twin**.

Our GNN sees privileged particle states from a synthetic Hookean lattice.
That is a different data regime. This file is a *within-simulator* sanity
check that AC-JEPA is compared against an established particle-graph
baseline family, not a claim of outperforming those papers on their
captures.
"""


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    label: str
    loss_type: str
    condition_on_action: bool = True


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec("ac_jepa_vicreg", "AC-JEPA (VICReg)", "vicreg"),
    BaselineSpec("ac_jepa_no_cov", "AC-JEPA (no covariance)", "vicreg_no_cov"),
    BaselineSpec("ac_jepa_mse", "AC-JEPA (MSE-only)", "mse"),
    BaselineSpec("pixel_ae", "Pixel autoencoder", "pixel_ae"),
    BaselineSpec("no_action", "No action conditioning", "vicreg", condition_on_action=False),
    BaselineSpec("particle_gnn", "Particle GNN (GNS/DPI-Net family)", "gnn"),
)


def build_baseline_config(
    *,
    epochs: int = 8,
    n_episodes: int = 160,
    output_dir: str = "outputs",
    device: str = "cpu",
) -> ExperimentConfig:
    """Shared mass-spring split long enough for ``k=10`` rollouts."""
    cfg = build_smoke_config(epochs=epochs, output_dir=output_dir)
    data = replace(
        cfg.data,
        n_episodes=n_episodes,
        episode_length=12,
        n_grid=8,
        val_fraction=0.25,
        normalize_actions=False,
        augment=False,
        pin_memory=False,
        n_stiffness_classes=3,
        seed=0,
    )
    model = replace(cfg.model, action_dim=data.action_dim, img_size=data.img_size)
    training = replace(
        cfg.training,
        epochs=epochs,
        batch_size=32,
        device=device,
        amp=False,
        num_workers=0,
        warmup_epochs=min(1, max(epochs - 1, 0)),
        output_dir=str(Path(output_dir) / "baselines"),
        log_interval=50,
        ckpt_interval=max(epochs, 1),
    )
    return ExperimentConfig(model=model, training=training, data=data)


def _spec_config(cfg: ExperimentConfig, spec: BaselineSpec) -> ExperimentConfig:
    return replace(
        cfg,
        training=replace(cfg.training, loss_type=spec.loss_type),  # type: ignore[arg-type]
        model=replace(cfg.model, condition_on_action=spec.condition_on_action),
    )


def write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "method_label",
        "horizon",
        "rollout_mse",
        "stiffness_probe_acc",
        "contact_probe_acc",
        "stiffness_probe_train_acc",
        "contact_probe_train_acc",
        "n_train",
        "n_val",
        "chance_stiffness",
        "chance_contact",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_readme(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(README_TEXT, encoding="utf-8")
    return path


def run_baseline_comparison(
    cfg: ExperimentConfig,
    *,
    csv_path: Path,
    readme_path: Path,
    max_horizon: int = 10,
    probe_epochs: int = 40,
) -> dict[str, Any]:
    seed_everything(cfg.training.seed)
    dataset = DeformableObjectDataset(cfg.data, split="all")
    n_train, n_val = dataset.split_sizes()
    LOGGER.info(
        "Shared split  episodes=%d  train/val=(%d, %d)  n_grid=%d  edges=%d",
        dataset.n_episodes,
        n_train,
        n_val,
        cfg.data.n_grid,
        int(_edge_list(cfg.data.n_grid, cfg.data.spatial_dims).shape[0]),
    )
    if dataset.positions is None:
        raise RuntimeError("Baseline comparison requires stored particle trajectories.")
    device = cfg.training.resolve_device()
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    for spec in BASELINES:
        run_cfg = _spec_config(cfg, spec)
        LOGGER.info("Training %s  loss=%s  action=%s", spec.key, spec.loss_type, spec.condition_on_action)
        trainer = ACJEPATrainer(run_cfg, run_name=spec.key, dataset=dataset)
        trainer.fit()
        model = trainer.model.to(device)
        model.eval()
        payload = evaluate_model(
            model,
            dataset,
            device,
            max_horizon=max_horizon,
            probe_epochs=probe_epochs,
            batch_size=8,
        )
        payloads[spec.key] = {
            key: payload[key]
            for key in (
                "horizons",
                "rollout_mse",
                "stiffness_probe_acc",
                "contact_probe_acc",
                "n_train",
                "n_val",
                "chance_stiffness",
                "chance_contact",
            )
        }
        for i, horizon in enumerate(payload["horizons"]):
            rows.append(
                {
                    "method": spec.key,
                    "method_label": spec.label,
                    "horizon": int(horizon),
                    "rollout_mse": payload["rollout_mse"][i],
                    "stiffness_probe_acc": payload["stiffness_probe_acc"][i],
                    "contact_probe_acc": payload["contact_probe_acc"][i],
                    "stiffness_probe_train_acc": payload["stiffness_probe_train_acc"][i],
                    "contact_probe_train_acc": payload["contact_probe_train_acc"][i],
                    "n_train": payload["n_train"],
                    "n_val": payload["n_val"],
                    "chance_stiffness": payload["chance_stiffness"],
                    "chance_contact": payload["chance_contact"],
                }
            )
    csv_out = write_comparison_csv(rows, csv_path)
    readme_out = write_readme(readme_path)
    report = {
        "n_train": n_train,
        "n_val": n_val,
        "n_grid": cfg.data.n_grid,
        "epochs": cfg.training.epochs,
        "device": str(device),
        "csv": str(csv_out),
        "readme": str(readme_out),
        "models": payloads,
        "domain_gap": (
            "Within-simulator only. Not a PhysWorld/PhysTwin comparison on captured video."
        ),
    }
    (csv_path.parent / "baseline_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    LOGGER.info("Wrote %s and %s", csv_out, readme_out)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Within-simulator AC-JEPA vs GNN / no-action baselines.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--n-episodes", type=int, default=160)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--readme", default=None)
    parser.add_argument("--probe-epochs", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = build_baseline_config(
        epochs=args.epochs,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        device=args.device,
    )
    csv_path = Path(args.csv) if args.csv else Path(args.output_dir) / "baseline_comparison.csv"
    readme_path = Path(args.readme) if args.readme else Path(args.output_dir) / "baseline_comparison.md"
    return run_baseline_comparison(
        cfg,
        csv_path=csv_path,
        readme_path=readme_path,
        probe_epochs=args.probe_epochs,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
