"""Automated AC-JEPA ablation suite and long-horizon representation-drift engine.

Variants
--------
* **Full AC-JEPA** — VICReg invariance + variance + covariance.
* **Variant A** — invariance + variance (no covariance penalty).
* **Variant B** — MSE-only latent predictor (no VICReg regularizers).
* **Variant C** — pixel-space autoencoder baseline (decode ``Predictor(z_t, a_t)``).

Each trained model is rolled out autoregressively for ``k ∈ {1, 2, 5, 10, 20}``:

    ẑ_{t+k} = Predictor(ẑ_{t+k-1}, a_{t+k-1})

and scored against the EMA/target embedding ``z_{t+k}`` with MSE and cosine
similarity. Results are written to ``outputs/ablation_results.csv`` plus a
high-resolution drift figure.

Run::

    python src/ablation.py
    python src/ablation.py --epochs 3 --output-dir outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

if __package__:
    from .config import ExperimentConfig, seed_everything
    from .dataset import build_dataset
    from .eval import rollout_predictions
    from .train import ACJEPATrainer, apply_dataset_path, build_smoke_config
else:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.config import ExperimentConfig, seed_everything
    from src.dataset import build_dataset
    from src.eval import rollout_predictions
    from src.train import ACJEPATrainer, apply_dataset_path, build_smoke_config

LOGGER = logging.getLogger("ac_jepa.ablation")

HORIZONS: tuple[int, ...] = (1, 2, 5, 10, 20)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None


@dataclass(frozen=True)
class AblationVariant:
    """One trained world-model family in the ablation grid."""

    key: str
    label: str
    loss_type: str
    color: str


VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("full_vicreg", "Full AC-JEPA (VICReg)", "vicreg", "#1f77b4"),
    AblationVariant("variant_a_no_cov", "Variant A (No Covariance)", "vicreg_no_cov", "#2ca02c"),
    AblationVariant("variant_b_mse", "Variant B (MSE-only)", "mse", "#d62728"),
    AblationVariant("variant_c_pixel_ae", "Variant C (Pixel Autoencoder)", "pixel_ae", "#9467bd"),
)


def build_ablation_config(
    epochs: int = 3,
    output_dir: str = "outputs",
    max_horizon: int = 20,
) -> ExperimentConfig:
    """Smoke-scale config with episodes long enough for ``k=20`` rollouts."""
    cfg = build_smoke_config(epochs=epochs, output_dir=output_dir)
    episode_length = max(cfg.data.episode_length, max_horizon + 1)
    data = replace(cfg.data, episode_length=episode_length, n_episodes=max(cfg.data.n_episodes, 32))
    training = replace(cfg.training, output_dir=str(Path(output_dir) / "ablation"))
    return ExperimentConfig(model=cfg.model, training=training, data=data)


def _policy_actions(dataset: Any) -> Tensor:
    return getattr(dataset, "model_actions", dataset.actions)


@torch.no_grad()
def representation_drift(
    model: Any,
    dataset: Any,
    device: torch.device,
    horizons: Sequence[int] = HORIZONS,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Autoregressive latent rollout drift at selected horizons.

    ``ẑ_{t+k}`` is produced by iterating the action-conditioned predictor;
    ``z_{t+k}`` is the stop-gradient target encoder of the true future frame.
    """
    model.eval()
    n_train, n_val = dataset.split_sizes()
    if n_val < 1:
        raise ValueError("Ablation drift evaluation requires a non-empty val split.")
    observations = dataset.observations[n_train : n_train + n_val]
    actions = _policy_actions(dataset)[n_train : n_train + n_val]
    available = int(observations.shape[1] - 1)
    requested = [int(k) for k in horizons if 1 <= int(k) <= available]
    if not requested:
        raise ValueError(
            f"No requested horizon fits episode length {observations.shape[1]} "
            f"(need T > max k). Asked {list(horizons)}."
        )
    k_max = max(requested)
    z_hat, z_tgt = rollout_predictions(
        model, observations, actions, k_max, device, batch_size=batch_size
    )
    mse_k: list[float] = []
    cos_k: list[float] = []
    for horizon in requested:
        pred = z_hat[:, horizon - 1]
        tgt = z_tgt[:, horizon - 1]
        mse_k.append(float((pred - tgt).pow(2).mean().item()))
        cos_k.append(float(F.cosine_similarity(pred, tgt, dim=-1).mean().item()))
    return {
        "horizons": requested,
        "mse": mse_k,
        "cosine": cos_k,
        "n_val": n_val,
        "k_max": k_max,
    }


def write_ablation_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "variant_label",
        "loss_type",
        "horizon",
        "mse",
        "cosine_similarity",
        "n_val",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %s (%d rows)", path, len(rows))
    return path


def plot_representation_drift(
    results: dict[str, dict[str, Any]],
    variants: Sequence[AblationVariant],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """High-resolution MSE and cosine-similarity vs horizon."""
    if plt is None:
        raise ImportError(
            "matplotlib is required to write ablation plots. `pip install matplotlib`."
        ) from _MATPLOTLIB_ERROR
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=dpi)
    ax_mse, ax_cos = axes
    lookup = {item.key: item for item in variants}
    for key, payload in results.items():
        variant = lookup.get(key)
        label = variant.label if variant is not None else key
        color = variant.color if variant is not None else None
        ax_mse.plot(
            payload["horizons"], payload["mse"], marker="o", color=color, label=label, linewidth=2.0
        )
        ax_cos.plot(
            payload["horizons"],
            payload["cosine"],
            marker="o",
            color=color,
            label=label,
            linewidth=2.0,
        )
    ax_mse.set_title("Representation drift (MSE)")
    ax_mse.set_xlabel("horizon k")
    ax_mse.set_ylabel(r"MSE$(\hat{z}_{t+k},\, z_{t+k})$")
    ax_mse.set_xticks(list(HORIZONS))
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(frameon=False, fontsize=8)

    ax_cos.set_title("Representation alignment (cosine)")
    ax_cos.set_xlabel("horizon k")
    ax_cos.set_ylabel(r"cos$(\hat{z}_{t+k},\, z_{t+k})$")
    ax_cos.set_xticks(list(HORIZONS))
    ax_cos.set_ylim(-0.05, 1.05)
    ax_cos.grid(True, alpha=0.3)
    ax_cos.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Wrote %s", output_path)
    return output_path


def run_ablation_suite(
    cfg: ExperimentConfig,
    *,
    horizons: Sequence[int] = HORIZONS,
    csv_path: Path | None = None,
    plot_path: Path | None = None,
) -> dict[str, Any]:
    """Train every variant, score multi-step drift, and export CSV + plots."""
    output_root = (
        Path(cfg.training.output_dir).parent
        if Path(cfg.training.output_dir).name == "ablation"
        else Path(cfg.training.output_dir)
    )
    csv_path = Path(csv_path) if csv_path is not None else output_root / "ablation_results.csv"
    plot_path = Path(plot_path) if plot_path is not None else output_root / "ablation_drift.png"
    seed_everything(cfg.training.seed)
    dataset = build_dataset(cfg.data, split="all")
    LOGGER.info(
        "Ablation dataset  source=%s  episodes=%d  T=%d  train/val=%s",
        cfg.data.dataset_path or "synthetic",
        dataset.n_episodes,
        cfg.data.episode_length,
        dataset.split_sizes(),
    )
    device = cfg.training.resolve_device()
    models: dict[str, Any] = {}
    summaries: dict[str, Any] = {}

    for variant in VARIANTS:
        LOGGER.info("Training %s  (%s)", variant.label, variant.loss_type)
        run_cfg = replace(
            cfg,
            training=replace(cfg.training, loss_type=variant.loss_type),  # type: ignore[arg-type]
        )
        trainer = ACJEPATrainer(run_cfg, run_name=variant.key, dataset=dataset)
        summary = trainer.fit()
        models[variant.key] = trainer.model
        summaries[variant.key] = {
            "checkpoint": summary.get("checkpoint"),
            "seconds": summary.get("seconds"),
            "best_val": summary.get("best_val"),
        }

    drift: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        metrics = representation_drift(models[variant.key], dataset, device, horizons=horizons)
        drift[variant.key] = metrics
        LOGGER.info(
            "%s  MSE=%s  cos=%s",
            variant.key,
            ", ".join(f"{v:.4f}" for v in metrics["mse"]),
            ", ".join(f"{v:.3f}" for v in metrics["cosine"]),
        )
        for horizon, mse, cosine in zip(
            metrics["horizons"], metrics["mse"], metrics["cosine"], strict=True
        ):
            rows.append(
                {
                    "variant": variant.key,
                    "variant_label": variant.label,
                    "loss_type": variant.loss_type,
                    "horizon": horizon,
                    "mse": mse,
                    "cosine_similarity": cosine,
                    "n_val": metrics["n_val"],
                }
            )

    write_ablation_csv(rows, csv_path)
    plot_representation_drift(drift, VARIANTS, plot_path)
    report = {
        "variants": {v.key: v.label for v in VARIANTS},
        "training": summaries,
        "drift": drift,
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    (csv_path.parent / "ablation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AC-JEPA automated ablation + long-horizon drift.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--csv", default=None, help="Defaults to <output-dir>/ablation_results.csv")
    parser.add_argument("--plot", default=None, help="Defaults to <output-dir>/ablation_drift.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset_path", "--dataset-path", dest="dataset_path", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--horizons",
        default="1,2,5,10,20",
        help="Comma-separated rollout horizons (default 1,2,5,10,20).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    horizons = tuple(int(part) for part in args.horizons.split(",") if part.strip())
    cfg = build_ablation_config(
        epochs=args.epochs,
        output_dir=args.output_dir,
        max_horizon=max(horizons) if horizons else 20,
    )
    cfg.training.device = args.device
    if args.dataset_path:
        cfg = apply_dataset_path(cfg, args.dataset_path, args.num_workers)
        cfg.training.device = args.device
    csv_path = Path(args.csv) if args.csv else Path(args.output_dir) / "ablation_results.csv"
    plot_path = Path(args.plot) if args.plot else Path(args.output_dir) / "ablation_drift.png"
    report = run_ablation_suite(
        cfg,
        horizons=horizons,
        csv_path=csv_path,
        plot_path=plot_path,
    )
    LOGGER.info("Ablation complete  csv=%s  plot=%s", report["csv"], report["plot"])
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
