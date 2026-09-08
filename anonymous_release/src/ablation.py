"""Automated AC-JEPA ablation suite and long-horizon representation-drift engine.

Variants
--------
* **Full AC-JEPA** — VICReg invariance + variance + covariance.
* **Variant A** — invariance + variance (no covariance penalty).
* **Variant B** — MSE-only latent predictor (no VICReg regularizers).
* **Variant C** — pixel-space autoencoder baseline (decode ``Predictor(z_t, a_t)``).
* **No-action** — predictor sees ``z_t`` only (``condition_on_action=False``).
* **EMA sweep** — ``ema_momentum ∈ {0.99, 0.996, 0.999}`` on Full AC-JEPA.

Workshop-scale entry (Prompt-2 config + Prompt-1 physics dump, ≥3 seeds)::

    python src/ablation.py --scale workshop --seeds 42,43,44

Smoke-scale (CI / dummy lattice)::

    python src/ablation.py --scale smoke --epochs 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

if __package__:
    from .config import ExperimentConfig, build_workshop_config, seed_everything
    from .dataset import build_dataset
    from .eval import evaluate_model, rollout_predictions
    from .train import ACJEPATrainer, apply_dataset_path, build_smoke_config, load_model_from_checkpoint
else:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.config import ExperimentConfig, build_workshop_config, seed_everything
    from src.dataset import build_dataset
    from src.eval import evaluate_model, rollout_predictions
    from src.train import ACJEPATrainer, apply_dataset_path, build_smoke_config, load_model_from_checkpoint

LOGGER = logging.getLogger("ac_jepa.ablation")

HORIZONS: tuple[int, ...] = (1, 2, 5, 10, 20)
REPORT_HORIZONS: tuple[int, ...] = (1, 5, 10)
ABLATION_SEEDS: tuple[int, ...] = (42, 43, 44)
EMA_MOMENTA: tuple[float, ...] = (0.99, 0.996, 0.999)

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
    condition_on_action: bool = True
    ema_momentum: float | None = None


VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("full_vicreg", "Full AC-JEPA (VICReg)", "vicreg", "#1f77b4"),
    AblationVariant("variant_a_no_cov", "Variant A (No Covariance)", "vicreg_no_cov", "#2ca02c"),
    AblationVariant("variant_b_mse", "Variant B (MSE-only)", "mse", "#d62728"),
    AblationVariant("variant_c_pixel_ae", "Variant C (Pixel Autoencoder)", "pixel_ae", "#9467bd"),
    AblationVariant(
        "no_action",
        "No action conditioning",
        "vicreg",
        "#ff7f0e",
        condition_on_action=False,
    ),
)


def ema_sweep_variants(
    momenta: Sequence[float] = EMA_MOMENTA,
) -> tuple[AblationVariant, ...]:
    """Full AC-JEPA copies that differ only in ``ema_momentum``."""
    palette = {0.99: "#8c564b", 0.996: "#1f77b4", 0.999: "#17becf"}
    variants: list[AblationVariant] = []
    for tau in momenta:
        key = "ema_" + str(tau).replace(".", "p")
        variants.append(
            AblationVariant(
                key=key,
                label=f"EMA tau={tau:g}",
                loss_type="vicreg",
                color=palette.get(float(tau), "#bcbd22"),
                ema_momentum=float(tau),
            )
        )
    return tuple(variants)


def study_variants(
    *,
    include_ema_sweep: bool = True,
    ema_momenta: Sequence[float] = EMA_MOMENTA,
) -> tuple[AblationVariant, ...]:
    """Loss / action ablations plus the EMA-τ sweep on Full AC-JEPA."""
    variants = list(VARIANTS)
    if include_ema_sweep:
        variants.extend(ema_sweep_variants(ema_momenta))
    return tuple(variants)


def ema_variant_key(momentum: float) -> str:
    return "ema_" + str(momentum).replace(".", "p")


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Sample mean and sample standard deviation (0.0 for a single value)."""
    xs = [float(v) for v in values]
    if not xs:
        raise ValueError("mean_std requires at least one value.")
    mean = float(statistics.fmean(xs))
    std = 0.0 if len(xs) < 2 else float(statistics.stdev(xs))
    return mean, std


def _is_default_ema_control(variant: AblationVariant, default_tau: float = 0.996) -> bool:
    """True when this EMA-sweep row is the same recipe as ``full_vicreg``."""
    return (
        variant.ema_momentum is not None
        and abs(float(variant.ema_momentum) - float(default_tau)) < 1e-12
        and variant.loss_type == "vicreg"
        and variant.condition_on_action
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


def build_workshop_ablation_config(
    output_dir: str = "outputs",
    *,
    dataset_path: str | None = None,
) -> ExperimentConfig:
    """Prompt-2 workshop schedule on the Prompt-1 physics dump."""
    cfg = build_workshop_config(dataset_path=dataset_path)
    training = replace(
        cfg.training,
        seed=42,
        output_dir=str(Path(output_dir) / "ablation"),
    )
    return ExperimentConfig(model=cfg.model, training=training, data=cfg.data)


def apply_variant(
    cfg: ExperimentConfig, variant: AblationVariant, *, seed: int | None = None
) -> ExperimentConfig:
    """Copy ``cfg`` with only the ablated factor (and optional seed) changed."""
    tau = cfg.model.ema_momentum if variant.ema_momentum is None else float(variant.ema_momentum)
    model = replace(
        cfg.model,
        condition_on_action=variant.condition_on_action,
        ema_momentum=tau,
    )
    training = replace(
        cfg.training,
        loss_type=variant.loss_type,  # type: ignore[arg-type]
        seed=cfg.training.seed if seed is None else int(seed),
    )
    return ExperimentConfig(model=model, training=training, data=cfg.data)


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


def plot_multiseed_drift(
    summary: dict[str, Any],
    output_path: Path,
    *,
    variants: Sequence[AblationVariant] | None = None,
    dpi: int = 300,
) -> Path:
    """Error-band drift plot from ``ablation_multiseed.json`` (mean ± sample std).

    Left: latent rollout MSE. Right: contact-probe accuracy when present,
    otherwise cosine similarity from the smoke-scale drift engine.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required to write ablation plots. `pip install matplotlib`."
        ) from _MATPLOTLIB_ERROR
    variants = tuple(study_variants() if variants is None else variants)
    lookup = {item.key: item for item in variants}
    blob = summary.get("variants", summary)
    if not isinstance(blob, dict) or not blob:
        raise ValueError("plot_multiseed_drift expected a variants mapping.")
    skip_ema_control = "full_vicreg" in blob
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=dpi)
    ax_mse, ax_right = axes
    n_seeds = summary.get("n_seeds") or len(summary.get("seeds") or [])
    dataset = summary.get("dataset_path") or "synthetic"
    for key, payload in blob.items():
        if not isinstance(payload, dict):
            continue
        if skip_ema_control and str(key).startswith("ema_"):
            tau = payload.get("ema_momentum")
            if tau is not None and abs(float(tau) - 0.996) < 1e-12:
                continue
        by_horizon = payload.get("by_horizon") or {}
        if not by_horizon:
            continue
        horizons = sorted(int(h) for h in by_horizon)
        mse_mean = [float(by_horizon[str(h)]["rollout_mse_mean"]) for h in horizons]
        mse_std = [float(by_horizon[str(h)].get("rollout_mse_std") or 0.0) for h in horizons]
        variant = lookup.get(key)
        label = str(payload.get("label") or (variant.label if variant else key))
        color = variant.color if variant is not None else None
        ax_mse.plot(horizons, mse_mean, marker="o", color=color, label=label, linewidth=2.0)
        ax_mse.fill_between(
            horizons,
            [m - s for m, s in zip(mse_mean, mse_std, strict=True)],
            [m + s for m, s in zip(mse_mean, mse_std, strict=True)],
            color=color,
            alpha=0.22,
            linewidth=0,
        )
        if "contact_probe_acc_mean" in by_horizon[str(horizons[0])]:
            c_mean = [float(by_horizon[str(h)]["contact_probe_acc_mean"]) for h in horizons]
            c_std = [float(by_horizon[str(h)].get("contact_probe_acc_std") or 0.0) for h in horizons]
            ax_right.plot(horizons, c_mean, marker="o", color=color, label=label, linewidth=2.0)
            ax_right.fill_between(
                horizons,
                [m - s for m, s in zip(c_mean, c_std, strict=True)],
                [m + s for m, s in zip(c_mean, c_std, strict=True)],
                color=color,
                alpha=0.22,
                linewidth=0,
            )
        elif "cosine_mean" in by_horizon[str(horizons[0])]:
            c_mean = [float(by_horizon[str(h)]["cosine_mean"]) for h in horizons]
            c_std = [float(by_horizon[str(h)].get("cosine_std") or 0.0) for h in horizons]
            ax_right.plot(horizons, c_mean, marker="o", color=color, label=label, linewidth=2.0)
            ax_right.fill_between(
                horizons,
                [m - s for m, s in zip(c_mean, c_std, strict=True)],
                [m + s for m, s in zip(c_mean, c_std, strict=True)],
                color=color,
                alpha=0.22,
                linewidth=0,
            )
    seed_note = f"{n_seeds} seeds" if n_seeds else "multi-seed"
    ax_mse.set_title(f"Representation drift (MSE, mean ± std, {seed_note})")
    ax_mse.set_xlabel("horizon k")
    ax_mse.set_ylabel(r"MSE$(\hat{z}_{t+k},\, z_{t+k})$")
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(frameon=False, fontsize=8)
    has_contact = any(
        isinstance(p, dict)
        and (p.get("by_horizon") or {})
        and "contact_probe_acc_mean" in next(iter((p.get("by_horizon") or {}).values()), {})
        for p in blob.values()
    )
    if has_contact:
        ax_right.set_title(f"Contact probe (mean ± std, {seed_note})")
        ax_right.set_ylabel("linear-probe accuracy")
        ax_right.set_ylim(-0.05, 1.05)
    else:
        ax_right.set_title(f"Representation alignment (cosine, mean ± std, {seed_note})")
        ax_right.set_ylabel(r"cos$(\hat{z}_{t+k},\, z_{t+k})$")
        ax_right.set_ylim(-0.05, 1.05)
    ax_right.set_xlabel("horizon k")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Multi-seed AC-JEPA ablation  ({dataset})", fontsize=11)
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
    variants: Sequence[AblationVariant] | None = None,
) -> dict[str, Any]:
    """Train every variant, score multi-step drift, and export CSV + plots.

    Smoke-scale helper (dummy lattice, single seed). Workshop rebuttals should
    use :func:`run_multiseed_ablation` instead.
    """
    variants = tuple(VARIANTS if variants is None else variants)
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

    for variant in variants:
        LOGGER.info(
            "Training %s  loss=%s  action=%s  ema=%s",
            variant.label,
            variant.loss_type,
            variant.condition_on_action,
            variant.ema_momentum,
        )
        run_cfg = apply_variant(cfg, variant)
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
    for variant in variants:
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
    plot_representation_drift(drift, variants, plot_path)
    report = {
        "variants": {v.key: v.label for v in variants},
        "training": summaries,
        "drift": drift,
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    (csv_path.parent / "ablation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _output_root(cfg: ExperimentConfig) -> Path:
    out = Path(cfg.training.output_dir)
    return out.parent if out.name == "ablation" else out


def _checkpoint_complete(run_dir: Path, epochs: int) -> bool:
    history = run_dir / "history.json"
    ckpt = run_dir / "last.pt"
    if not history.is_file() or not ckpt.is_file():
        return False
    try:
        blob = json.loads(history.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(blob.get("epochs", 0)) >= int(epochs)


def _slice_eval(
    payload: dict[str, Any],
    report_horizons: Sequence[int],
) -> dict[str, Any]:
    index = {int(h): i for i, h in enumerate(payload["horizons"])}
    sliced = {
        "horizons": [],
        "rollout_mse": [],
        "stiffness_probe_acc": [],
        "contact_probe_acc": [],
        "n_train": payload.get("n_train"),
        "n_val": payload.get("n_val"),
        "chance_stiffness": payload.get("chance_stiffness"),
        "chance_contact": payload.get("chance_contact"),
        "k_max": payload.get("k_max"),
    }
    for horizon in report_horizons:
        if int(horizon) not in index:
            continue
        i = index[int(horizon)]
        sliced["horizons"].append(int(horizon))
        sliced["rollout_mse"].append(float(payload["rollout_mse"][i]))
        sliced["stiffness_probe_acc"].append(float(payload["stiffness_probe_acc"][i]))
        sliced["contact_probe_acc"].append(float(payload["contact_probe_acc"][i]))
    if not sliced["horizons"]:
        raise ValueError(
            f"evaluate_model returned horizons {payload.get('horizons')} "
            f"which do not overlap report horizons {list(report_horizons)}."
        )
    return sliced


def _eval_rows(
    variant: AblationVariant,
    seed: int,
    payload: dict[str, Any],
    *,
    ema_momentum: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, horizon in enumerate(payload["horizons"]):
        rows.append(
            {
                "variant": variant.key,
                "variant_label": variant.label,
                "loss_type": variant.loss_type,
                "condition_on_action": variant.condition_on_action,
                "ema_momentum": ema_momentum,
                "seed": int(seed),
                "horizon": int(horizon),
                "rollout_mse": float(payload["rollout_mse"][i]),
                "stiffness_probe_acc": float(payload["stiffness_probe_acc"][i]),
                "contact_probe_acc": float(payload["contact_probe_acc"][i]),
                "n_train": payload.get("n_train"),
                "n_val": payload.get("n_val"),
            }
        )
    return rows


def write_multiseed_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "variant_label",
        "loss_type",
        "condition_on_action",
        "ema_momentum",
        "seed",
        "horizon",
        "rollout_mse",
        "stiffness_probe_acc",
        "contact_probe_acc",
        "n_train",
        "n_val",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(list(rows))
    LOGGER.info("Wrote %s (%d rows)", path, len(rows))
    return path


def write_ablation_summary_csv(summary: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "variant_label",
        "loss_type",
        "condition_on_action",
        "ema_momentum",
        "horizon",
        "n_seeds",
        "n_val",
        "rollout_mse_mean",
        "rollout_mse_std",
        "stiffness_probe_acc_mean",
        "stiffness_probe_acc_std",
        "contact_probe_acc_mean",
        "contact_probe_acc_std",
    ]
    rows: list[dict[str, Any]] = []
    for key, payload in summary.get("variants", {}).items():
        for horizon_key, stats in payload.get("by_horizon", {}).items():
            rows.append(
                {
                    "variant": key,
                    "variant_label": payload.get("label", key),
                    "loss_type": payload.get("loss_type"),
                    "condition_on_action": payload.get("condition_on_action"),
                    "ema_momentum": payload.get("ema_momentum"),
                    "horizon": int(horizon_key),
                    "n_seeds": summary.get("n_seeds"),
                    "n_val": summary.get("n_val"),
                    **stats,
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %s (%d rows)", path, len(rows))
    return path


def aggregate_seed_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    variants: Sequence[AblationVariant] | None = None,
    seeds: Sequence[int] | None = None,
    horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Collapse per-(variant, seed, horizon) rows into mean ± sample std."""
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {
            "rollout_mse": [],
            "stiffness_probe_acc": [],
            "contact_probe_acc": [],
            "seeds": [],
            "n_val": [],
        }
    )
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["variant"])
        horizon = int(row["horizon"])
        bucket = grouped[(key, horizon)]
        bucket["rollout_mse"].append(float(row["rollout_mse"]))
        bucket["stiffness_probe_acc"].append(float(row["stiffness_probe_acc"]))
        bucket["contact_probe_acc"].append(float(row["contact_probe_acc"]))
        bucket["seeds"].append(int(row["seed"]))
        if row.get("n_val") is not None:
            bucket["n_val"].append(int(row["n_val"]))
        meta[key] = {
            "label": row.get("variant_label", key),
            "loss_type": row.get("loss_type"),
            "condition_on_action": row.get("condition_on_action"),
            "ema_momentum": row.get("ema_momentum"),
        }
    lookup = {item.key: item for item in (variants or ())}
    order: list[str] = []
    for item in variants or ():
        if item.key in meta and item.key not in order:
            order.append(item.key)
    for key in meta:
        if key not in order:
            order.append(key)
    if horizons is None:
        horizon_set = sorted({h for _, h in grouped})
    else:
        horizon_set = [int(h) for h in horizons]
    variants_out: dict[str, Any] = {}
    n_val = None
    for key in order:
        item = lookup.get(key)
        info = meta.get(key, {})
        by_horizon: dict[str, Any] = {}
        for horizon in horizon_set:
            bucket = grouped.get((key, int(horizon)))
            if bucket is None or not bucket["rollout_mse"]:
                continue
            mse_mean, mse_std = mean_std(bucket["rollout_mse"])
            stiff_mean, stiff_std = mean_std(bucket["stiffness_probe_acc"])
            contact_mean, contact_std = mean_std(bucket["contact_probe_acc"])
            if bucket["n_val"]:
                n_val = int(bucket["n_val"][-1])
            by_horizon[str(int(horizon))] = {
                "rollout_mse_mean": mse_mean,
                "rollout_mse_std": mse_std,
                "stiffness_probe_acc_mean": stiff_mean,
                "stiffness_probe_acc_std": stiff_std,
                "contact_probe_acc_mean": contact_mean,
                "contact_probe_acc_std": contact_std,
                "n_seeds": len(bucket["rollout_mse"]),
                "seeds": bucket["seeds"],
            }
        variants_out[key] = {
            "label": item.label if item is not None else info.get("label", key),
            "loss_type": item.loss_type if item is not None else info.get("loss_type"),
            "condition_on_action": (
                item.condition_on_action if item is not None else info.get("condition_on_action")
            ),
            "ema_momentum": (
                item.ema_momentum if item is not None and item.ema_momentum is not None
                else info.get("ema_momentum")
            ),
            "by_horizon": by_horizon,
        }
    seed_list = list(seeds) if seeds is not None else sorted({int(r["seed"]) for r in rows})
    return {
        "seeds": [int(s) for s in seed_list],
        "n_seeds": len(seed_list),
        "horizons": [int(h) for h in horizon_set],
        "n_val": n_val,
        "variants": variants_out,
    }


def _legacy_rows_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, payload in summary.get("variants", {}).items():
        for horizon_key, stats in payload.get("by_horizon", {}).items():
            rows.append(
                {
                    "variant": key,
                    "variant_label": payload.get("label", key),
                    "loss_type": payload.get("loss_type"),
                    "horizon": int(horizon_key),
                    "mse": stats["rollout_mse_mean"],
                    "cosine_similarity": float("nan"),
                    "n_val": summary.get("n_val"),
                }
            )
    return rows


def _persist_multiseed(
    rows: Sequence[dict[str, Any]],
    *,
    variants: Sequence[AblationVariant],
    seeds: Sequence[int],
    horizons: Sequence[int],
    output_root: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = aggregate_seed_metrics(rows, variants=variants, seeds=seeds, horizons=horizons)
    if extra:
        summary.update(extra)
    json_path = output_root / "ablation_multiseed.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = write_multiseed_csv(rows, output_root / "ablation_multiseed.csv")
    summary_csv = write_ablation_summary_csv(summary, output_root / "ablation_summary.csv")
    legacy = _legacy_rows_from_summary(summary)
    if legacy:
        write_ablation_csv(legacy, output_root / "ablation_results.csv")
    plot_path = plot_multiseed_drift(summary, output_root / "ablation_drift.png", variants=variants)
    LOGGER.info("Updated multi-seed artifacts  json=%s  csv=%s  plot=%s", json_path, csv_path, plot_path)
    return {
        "summary": summary,
        "json": str(json_path),
        "csv": str(csv_path),
        "summary_csv": str(summary_csv),
        "plot": str(plot_path),
    }


def _evaluate_variant(
    model: Any,
    dataset: Any,
    device: torch.device,
    report_horizons: Sequence[int],
    probe_epochs: int,
) -> dict[str, Any]:
    k_max = max(int(k) for k in report_horizons)
    payload = evaluate_model(
        model,
        dataset,
        device,
        max_horizon=k_max,
        probe_epochs=probe_epochs,
    )
    return _slice_eval(payload, report_horizons)


def run_multiseed_ablation(
    cfg: ExperimentConfig,
    *,
    seeds: Sequence[int] = ABLATION_SEEDS,
    variants: Sequence[AblationVariant] | None = None,
    report_horizons: Sequence[int] = REPORT_HORIZONS,
    probe_epochs: int = 80,
    skip_completed: bool = True,
) -> dict[str, Any]:
    """Train each ablation on the shared dataset across seeds; write mean ± std."""
    variants = tuple(study_variants() if variants is None else variants)
    seeds = tuple(int(s) for s in seeds)
    report_horizons = tuple(int(k) for k in report_horizons)
    output_root = _output_root(cfg)
    ablation_root = Path(cfg.training.output_dir)
    if ablation_root.name != "ablation":
        ablation_root = Path(cfg.training.output_dir) / "ablation"
        cfg = replace(cfg, training=replace(cfg.training, output_dir=str(ablation_root)))
    dataset = build_dataset(cfg.data, split="all")
    n_train, n_val = dataset.split_sizes()
    LOGGER.info(
        "Multi-seed ablation  source=%s  episodes=%d  T=%d  train/val=(%d, %d)  "
        "epochs=%d  seeds=%s  variants=%s  horizons=%s",
        cfg.data.dataset_path or "synthetic",
        dataset.n_episodes,
        int(dataset.observations.shape[1]),
        n_train,
        n_val,
        cfg.training.epochs,
        list(seeds),
        [v.key for v in variants],
        list(report_horizons),
    )
    device = cfg.training.resolve_device()
    rows: list[dict[str, Any]] = []
    evals: dict[tuple[str, int], dict[str, Any]] = {}
    training: dict[str, Any] = {}

    for seed in seeds:
        seed_everything(seed)
        for variant in variants:
            tau = (
                float(cfg.model.ema_momentum)
                if variant.ema_momentum is None
                else float(variant.ema_momentum)
            )
            run_cfg = apply_variant(cfg, variant, seed=seed)
            run_cfg = replace(
                run_cfg,
                training=replace(run_cfg.training, output_dir=str(ablation_root / f"seed{seed}")),
            )
            run_dir = Path(run_cfg.training.output_dir) / variant.key
            eval_path = run_dir / "eval.json"

            if _is_default_ema_control(variant):
                reused = None
                source_meta = training.get("full_vicreg", {}).get(str(seed), {})
                if skip_completed and eval_path.is_file():
                    reused = json.loads(eval_path.read_text(encoding="utf-8"))
                elif ("full_vicreg", seed) in evals:
                    reused = dict(evals[("full_vicreg", seed)])
                    run_dir.mkdir(parents=True, exist_ok=True)
                    eval_path.write_text(json.dumps(reused, indent=2), encoding="utf-8")
                if reused is not None:
                    LOGGER.info(
                        "Reusing full_vicreg seed=%d for %s (ema_momentum=%s)",
                        seed,
                        variant.key,
                        tau,
                    )
                    evals[(variant.key, seed)] = reused
                    rows.extend(_eval_rows(variant, seed, reused, ema_momentum=tau))
                    training.setdefault(variant.key, {})[str(seed)] = {
                        "reused_from": "full_vicreg",
                        "checkpoint": source_meta.get("checkpoint"),
                    }
                    _persist_multiseed(
                        rows,
                        variants=variants,
                        seeds=seeds,
                        horizons=report_horizons,
                        output_root=output_root,
                        extra={
                            "dataset_path": cfg.data.dataset_path,
                            "epochs": cfg.training.epochs,
                            "training": training,
                        },
                    )
                    continue

            if skip_completed and eval_path.is_file() and _checkpoint_complete(run_dir, cfg.training.epochs):
                LOGGER.info("Skipping completed %s seed=%d  (%s)", variant.key, seed, run_dir)
                sliced = json.loads(eval_path.read_text(encoding="utf-8"))
            else:
                if skip_completed and _checkpoint_complete(run_dir, cfg.training.epochs):
                    LOGGER.info("Evaluating cached checkpoint %s seed=%d", variant.key, seed)
                    model, _ = load_model_from_checkpoint(run_dir / "last.pt", device)
                    seconds = None
                    checkpoint = str(run_dir / "last.pt")
                else:
                    LOGGER.info(
                        "Training %s seed=%d  loss=%s  action=%s  ema=%s  epochs=%d",
                        variant.key,
                        seed,
                        variant.loss_type,
                        variant.condition_on_action,
                        tau,
                        run_cfg.training.epochs,
                    )
                    trainer = ACJEPATrainer(run_cfg, run_name=variant.key, dataset=dataset)
                    summary = trainer.fit()
                    model = trainer.model
                    seconds = summary.get("seconds")
                    checkpoint = summary.get("checkpoint")
                sliced = _evaluate_variant(
                    model, dataset, device, report_horizons, probe_epochs
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                eval_path.write_text(json.dumps(sliced, indent=2), encoding="utf-8")
                training.setdefault(variant.key, {})[str(seed)] = {
                    "checkpoint": checkpoint,
                    "seconds": seconds,
                }
                del model
                if device.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            evals[(variant.key, seed)] = sliced
            rows.extend(_eval_rows(variant, seed, sliced, ema_momentum=tau))
            _persist_multiseed(
                rows,
                variants=variants,
                seeds=seeds,
                horizons=report_horizons,
                output_root=output_root,
                extra={
                    "dataset_path": cfg.data.dataset_path,
                    "epochs": cfg.training.epochs,
                    "training": training,
                },
            )

    persisted = _persist_multiseed(
        rows,
        variants=variants,
        seeds=seeds,
        horizons=report_horizons,
        output_root=output_root,
        extra={
            "dataset_path": cfg.data.dataset_path,
            "epochs": cfg.training.epochs,
            "training": training,
        },
    )
    report = {
        "variants": {v.key: v.label for v in variants},
        "seeds": list(seeds),
        "horizons": list(report_horizons),
        "n_val": n_val,
        "training": training,
        **persisted,
    }
    (output_root / "ablation_report.json").write_text(
        json.dumps(
            {
                "variants": report["variants"],
                "seeds": report["seeds"],
                "horizons": report["horizons"],
                "n_val": n_val,
                "training": training,
                "csv": persisted["csv"],
                "json": persisted["json"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AC-JEPA automated ablation + long-horizon drift.")
    parser.add_argument(
        "--scale",
        choices=("smoke", "workshop"),
        default="workshop",
        help="smoke: dummy lattice, few epochs. workshop: Prompt-2 config + physics dump, ≥3 seeds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs. Smoke defaults to 3; workshop keeps the YAML schedule (200).",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--csv", default=None, help="Smoke-scale CSV path.")
    parser.add_argument("--plot", default=None, help="Smoke-scale drift plot path.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset_path", "--dataset-path", dest="dataset_path", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--horizons",
        default=None,
        help="Smoke-scale drift horizons (default 1,2,5,10,20).",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Workshop-scale RNG seeds (default 42,43,44).",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variant keys (e.g. full_vicreg,variant_b_mse). Default: all.",
    )
    parser.add_argument(
        "--report-horizons",
        default="1,5,10",
        help="Horizons written to the multi-seed table (physics dump T=16 cannot do k=20).",
    )
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument(
        "--ema-sweep",
        default="",
        help="Comma-separated EMA momenta to add as extra variants. Empty disables the sweep.",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Retrain even when a finished last.pt + history.json already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.scale == "smoke":
        horizons = _parse_int_tuple(args.horizons or "1,2,5,10,20")
        cfg = build_ablation_config(
            epochs=args.epochs if args.epochs is not None else 3,
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

    cfg = build_workshop_ablation_config(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
    )
    cfg.training.device = args.device
    if args.epochs is not None:
        cfg = replace(cfg, training=replace(cfg.training, epochs=int(args.epochs)))
        cfg.training.device = args.device
    if args.dataset_path:
        cfg = apply_dataset_path(cfg, args.dataset_path, args.num_workers)
        cfg.training.device = args.device
    if args.num_workers is not None:
        cfg.training.num_workers = int(args.num_workers)
    seeds = _parse_int_tuple(args.seeds)
    ema_momenta = _parse_float_tuple(args.ema_sweep)
    report_horizons = _parse_int_tuple(args.report_horizons)
    variants = study_variants(
        include_ema_sweep=bool(ema_momenta),
        ema_momenta=ema_momenta,
    )
    if args.variants:
        wanted = {part.strip() for part in args.variants.split(",") if part.strip()}
        variants = tuple(item for item in variants if item.key in wanted)
        if not variants:
            raise ValueError(f"No variants matched --variants={args.variants!r}.")
    report = run_multiseed_ablation(
        cfg,
        seeds=seeds,
        variants=variants,
        report_horizons=report_horizons,
        probe_epochs=args.probe_epochs,
        skip_completed=not args.no_skip_completed,
    )
    LOGGER.info(
        "Multi-seed ablation complete  json=%s  csv=%s",
        report["json"],
        report["csv"],
    )
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
