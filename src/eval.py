"""Archival evaluation probes for Action-Conditioned JEPA.

* Multi-step open-loop latent rollout MSE at horizons ``k = 1 … K``.
* Frozen linear probes on ``ẑ_{t+k}`` for stiffness class and contact quadrant.
* MSE-only vs VICReg ablation overlay.
* Plot/report writer targeting ``outputs/metrics.png``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

if __package__:
    from .dataset import DeformableObjectDataset
    from .models.ac_jepa import ACJEPA
else:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.dataset import DeformableObjectDataset
    from src.models.ac_jepa import ACJEPA

LOGGER = logging.getLogger("ac_jepa.eval")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional until requirements install
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None


def contact_classes(actions: Tensor, bins: int = 2) -> Tensor:
    """Quantize 2D contact ``(cx, cy)`` into a ``bins x bins`` class grid.

    Args:
        actions: ``(..., A_dim)`` with contact in the first two components.
        bins: Bins per axis. ``2`` yields four quadrants.

    Returns:
        Long tensor of class ids with shape ``actions.shape[:-1]``.
    """
    if bins < 2:
        raise ValueError("bins must be >= 2.")
    cx = actions[..., 0].clamp(0.0, 1.0 - 1e-6)
    cy = actions[..., 1].clamp(0.0, 1.0 - 1e-6)
    bx = (cx * bins).long()
    by = (cy * bins).long()
    return by * bins + bx


@torch.no_grad()
def rollout_predictions(
    model: Any,
    observations: Tensor,
    actions: Tensor,
    max_horizon: int,
    device: torch.device,
    batch_size: int = 8,
) -> tuple[Tensor, Tensor]:
    """Predict ``ẑ_{t+k}`` and encode EMA targets ``z_{t+k}`` for ``k=1..K``.

    Args:
        observations: Episode frames ``(N, T, C, H, W)``.
        actions: Episode actions ``(N, T-1, A)``.
        max_horizon: ``K``. Capped at ``T - 1``.

    Returns:
        ``(z_hat, z_tgt)`` each of shape ``(N, K, D)``.
    """
    model.eval()
    n_episodes, horizon_limit = observations.shape[0], observations.shape[1] - 1
    k_max = min(max_horizon, horizon_limit)
    if k_max < 1:
        raise ValueError("Need episode_length >= 2 for rollouts.")
    latent_dim = model.latent_dim
    z_hat = torch.empty(n_episodes, k_max, latent_dim, device=device)
    z_tgt = torch.empty(n_episodes, k_max, latent_dim, device=device)
    for start in range(0, n_episodes, batch_size):
        end = min(start + batch_size, n_episodes)
        frames = observations[start:end].to(device)
        acts = actions[start:end, :k_max].to(device)
        z_0 = model.encode(frames[:, 0], target=False)
        preds = model.rollout(z_0, acts)
        future = frames[:, 1 : k_max + 1].reshape(-1, *frames.shape[2:])
        targets = model.encode(future, target=True).reshape(end - start, k_max, latent_dim)
        z_hat[start:end] = preds
        z_tgt[start:end] = targets
    return z_hat, z_tgt


def horizon_mse(z_hat: Tensor, z_tgt: Tensor) -> list[float]:
    """Per-horizon mean squared error (averaged over batch and latent dim)."""
    err = (z_hat - z_tgt).pow(2).mean(dim=(0, 2))
    return [float(v) for v in err.detach().cpu()]


def _train_linear_probe(
    features: Tensor,
    labels: Tensor,
    n_classes: int,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> nn.Linear:
    """Fit a single linear layer on frozen embeddings."""
    probe = nn.Linear(features.shape[1], n_classes).to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    feats = features.detach()
    y = labels.long()
    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(feats), y)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


@torch.no_grad()
def _accuracy(probe: nn.Linear, features: Tensor, labels: Tensor) -> float:
    pred = probe(features).argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def linear_probe_over_horizon(
    z_hat_train: Tensor,
    z_hat_val: Tensor,
    labels_train: Tensor,
    labels_val: Tensor,
    n_classes: int,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Train one frozen linear probe per horizon and return val accuracies.

    ``z_hat_*`` are ``(N, K, D)``; ``labels_*`` are either ``(N,)`` (episode-level)
    or ``(N, K)`` (step-level).
    """
    k_max = z_hat_train.shape[1]
    accuracies: list[float] = []
    for horizon in range(k_max):
        feat_tr = z_hat_train[:, horizon]
        feat_va = z_hat_val[:, horizon]
        if labels_train.ndim == 1:
            y_tr = labels_train
            y_va = labels_val
        else:
            y_tr = labels_train[:, horizon]
            y_va = labels_val[:, horizon]
        n_present = int(y_tr.max().item()) + 1 if y_tr.numel() else n_classes
        classes = max(n_classes, n_present)
        probe = _train_linear_probe(
            feat_tr, y_tr, classes, epochs=epochs, lr=lr, device=device
        )
        accuracies.append(_accuracy(probe, feat_va, y_va))
    return accuracies


def evaluate_model(
    model: ACJEPA,
    dataset: DeformableObjectDataset,
    device: torch.device,
    *,
    max_horizon: int = 10,
    probe_epochs: int = 80,
    probe_lr: float = 1e-2,
    contact_bins: int = 2,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run rollout MSE + stiffness/contact linear probes for one model."""
    n_train, n_val = dataset.split_sizes()
    if n_val < 1:
        raise ValueError("Need a non-empty val split for archival probes.")
    k_max = min(max_horizon, dataset.pairs_per_episode)
    observations = dataset.observations
    raw_actions = dataset.actions
    policy_actions = getattr(dataset, "model_actions", raw_actions)
    stiffness = dataset.stiffness_labels.to(device)
    z_hat, z_tgt = rollout_predictions(
        model, observations, policy_actions, k_max, device, batch_size=batch_size
    )
    mse = horizon_mse(z_hat, z_tgt)
    z_tr, z_va = z_hat[:n_train], z_hat[n_train : n_train + n_val]
    y_stiff_tr, y_stiff_va = stiffness[:n_train], stiffness[n_train : n_train + n_val]
    contact = contact_classes(raw_actions[:, :k_max].to(device), bins=contact_bins)
    contact_tr, contact_va = contact[:n_train], contact[n_train : n_train + n_val]
    n_stiff = int(max(int(stiffness.max().item()) + 1, dataset.cfg.n_stiffness_classes))
    n_contact = contact_bins * contact_bins
    stiff_acc = linear_probe_over_horizon(
        z_tr, z_va, y_stiff_tr, y_stiff_va, n_stiff,
        epochs=probe_epochs, lr=probe_lr, device=device,
    )
    contact_acc = linear_probe_over_horizon(
        z_tr, z_va, contact_tr, contact_va, n_contact,
        epochs=probe_epochs, lr=probe_lr, device=device,
    )
    chance_stiff = 1.0 / max(n_stiff, 1)
    chance_contact = 1.0 / max(n_contact, 1)
    return {
        "horizons": list(range(1, k_max + 1)),
        "rollout_mse": mse,
        "stiffness_probe_acc": stiff_acc,
        "contact_probe_acc": contact_acc,
        "chance_stiffness": chance_stiff,
        "chance_contact": chance_contact,
        "n_train": n_train,
        "n_val": n_val,
        "k_max": k_max,
    }


def plot_horizon_metrics(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write a two-panel figure: rollout MSE and linear-probe accuracy vs ``k``."""
    if plt is None:
        raise ImportError(
            "matplotlib is required to write metrics.png. "
            "Install it with `pip install matplotlib`."
        ) from _MATPLOTLIB_ERROR

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    palette = {
        "vicreg": "#1f77b4",
        "mse": "#d62728",
        "mse_baseline": "#d62728",
    }
    labels = {
        "vicreg": "AC-JEPA (VICReg)",
        "mse": "MSE-only baseline",
        "mse_baseline": "MSE-only baseline",
    }

    ax_mse, ax_acc = axes
    for name, payload in results.items():
        horizons = payload["horizons"]
        color = palette.get(name, None)
        legend = labels.get(name, name)
        ax_mse.plot(horizons, payload["rollout_mse"], marker="o", color=color, label=legend)
        ax_acc.plot(
            horizons,
            payload["stiffness_probe_acc"],
            marker="o",
            color=color,
            label=f"{legend} — stiffness",
        )
        ax_acc.plot(
            horizons,
            payload["contact_probe_acc"],
            marker="s",
            linestyle="--",
            color=color,
            alpha=0.75,
            label=f"{legend} — contact",
        )

    first = next(iter(results.values()))
    ax_mse.set_title("Multi-step latent prediction MSE")
    ax_mse.set_xlabel("horizon k")
    ax_mse.set_ylabel(r"MSE$(\hat{z}_{t+k}, z_{t+k})$")
    ax_mse.set_xticks(first["horizons"])
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(frameon=False)

    ax_acc.set_title("Frozen linear probe accuracy")
    ax_acc.set_xlabel("horizon k")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_xticks(first["horizons"])
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.axhline(first["chance_stiffness"], color="#7f7f7f", linestyle=":", linewidth=1.0, label="chance (stiffness)")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Wrote %s", output_path)
    return output_path


def run_archival_evaluation(
    models: dict[str, ACJEPA],
    dataset: DeformableObjectDataset,
    device: torch.device,
    metrics_png: Path,
    max_horizon: int = 10,
    probe_epochs: int = 80,
    probe_lr: float = 1e-2,
    contact_bins: int = 2,
) -> dict[str, Any]:
    """Evaluate every named model and export ``outputs/metrics.png``."""
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        LOGGER.info("Archival eval: %s  (K<=%d)", name, max_horizon)
        results[name] = evaluate_model(
            model,
            dataset,
            device,
            max_horizon=max_horizon,
            probe_epochs=probe_epochs,
            probe_lr=probe_lr,
            contact_bins=contact_bins,
        )
        mse_str = ", ".join(f"{v:.4f}" for v in results[name]["rollout_mse"])
        acc_str = ", ".join(f"{v:.3f}" for v in results[name]["stiffness_probe_acc"])
        LOGGER.info("  rollout MSE[k]: %s", mse_str)
        LOGGER.info("  stiffness acc[k]: %s", acc_str)

    png_path = plot_horizon_metrics(results, Path(metrics_png))
    json_path = Path(metrics_png).with_suffix(".json")
    serializable = {
        name: {
            key: value
            for key, value in payload.items()
        }
        for name, payload in results.items()
    }
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", json_path)
    return {
        "models": serializable,
        "metrics_png": str(png_path),
        "metrics_json": str(json_path),
    }


def main(argv: list[str] | None = None) -> None:
    """Standalone eval entrypoint; prefer ``python src/train.py --mode eval``."""
    del argv
    raise SystemExit("Use `python src/train.py --mode eval --ckpt PATH` instead.")


if __name__ == "__main__":
    main(sys.argv[1:])
