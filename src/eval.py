"""Archival evaluation probes for Action-Conditioned JEPA.

* Multi-step open-loop latent rollout MSE at horizons ``k = 1 … K``.
* Frozen linear probes on ``ẑ_{t+k}`` for stiffness class and contact quadrant.
* Linear regression from ``z_t`` onto continuous stiffness (Young's modulus).
* Probe diagnostics: label coverage, train vs val accuracy, ``z_target`` health.
* MSE-only vs VICReg ablation overlay.
* Plot/report writer targeting ``outputs/metrics.png``.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Sequence

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
    particle_positions: Tensor | None = None,
    particle_velocities: Tensor | None = None,
    start_frame: int = 0,
) -> tuple[Tensor, Tensor]:
    """Predict ``ẑ_{t+k}`` and encode EMA targets ``z_{t+k}`` for ``k=1..K``.

    Args:
        observations: Episode frames ``(N, T, C, H, W)``.
        actions: Episode actions ``(N, T-1, A)``.
        max_horizon: ``K``. Capped at ``T - 1 - start_frame``.
        start_frame: Encode this index as ``z_t`` instead of always frame 0.
            Use ``t>0`` when stiffness is only visible after the object has
            already deformed.
        particle_positions: Optional lattice ``(N, T, P, D)`` for the GNN baseline.
        particle_velocities: Optional lattice velocities ``(N, T, P, D)``.

    Returns:
        ``(z_hat, z_tgt)`` each of shape ``(N, K, D)``.
    """
    model.eval()
    n_episodes, n_frames = observations.shape[0], observations.shape[1]
    if start_frame < 0 or start_frame >= n_frames - 1:
        raise ValueError(
            f"start_frame={start_frame} needs 0 <= start_frame < T-1 (T={n_frames})."
        )
    horizon_limit = n_frames - 1 - start_frame
    k_max = min(max_horizon, horizon_limit)
    if k_max < 1:
        raise ValueError("Need episode_length >= 2 for rollouts.")
    if getattr(model, "requires_particles", False):
        if particle_positions is None or particle_velocities is None:
            raise ValueError("Particle GNN eval requires dataset.positions and dataset.velocities.")
        return model.rollout_from_particles(
            particle_positions[:, start_frame:].to(device),
            particle_velocities[:, start_frame:].to(device),
            actions[:, start_frame : start_frame + k_max].to(device),
            k_max,
            batch_size=batch_size,
        )
    if getattr(model, "architecture", None) == "rssm":
        from src.rssm_eval import rssm_rollout_predictions

        return rssm_rollout_predictions(
            model,
            observations,
            actions,
            max_horizon,
            device,
            batch_size=batch_size,
            start_frame=start_frame,
        )
    latent_dim = model.latent_dim
    z_hat = torch.empty(n_episodes, k_max, latent_dim, device=device)
    z_tgt = torch.empty(n_episodes, k_max, latent_dim, device=device)
    for start in range(0, n_episodes, batch_size):
        end = min(start + batch_size, n_episodes)
        frames = observations[start:end].to(device)
        acts = actions[start:end, start_frame : start_frame + k_max].to(device)
        z_0 = model.encode(frames[:, start_frame], target=False)
        preds = model.rollout(z_0, acts)
        future = frames[:, start_frame + 1 : start_frame + 1 + k_max].reshape(-1, *frames.shape[2:])
        targets = model.encode(future, target=True).reshape(end - start, k_max, latent_dim)
        z_hat[start:end] = preds
        z_tgt[start:end] = targets
    return z_hat, z_tgt


@torch.no_grad()
def encode_frame_latents(
    model: Any,
    observations: Tensor,
    frame_index: int,
    device: torch.device,
    *,
    batch_size: int = 8,
    target: bool = False,
) -> Tensor:
    """Encode a single frame index for every episode: ``(N, D)``."""
    model.eval()
    n_episodes = observations.shape[0]
    latent_dim = model.latent_dim
    out = torch.empty(n_episodes, latent_dim, device=device)
    for start in range(0, n_episodes, batch_size):
        end = min(start + batch_size, n_episodes)
        frames = observations[start:end, frame_index].to(device)
        out[start:end] = model.encode(frames, target=target)
    return out


def horizon_mse(z_hat: Tensor, z_tgt: Tensor) -> list[float]:
    """Per-horizon mean squared error (averaged over batch and latent dim)."""
    err = (z_hat - z_tgt).pow(2).mean(dim=(0, 2))
    return [float(v) for v in err.detach().cpu()]


def _horizon_labels(labels: Tensor, horizon: int) -> Tensor:
    if labels.ndim == 1:
        return labels
    return labels[:, horizon]


def label_coverage(labels: Tensor, n_classes: int) -> dict[str, Any]:
    """How many distinct classes a split actually contains (and their counts)."""
    flat = labels.reshape(-1).detach().cpu().long()
    n_examples = int(flat.numel())
    if n_examples == 0:
        return {
            "n_examples": 0,
            "n_distinct": 0,
            "class_ids": [],
            "counts": [],
            "n_classes_configured": int(n_classes),
            "missing_class_ids": list(range(int(n_classes))),
        }
    n_bins = max(int(n_classes), int(flat.max().item()) + 1)
    counts = torch.bincount(flat, minlength=n_bins)
    present = torch.nonzero(counts, as_tuple=False).view(-1)
    missing = [i for i in range(int(n_classes)) if i >= counts.numel() or int(counts[i]) == 0]
    return {
        "n_examples": n_examples,
        "n_distinct": int(present.numel()),
        "class_ids": [int(v) for v in present.tolist()],
        "counts": [int(v) for v in counts[: max(int(n_classes), n_bins)].tolist()],
        "n_classes_configured": int(n_classes),
        "missing_class_ids": missing,
    }


def latent_health(z: Tensor, *, gamma: float = 1.0, eps: float = 1e-4) -> dict[str, Any]:
    """Norms and per-dimension spread of a latent tensor ``(..., D)``.

    VICReg's variance hinge is ``mean_j ReLU(gamma - std_j)``. If most
    dimensions sit at ``std_j >= gamma``, the hinge is not binding.
    """
    if z.numel() == 0:
        raise ValueError("latent_health received an empty tensor.")
    flat = z.reshape(-1, z.shape[-1]).float()
    n_vectors, dim = int(flat.shape[0]), int(flat.shape[1])
    norms = flat.norm(dim=-1)
    per_dim_var = flat.var(dim=0, unbiased=False)
    per_dim_std = torch.sqrt(per_dim_var + eps)
    hinge = torch.relu(gamma - per_dim_std)
    below = per_dim_std < gamma
    return {
        "n_vectors": n_vectors,
        "latent_dim": dim,
        "gamma": float(gamma),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "per_dim_variance_mean": float(per_dim_var.mean().item()),
        "per_dim_variance_median": float(per_dim_var.median().item()),
        "per_dim_std_mean": float(per_dim_std.mean().item()),
        "per_dim_std_min": float(per_dim_std.min().item()),
        "n_dims_std_below_gamma": int(below.sum().item()),
        "frac_dims_std_below_gamma": float(below.float().mean().item()),
        "variance_hinge_mean": float(hinge.mean().item()),
        "variance_hinge_binding": bool(float(hinge.mean().item()) > 1e-3),
    }


def effective_rank(z: Tensor, *, eps: float = 1e-12) -> dict[str, Any]:
    """Participation ratio of the latent covariance: ``(sum λ)² / sum λ²``.

    ``z`` is flattened to ``(N, D)``. Values near 1 are rank-1 collapse;
    values near ``D`` are isotropic. This is not the VICReg variance hinge:
    per-dimension std can sit above ``γ`` while a few eigenmodes still
    dominate the covariance.
    """
    if z.numel() == 0:
        raise ValueError("effective_rank received an empty tensor.")
    flat = z.reshape(-1, z.shape[-1]).float()
    n_vectors, dim = int(flat.shape[0]), int(flat.shape[1])
    if n_vectors < 2:
        raise ValueError("effective_rank needs at least two vectors.")
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (centered.transpose(0, 1) @ centered) / float(n_vectors - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    eig_sum = float(eig.sum().item())
    eig_sumsq = float(eig.square().sum().item())
    pr = (eig_sum * eig_sum) / max(eig_sumsq, eps)
    eig_max = float(eig.max().item())
    return {
        "participation_ratio": pr,
        "n_vectors": n_vectors,
        "latent_dim": dim,
        "eig_sum": eig_sum,
        "eig_sumsq": eig_sumsq,
        "eig_max": eig_max,
        "eig_frac_top1": eig_max / max(eig_sum, eps),
        "split": "val",
    }


def effective_rank_record_is_final(rank: dict[str, Any] | None, *, min_epochs: int = 200) -> bool:
    """True only when JSON provenance says the rank came from a finished run."""
    if not isinstance(rank, dict) or "z_target" not in rank:
        return False
    if not rank.get("final"):
        return False
    try:
        ckpt_ep = int(rank.get("checkpoint_epoch") or 0)
        jsonl_ep = int(rank.get("metrics_jsonl_train_epochs") or 0)
    except (TypeError, ValueError):
        return False
    return ckpt_ep >= min_epochs and jsonl_ep >= min_epochs


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


def _train_linear_regressor(
    features: Tensor,
    targets: Tensor,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Linear, float, float]:
    """Fit ``D → 1`` on frozen embeddings. Targets are z-scored for stability."""
    y = targets.reshape(-1).float()
    y_mean = float(y.mean().item())
    y_std = float(y.std(unbiased=False).clamp_min(1e-6).item())
    y_n = (y - y_mean) / y_std
    probe = nn.Linear(features.shape[1], 1).to(device)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    feats = features.detach()
    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = probe(feats).squeeze(-1)
        loss = F.mse_loss(pred, y_n)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe, y_mean, y_std


@torch.no_grad()
def _accuracy(probe: nn.Linear, features: Tensor, labels: Tensor) -> float:
    pred = probe(features).argmax(dim=-1)
    return float((pred == labels).float().mean().item())


@torch.no_grad()
def _regression_scores(
    probe: nn.Linear, features: Tensor, targets: Tensor, y_mean: float, y_std: float
) -> dict[str, float]:
    pred = probe(features).squeeze(-1) * y_std + y_mean
    y = targets.reshape(-1).float()
    mse = float((pred - y).pow(2).mean().item())
    ss_res = float((y - pred).pow(2).sum().item())
    ss_tot = float((y - y.mean()).pow(2).sum().item())
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {"mse": mse, "r2": r2}


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
) -> dict[str, Any]:
    """Train one frozen linear probe per horizon.

    Returns val accuracies (paper-facing) plus train accuracies and the
    per-split label coverage the probe actually saw.

    ``z_hat_*`` are ``(N, K, D)``; ``labels_*`` are either ``(N,)`` (episode-level)
    or ``(N, K)`` (step-level).
    """
    k_max = z_hat_train.shape[1]
    val_acc: list[float] = []
    train_acc: list[float] = []
    per_horizon: list[dict[str, Any]] = []
    for horizon in range(k_max):
        feat_tr = z_hat_train[:, horizon]
        feat_va = z_hat_val[:, horizon]
        y_tr = _horizon_labels(labels_train, horizon)
        y_va = _horizon_labels(labels_val, horizon)
        n_present = int(y_tr.max().item()) + 1 if y_tr.numel() else n_classes
        classes = max(n_classes, n_present)
        probe = _train_linear_probe(
            feat_tr, y_tr, classes, epochs=epochs, lr=lr, device=device
        )
        acc_tr = _accuracy(probe, feat_tr, y_tr)
        acc_va = _accuracy(probe, feat_va, y_va)
        train_acc.append(acc_tr)
        val_acc.append(acc_va)
        coverage_tr = label_coverage(y_tr, n_classes)
        coverage_va = label_coverage(y_va, n_classes)
        per_horizon.append(
            {
                "horizon": horizon + 1,
                "train_acc": acc_tr,
                "val_acc": acc_va,
                "train_coverage": coverage_tr,
                "val_coverage": coverage_va,
            }
        )
        LOGGER.info(
            "  probe k=%d  train_acc=%.3f  val_acc=%.3f  "
            "train_distinct=%d/%d n=%d  val_distinct=%d/%d n=%d  val_counts=%s",
            horizon + 1,
            acc_tr,
            acc_va,
            coverage_tr["n_distinct"],
            n_classes,
            coverage_tr["n_examples"],
            coverage_va["n_distinct"],
            n_classes,
            coverage_va["n_examples"],
            coverage_va["counts"],
        )
    return {
        "val_acc": val_acc,
        "train_acc": train_acc,
        "per_horizon": per_horizon,
        "train_coverage": label_coverage(labels_train, n_classes),
        "val_coverage": label_coverage(labels_val, n_classes),
    }


def linear_regression_over_horizon(
    z_train: Tensor,
    z_val: Tensor,
    y_train: Tensor,
    y_val: Tensor,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, Any]:
    """Frozen linear regression from ``z`` onto a continuous stiffness scalar."""
    k_max = z_train.shape[1]
    train_r2: list[float] = []
    val_r2: list[float] = []
    train_mse: list[float] = []
    val_mse: list[float] = []
    per_horizon: list[dict[str, Any]] = []
    y_tr = y_train.reshape(-1).float()
    y_va = y_val.reshape(-1).float()
    for horizon in range(k_max):
        probe, y_mean, y_std = _train_linear_regressor(
            z_train[:, horizon], y_tr, epochs=epochs, lr=lr, device=device
        )
        tr = _regression_scores(probe, z_train[:, horizon], y_tr, y_mean, y_std)
        va = _regression_scores(probe, z_val[:, horizon], y_va, y_mean, y_std)
        train_r2.append(tr["r2"])
        val_r2.append(va["r2"])
        train_mse.append(tr["mse"])
        val_mse.append(va["mse"])
        per_horizon.append(
            {
                "horizon": horizon + 1,
                "train_mse": tr["mse"],
                "val_mse": va["mse"],
                "train_r2": tr["r2"],
                "val_r2": va["r2"],
            }
        )
        LOGGER.info(
            "  stiffness-reg k=%d  train_r2=%.3f  val_r2=%.3f  train_mse=%.4g  val_mse=%.4g",
            horizon + 1,
            tr["r2"],
            va["r2"],
            tr["mse"],
            va["mse"],
        )
    return {
        "train_r2": train_r2,
        "val_r2": val_r2,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "per_horizon": per_horizon,
        "n_unique_train": int(torch.unique(y_tr).numel()),
        "n_unique_val": int(torch.unique(y_va).numel()),
        "value_mean": float(y_tr.mean().item()),
        "value_std": float(y_tr.std(unbiased=False).item()),
    }


def _chance_finding(val_acc: list[float], chance: float, n_val: int, name: str) -> str:
    if not val_acc:
        return f"{name}: no probe accuracies recorded."
    mean_acc = float(sum(val_acc) / len(val_acc))
    se = math.sqrt(chance * (1.0 - chance) / max(n_val, 1))
    z_score = (mean_acc - chance) / max(se, 1e-8)
    if n_val < 50:
        return (
            f"{name}: n_val={n_val} is too small for a {len(val_acc)}-horizon linear "
            f"probe (binomial SE≈{se:.3f} at chance={chance:.3f}). Mean val acc="
            f"{mean_acc:.3f} is indistinguishable from sampling noise — not a "
            f"representation-quality result."
        )
    if abs(z_score) < 2.0:
        return (
            f"{name}: after scaling to n_val={n_val}, mean val acc={mean_acc:.3f} is "
            f"still consistent with chance={chance:.3f} (z={z_score:.2f}, SE≈{se:.3f}). "
            f"This is a real limitation: a linear readout of {name} from the latents "
            f"does not beat chance."
        )
    direction = "above" if z_score > 0 else "below"
    return (
        f"{name}: mean val acc={mean_acc:.3f} is {direction} chance={chance:.3f} "
        f"(z={z_score:.2f}, n_val={n_val})."
    )


def evaluate_model(
    model: Any,
    dataset: Any,
    device: torch.device,
    *,
    max_horizon: int = 10,
    probe_epochs: int = 80,
    probe_lr: float = 1e-2,
    contact_bins: int = 2,
    batch_size: int = 8,
    start_frames: Sequence[int] = (0,),
) -> dict[str, Any]:
    """Run rollout MSE + stiffness/contact linear probes for one model.

    ``start_frames`` encodes ``z_t`` at those indices (default frame 0). Extra
    starts (e.g. 3, 5, 7) test stiffness after the object has already moved.
    The top-level payload is always ``start_frames[0]`` so existing tables stay
    comparable; other starts live under ``by_start_frame``.
    """
    n_train, n_val = dataset.split_sizes()
    if n_val < 1:
        raise ValueError("Need a non-empty val split for archival probes.")
    observations = dataset.observations
    raw_actions = dataset.actions
    policy_actions = getattr(dataset, "model_actions", raw_actions)
    stiffness = dataset.stiffness_labels.to(device)
    n_stiff = int(max(int(stiffness.max().item()) + 1, dataset.cfg.n_stiffness_classes))
    n_contact = contact_bins * contact_bins
    y_stiff_tr, y_stiff_va = stiffness[:n_train], stiffness[n_train : n_train + n_val]
    geometry = getattr(dataset, "geometry_labels", None)
    y_geo_tr = y_geo_va = None
    n_geo = 0
    if geometry is not None:
        geo = geometry.to(device).long()
        n_geo = int(max(int(geo.max().item()) + 1, 2))
        y_geo_tr, y_geo_va = geo[:n_train], geo[n_train : n_train + n_val]
    stiffness_values = getattr(dataset, "stiffness_values", None)
    particle_positions = getattr(dataset, "positions", None)
    particle_velocities = getattr(dataset, "velocities", None)
    t_limit = int(observations.shape[1]) - 1
    starts = []
    for raw in start_frames:
        start = int(raw)
        if 0 <= start < t_limit and start not in starts:
            starts.append(start)
    if not starts:
        raise ValueError(f"No valid start_frames in {tuple(start_frames)} (T-1={t_limit}).")

    def _pack(start_frame: int) -> dict[str, Any]:
        z_hat, z_tgt = rollout_predictions(
            model,
            observations,
            policy_actions,
            max_horizon,
            device,
            batch_size=batch_size,
            particle_positions=particle_positions,
            particle_velocities=particle_velocities,
            start_frame=start_frame,
        )
        k_here = int(z_hat.shape[1])
        mse = horizon_mse(z_hat, z_tgt)
        z_tr, z_va = z_hat[:n_train], z_hat[n_train : n_train + n_val]
        z_tgt_tr, z_tgt_va = z_tgt[:n_train], z_tgt[n_train : n_train + n_val]
        contact = contact_classes(
            raw_actions[:, start_frame : start_frame + k_here].to(device),
            bins=contact_bins,
        )
        contact_tr, contact_va = contact[:n_train], contact[n_train : n_train + n_val]
        LOGGER.info(
            "Probes start_frame=%d  n_train=%d  n_val=%d  k_max=%d",
            start_frame, n_train, n_val, k_here,
        )
        z_tgt_stats = latent_health(z_tgt)
        z_hat_stats = latent_health(z_hat)
        rank_tgt = effective_rank(z_tgt_va)
        rank_hat = effective_rank(z_va)
        stiff = linear_probe_over_horizon(
            z_tr, z_va, y_stiff_tr, y_stiff_va, n_stiff,
            epochs=probe_epochs, lr=probe_lr, device=device,
        )
        contact_probe = linear_probe_over_horizon(
            z_tr, z_va, contact_tr, contact_va, n_contact,
            epochs=probe_epochs, lr=probe_lr, device=device,
        )
        stiff_tgt = linear_probe_over_horizon(
            z_tgt_tr, z_tgt_va, y_stiff_tr, y_stiff_va, n_stiff,
            epochs=probe_epochs, lr=probe_lr, device=device,
        )
        stiffness_reg: dict[str, Any] | None = None
        if stiffness_values is not None:
            y_val_cont = stiffness_values.to(device).float()
            y_tr_c, y_va_c = y_val_cont[:n_train], y_val_cont[n_train : n_train + n_val]
            stiffness_reg = linear_regression_over_horizon(
                z_tr, z_va, y_tr_c, y_va_c, epochs=probe_epochs, lr=probe_lr, device=device
            )
        geo_probe = geo_tgt = None
        if y_geo_tr is not None and y_geo_va is not None:
            geo_probe = linear_probe_over_horizon(
                z_tr, z_va, y_geo_tr, y_geo_va, n_geo,
                epochs=probe_epochs, lr=probe_lr, device=device,
            )
            geo_tgt = linear_probe_over_horizon(
                z_tgt_tr, z_tgt_va, y_geo_tr, y_geo_va, n_geo,
                epochs=probe_epochs, lr=probe_lr, device=device,
            )
        context_stiff = context_geo = None
        if not getattr(model, "requires_particles", False):
            z_ctx = encode_frame_latents(
                model, observations, start_frame, device, batch_size=batch_size, target=False
            )
            z_ctx_tr, z_ctx_va = z_ctx[:n_train], z_ctx[n_train : n_train + n_val]
            context_stiff = linear_probe_over_horizon(
                z_ctx_tr.unsqueeze(1), z_ctx_va.unsqueeze(1),
                y_stiff_tr, y_stiff_va, n_stiff,
                epochs=probe_epochs, lr=probe_lr, device=device,
            )
            if y_geo_tr is not None and y_geo_va is not None:
                context_geo = linear_probe_over_horizon(
                    z_ctx_tr.unsqueeze(1), z_ctx_va.unsqueeze(1),
                    y_geo_tr, y_geo_va, n_geo,
                    epochs=probe_epochs, lr=probe_lr, device=device,
                )
        chance_stiff = 1.0 / max(n_stiff, 1)
        chance_contact = 1.0 / max(n_contact, 1)
        chance_geo = 1.0 / max(n_geo, 1) if n_geo else None
        findings = [
            _chance_finding(
                stiff["val_acc"], chance_stiff, n_val, f"stiffness class (start={start_frame})"
            ),
            _chance_finding(
                contact_probe["val_acc"], chance_contact, n_val, f"contact quadrant (start={start_frame})"
            ),
            _chance_finding(
                stiff_tgt["val_acc"], chance_stiff, n_val,
                f"stiffness class z_target (start={start_frame})",
            ),
        ]
        if context_stiff is not None:
            findings.append(
                _chance_finding(
                    context_stiff["val_acc"], chance_stiff, n_val,
                    f"stiffness from z_t (start={start_frame})",
                )
            )
        if geo_probe is not None:
            findings.append(
                _chance_finding(
                    geo_probe["val_acc"], float(chance_geo or 0.5), n_val,
                    f"geometry (start={start_frame})",
                )
            )
            if context_geo is not None:
                findings.append(
                    _chance_finding(
                        context_geo["val_acc"], float(chance_geo or 0.5), n_val,
                        f"geometry from z_t (start={start_frame})",
                    )
                )
        return {
            "start_frame": start_frame,
            "horizons": list(range(1, k_here + 1)),
            "rollout_mse": mse,
            "k_max": k_here,
            "stiffness_probe_acc": stiff["val_acc"],
            "contact_probe_acc": contact_probe["val_acc"],
            "stiffness_probe_train_acc": stiff["train_acc"],
            "contact_probe_train_acc": contact_probe["train_acc"],
            "stiffness_target_probe_acc": stiff_tgt["val_acc"],
            "stiffness_reg_val_r2": None if stiffness_reg is None else stiffness_reg["val_r2"],
            "stiffness_reg_train_r2": None if stiffness_reg is None else stiffness_reg["train_r2"],
            "stiffness_reg_val_mse": None if stiffness_reg is None else stiffness_reg["val_mse"],
            "geometry_probe_acc": None if geo_probe is None else geo_probe["val_acc"],
            "geometry_target_probe_acc": None if geo_tgt is None else geo_tgt["val_acc"],
            "context_stiffness_val_acc": None if context_stiff is None else context_stiff["val_acc"],
            "context_geometry_val_acc": None if context_geo is None else context_geo["val_acc"],
            "chance_stiffness": chance_stiff,
            "chance_contact": chance_contact,
            "chance_geometry": chance_geo,
            "effective_rank": {"z_target": rank_tgt, "z_hat": rank_hat},
            "diagnostics": {
                "start_frame": start_frame,
                "n_train": n_train,
                "n_val": n_val,
                "k_max": k_here,
                "stiffness_val_coverage": stiff["val_coverage"],
                "stiffness_train_coverage": stiff["train_coverage"],
                "contact_val_coverage": contact_probe["val_coverage"],
                "contact_train_coverage": contact_probe["train_coverage"],
                "z_target": z_tgt_stats,
                "z_hat": z_hat_stats,
                "effective_rank": {"z_target": rank_tgt, "z_hat": rank_hat},
                "stiffness_probe": stiff,
                "contact_probe": contact_probe,
                "stiffness_target_encoder_probe": stiff_tgt,
                "stiffness_regression": stiffness_reg,
                "geometry_probe": geo_probe,
                "context_stiffness_probe": context_stiff,
                "context_geometry_probe": context_geo,
                "findings": findings,
            },
            "findings": findings,
        }

    packed = {start: _pack(start) for start in starts}
    primary = packed[starts[0]]
    LOGGER.info("Probe findings: %s", " | ".join(primary["findings"]))
    payload = {
        "horizons": primary["horizons"],
        "rollout_mse": primary["rollout_mse"],
        "stiffness_probe_acc": primary["stiffness_probe_acc"],
        "contact_probe_acc": primary["contact_probe_acc"],
        "stiffness_probe_train_acc": primary["stiffness_probe_train_acc"],
        "contact_probe_train_acc": primary["contact_probe_train_acc"],
        "stiffness_target_probe_acc": primary["stiffness_target_probe_acc"],
        "stiffness_reg_val_r2": primary["stiffness_reg_val_r2"],
        "stiffness_reg_train_r2": primary["stiffness_reg_train_r2"],
        "stiffness_reg_val_mse": primary["stiffness_reg_val_mse"],
        "geometry_probe_acc": primary["geometry_probe_acc"],
        "geometry_target_probe_acc": primary["geometry_target_probe_acc"],
        "context_stiffness_val_acc": primary["context_stiffness_val_acc"],
        "context_geometry_val_acc": primary["context_geometry_val_acc"],
        "chance_stiffness": primary["chance_stiffness"],
        "chance_contact": primary["chance_contact"],
        "chance_geometry": primary["chance_geometry"],
        "n_train": n_train,
        "n_val": n_val,
        "k_max": primary["k_max"],
        "start_frame": starts[0],
        "diagnostics": primary["diagnostics"],
        "effective_rank": primary["effective_rank"],
        "by_start_frame": {str(start): packed[start] for start in starts},
    }
    return payload


def write_probe_diagnostics(
    results: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``outputs/probe_diagnostics.json`` from ``evaluate_model`` payloads."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    limitations: list[str] = []
    for name, payload in results.items():
        diag = payload.get("diagnostics") or {}
        rank = payload.get("effective_rank") or diag.get("effective_rank")
        if isinstance(rank, dict):
            rank = dict(rank)
            rank["checkpoint_path"] = payload.get("checkpoint_source") or payload.get("checkpoint")
            rank["checkpoint_epoch"] = payload.get("checkpoint_epoch")
            rank["metrics_jsonl_train_epochs"] = payload.get("metrics_jsonl_train_epochs")
            rank["final"] = bool(payload.get("final"))
        models[name] = {
            "n_train": payload.get("n_train"),
            "n_val": payload.get("n_val"),
            "k_max": payload.get("k_max"),
            "chance_stiffness": payload.get("chance_stiffness"),
            "chance_contact": payload.get("chance_contact"),
            "stiffness_probe_acc": payload.get("stiffness_probe_acc"),
            "stiffness_probe_train_acc": payload.get("stiffness_probe_train_acc"),
            "contact_probe_acc": payload.get("contact_probe_acc"),
            "contact_probe_train_acc": payload.get("contact_probe_train_acc"),
            "stiffness_target_probe_acc": payload.get("stiffness_target_probe_acc"),
            "stiffness_reg_val_r2": payload.get("stiffness_reg_val_r2"),
            "stiffness_reg_train_r2": payload.get("stiffness_reg_train_r2"),
            "stiffness_reg_val_mse": payload.get("stiffness_reg_val_mse"),
            "geometry_probe_acc": payload.get("geometry_probe_acc"),
            "geometry_target_probe_acc": payload.get("geometry_target_probe_acc"),
            "context_stiffness_val_acc": payload.get("context_stiffness_val_acc"),
            "context_geometry_val_acc": payload.get("context_geometry_val_acc"),
            "chance_geometry": payload.get("chance_geometry"),
            "start_frame": payload.get("start_frame", 0),
            "by_start_frame": payload.get("by_start_frame"),
            "stiffness_val_coverage": diag.get("stiffness_val_coverage"),
            "stiffness_train_coverage": diag.get("stiffness_train_coverage"),
            "contact_val_coverage": diag.get("contact_val_coverage"),
            "z_target": diag.get("z_target"),
            "z_hat": diag.get("z_hat"),
            "stiffness_regression": diag.get("stiffness_regression"),
            "effective_rank": rank,
            "checkpoint_path": payload.get("checkpoint_source") or payload.get("checkpoint"),
            "checkpoint_epoch": payload.get("checkpoint_epoch"),
            "metrics_jsonl_train_epochs": payload.get("metrics_jsonl_train_epochs"),
            "final": bool(payload.get("final")),
            "findings": diag.get("findings", []),
        }
        limitations.extend(str(item) for item in diag.get("findings", []))
    document: dict[str, Any] = {
        "models": models,
        "limitation": limitations,
    }
    if extra:
        document.update(extra)
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", output_path)
    return output_path


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
    models: dict[str, Any],
    dataset: Any,
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
        name: {key: value for key, value in payload.items() if key != "diagnostics"}
        for name, payload in results.items()
    }
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", json_path)
    diag_path = Path(metrics_png).resolve().parent / "probe_diagnostics.json"
    write_probe_diagnostics(results, diag_path)
    return {
        "models": serializable,
        "metrics_png": str(png_path),
        "metrics_json": str(json_path),
        "probe_diagnostics": str(diag_path),
    }


def main(argv: list[str] | None = None) -> None:
    """Standalone eval entrypoint; prefer ``python src/train.py --mode eval``."""
    del argv
    raise SystemExit("Use `python src/train.py --mode eval --ckpt PATH` instead.")


if __name__ == "__main__":
    main(sys.argv[1:])
