"""Transform AC-JEPA experimental outputs into camera-ready paper assets.

1. **LaTeX table** — multi-step rollout MSE (k=1,5,10,20), stiffness / friction
   linear-probe accuracy, and inference latency (FPS and ms/frame).
2. **Trajectory visualizer** — side-by-side PCA and t-SNE of ground-truth
   physical latent paths versus AC-JEPA open-loop predictions.
3. **Exporter** — writes ``outputs/paper_tables.tex`` ready to ``\\input{}``
   into a NeurIPS / PhysWorld template.

Run::

    python src/utils/export_paper_artifacts.py
    python src/utils/export_paper_artifacts.py --csv outputs/ablation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

if __package__:
    from ..ablation import VARIANTS
    from ..config import resolve_device, seed_everything
    from ..dataset import build_dataset
    from ..eval import evaluate_model, rollout_predictions
    from ..train import load_model_from_checkpoint
else:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.ablation import VARIANTS
    from src.config import resolve_device, seed_everything
    from src.dataset import build_dataset
    from src.eval import evaluate_model, rollout_predictions
    from src.train import load_model_from_checkpoint

LOGGER = logging.getLogger("ac_jepa.paper")

TABLE_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)
PROBE_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except ImportError as exc:  # pragma: no cover
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    LineCollection = None  # type: ignore[assignment]
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None

FloatArr = npt.NDArray[np.float64]

_METRICS_ALIASES: dict[str, tuple[str, ...]] = {
    "full_vicreg": ("full_vicreg", "vicreg"),
    "variant_a_no_cov": ("variant_a_no_cov", "vicreg_no_cov"),
    "variant_b_mse": ("variant_b_mse", "mse", "mse_baseline"),
    "variant_c_pixel_ae": ("variant_c_pixel_ae", "pixel_ae"),
}


@dataclass
class VariantPaperRow:
    """One method row in the camera-ready ablation table."""

    key: str
    label: str
    mse: dict[int, float] = field(default_factory=dict)
    stiffness_acc: float | None = None
    friction_acc: float | None = None
    fps: float | None = None
    ms_per_frame: float | None = None


def load_ablation_csv(path: Path | str) -> list[VariantPaperRow]:
    """Parse ``ablation_results.csv`` into one row per variant."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ablation CSV not found: {path}")
    grouped: dict[str, VariantPaperRow] = {}
    order: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            key = str(raw.get("variant") or raw.get("key") or "").strip()
            if not key:
                continue
            if key not in grouped:
                label = str(raw.get("variant_label") or raw.get("label") or key)
                grouped[key] = VariantPaperRow(key=key, label=label)
                order.append(key)
            horizon = int(float(raw["horizon"]))
            grouped[key].mse[horizon] = float(raw["mse"])
    if not order:
        raise ValueError(f"No ablation rows parsed from {path}")
    return [grouped[key] for key in order]


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _fmt_mse(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        return r"---"
    body = f"{value:.3f}"
    return rf"\textbf{{{body}}}" if bold else body


def _fmt_pct(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        return r"---"
    body = f"{100.0 * value:.1f}"
    return rf"\textbf{{{body}}}" if bold else body


def _fmt_fps(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        return r"---"
    body = f"{value:.1f}"
    return rf"\textbf{{{body}}}" if bold else body


def _fmt_ms(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        return r"---"
    body = f"{value:.2f}"
    return rf"\textbf{{{body}}}" if bold else body


def _best_mask(values: Sequence[float | None], *, minimize: bool) -> list[bool]:
    present = [v for v in values if v is not None]
    if not present:
        return [False] * len(values)
    target = min(present) if minimize else max(present)
    return [v is not None and abs(v - target) <= 1e-12 for v in values]


def render_ablation_table(
    rows: Sequence[VariantPaperRow],
    *,
    horizons: Sequence[int] = TABLE_HORIZONS,
    caption: str | None = None,
    label: str = "tab:ac-jepa-ablation",
) -> str:
    """NeurIPS-style ``booktabs`` table spanning MSE, probes, and latency."""
    if not rows:
        raise ValueError("render_ablation_table requires at least one variant.")
    caption = caption or (
        "Ablation of AC-JEPA world-model variants. Multi-step open-loop latent "
        r"rollout MSE ($\downarrow$) at horizons $k\in\{1,5,10,20\}$. "
        r"Physical-property linear-probe accuracy ($\uparrow$, \%) on stiffness "
        r"class and frictional contact (contact-quadrant labels). "
        r"Inference cost is next-step world-model latency: frames per second "
        r"($\uparrow$) and wall-clock milliseconds per frame ($\downarrow$). "
        r"Best result in each column is bold."
    )
    mse_cols = {k: _best_mask([row.mse.get(k) for row in rows], minimize=True) for k in horizons}
    stiff_best = _best_mask([row.stiffness_acc for row in rows], minimize=False)
    fric_best = _best_mask([row.friction_acc for row in rows], minimize=False)
    fps_best = _best_mask([row.fps for row in rows], minimize=False)
    ms_best = _best_mask([row.ms_per_frame for row in rows], minimize=True)
    mse_headers = " & ".join(rf"$k{{=}}{k}$" for k in horizons)
    n_mse = len(tuple(horizons))
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{l{'c' * n_mse}ccrr}}",
        r"    \toprule",
        rf"    & \multicolumn{{{n_mse}}}{{c}}{{Rollout MSE $\downarrow$}}",
        r"      & \multicolumn{2}{c}{Probe acc.\ $\uparrow$ (\%)}",
        r"      & \multicolumn{2}{c}{Latency} \\",
        rf"    \cmidrule(lr){{2-{1 + n_mse}}}",
        rf"    \cmidrule(lr){{{2 + n_mse}-{3 + n_mse}}}",
        rf"    \cmidrule(lr){{{4 + n_mse}-{5 + n_mse}}}",
        rf"    Method & {mse_headers} & Stiffness & Friction & FPS $\uparrow$ & ms/frame $\downarrow$ \\",
        r"    \midrule",
    ]
    for i, row in enumerate(rows):
        mse_cells = [_fmt_mse(row.mse.get(k), bold=mse_cols[k][i]) for k in horizons]
        cells = [
            _latex_escape(row.label),
            *mse_cells,
            _fmt_pct(row.stiffness_acc, bold=stiff_best[i]),
            _fmt_pct(row.friction_acc, bold=fric_best[i]),
            _fmt_fps(row.fps, bold=fps_best[i]),
            _fmt_ms(row.ms_per_frame, bold=ms_best[i]),
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_trajectory_figure(
    figure_name: str = "paper_latent_trajectories.pdf",
    *,
    caption: str | None = None,
    label: str = "fig:ac-jepa-latent-traj",
) -> str:
    """LaTeX figure environment pointing at the PCA / t-SNE comparison plot."""
    caption = caption or (
        "Side-by-side latent trajectories of ground-truth physical rollouts "
        r"(target-encoder embeddings $z_{t:t+k}$) versus AC-JEPA open-loop "
        r"predictions $\hat{z}_{t:t+k}$. Left column: 2D PCA. Right column: t-SNE. "
        "Each polyline is one validation episode; color denotes time along the path."
    )
    return "\n".join(
        [
            r"\begin{figure}[t]",
            r"  \centering",
            rf"  \includegraphics[width=\linewidth]{{{figure_name}}}",
            rf"  \caption{{{caption}}}",
            rf"  \label{{{label}}}",
            r"\end{figure}",
            "",
        ]
    )


def pca_embed(features: npt.ArrayLike, n_components: int = 2) -> FloatArr:
    """Project ``(N, D)`` features onto the top principal components via SVD."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"pca_embed expects (N, D), got {x.shape}.")
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    if x.shape[0] == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    rank = min(centered.shape[0], centered.shape[1], n_components)
    if rank == 0:
        return np.zeros((x.shape[0], n_components), dtype=np.float64)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:rank].T
    if rank < n_components:
        pad = np.zeros((x.shape[0], n_components - rank), dtype=np.float64)
        projected = np.concatenate([projected, pad], axis=1)
    return projected


def _pairwise_sq_dists(x: FloatArr) -> FloatArr:
    gram = x @ x.T
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * gram, 0.0)
    np.fill_diagonal(d2, 0.0)
    return d2


def _binary_search_beta(d2_row: FloatArr, perplexity: float, max_iter: int = 50) -> FloatArr:
    """Return a softmax affinity row with entropy close to ``log(perplexity)``."""
    target = np.log(perplexity)
    beta = 1.0
    beta_min, beta_max = -np.inf, np.inf
    p = np.ones_like(d2_row) / max(d2_row.size, 1)
    for _ in range(max_iter):
        log_p = -d2_row * beta
        log_p -= np.max(log_p)
        weights = np.exp(log_p)
        mass = np.sum(weights)
        if mass <= 1e-12:
            beta *= 2.0
            continue
        p = weights / mass
        entropy = -np.sum(p * np.log(p + 1e-12))
        error = entropy - target
        if abs(error) < 1e-5:
            break
        if error > 0:
            beta_min = beta
            beta = beta * 2.0 if not np.isfinite(beta_max) else 0.5 * (beta + beta_max)
        else:
            beta_max = beta
            beta = beta / 2.0 if not np.isfinite(beta_min) else 0.5 * (beta + beta_min)
    return p


def _joint_probabilities(x: FloatArr, perplexity: float) -> FloatArr:
    n = x.shape[0]
    d2 = _pairwise_sq_dists(x)
    p_cond = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        p_cond[i, mask] = _binary_search_beta(d2[i, mask], perplexity)
    joint = (p_cond + p_cond.T) / (2.0 * n)
    joint = np.maximum(joint, 1e-12)
    joint /= joint.sum()
    return joint


def _tsne_numpy(
    x: FloatArr,
    n_components: int,
    perplexity: float,
    n_iter: int,
    learning_rate: float,
    seed: int,
) -> FloatArr:
    """Exact (non-Barnes-Hut) t-SNE for modest ``N`` (paper-scale episode sets)."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    perp = float(min(max(perplexity, 2.0), max((n - 1) / 3.0, 2.0)))
    p = _joint_probabilities(x, perp)
    y = pca_embed(x, n_components)
    scale = 1e-4 / (np.std(y) + 1e-12)
    y = y * scale
    y += rng.normal(0.0, 1e-6, size=y.shape)
    velocity = np.zeros_like(y)
    gains = np.ones_like(y)
    for step in range(n_iter):
        p_use = p * 12.0 if step < min(50, max(n_iter // 4, 1)) else p
        d2 = _pairwise_sq_dists(y)
        q_unnorm = 1.0 / (1.0 + d2)
        np.fill_diagonal(q_unnorm, 0.0)
        q_sum = float(q_unnorm.sum())
        q = np.maximum(q_unnorm / max(q_sum, 1e-12), 1e-12)
        pq = p_use - q
        grad = np.zeros_like(y)
        for i in range(n):
            diff = y[i] - y
            grad[i] = 4.0 * np.sum((pq[i] * q_unnorm[i])[:, None] * diff, axis=0)
        same_sign = np.sign(grad) == np.sign(velocity)
        gains = np.where(same_sign, gains * 0.8, gains + 0.2)
        gains = np.maximum(gains, 0.01)
        momentum = 0.5 if step < 20 else 0.8
        velocity = momentum * velocity - learning_rate * gains * grad
        y = y + velocity
        y -= y.mean(axis=0, keepdims=True)
    return y


def tsne_embed(
    features: npt.ArrayLike,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 400,
    learning_rate: float = 100.0,
    seed: int = 0,
) -> FloatArr:
    """2D t-SNE embedding. Uses scikit-learn when installed, else a NumPy fallback."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"tsne_embed expects (N, D), got {x.shape}.")
    if x.shape[0] < 4:
        return pca_embed(x, n_components)
    # sklearn requires max_iter >= 250; keep the NumPy path for short debug runs.
    if n_iter >= 250:
        try:
            from sklearn.manifold import TSNE

            perp = min(perplexity, max((x.shape[0] - 1) / 3.0, 2.0))
            kwargs: dict[str, Any] = {
                "n_components": n_components,
                "perplexity": perp,
                "init": "pca",
                "learning_rate": learning_rate,
                "random_state": seed,
            }
            try:
                embedder = TSNE(max_iter=n_iter, **kwargs)
            except TypeError:
                embedder = TSNE(n_iter=n_iter, **kwargs)  # type: ignore[call-arg]
            return np.asarray(embedder.fit_transform(x), dtype=np.float64)
        except ImportError:
            pass
    return _tsne_numpy(
        x,
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        seed=seed,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _policy_actions(dataset: Any) -> Tensor:
    return getattr(dataset, "model_actions", dataset.actions)


@torch.no_grad()
def measure_inference_latency(
    model: Any,
    frames: Tensor,
    actions: Tensor,
    device: torch.device,
    *,
    warmup: int = 10,
    steps: int = 50,
) -> dict[str, float]:
    """Wall-clock next-step inference: ``Predictor(Encoder(x_t), a_t)``.

    Returns milliseconds per frame and frames per second for batch size 1.
    """
    model.eval()
    x = frames[:1].to(device, non_blocking=True)
    a = actions[:1].to(device, non_blocking=True)

    def _step() -> None:
        latent = model.encode(x, target=False)
        model.predictor(latent, a)

    for _ in range(max(warmup, 0)):
        _step()
    _sync(device)
    started = time.perf_counter()
    for _ in range(max(steps, 1)):
        _step()
    _sync(device)
    elapsed = time.perf_counter() - started
    n_frames = max(steps, 1) * int(x.shape[0])
    ms_per_frame = 1000.0 * elapsed / n_frames
    fps = 1000.0 / ms_per_frame if ms_per_frame > 0 else 0.0
    return {
        "ms_per_frame": ms_per_frame,
        "fps": fps,
        "seconds": elapsed,
        "n_frames": float(n_frames),
    }


def _mean_at_horizons(values: Sequence[float], horizons: Sequence[int]) -> float | None:
    picked: list[float] = []
    for horizon in horizons:
        idx = int(horizon) - 1
        if 0 <= idx < len(values):
            picked.append(float(values[idx]))
    if not picked:
        return None
    return float(sum(picked) / len(picked))


def _lookup_cached_metrics(blob: dict[str, Any], variant_key: str) -> dict[str, Any] | None:
    aliases = _METRICS_ALIASES.get(variant_key, (variant_key,))
    models = blob.get("models", blob)
    if not isinstance(models, dict):
        return None
    for alias in aliases:
        payload = models.get(alias)
        if isinstance(payload, dict) and (
            "stiffness_probe_acc" in payload or "rollout_mse" in payload
        ):
            return payload
    return None


def load_cached_probes(paths: Sequence[Path]) -> dict[str, dict[str, float | None]]:
    """Read stiffness / contact probes from ``metrics.json`` or ``run_report.json``."""
    cached: dict[str, dict[str, float | None]] = {}
    for path in paths:
        if not path.is_file():
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        if "evaluation" in blob and isinstance(blob["evaluation"], dict):
            blob = blob["evaluation"]
        for variant in VARIANTS:
            payload = _lookup_cached_metrics(blob, variant.key)
            if payload is None:
                continue
            cached[variant.key] = {
                "stiffness_acc": _mean_at_horizons(
                    payload.get("stiffness_probe_acc", []), PROBE_HORIZONS
                ),
                "friction_acc": _mean_at_horizons(
                    payload.get("contact_probe_acc", []), PROBE_HORIZONS
                ),
            }
    return cached


def discover_checkpoints(output_dir: Path) -> dict[str, Path]:
    """Locate ablation ``last.pt`` files from the report or the default layout."""
    found: dict[str, Path] = {}
    report_path = output_dir / "ablation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for key, meta in report.get("training", {}).items():
            ckpt = Path(str(meta.get("checkpoint", "")))
            if ckpt.is_file():
                found[str(key)] = ckpt
    for variant in VARIANTS:
        fallback = output_dir / "ablation" / variant.key / "last.pt"
        if variant.key not in found and fallback.is_file():
            found[variant.key] = fallback
    return found


@torch.no_grad()
def collect_latent_trajectories(
    model: Any,
    dataset: Any,
    device: torch.device,
    *,
    n_episodes: int = 6,
    max_horizon: int = 20,
    batch_size: int = 8,
) -> tuple[FloatArr, FloatArr]:
    """Ground-truth target embeddings vs autoregressive predicted latents.

    Returns ``(z_gt, z_pred)`` each shaped ``(E, T, D)`` with ``T = K+1``
    (the initial state plus ``K`` predicted / encoded future steps).
    """
    model.eval()
    n_train, n_val = dataset.split_sizes()
    if n_val < 1:
        raise ValueError("Trajectory export requires a non-empty validation split.")
    n_use = max(1, min(int(n_episodes), n_val))
    observations = dataset.observations[n_train : n_train + n_use]
    actions = _policy_actions(dataset)[n_train : n_train + n_use]
    k_max = min(int(max_horizon), int(observations.shape[1] - 1))
    if k_max < 1:
        raise ValueError("Need episode_length >= 2 for latent trajectories.")
    z_hat, z_tgt = rollout_predictions(
        model, observations, actions, k_max, device, batch_size=batch_size
    )
    z_init_online = []
    z_init_target = []
    for start in range(0, n_use, batch_size):
        end = min(start + batch_size, n_use)
        frame0 = observations[start:end, 0].to(device)
        z_init_online.append(model.encode(frame0, target=False))
        z_init_target.append(model.encode(frame0, target=True))
    z_online = torch.cat(z_init_online, dim=0)
    z_target0 = torch.cat(z_init_target, dim=0)
    z_gt = torch.cat([z_target0[:, None], z_tgt], dim=1)
    z_pred = torch.cat([z_online[:, None], z_hat], dim=1)
    return (
        z_gt.detach().cpu().numpy().astype(np.float64),
        z_pred.detach().cpu().numpy().astype(np.float64),
    )


def _colorline(ax: Any, xy: FloatArr, cmap: str, linewidth: float) -> None:
    """Draw a polyline whose color tracks normalized time."""
    if xy.shape[0] < 2:
        ax.scatter(xy[:, 0], xy[:, 1], c=[0.0], cmap=cmap, s=18)
        return
    points = xy.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    values = np.linspace(0.0, 1.0, xy.shape[0] - 1)
    collection = LineCollection(
        segs, cmap=cmap, norm=plt.Normalize(0.0, 1.0), linewidth=linewidth
    )
    collection.set_array(values)
    ax.add_collection(collection)
    ax.scatter(xy[0, 0], xy[0, 1], s=22, c="k", zorder=3)
    ax.scatter(xy[-1, 0], xy[-1, 1], s=28, marker="*", c="k", zorder=3)


def _axis_limits(xy: FloatArr, pad_frac: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-6)
    pad = pad_frac * span
    xlim = (float(xy[:, 0].min() - pad[0]), float(xy[:, 0].max() + pad[0]))
    ylim = (float(xy[:, 1].min() - pad[1]), float(xy[:, 1].max() + pad[1]))
    return xlim, ylim


def plot_latent_trajectories(
    z_gt: npt.ArrayLike,
    z_pred: npt.ArrayLike,
    output_path: Path | str,
    *,
    dpi: int = 300,
    tsne_iter: int = 400,
    perplexity: float = 30.0,
    title_gt: str = "Ground truth",
    title_pred: str = "AC-JEPA predicted",
) -> Path:
    """Side-by-side PCA / t-SNE comparison of physical vs predicted latent paths.

    ``z_gt`` and ``z_pred`` are ``(E, T, D)``. Embeddings are fit on the pooled
    cloud so both views share a coordinate system.
    """
    if plt is None or LineCollection is None:
        raise ImportError(
            "matplotlib is required to write trajectory figures. `pip install matplotlib`."
        ) from _MATPLOTLIB_ERROR
    z_gt_arr = np.asarray(z_gt, dtype=np.float64)
    z_pred_arr = np.asarray(z_pred, dtype=np.float64)
    if z_gt_arr.shape != z_pred_arr.shape or z_gt_arr.ndim != 3:
        raise ValueError(
            f"Expected matching (E, T, D) trajectory tensors, got {z_gt_arr.shape} vs {z_pred_arr.shape}."
        )
    n_ep, n_t, _ = z_gt_arr.shape
    stacked = np.concatenate(
        [
            z_gt_arr.reshape(-1, z_gt_arr.shape[-1]),
            z_pred_arr.reshape(-1, z_pred_arr.shape[-1]),
        ],
        axis=0,
    )
    pca_xy = pca_embed(stacked, 2)
    tsne_xy = tsne_embed(stacked, 2, perplexity=perplexity, n_iter=tsne_iter, seed=0)
    pca_gt = pca_xy[: n_ep * n_t].reshape(n_ep, n_t, 2)
    pca_pr = pca_xy[n_ep * n_t :].reshape(n_ep, n_t, 2)
    tsne_gt = tsne_xy[: n_ep * n_t].reshape(n_ep, n_t, 2)
    tsne_pr = tsne_xy[n_ep * n_t :].reshape(n_ep, n_t, 2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0), dpi=dpi)
    panels = (
        (axes[0, 0], pca_gt, f"PCA — {title_gt}"),
        (axes[0, 1], pca_pr, f"PCA — {title_pred}"),
        (axes[1, 0], tsne_gt, f"t-SNE — {title_gt}"),
        (axes[1, 1], tsne_pr, f"t-SNE — {title_pred}"),
    )
    cmaps = ("viridis", "plasma", "cividis", "magma", "cool", "spring")
    pca_lim = _axis_limits(np.concatenate([pca_gt.reshape(-1, 2), pca_pr.reshape(-1, 2)], axis=0))
    tsne_lim = _axis_limits(np.concatenate([tsne_gt.reshape(-1, 2), tsne_pr.reshape(-1, 2)], axis=0))
    row_limits = (pca_lim, pca_lim, tsne_lim, tsne_lim)
    for (ax, cloud, title), limits in zip(panels, row_limits, strict=True):
        for ep in range(n_ep):
            _colorline(ax, cloud[ep], cmap=cmaps[ep % len(cmaps)], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Latent trajectories: ground-truth physics vs AC-JEPA rollout", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    LOGGER.info("Wrote %s", output_path)
    return output_path


def _apply_cached_probes(
    rows: list[VariantPaperRow], cached: dict[str, dict[str, float | None]]
) -> None:
    for row in rows:
        payload = cached.get(row.key)
        if not payload:
            continue
        if row.stiffness_acc is None:
            row.stiffness_acc = payload.get("stiffness_acc")
        if row.friction_acc is None:
            row.friction_acc = payload.get("friction_acc")


def _order_like_variants(rows: list[VariantPaperRow]) -> list[VariantPaperRow]:
    lookup = {row.key: row for row in rows}
    ordered: list[VariantPaperRow] = []
    seen: set[str] = set()
    for variant in VARIANTS:
        if variant.key in lookup:
            if lookup[variant.key].label == variant.key:
                lookup[variant.key].label = variant.label
            ordered.append(lookup[variant.key])
            seen.add(variant.key)
    for row in rows:
        if row.key not in seen:
            ordered.append(row)
    return ordered


def _enrich_from_checkpoints(
    rows: list[VariantPaperRow],
    checkpoints: dict[str, Path],
    device: torch.device,
    *,
    max_horizon: int,
    probe_epochs: int,
    n_traj_episodes: int,
    tsne_iter: int,
    figure_path: Path,
) -> Path | None:
    """Live probes, latency, and (for Full AC-JEPA) trajectory figure."""
    if not checkpoints:
        return None
    first_ckpt = next(iter(checkpoints.values()))
    _, cfg = load_model_from_checkpoint(first_ckpt, device)
    seed_everything(cfg.training.seed)
    dataset = build_dataset(cfg.data, split="all")
    n_train, n_val = dataset.split_sizes()
    n_lat = max(n_val, 1)
    frame0 = dataset.observations[n_train : n_train + n_lat, 0]
    act0 = _policy_actions(dataset)[n_train : n_train + n_lat, 0]
    figure: Path | None = None
    for row in rows:
        ckpt = checkpoints.get(row.key)
        if ckpt is None:
            continue
        LOGGER.info("Paper metrics from checkpoint %s (%s)", ckpt, row.key)
        model, _ = load_model_from_checkpoint(ckpt, device)
        k_cap = min(max_horizon, int(dataset.observations.shape[1] - 1))
        probes = evaluate_model(
            model,
            dataset,
            device,
            max_horizon=k_cap,
            probe_epochs=probe_epochs,
        )
        row.stiffness_acc = _mean_at_horizons(probes["stiffness_probe_acc"], PROBE_HORIZONS)
        row.friction_acc = _mean_at_horizons(probes["contact_probe_acc"], PROBE_HORIZONS)
        latency = measure_inference_latency(model, frame0, act0, device)
        row.fps = latency["fps"]
        row.ms_per_frame = latency["ms_per_frame"]
        want_figure = row.key == "full_vicreg" or (figure is None and row.key == rows[0].key)
        if want_figure:
            z_gt, z_pred = collect_latent_trajectories(
                model,
                dataset,
                device,
                n_episodes=n_traj_episodes,
                max_horizon=k_cap,
            )
            figure = plot_latent_trajectories(
                z_gt,
                z_pred,
                figure_path,
                tsne_iter=tsne_iter,
            )
    return figure


def write_paper_tables_tex(
    rows: Sequence[VariantPaperRow],
    output_path: Path,
    *,
    figure_name: str | None = None,
    extra_blocks: Sequence[str] = (),
) -> Path:
    """Write NeurIPS-ready table (and optional figure) fragments."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "% Auto-generated by src/utils/export_paper_artifacts.py",
        "% Drop into a NeurIPS / PhysWorld source file that already loads:",
        "%   \\usepackage{booktabs,graphicx,amsmath}",
        "% If compiling this file from the outputs/ directory, figures resolve by basename.",
        "",
    ]
    body = [render_ablation_table(rows)]
    if figure_name:
        body.append(render_trajectory_figure(figure_name))
    body.extend(extra_blocks)
    output_path.write_text("\n".join(header + body), encoding="utf-8")
    LOGGER.info("Wrote %s", output_path)
    return output_path


def export_paper_artifacts(
    csv_path: Path | str = "outputs/ablation_results.csv",
    output_dir: Path | str = "outputs",
    *,
    tex_name: str = "paper_tables.tex",
    figure_name: str = "paper_latent_trajectories.png",
    device: str = "auto",
    live_eval: bool = True,
    probe_epochs: int = 80,
    max_horizon: int = 20,
    n_traj_episodes: int = 6,
    tsne_iter: int = 400,
) -> dict[str, Any]:
    """Build the camera-ready table + trajectory figure from experimental outputs."""
    output_dir = Path(output_dir)
    csv_path = Path(csv_path)
    rows = _order_like_variants(load_ablation_csv(csv_path))
    cached = load_cached_probes(
        [output_dir / "metrics.json", output_dir / "run_report.json"]
    )
    _apply_cached_probes(rows, cached)

    torch_device = resolve_device(device)
    figure_path = output_dir / figure_name
    figure: Path | None = None
    if live_eval:
        checkpoints = discover_checkpoints(output_dir)
        if checkpoints:
            figure = _enrich_from_checkpoints(
                rows,
                checkpoints,
                torch_device,
                max_horizon=max_horizon,
                probe_epochs=probe_epochs,
                n_traj_episodes=n_traj_episodes,
                tsne_iter=tsne_iter,
                figure_path=figure_path,
            )
        else:
            LOGGER.warning(
                "No ablation checkpoints found under %s; probes/latency use cache only.",
                output_dir,
            )

    tex_path = write_paper_tables_tex(
        rows,
        output_dir / tex_name,
        figure_name=(figure_path.with_suffix(".pdf").name if figure is not None else None),
    )
    return {
        "tex": str(tex_path),
        "figure": None if figure is None else str(figure),
        "rows": [
            {
                "variant": row.key,
                "label": row.label,
                "mse": {str(k): v for k, v in row.mse.items()},
                "stiffness_acc": row.stiffness_acc,
                "friction_acc": row.friction_acc,
                "fps": row.fps,
                "ms_per_frame": row.ms_per_frame,
            }
            for row in rows
        ],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export camera-ready AC-JEPA paper assets.")
    parser.add_argument("--csv", default="outputs/ablation_results.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tex", default="paper_tables.tex")
    parser.add_argument("--figure", default="paper_latent_trajectories.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--max-horizon", type=int, default=20)
    parser.add_argument("--n-traj-episodes", type=int, default=6)
    parser.add_argument("--tsne-iter", type=int, default=400)
    parser.add_argument(
        "--no-live-eval",
        action="store_true",
        help="Skip checkpoint loading; table uses CSV + cached metrics.json only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    report = export_paper_artifacts(
        csv_path=args.csv,
        output_dir=args.output_dir,
        tex_name=args.tex,
        figure_name=args.figure,
        device=args.device,
        live_eval=not args.no_live_eval,
        probe_epochs=args.probe_epochs,
        max_horizon=args.max_horizon,
        n_traj_episodes=args.n_traj_episodes,
        tsne_iter=args.tsne_iter,
    )
    LOGGER.info("Paper artifacts ready  tex=%s  figure=%s", report["tex"], report["figure"])
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
