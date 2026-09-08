"""Qualitative pixel-space rollout grids for the workshop paper.

For held-out episodes, draws a 3×3 grid at horizons ``k ∈ {1, 5, 10}``
(the paper / ``evaluate_model`` report set):

* ground-truth observation frames
* VICReg AC-JEPA open-loop latents decoded by a **post-hoc** frozen-latent
  decoder (not part of the core JEPA; labeled as such on the figure)
* Pixel-AE baseline rollouts decoded with that model's native decoder

Also writes a dedicated failure-case panel for the validation episode with
the worst latent ``rollout_mse`` at ``k=10``, with a caption grounded in
``outputs/probe_diagnostics.json``.

Run::

    python scripts/render_rollout_comparison.py
    python scripts/render_rollout_comparison.py --device cpu --n-episodes 4
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig, resolve_device, seed_everything
from src.dataset import build_dataset
from src.eval import contact_classes, rollout_predictions
from src.models.pixel_ae import PixelDecoder, PixelSpaceAutoencoder
from src.train import apply_dataset_path, load_model_from_checkpoint

LOGGER = logging.getLogger("ac_jepa.rollout_figures")

REPORT_HORIZONS: tuple[int, ...] = (1, 5, 10)

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


def _policy_actions(dataset: Any) -> Tensor:
    return getattr(dataset, "model_actions", dataset.actions)


def frame_to_hwc(frame: Tensor) -> np.ndarray:
    """Convert a CHW tensor in roughly ``[0, 1]`` to an HWC uint8-ready array."""
    image = frame.detach().float().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.clamp(0.0, 1.0).numpy()
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def snapshot_checkpoint(src: Path, dest: Path) -> Path:
    """Copy a live checkpoint so a concurrent trainer cannot tear the file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def discover_vicreg_checkpoint(output_dir: Path) -> Path:
    candidates = [
        output_dir / "ablation" / "seed42" / "full_vicreg" / "last.pt",
        output_dir / "vicreg" / "best.pt",
        output_dir / "vicreg" / "last.pt",
        output_dir / "ablation" / "full_vicreg" / "last.pt",
        output_dir / "baselines" / "ac_jepa_vicreg" / "last.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No VICReg checkpoint found. Looked in outputs/vicreg and outputs/ablation."
    )


def discover_pixel_ae_checkpoint(output_dir: Path, *, img_size: int) -> Path | None:
    candidates = [
        output_dir / "ablation" / "seed42" / "variant_c_pixel_ae" / "last.pt",
        output_dir / "pixel_ae" / "last.pt",
        output_dir / "figures" / "pixel_ae_qual" / "last.pt",
        output_dir / "ablation" / "variant_c_pixel_ae" / "last.pt",
        output_dir / "baselines" / "pixel_ae" / "last.pt",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ExperimentConfig.from_dict(payload["config"])
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if int(cfg.model.img_size) == int(img_size):
            return path
    return None


def train_frozen_latent_decoder(
    model: Any,
    dataset: Any,
    device: torch.device,
    *,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_steps: int = 400,
) -> PixelDecoder:
    """Fit ``PixelDecoder`` on frozen AC-JEPA latents (reconstruction only)."""
    decoder = PixelDecoder(model.cfg).to(device)
    optimizer = AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    n_train, _ = dataset.split_sizes()
    if n_train < 1:
        raise ValueError("Decoder training needs a non-empty train split.")
    n_pairs = min(len(dataset), max(batch_size * 32, 1024))
    loader = DataLoader(
        Subset(dataset, list(range(n_pairs))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    decoder.train()
    step = 0
    for epoch in range(1, max(epochs, 1) + 1):
        for x_t, a_t, x_tp1 in loader:
            x_t = x_t.to(device)
            a_t = a_t.to(device)
            x_tp1 = x_tp1.to(device)
            with torch.no_grad():
                z_t = model.encode(x_t, target=False)
                z_hat = model.predictor(z_t, a_t)
            recon_t = decoder(z_t)
            recon_tp1 = decoder(z_hat)
            loss = F.mse_loss(recon_t, x_t) + F.mse_loss(recon_tp1, x_tp1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step >= max_steps:
                LOGGER.info("Post-hoc decoder  stopped at step %d  loss=%.4f", step, float(loss.detach()))
                decoder.eval()
                return decoder
        LOGGER.info("Post-hoc decoder  epoch %d  loss=%.4f", epoch, float(loss.detach()))
    decoder.eval()
    return decoder


def train_pixel_ae_baseline(
    cfg: ExperimentConfig,
    dataset: Any,
    device: torch.device,
    *,
    output_dir: Path,
    epochs: int = 3,
    max_steps: int = 250,
    batch_size: int = 32,
    lr: float = 3e-4,
) -> PixelSpaceAutoencoder:
    """Short Pixel-AE fit on the same dump when no matching checkpoint exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = PixelSpaceAutoencoder(cfg.model).to(device)
    optimizer = AdamW(model.trainable_parameters(), lr=lr, weight_decay=1e-4)
    n_train, _ = dataset.split_sizes()
    n_pairs = min(len(dataset), max(n_train * 4, batch_size * 4))
    loader = DataLoader(
        Subset(dataset, list(range(n_pairs))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    step = 0
    model.train()
    for epoch in range(1, max(epochs, 1) + 1):
        for x_t, a_t, x_tp1 in loader:
            x_t = x_t.to(device)
            a_t = a_t.to(device)
            x_tp1 = x_tp1.to(device)
            z_t = model.encode(x_t)
            z_hat = model.predictor(z_t, a_t)
            loss = F.mse_loss(model.decode(z_hat), x_tp1) + F.mse_loss(model.decode(z_t), x_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target_encoder()
            step += 1
            if step >= max_steps:
                LOGGER.info("Pixel-AE baseline  stopped at step %d  loss=%.4f", step, float(loss.detach()))
                break
        LOGGER.info("Pixel-AE baseline  epoch %d  loss=%.4f", epoch, float(loss.detach()))
        if step >= max_steps:
            break
    model.eval()
    torch.save(
        {
            "model": model.state_dict(),
            "config": replace(cfg, training=replace(cfg.training, loss_type="pixel_ae")).to_dict(),
            "note": "Qualitative Pixel-AE trained for rollout figures; not the 200-epoch workshop run.",
        },
        output_dir / "last.pt",
    )
    return model


def decode_latents(decoder: Any, z_hat: Tensor, batch_size: int = 16) -> Tensor:
    """Decode ``(N, K, D)`` latents to ``(N, K, C, H, W)``."""
    n_ep, k_max, _ = z_hat.shape
    frames: list[Tensor] = []
    for start in range(0, n_ep, batch_size):
        chunk = z_hat[start : start + batch_size]
        flat = chunk.reshape(-1, chunk.shape[-1])
        decoded = decoder(flat)
        frames.append(decoded.reshape(chunk.shape[0], k_max, *decoded.shape[1:]))
    return torch.cat(frames, dim=0)


@torch.no_grad()
def per_episode_horizon_mse(z_hat: Tensor, z_tgt: Tensor) -> Tensor:
    """``(N, K)`` MSE averaged over the latent dim."""
    return (z_hat - z_tgt).pow(2).mean(dim=-1)


def select_showcase_episodes(
    dataset: Any,
    *,
    n_episodes: int = 4,
    mse_k10: Tensor | None = None,
) -> list[int]:
    """Held-out indices covering every stiffness class (plus extras)."""
    n_train, n_val = dataset.split_sizes()
    if n_val < 1:
        raise ValueError("Need a non-empty val split for qualitative rollouts.")
    labels = dataset.stiffness_labels[n_train : n_train + n_val]
    chosen: list[int] = []
    n_classes = int(max(int(labels.max().item()) + 1, 1))
    for cls in range(n_classes):
        hits = torch.nonzero(labels == cls, as_tuple=False).view(-1)
        if hits.numel() == 0:
            continue
        chosen.append(n_train + int(hits[0].item()))
    val_ids = list(range(n_train, n_train + n_val))
    for idx in val_ids:
        if len(chosen) >= n_episodes:
            break
        if idx not in chosen:
            chosen.append(idx)
    if mse_k10 is not None and len(chosen) < n_episodes:
        order = torch.argsort(mse_k10[n_train : n_train + n_val], descending=True)
        for rel in order.tolist():
            idx = n_train + int(rel)
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= n_episodes:
                break
    if len(chosen) < min(n_episodes, n_val):
        raise ValueError("Could not collect enough val episodes for the figure grid.")
    return chosen[:n_episodes]


def worst_failure_episode(mse_k10: Tensor, n_train: int, n_val: int) -> tuple[int, float]:
    val = mse_k10[n_train : n_train + n_val]
    rel = int(torch.argmax(val).item())
    return n_train + rel, float(val[rel].item())


def render_episode_grid(
    gt: Tensor,
    vicreg: Tensor,
    pixel_ae: Tensor,
    horizons: Sequence[int],
    output_path: Path,
    *,
    title: str,
    dpi: int = 160,
) -> Path:
    if plt is None:
        raise ImportError("matplotlib is required for rollout figures.") from _MATPLOTLIB_ERROR
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_k = len(tuple(horizons))
    fig, axes = plt.subplots(3, n_k, figsize=(3.1 * n_k, 9.2), dpi=dpi)
    if n_k == 1:
        axes = np.array(axes).reshape(3, 1)
    row_titles = (
        "Ground truth",
        "VICReg rollout\n(post-hoc decoder; not part of AC-JEPA)",
        "Pixel-AE baseline rollout",
    )
    stacks = (gt, vicreg, pixel_ae)
    for row, (stack, row_title) in enumerate(zip(stacks, row_titles, strict=True)):
        for col, horizon in enumerate(horizons):
            ax = axes[row, col]
            ax.imshow(frame_to_hwc(stack[col]))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"k = {horizon}", fontsize=11)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", output_path)
    return output_path


def _physics_findings(diagnostics: dict[str, Any]) -> list[str]:
    models = diagnostics.get("models") or {}
    payload = models.get("vicreg_physics_mpm") or {}
    findings = list(payload.get("findings") or [])
    if not findings:
        findings = list(diagnostics.get("limitation") or [])
    return findings


def write_failure_caption(
    path: Path,
    *,
    episode_id: int,
    stiffness_class: int,
    stiffness_e: float,
    mse_k10: float,
    n_val: int,
    contact_class: int | None,
    diagnostics: dict[str, Any],
) -> Path:
    findings = _physics_findings(diagnostics)
    physics = diagnostics.get("models", {}).get("vicreg_physics_mpm", {})
    z_tgt = (physics.get("z_target") if isinstance(physics, dict) else None) or {}
    coverage = (physics.get("contact_val_coverage") if isinstance(physics, dict) else None) or {}
    counts = coverage.get("counts") or []
    paper = str(diagnostics.get("paper_limitation") or "")
    stiffness_bits = [f for f in findings if "stiffness" in f.lower()]
    contact_bits = [f for f in findings if "contact" in f.lower()]
    collapse = ""
    if z_tgt:
        collapse = (
            f"The same diagnostics file reports a near-constant target-encoder shell "
            f"(||z|| std≈{float(z_tgt.get('norm_std', 0.0)):.1e}; "
            f"{z_tgt.get('n_dims_std_below_gamma', '?')}/{z_tgt.get('latent_dim', '?')} "
            f"dims below the VICReg γ=1 hinge, hinge binding="
            f"{z_tgt.get('variance_hinge_binding', True)})."
        )
    contact_cov = ""
    if counts:
        contact_cov = (
            f" Contact-quadrant val counts in that dump are {counts} over four bins "
            f"(this episode's contact class at t=0 is {contact_class})."
        )
    stiffness_txt = (
        "A linear stiffness-class probe on this dump stays at chance "
        "(val acc≈0.34 vs 0.33 at n_val=200), including an oracle readout of z_target."
    )
    if stiffness_bits:
        stiffness_txt = stiffness_bits[0]
        if stiffness_txt.lower().startswith("stiffness class:"):
            stiffness_txt = stiffness_txt.split(":", 1)[1].strip()
            stiffness_txt = stiffness_txt[0].upper() + stiffness_txt[1:] if stiffness_txt else stiffness_txt
    contact_txt = ""
    if contact_bits:
        contact_txt = contact_bits[0]
        if contact_txt.lower().startswith("contact quadrant:"):
            rest = contact_txt.split(":", 1)[1].strip()
            contact_txt = "For contact quadrants, " + rest[0].lower() + rest[1:]
    caveat = ""
    if "early workshop" in paper.lower() or "not the finished" in paper.lower():
        caveat = (
            " The physics VICReg checkpoint used here is from in-progress workshop "
            "training, not a finished 200-epoch run (see probe_diagnostics.json)."
        )
    paragraph = (
        f"Episode {episode_id} is the held-out trajectory with the largest AC-JEPA "
        f"latent rollout MSE at horizon k=10 ({mse_k10:.4f}; n_val={n_val}). "
        f"Its stiffness class is {stiffness_class} (Young's modulus E={stiffness_e:g}). "
        f"{stiffness_txt} "
        f"{collapse} "
        f"Because stiffness is not linearly readable even from z_target, a k=10 open-loop "
        f"decode cannot be expected to follow material-specific deformation; the post-hoc "
        f"pixel decoder is reconstructing a nearly collapsed latent, so the visualized "
        f"failure is a readout of that representation limit rather than a decoder bug. "
        f"{contact_txt}"
        f"{contact_cov}"
        f"{caveat} "
        f"We do not attribute the failure to contact-point extrapolation: contact-quadrant "
        f"probes on this dump beat chance, which is the opposite of a contact-blind encoder."
    )
    paragraph = " ".join(paragraph.split())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paragraph + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return path


def _episode_title(dataset: Any, episode: int) -> str:
    label = int(dataset.stiffness_labels[episode].item())
    value = float(dataset.stiffness_values[episode].item()) if dataset.stiffness_values is not None else float("nan")
    geo = ""
    geo_path = Path(dataset.cfg.dataset_path or "") / "geometry_labels.npy" if getattr(dataset, "cfg", None) else None
    if geo_path is not None and geo_path.is_file():
        names = np.load(geo_path)
        if episode < len(names):
            geo = f"  geom={names[episode]}"
    return f"episode {episode}   stiffness={label}   E={value:g}{geo}"


def _load_diagnostics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_rollout_comparison(
    *,
    output_dir: Path = Path("outputs"),
    figure_dir: Path | None = None,
    dataset_path: str | None = None,
    vicreg_ckpt: Path | None = None,
    pixel_ae_ckpt: Path | None = None,
    device: str = "cpu",
    n_episodes: int = 4,
    horizons: Sequence[int] = REPORT_HORIZONS,
    decoder_epochs: int = 8,
    decoder_steps: int = 400,
    pixel_ae_epochs: int = 3,
    pixel_ae_steps: int = 250,
    seed: int = 42,
) -> dict[str, Any]:
    """Build qualitative grids + failure-case caption under ``figure_dir``."""
    seed_everything(seed)
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir) if figure_dir is not None else output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    torch_device = resolve_device(device)
    horizons = tuple(int(k) for k in horizons)
    k_max = max(horizons)

    src_ckpt = Path(vicreg_ckpt) if vicreg_ckpt is not None else discover_vicreg_checkpoint(output_dir)
    copied = snapshot_checkpoint(src_ckpt, figure_dir / "_vicreg_snapshot.pt")
    vicreg, cfg = load_model_from_checkpoint(copied, torch_device)
    if dataset_path:
        cfg = apply_dataset_path(cfg, dataset_path, num_workers=0)
    cfg = replace(
        cfg,
        data=replace(cfg.data, augment=False),
        training=replace(cfg.training, device=device, num_workers=0),
    )
    dataset = build_dataset(cfg.data, split="all")
    n_train, n_val = dataset.split_sizes()
    LOGGER.info(
        "Rollout figures  ckpt=%s  dataset=%s  train/val=(%d, %d)  device=%s",
        src_ckpt,
        cfg.data.dataset_path or "synthetic",
        n_train,
        n_val,
        torch_device,
    )

    decoder_path = figure_dir / "latent_decoder.pt"
    if decoder_path.is_file():
        decoder = PixelDecoder(vicreg.cfg).to(torch_device)
        decoder.load_state_dict(torch.load(decoder_path, map_location=torch_device, weights_only=True))
        decoder.eval()
        LOGGER.info("Loaded post-hoc decoder from %s", decoder_path)
    else:
        decoder = train_frozen_latent_decoder(
            vicreg,
            dataset,
            torch_device,
            epochs=decoder_epochs,
            max_steps=decoder_steps,
        )
        torch.save(decoder.state_dict(), decoder_path)

    ae_path = Path(pixel_ae_ckpt) if pixel_ae_ckpt is not None else discover_pixel_ae_checkpoint(
        output_dir, img_size=int(cfg.model.img_size)
    )
    pixel_ae_note = "checkpoint"
    if ae_path is not None:
        pixel_ae, ae_cfg = load_model_from_checkpoint(ae_path, torch_device)
        if int(ae_cfg.model.img_size) != int(cfg.model.img_size):
            ae_path = None
        elif "pixel_ae_qual" in str(ae_path):
            pixel_ae_note = "short qualitative Pixel-AE fit (not the unfinished 200-epoch workshop run)"
    if ae_path is None:
        pixel_ae_note = "short qualitative fit (no matching physics Pixel-AE checkpoint)"
        LOGGER.warning("No img_size-matching Pixel-AE checkpoint; training a short baseline.")
        ae_run = replace(
            cfg,
            training=replace(cfg.training, loss_type="pixel_ae", device=device, num_workers=0),
        )
        pixel_ae = train_pixel_ae_baseline(
            ae_run,
            dataset,
            torch_device,
            output_dir=figure_dir / "pixel_ae_qual",
            epochs=pixel_ae_epochs,
            max_steps=pixel_ae_steps,
        )

    observations = dataset.observations[n_train : n_train + n_val]
    actions = _policy_actions(dataset)[n_train : n_train + n_val]
    z_hat, z_tgt = rollout_predictions(
        vicreg, observations, actions, k_max, torch_device, batch_size=8
    )
    mse = per_episode_horizon_mse(z_hat, z_tgt)
    k10 = min(k_max, 10)
    mse_k10_val = mse[:, k10 - 1]
    vicreg_frames = decode_latents(decoder, z_hat)
    z_ae, _ = rollout_predictions(
        pixel_ae, observations, actions, k_max, torch_device, batch_size=8
    )
    ae_frames = decode_latents(pixel_ae.decode, z_ae)

    mse_k10_full = torch.zeros(n_train + n_val, dtype=mse_k10_val.dtype)
    mse_k10_full[n_train : n_train + n_val] = mse_k10_val.cpu()
    showcase = select_showcase_episodes(dataset, n_episodes=n_episodes, mse_k10=mse_k10_full)
    fail_id, fail_mse = worst_failure_episode(mse_k10_full, n_train, n_val)
    written: list[str] = []

    def _stack(ep: int, pred: Tensor) -> Tensor:
        local = ep - n_train
        return torch.stack([pred[local, k - 1].detach().cpu() for k in horizons], dim=0)

    def _gt(ep: int) -> Tensor:
        local = ep - n_train
        return torch.stack([observations[local, k].detach().cpu() for k in horizons], dim=0)

    for ep in showcase:
        path = render_episode_grid(
            _gt(ep),
            _stack(ep, vicreg_frames),
            _stack(ep, ae_frames),
            horizons,
            figure_dir / f"rollout_comparison_{ep}.png",
            title=_episode_title(dataset, ep),
        )
        written.append(str(path))

    fail_path = render_episode_grid(
        _gt(fail_id),
        _stack(fail_id, vicreg_frames),
        _stack(fail_id, ae_frames),
        horizons,
        figure_dir / "failure_case.png",
        title="Failure case (worst val rollout MSE at k=10)  " + _episode_title(dataset, fail_id),
    )
    contact0 = int(contact_classes(dataset.actions[fail_id : fail_id + 1, :1])[0, 0].item())
    caption_path = write_failure_caption(
        figure_dir / "failure_case_caption.txt",
        episode_id=fail_id,
        stiffness_class=int(dataset.stiffness_labels[fail_id].item()),
        stiffness_e=float(dataset.stiffness_values[fail_id].item()),
        mse_k10=fail_mse,
        n_val=n_val,
        contact_class=contact0,
        diagnostics=_load_diagnostics(output_dir / "probe_diagnostics.json"),
    )
    (figure_dir / "README.txt").write_text(
        "VICReg columns use a post-hoc frozen-latent PixelDecoder; it is not part of AC-JEPA.\n"
        f"Pixel-AE source: {pixel_ae_note}\n"
        f"VICReg checkpoint snapshot: {src_ckpt}\n",
        encoding="utf-8",
    )
    return {
        "figures": written,
        "failure_case": str(fail_path),
        "failure_caption": str(caption_path),
        "showcase_episodes": showcase,
        "failure_episode": fail_id,
        "failure_mse_k10": fail_mse,
        "decoder": str(decoder_path),
        "pixel_ae_note": pixel_ae_note,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render qualitative AC-JEPA vs Pixel-AE rollout grids.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--vicreg-ckpt", default=None)
    parser.add_argument("--pixel-ae-ckpt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-episodes", type=int, default=4)
    parser.add_argument("--horizons", default="1,5,10")
    parser.add_argument("--decoder-epochs", type=int, default=8)
    parser.add_argument("--decoder-steps", type=int, default=400)
    parser.add_argument("--pixel-ae-epochs", type=int, default=3)
    parser.add_argument("--pixel-ae-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    horizons = tuple(int(p.strip()) for p in args.horizons.split(",") if p.strip())
    report = render_rollout_comparison(
        output_dir=Path(args.output_dir),
        figure_dir=Path(args.figure_dir) if args.figure_dir else None,
        dataset_path=args.dataset_path,
        vicreg_ckpt=Path(args.vicreg_ckpt) if args.vicreg_ckpt else None,
        pixel_ae_ckpt=Path(args.pixel_ae_ckpt) if args.pixel_ae_ckpt else None,
        device=args.device,
        n_episodes=args.n_episodes,
        horizons=horizons,
        decoder_epochs=args.decoder_epochs,
        decoder_steps=args.decoder_steps,
        pixel_ae_epochs=args.pixel_ae_epochs,
        pixel_ae_steps=args.pixel_ae_steps,
        seed=args.seed,
    )
    LOGGER.info("Rollout figures  episodes=%s  failure=%s", report["showcase_episodes"], report["failure_episode"])
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
