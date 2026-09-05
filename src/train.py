"""Clean PyTorch training loop for Action-Conditioned JEPA.

Supports AMP, AdamW, cosine decay with warmup, EMA target updates, and
JSON/stdout logging of the VICReg constituents (total, invariance, variance,
covariance). ``python src/train.py --mode train_and_eval`` runs a 5-epoch
smoke test plus the archival evaluation suite.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset as TorchDataset

if __package__:
    from .config import (
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
        resolve_device,
        seed_everything,
    )
    from .dataset import (
        DeformableObjectDataset,
        DeformablePhysicsDataset,
        build_dataset,
        dataloader_worker_kwargs,
        inspect_physics_source,
    )
    from .losses import prediction_objective, vicreg_loss
    from .models.ac_jepa import ACJEPA
    from .models.pixel_ae import PixelSpaceAutoencoder
else:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.config import (
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
        resolve_device,
        seed_everything,
    )
    from src.dataset import (
        DeformableObjectDataset,
        DeformablePhysicsDataset,
        build_dataset,
        dataloader_worker_kwargs,
        inspect_physics_source,
    )
    from src.losses import prediction_objective, vicreg_loss
    from src.models.ac_jepa import ACJEPA
    from src.models.pixel_ae import PixelSpaceAutoencoder

LOGGER = logging.getLogger("ac_jepa.train")
RunMode = Literal["train", "eval", "train_and_eval"]
WorldModel = Union[ACJEPA, PixelSpaceAutoencoder]


def _load_eval():
    if __package__:
        from .eval import run_archival_evaluation
    else:
        from src.eval import run_archival_evaluation
    return run_archival_evaluation


def build_smoke_config(epochs: int = 5, output_dir: str = "outputs") -> ExperimentConfig:
    """Compact config that finishes a 5-epoch end-to-end run on CPU."""
    model = ModelConfig(
        in_channels=3,
        img_size=32,
        patch_size=8,
        embed_dim=64,
        encoder_depth=2,
        encoder_num_heads=4,
        mlp_ratio=2.0,
        action_dim=4,
        action_embed_dim=32,
        predictor_embed_dim=64,
        predictor_depth=2,
        predictor_num_heads=4,
        latent_dim=0,
        ema_momentum=0.996,
        dropout=0.0,
        drop_path=0.0,
        backbone="vit",
        fuse_mode="concat",
    )
    training = TrainingConfig(
        batch_size=8,
        epochs=epochs,
        lr=1e-3,
        min_lr=1e-5,
        weight_decay=0.05,
        warmup_epochs=min(1, max(epochs - 1, 0)),
        grad_clip=1.0,
        loss_type="vicreg",
        amp=True,
        seed=42,
        num_workers=0,
        log_interval=5,
        ckpt_interval=max(epochs, 1),
        output_dir=output_dir,
        device="auto",
        ema_sync_every=1,
    )
    data = DataConfig(
        spatial_dims=2,
        img_size=32,
        in_channels=3,
        action_dim=4,
        n_episodes=48,
        episode_length=11,
        n_grid=4,
        seed=0,
        val_fraction=0.25,
        pin_memory=False,
        n_stiffness_classes=3,
    )
    return ExperimentConfig(model=model, training=training, data=data)


def build_world_model(cfg: ExperimentConfig) -> WorldModel:
    """Instantiate AC-JEPA or the pixel autoencoder according to ``loss_type``."""
    if cfg.training.loss_type == "pixel_ae":
        return PixelSpaceAutoencoder(cfg.model)
    return ACJEPA(cfg.model)


def _amp_enabled(device: torch.device, requested: bool) -> bool:
    return bool(requested and device.type in ("cuda", "mps"))


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _grad_scaler(device: torch.device, enabled: bool):
    cuda_amp = bool(enabled and device.type == "cuda")
    if not cuda_amp:
        return _DisabledScaler()
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (TypeError, AttributeError, RuntimeError):
        from torch.cuda.amp import GradScaler

        return GradScaler(enabled=True)


class _DisabledScaler:
    """Identity GradScaler for CPU / MPS (autocast does not need loss scaling)."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: Optimizer) -> None:
        return None

    def step(self, optimizer: Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        return None


def _cosine_warmup_lambda(
    warmup_steps: int, total_steps: int, min_lr: float, base_lr: float
):
    floor = min_lr / base_lr if base_lr > 0 else 0.0

    def _lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if step >= total_steps:
            return floor
        span = max(total_steps - warmup_steps, 1)
        progress = (step - warmup_steps) / span
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return floor + (1.0 - floor) * cosine

    return _lr_lambda


class _AverageMeter:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, metrics: dict[str, float], batch_size: int = 1) -> None:
        self.count += batch_size
        for key, value in metrics.items():
            self.totals[key] += float(value) * batch_size

    def mean(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}


def _transition_subset(dataset: DeformableObjectDataset | DeformablePhysicsDataset, episode_slice: slice) -> Subset:
    start, stop, _ = episode_slice.indices(dataset.n_episodes)
    indices = dataset.indices_for_episodes(start, stop)
    return Subset(dataset, indices)


def build_dataloaders(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    *,
    full_dataset: DeformableObjectDataset | DeformablePhysicsDataset | None = None,
) -> tuple[DataLoader, DataLoader | None, DeformableObjectDataset | DeformablePhysicsDataset]:
    """Build train/val loaders from a single simulated or physics episode archive."""
    dataset = full_dataset or build_dataset(data_cfg, split="all")
    n_train, n_val = dataset.split_sizes()
    worker_kwargs = dataloader_worker_kwargs(data_cfg, train_cfg)
    train_set: TorchDataset = _transition_subset(dataset, slice(0, n_train))
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        **worker_kwargs,
    )
    val_loader: DataLoader | None = None
    if n_val > 0:
        val_set = _transition_subset(dataset, slice(n_train, n_train + n_val))
        val_loader = DataLoader(
            val_set,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            **worker_kwargs,
        )
    return train_loader, val_loader, dataset


class ACJEPATrainer:
    """AdamW + AMP trainer for a single AC-JEPA loss variant."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        model: WorldModel | None = None,
        *,
        run_name: str = "vicreg",
        dataset: DeformableObjectDataset | DeformablePhysicsDataset | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self.device = cfg.training.resolve_device()
        self.use_amp = _amp_enabled(self.device, cfg.training.amp)
        seed_everything(cfg.training.seed)
        self.model = (model if model is not None else build_world_model(cfg)).to(self.device)
        self.train_loader, self.val_loader, self.dataset = build_dataloaders(
            cfg.data, cfg.training, full_dataset=dataset
        )
        self.optimizer = AdamW(
            list(self.model.trainable_parameters()),
            lr=cfg.training.lr,
            betas=(cfg.training.beta1, cfg.training.beta2),
            weight_decay=cfg.training.weight_decay,
        )
        steps_per_epoch = max(len(self.train_loader), 1)
        warmup_epochs = min(cfg.training.warmup_epochs, cfg.training.epochs)
        self.scheduler = LambdaLR(
            self.optimizer,
            _cosine_warmup_lambda(
                warmup_steps=warmup_epochs * steps_per_epoch,
                total_steps=cfg.training.epochs * steps_per_epoch,
                min_lr=cfg.training.min_lr,
                base_lr=cfg.training.lr,
            ),
        )
        self.scaler = _grad_scaler(self.device, self.use_amp)
        self.output_dir = Path(cfg.training.output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []
        self.global_step = 0

    def _objective(self, z_hat: torch.Tensor, z_target: torch.Tensor):
        train_cfg = self.cfg.training
        return prediction_objective(
            z_hat,
            z_target,
            loss_type=train_cfg.loss_type,
            sim_coeff=train_cfg.vicreg_sim_coeff,
            std_coeff=train_cfg.vicreg_std_coeff,
            cov_coeff=train_cfg.vicreg_cov_coeff,
        )

    def _pixel_ae_objective(
        self, x_t: torch.Tensor, a_t: torch.Tensor, x_tp1: torch.Tensor
    ):
        """Pixel reconstruction + next-frame MSE; VICReg terms are logging-only."""
        z_t = self.model.encode(x_t)
        z_hat = self.model.predictor(z_t, a_t)
        x_hat = self.model.decode(z_hat)
        x_rec = self.model.decode(z_t)
        pixel_next = F.mse_loss(x_hat, x_tp1)
        pixel_rec = F.mse_loss(x_rec, x_t)
        loss = pixel_next + pixel_rec
        with torch.no_grad():
            z_tp1 = self.model.encode(x_tp1, target=True)
            _, terms = vicreg_loss(z_hat.detach(), z_tp1)
        terms["loss"] = loss
        terms["invariance"] = pixel_next
        return loss, terms

    def _forward_loss(
        self, x_t: torch.Tensor, a_t: torch.Tensor, x_tp1: torch.Tensor
    ):
        if self.cfg.training.loss_type == "pixel_ae":
            return self._pixel_ae_objective(x_t, a_t, x_tp1)
        output = self.model(x_t, a_t, x_tp1)
        assert output.z_tp1 is not None
        return self._objective(output.z_hat_tp1, output.z_tp1)

    def _step_metrics(self, terms: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            "loss": float(terms["loss"].detach().item()),
            "invariance": float(terms["invariance"].detach().item()),
            "variance": float(terms["variance"].detach().item()),
            "covariance": float(terms["covariance"].detach().item()),
        }

    def training_step(
        self, x_t: torch.Tensor, a_t: torch.Tensor, x_tp1: torch.Tensor
    ) -> dict[str, float]:
        self.model.train()
        x_t = x_t.to(self.device, non_blocking=True)
        a_t = a_t.to(self.device, non_blocking=True)
        x_tp1 = x_tp1.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.device, self.use_amp):
            loss, terms = self._forward_loss(x_t, a_t, x_tp1)
        self.scaler.scale(loss).backward()
        if self.cfg.training.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.trainable_parameters()), self.cfg.training.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.global_step % self.cfg.training.ema_sync_every == 0:
            self.model.update_target_encoder()
        self.global_step += 1
        metrics = self._step_metrics(terms)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        meter = _AverageMeter()
        for x_t, a_t, x_tp1 in loader:
            x_t = x_t.to(self.device, non_blocking=True)
            a_t = a_t.to(self.device, non_blocking=True)
            x_tp1 = x_tp1.to(self.device, non_blocking=True)
            with _autocast(self.device, self.use_amp):
                _, terms = self._forward_loss(x_t, a_t, x_tp1)
            meter.update(self._step_metrics(terms), batch_size=x_t.shape[0])
        return meter.mean()

    def _log(self, epoch: int, step: int, metrics: dict[str, float], split: str) -> None:
        payload = {
            "run": self.run_name,
            "split": split,
            "epoch": epoch,
            "step": step,
            **metrics,
        }
        LOGGER.info(
            "[%s] epoch %d step %d  loss=%.4f  inv=%.4f  var=%.4f  cov=%.4f  lr=%.2e",
            self.run_name,
            epoch,
            step,
            metrics.get("loss", float("nan")),
            metrics.get("invariance", float("nan")),
            metrics.get("variance", float("nan")),
            metrics.get("covariance", float("nan")),
            metrics.get("lr", self.optimizer.param_groups[0]["lr"]),
        )
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def save_checkpoint(self, path: Path, epoch: int, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "epoch": epoch,
            "run_name": self.run_name,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.cfg.to_dict(),
            "history": self.history,
            **(extra or {}),
        }
        torch.save(payload, path)

    def fit(self) -> dict[str, Any]:
        train_cfg = self.cfg.training
        LOGGER.info(
            "Training %s on %s  (AMP=%s, epochs=%d, steps/epoch=%d, N_train=%d)",
            self.run_name,
            self.device,
            self.use_amp,
            train_cfg.epochs,
            len(self.train_loader),
            len(self.train_loader.dataset),  # type: ignore[arg-type]
        )
        best_val = float("inf")
        started = time.perf_counter()
        for epoch in range(1, train_cfg.epochs + 1):
            epoch_meter = _AverageMeter()
            for step, (x_t, a_t, x_tp1) in enumerate(self.train_loader, start=1):
                metrics = self.training_step(x_t, a_t, x_tp1)
                epoch_meter.update(metrics, batch_size=x_t.shape[0])
                if step % train_cfg.log_interval == 0 or step == 1:
                    self._log(epoch, self.global_step, metrics, split="train_step")
            epoch_stats = {f"train_{k}": v for k, v in epoch_meter.mean().items()}
            if self.val_loader is not None:
                val_stats = self.evaluate_loader(self.val_loader)
                epoch_stats.update({f"val_{k}": v for k, v in val_stats.items()})
                val_loss = val_stats.get("loss", float("inf"))
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint(self.output_dir / "best.pt", epoch, {"best_val": best_val})
            self._log(epoch, self.global_step, epoch_meter.mean(), split="train_epoch")
            self.history.append({"epoch": epoch, **epoch_stats})
            if epoch % train_cfg.ckpt_interval == 0 or epoch == train_cfg.epochs:
                self.save_checkpoint(self.output_dir / "last.pt", epoch)
        elapsed = time.perf_counter() - started
        summary = {
            "run_name": self.run_name,
            "epochs": train_cfg.epochs,
            "device": str(self.device),
            "amp": self.use_amp,
            "seconds": elapsed,
            "history": self.history,
            "best_val": best_val if best_val < float("inf") else None,
            "checkpoint": str(self.output_dir / "last.pt"),
        }
        (self.output_dir / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("Finished %s in %.1fs  last ckpt=%s", self.run_name, elapsed, summary["checkpoint"])
        return summary


def load_model_from_checkpoint(path: Path, device: torch.device) -> tuple[WorldModel, ExperimentConfig]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ExperimentConfig.from_dict(payload["config"])
    model = build_world_model(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate AC-JEPA.")
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "train_and_eval"),
        default="train_and_eval",
        help="train, eval a checkpoint, or run the 5-epoch smoke + archival suite.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--metrics-png", default="outputs/metrics.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ckpt", default=None, help="Checkpoint for --mode eval (VICReg run).")
    parser.add_argument("--baseline-ckpt", default=None, help="Optional MSE-baseline checkpoint.")
    parser.add_argument(
        "--loss-type",
        choices=("vicreg", "vicreg_no_cov", "mse", "smooth_l1", "cosine", "pixel_ae"),
        default=None,
    )
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument("--skip-baseline", action="store_true", help="Do not train the MSE ablation.")
    parser.add_argument(
        "--dataset_path",
        "--dataset-path",
        dest="dataset_path",
        default=None,
        help=(
            "Path to an HDF5 / NPY-directory / NPZ physics dump. "
            "When omitted, the mass-spring simulator generates synthetic tensors."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults to 2 when --dataset_path is set, else 0.",
    )
    return parser.parse_args(argv)


def _train_variant(
    cfg: ExperimentConfig,
    loss_type: str,
    run_name: str,
    dataset: DeformableObjectDataset | DeformablePhysicsDataset,
) -> tuple[WorldModel, dict[str, Any]]:
    run_cfg = replace(cfg, training=replace(cfg.training, loss_type=loss_type))  # type: ignore[arg-type]
    trainer = ACJEPATrainer(run_cfg, run_name=run_name, dataset=dataset)
    summary = trainer.fit()
    return trainer.model, summary


def apply_dataset_path(cfg: ExperimentConfig, dataset_path: str, num_workers: int | None) -> ExperimentConfig:
    """Point the run at a physics dump and align model tensor shapes."""
    meta = inspect_physics_source(dataset_path)
    workers = 2 if num_workers is None else num_workers
    data = replace(
        cfg.data,
        dataset_path=dataset_path,
        in_channels=meta["in_channels"],
        action_dim=meta["action_dim"],
        n_episodes=meta["n_episodes"],
        episode_length=meta["episode_length"],
        pin_memory=True,
        augment=True,
        normalize_actions=True,
    )
    model = replace(
        cfg.model,
        in_channels=meta["in_channels"],
        action_dim=meta["action_dim"],
    )
    if cfg.model.img_size % cfg.model.patch_size != 0:
        raise ValueError("img_size must remain divisible by patch_size after loading physics data.")
    training = replace(cfg.training, num_workers=workers)
    LOGGER.info(
        "Physics dump %s  N=%d T=%d C=%d A=%d  (frames resized to %d, workers=%d)",
        dataset_path,
        meta["n_episodes"],
        meta["episode_length"],
        meta["in_channels"],
        meta["action_dim"],
        cfg.model.img_size,
        workers,
    )
    return ExperimentConfig(model=model, training=training, data=data)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = build_smoke_config(epochs=args.epochs, output_dir=args.output_dir)
    cfg.training.device = args.device
    if args.loss_type is not None:
        cfg.training.loss_type = args.loss_type
    if args.num_workers is not None and args.dataset_path is None:
        cfg.training.num_workers = args.num_workers
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        run_archival_evaluation = _load_eval()

        device = resolve_device(args.device)
        if args.ckpt is None:
            args.ckpt = str(Path(args.output_dir) / "vicreg" / "last.pt")
        model, loaded_cfg = load_model_from_checkpoint(Path(args.ckpt), device)
        models = {"vicreg": model}
        if args.baseline_ckpt:
            baseline, _ = load_model_from_checkpoint(Path(args.baseline_ckpt), device)
            models["mse"] = baseline
        if args.dataset_path:
            loaded_cfg = apply_dataset_path(loaded_cfg, args.dataset_path, args.num_workers)
        dataset = build_dataset(loaded_cfg.data, split="all")
        report = run_archival_evaluation(
            models,
            dataset,
            device=device,
            metrics_png=Path(args.metrics_png),
            max_horizon=args.max_horizon,
        )
        return report

    if args.dataset_path:
        cfg = apply_dataset_path(cfg, args.dataset_path, args.num_workers)
        cfg.training.device = args.device

    seed_everything(cfg.training.seed)
    dataset = build_dataset(cfg.data, split="all")
    LOGGER.info(
        "Dataset  source=%s  episodes=%d  T=%d  transitions=%d  train/val=%s  workers=%d",
        cfg.data.dataset_path or "synthetic",
        dataset.n_episodes,
        cfg.data.episode_length,
        len(dataset),
        dataset.split_sizes(),
        cfg.training.num_workers,
    )

    models: dict[str, WorldModel] = {}
    summaries: dict[str, Any] = {}

    def _record(name: str, loss_type: str) -> WorldModel:
        model, summary = _train_variant(cfg, loss_type, name, dataset)
        models[name] = model
        summaries[name] = summary
        return model

    if args.mode == "train":
        loss_type = args.loss_type or "vicreg"
        run_name = "mse_baseline" if loss_type == "mse" else loss_type
        _record(run_name, loss_type)
    elif args.mode == "train_and_eval":
        _record("vicreg", "vicreg")
        if not args.skip_baseline:
            _record("mse", "mse")

    report: dict[str, Any] = {"training": summaries}
    if args.mode == "train_and_eval":
        run_archival_evaluation = _load_eval()

        device = cfg.training.resolve_device()
        eval_report = run_archival_evaluation(
            models,
            dataset,
            device=device,
            metrics_png=Path(args.metrics_png),
            max_horizon=args.max_horizon,
        )
        report["evaluation"] = eval_report
        (Path(args.output_dir) / "run_report.json").write_text(
            json.dumps(
                {
                    "training": {
                        name: {k: v for k, v in payload.items() if k != "history"}
                        | {"n_epochs_logged": len(payload.get("history", []))}
                        for name, payload in summaries.items()
                    },
                    "evaluation": eval_report,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Wrote %s and %s", args.metrics_png, Path(args.output_dir) / "run_report.json")
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
