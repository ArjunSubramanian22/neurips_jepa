"""Clean PyTorch training loop for Action-Conditioned JEPA.

Supports AMP, AdamW, cosine decay with warmup, EMA target updates, and
JSON/stdout logging of the VICReg constituents (total, invariance, variance,
covariance) plus encoder–target L2 distance. Optional W&B / TensorBoard
mirrors activate via ``WANDB_API_KEY`` / ``TENSORBOARD_LOGDIR``.

    python src/train.py --mode train_and_eval
    python src/train.py --config configs/workshop_vicreg.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
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
        load_experiment_config,
        resolve_device,
        resolve_warmup_epochs,
        seed_everything,
    )
    from .dataset import (
        DeformableObjectDataset,
        DeformablePhysicsDataset,
        ParticleTransitionDataset,
        build_dataset,
        dataloader_worker_kwargs,
        inspect_physics_source,
    )
    from .losses import prediction_objective, vicreg_loss
    from .models.ac_jepa import ACJEPA
    from .models.pixel_ae import PixelSpaceAutoencoder
    from .models.particle_gnn import ParticleGraphWorldModel
    from .models.rssm import CompactRSSM, RSSMWindowDataset
    from .models.latent_transformer import CausalLatentDynamics
    from .models.obs_pred import ActionConditionedUNet
else:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.config import (
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
        load_experiment_config,
        resolve_device,
        resolve_warmup_epochs,
        seed_everything,
    )
    from src.dataset import (
        DeformableObjectDataset,
        DeformablePhysicsDataset,
        ParticleTransitionDataset,
        build_dataset,
        dataloader_worker_kwargs,
        inspect_physics_source,
    )
    from src.losses import prediction_objective, vicreg_loss
    from src.models.ac_jepa import ACJEPA
    from src.models.pixel_ae import PixelSpaceAutoencoder
    from src.models.particle_gnn import ParticleGraphWorldModel
    from src.models.rssm import CompactRSSM, RSSMWindowDataset
    from src.models.latent_transformer import CausalLatentDynamics
    from src.models.obs_pred import ActionConditionedUNet

LOGGER = logging.getLogger("ac_jepa.train")
RunMode = Literal["train", "eval", "train_and_eval"]
WorldModel = Union[ACJEPA, PixelSpaceAutoencoder, ParticleGraphWorldModel, CompactRSSM, CausalLatentDynamics, ActionConditionedUNet]


class ExperimentLogger:
    """Always writes metrics.jsonl; optionally mirrors to W&B / TensorBoard.

    W&B activates when ``WANDB_API_KEY`` is set. TensorBoard activates when
    ``TENSORBOARD_LOGDIR`` is set. Missing packages or init failures are
    logged and ignored so local training never hard-fails.
    """

    def __init__(self, run_name: str, output_dir: Path, cfg: ExperimentConfig) -> None:
        self.run_name = run_name
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.jsonl_path.write_text("", encoding="utf-8")
        self._wandb: Any = None
        self._tb: Any = None
        api_key = os.environ.get("WANDB_API_KEY", "").strip()
        if api_key:
            try:
                import wandb

                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "ac-jepa"),
                    name=run_name,
                    config=cfg.to_dict(),
                    reinit=True,
                    settings=wandb.Settings(start_method="thread"),
                )
                self._wandb = wandb
                LOGGER.info("W&B logging enabled  project=%s run=%s", wandb.run.project, run_name)
            except Exception as exc:  # noqa: BLE001 — optional sink
                LOGGER.warning("W&B requested via WANDB_API_KEY but init failed: %s", exc)
        tb_root = os.environ.get("TENSORBOARD_LOGDIR", "").strip()
        if tb_root:
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = Path(tb_root) / run_name
                log_dir.mkdir(parents=True, exist_ok=True)
                self._tb = SummaryWriter(log_dir=str(log_dir))
                LOGGER.info("TensorBoard logging enabled  dir=%s", log_dir)
            except Exception as exc:  # noqa: BLE001 — optional sink
                LOGGER.warning("TENSORBOARD_LOGDIR set but TensorBoard init failed: %s", exc)

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        split = str(payload.get("split", "train"))
        scalars = {
            key: value
            for key, value in payload.items()
            if key not in {"run", "split"} and isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        if self._wandb is not None:
            try:
                self._wandb.log({f"{split}/{key}": value for key, value in scalars.items()}, step=step)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("W&B log failed: %s", exc)
        if self._tb is not None:
            try:
                for key, value in scalars.items():
                    self._tb.add_scalar(f"{split}/{key}", float(value), global_step=step)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("TensorBoard log failed: %s", exc)

    def close(self) -> None:
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


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
    """Instantiate AC-JEPA, pixel autoencoder, or particle GNN according to ``loss_type``."""
    if cfg.training.loss_type == "pixel_ae":
        return PixelSpaceAutoencoder(cfg.model)
    if cfg.training.loss_type == "gnn":
        return ParticleGraphWorldModel(cfg.model, cfg.data)
    if cfg.training.loss_type == "rssm":
        return CompactRSSM(cfg.model)
    if cfg.training.loss_type == "latent_tf":
        return CausalLatentDynamics(cfg.model)
    if cfg.training.loss_type == "obs_pred":
        return ActionConditionedUNet(cfg.model)
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
        self.counts: dict[str, int] = defaultdict(int)
        self.count = 0

    def update(self, metrics: dict[str, float], batch_size: int = 1) -> None:
        self.count += batch_size
        for key, value in metrics.items():
            self.totals[key] += float(value) * batch_size
            self.counts[key] += batch_size

    def mean(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: self.totals[key] / self.counts[key] for key in self.totals}


def _unpack_transition_batch(
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if len(batch) == 3:
        x_t, a_t, x_tp1 = batch
        return x_t, a_t, x_tp1, None, None, None
    if len(batch) != 6:
        raise ValueError(f"expected 3 or 6 transition tensors, got {len(batch)}.")
    x_t, a_t, x_tp1, a_prime, x_prime, cf_valid = batch
    return x_t, a_t, x_tp1, a_prime, x_prime, cf_valid


def _sequence_loaders(dataset: Any, train_cfg: TrainingConfig, window: int) -> tuple[DataLoader, DataLoader | None]:
    """Sequence windows. JEPA transition loaders are left unchanged."""
    n_train, n_val = dataset.split_sizes()
    worker_kwargs = dataloader_worker_kwargs(dataset.cfg, train_cfg)
    train_loader = DataLoader(
        RSSMWindowDataset(dataset, range(n_train), window),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        **worker_kwargs,
    )
    val_loader = None
    if n_val > 0:
        val_loader = DataLoader(
            RSSMWindowDataset(dataset, range(n_train, n_train + n_val), window),
            batch_size=train_cfg.batch_size,
            shuffle=False,
            **worker_kwargs,
        )
    return train_loader, val_loader


def _rssm_loaders(dataset: Any, train_cfg: TrainingConfig) -> tuple[DataLoader, DataLoader | None]:
    """Sequence windows for the RSSM."""
    return _sequence_loaders(dataset, train_cfg, train_cfg.rssm_window)


def _transition_subset(dataset: DeformableObjectDataset | DeformablePhysicsDataset, episode_slice: slice) -> Subset:
    start, stop, _ = episode_slice.indices(dataset.n_episodes)
    indices = dataset.indices_for_episodes(start, stop)
    return Subset(dataset, indices)


def build_dataloaders(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    *,
    full_dataset: DeformableObjectDataset | DeformablePhysicsDataset | None = None,
    counterfactual_cache: Any | None = None,
) -> tuple[DataLoader, DataLoader | None, DeformableObjectDataset | DeformablePhysicsDataset]:
    """Build train/val loaders from a single simulated or physics episode archive."""
    dataset = full_dataset or build_dataset(data_cfg, split="all")
    n_train, n_val = dataset.split_sizes()
    worker_kwargs = dataloader_worker_kwargs(data_cfg, train_cfg)
    if train_cfg.loss_type == "gnn":
        if getattr(dataset, "positions", None) is None:
            raise ValueError("GNN baseline requires the mass-spring simulator (no particle dump).")
        train_idx = dataset.indices_for_episodes(0, n_train)
        train_loader = DataLoader(
            ParticleTransitionDataset(dataset, train_idx),  # type: ignore[arg-type]
            batch_size=train_cfg.batch_size,
            shuffle=True,
            **worker_kwargs,
        )
        val_loader = None
        if n_val > 0:
            val_idx = dataset.indices_for_episodes(n_train, n_train + n_val)
            val_loader = DataLoader(
                ParticleTransitionDataset(dataset, val_idx),  # type: ignore[arg-type]
                batch_size=train_cfg.batch_size,
                shuffle=False,
                **worker_kwargs,
            )
        return train_loader, val_loader, dataset
    train_indices = dataset.indices_for_episodes(0, n_train)
    if counterfactual_cache is not None:
        from src.action_alignment import PairedCounterfactualDataset

        train_set = PairedCounterfactualDataset(
            dataset,
            counterfactual_cache,
            train_indices,
            fraction=train_cfg.cf_batch_fraction,
            seed=train_cfg.seed,
        )
    else:
        train_set = _transition_subset(dataset, slice(0, n_train))
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
        self.inverse_head = None
        if cfg.training.inverse_mode == "joint" and cfg.training.inverse_aux_weight > 0.0:
            if not cfg.data.normalize_actions:
                raise ValueError("inverse_aux_weight > 0 requires normalized actions.")
            from src.models.inverse_dynamics import InverseDynamicsHead

            self.inverse_head = InverseDynamicsHead(
                cfg.model.effective_latent_dim,
                cfg.model.action_dim,
                cfg.model.predictor_embed_dim,
            ).to(self.device)
        trainable = list(self.model.trainable_parameters())
        if self.inverse_head is not None:
            trainable.extend(self.inverse_head.parameters())
        self.train_loader, self.val_loader, self.dataset = build_dataloaders(
            cfg.data,
            cfg.training,
            full_dataset=dataset,
            counterfactual_cache=self._load_counterfactual_cache(),
        )
        if cfg.training.loss_type == "rssm":
            self.train_loader, self.val_loader = _rssm_loaders(self.dataset, cfg.training)
        elif cfg.training.loss_type == "latent_tf":
            self.train_loader, self.val_loader = _sequence_loaders(
                self.dataset, cfg.training, cfg.model.latent_tf_history + 1
            )
        elif cfg.training.loss_type == "obs_pred" and cfg.training.obs_pred_horizon > 1:
            self.train_loader, self.val_loader = _sequence_loaders(
                self.dataset, cfg.training, cfg.training.obs_pred_horizon + 1
            )
        self.optimizer = AdamW(
            trainable,
            lr=cfg.training.lr,
            betas=(cfg.training.beta1, cfg.training.beta2),
            weight_decay=cfg.training.weight_decay,
        )
        steps_per_epoch = max(len(self.train_loader), 1)
        warmup_epochs = resolve_warmup_epochs(
            cfg.training.epochs, cfg.training.warmup_epochs, cfg.training.warmup_ratio
        )
        warmup_epochs = min(warmup_epochs, cfg.training.epochs)
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
        self.logger = ExperimentLogger(run_name, self.output_dir, cfg)
        self.history: list[dict[str, Any]] = []
        self.global_step = 0

    def _alignment_enabled(self) -> bool:
        train_cfg = self.cfg.training
        return train_cfg.cf_lambda_dir > 0.0 or train_cfg.cf_lambda_mag > 0.0 or train_cfg.cf_lambda_vec > 0.0

    def _load_counterfactual_cache(self) -> Any | None:
        needs_fork = self._alignment_enabled() or self.cfg.training.target_mask in ("soft", "topk")
        if not needs_fork:
            return None
        from src.action_alignment import load_counterfactual_cache

        path = Path(str(self.cfg.training.cf_cache_path))
        if not path.is_file():
            raise FileNotFoundError(f"counterfactual cache does not exist: {path}")
        cache = load_counterfactual_cache(path)
        if cache.family != self.cfg.training.cf_intervention:
            raise ValueError(
                f"cache family {cache.family!r} does not match cf_intervention {self.cfg.training.cf_intervention!r}."
            )
        return cache

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

    def _gnn_objective(
        self,
        pos_t: torch.Tensor,
        vel_t: torch.Tensor,
        a_t: torch.Tensor,
        pos_tp1: torch.Tensor,
        vel_tp1: torch.Tensor,
    ):
        pred_pos, pred_vel, _ = self.model(pos_t, vel_t, a_t)
        pos_mse = F.mse_loss(pred_pos, pos_tp1)
        vel_mse = F.mse_loss(pred_vel, vel_tp1)
        loss = pos_mse + 0.25 * vel_mse
        zero = loss.new_zeros(())
        terms = {
            "loss": loss,
            "invariance": pos_mse,
            "variance": vel_mse,
            "covariance": zero,
        }
        return loss, terms

    def _forward_loss(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        x_tp1: torch.Tensor,
        a_prime: torch.Tensor | None = None,
        x_prime: torch.Tensor | None = None,
        cf_valid: torch.Tensor | None = None,
    ):
        if self.cfg.training.loss_type == "pixel_ae":
            return self._pixel_ae_objective(x_t, a_t, x_tp1)
        output = self.model(x_t, a_t, x_tp1)
        assert output.z_tp1 is not None
        if self._uses_experimental_target(x_prime):
            loss, terms = self._experimental_target_loss(output, x_t, a_t, x_tp1, x_prime, cf_valid)
        else:
            loss, terms = self._objective(output.z_hat_tp1, output.z_tp1)
        weight = float(self.cfg.training.inverse_aux_weight)
        if self.inverse_head is not None and weight > 0.0:
            from src.models.inverse_dynamics import combine_prediction_and_inverse, inverse_dynamics_loss

            z_tp1_online = self.model.encode(x_tp1)
            predicted_action = self.inverse_head(output.z_t, z_tp1_online)
            inverse = inverse_dynamics_loss(
                predicted_action,
                a_t,
                loss_type=self.cfg.training.inverse_loss_type,
            )
            loss = combine_prediction_and_inverse(loss, inverse["loss"], weight)
            terms["loss"] = loss
            terms["inverse"] = inverse["loss"].detach()
            terms["inverse_center"] = inverse["center"].detach()
            terms["inverse_force"] = inverse["force"].detach()
            for index, value in enumerate(inverse["per_coordinate"].detach()):
                terms[f"inverse_coord_{index}"] = value
        if self._alignment_enabled() and a_prime is not None and x_prime is not None and cf_valid is not None:
            from src.action_alignment import apply_action_alignment

            train_cfg = self.cfg.training
            loss, alignment_logs = apply_action_alignment(
                loss,
                self.model.predictor,
                output.z_t,
                output.z_hat_tp1,
                output.z_tp1,
                a_prime,
                x_prime,
                self.model.target_encoder,
                cf_valid,
                lambda_dir=train_cfg.cf_lambda_dir,
                lambda_mag=train_cfg.cf_lambda_mag,
                lambda_vec=train_cfg.cf_lambda_vec,
                norm_tau=train_cfg.cf_norm_tau,
                eps=train_cfg.cf_eps,
            )
            terms["loss"] = loss
            for key, value in alignment_logs.items():
                terms[key] = loss.new_tensor(value)
        return loss, terms

    def _uses_experimental_target(self, x_prime: torch.Tensor | None) -> bool:
        """Canonical CLS training never enters the spatial target."""
        if self.cfg.model.target_readout == "cls":
            return False
        if self.cfg.training.target_mask == "uniform":
            return True
        return x_prime is not None

    def _experimental_target_loss(
        self,
        output: Any,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        x_tp1: torch.Tensor,
        x_prime: torch.Tensor | None,
        cf_valid: torch.Tensor | None,
    ):
        from src.target_mask import mask_weights, weighted_patch_pool

        train_cfg = self.cfg.training
        readout = self.cfg.model.target_readout
        counterfactual = None
        if train_cfg.target_mask != "uniform":
            if x_prime is None or cf_valid is None:
                raise RuntimeError("a fork mask requires the counterfactual future frame.")
            valid = cf_valid.reshape(-1) > 0.5
            counterfactual = torch.where(valid.view(-1, 1, 1, 1), x_prime, x_tp1)
        tokens = self.model.encode_patches(x_tp1, target=True)
        weights, degenerate = mask_weights(
            x_tp1,
            counterfactual,
            mode=train_cfg.target_mask,
            patch_size=self.cfg.model.patch_size,
            n_patches=int(tokens.shape[1]),
            topk_fraction=train_cfg.target_mask_topk_fraction,
            temperature=train_cfg.target_mask_temperature,
        )
        weights = weights.detach()
        if readout == "weighted_pool":
            loss, terms = self._objective(output.z_hat_tp1, weighted_patch_pool(tokens, weights))
        elif readout == "local_head":
            loss, terms = self._objective(output.z_hat_tp1, output.z_tp1)
            predicted = self.model.local_head(self.model.encode_patches(x_t), a_t)
            aux = ((predicted - tokens).pow(2).mean(dim=-1) * weights).sum(dim=-1).mean()
            loss = loss + loss.new_tensor(float(train_cfg.target_mask_aux_weight)) * aux
            terms["loss"] = loss
            terms["mask_aux"] = aux.detach()
        elif readout == "patch_tokens":
            loss, terms = self._objective(output.z_hat_tp1, output.z_tp1)
            predicted = self.model.patch_predictor(output.z_hat_tp1, a_t)
            aux = ((predicted - tokens).pow(2).mean(dim=-1) * weights).sum(dim=-1).mean()
            loss = loss + loss.new_tensor(float(train_cfg.target_mask_aux_weight)) * aux
            terms["loss"] = loss
            terms["mask_aux"] = aux.detach()
        else:
            raise ValueError(f"unknown target_readout {readout!r}.")
        terms["mask_degenerate"] = degenerate.float().mean().detach()
        terms["mask_applied"] = loss.new_ones(())
        return loss, terms

    def _step_metrics(self, terms: dict[str, torch.Tensor]) -> dict[str, float]:
        metrics = {
            "loss": float(terms["loss"].detach().item()),
            "invariance": float(terms["invariance"].detach().item()),
            "variance": float(terms["variance"].detach().item()),
            "covariance": float(terms["covariance"].detach().item()),
        }
        for key, value in terms.items():
            if key.startswith("inverse") or key.startswith("cf_") or key.startswith("mask_"):
                metrics[key] = float(value.detach().item())
        return metrics

    def training_step(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        x_tp1: torch.Tensor,
        a_prime: torch.Tensor | None = None,
        x_prime: torch.Tensor | None = None,
        cf_valid: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.model.train()
        x_t = x_t.to(self.device, non_blocking=True)
        a_t = a_t.to(self.device, non_blocking=True)
        x_tp1 = x_tp1.to(self.device, non_blocking=True)
        if a_prime is not None:
            a_prime = a_prime.to(self.device, non_blocking=True)
        if x_prime is not None:
            x_prime = x_prime.to(self.device, non_blocking=True)
        if cf_valid is not None:
            cf_valid = cf_valid.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.device, self.use_amp):
            loss, terms = self._forward_loss(x_t, a_t, x_tp1, a_prime, x_prime, cf_valid)
        self.scaler.scale(loss).backward()
        params = list(self.model.trainable_parameters())
        if self.inverse_head is not None:
            params.extend(self.inverse_head.parameters())
        self.scaler.unscale_(self.optimizer)
        from src.models.ac_jepa import action_conditioning_grad_norm

        action_grad = action_conditioning_grad_norm(self.model.predictor)
        max_norm = self.cfg.training.grad_clip if self.cfg.training.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.global_step % self.cfg.training.ema_sync_every == 0:
            self.model.update_target_encoder()
        self.global_step += 1
        metrics = self._step_metrics(terms)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["grad_norm"] = float(grad_norm)
        metrics["action_cond_grad_norm"] = action_grad
        metrics["ema_param_l2"] = self._ema_param_l2()
        return metrics

    def rssm_training_step(self, frames: torch.Tensor, actions: torch.Tensor) -> dict[str, float]:
        self.model.train()
        frames = frames.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.device, self.use_amp):
            terms = self.model.loss(frames, actions)
        self.scaler.scale(terms["loss"]).backward()
        params = list(self.model.parameters())
        self.scaler.unscale_(self.optimizer)
        max_norm = self.cfg.training.grad_clip if self.cfg.training.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        return {
            "loss": float(terms["loss"].detach().item()),
            "recon": float(terms["recon"].detach().item()),
            "kl": float(terms["kl"].detach().item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "grad_norm": float(grad_norm),
        }

    @torch.no_grad()
    def _rssm_evaluate_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        meter = _AverageMeter()
        for frames, actions in loader:
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            terms = self.model.loss(frames, actions)
            meter.update(
                {
                    "loss": float(terms["loss"].detach().item()),
                    "recon": float(terms["recon"].detach().item()),
                    "kl": float(terms["kl"].detach().item()),
                },
                batch_size=int(frames.shape[0]),
            )
        return meter.mean()

    @torch.no_grad()
    def _latent_tf_evaluate_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        meter = _AverageMeter()
        for frames, actions in loader:
            terms = self.model.loss(frames.to(self.device), actions.to(self.device))
            meter.update({"loss": float(terms["loss"].detach().item())}, batch_size=int(frames.shape[0]))
        return meter.mean()

    def latent_tf_training_step(self, frames: torch.Tensor, actions: torch.Tensor) -> dict[str, float]:
        self.model.train()
        frames = frames.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.device, self.use_amp):
            terms = self.model.loss(frames, actions, record_attention=True)
        self.scaler.scale(terms["loss"]).backward()
        action_grad = self.model.action_embedding_grad_norm()
        params = list(self.model.trainable_parameters())
        self.scaler.unscale_(self.optimizer)
        max_norm = self.cfg.training.grad_clip if self.cfg.training.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.global_step % self.cfg.training.ema_sync_every == 0:
            self.model.update_target_encoder()
        self.global_step += 1
        replacement = actions.clone()
        replacement[:, :, 2] = -replacement[:, :, 2]
        sensitivity = self.model.action_replacement_sensitivity(frames, actions, replacement)
        return {
            "loss": float(terms["loss"].detach().item()),
            "action_embed_grad_norm": action_grad,
            "action_attention_mass": float(self.model.last_attention_action_mass or 0.0),
            "latent_attention_mass": float(self.model.last_attention_latent_mass or 0.0),
            "action_replacement_sensitivity": sensitivity,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "grad_norm": float(grad_norm),
        }

    def _require_raster_batch(self) -> None:
        if self.cfg.training.observation_space == "rgb":
            raise RuntimeError(
                "RGB training needs particle positions to render rgb_v1. "
                "The image dump is the physical raster, so observation_space must be raster."
            )

    def obs_pred_training_step(self, frames: torch.Tensor, actions: torch.Tensor) -> dict[str, float]:
        self._require_raster_batch()
        self.model.train()
        frames = frames.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.device, self.use_amp):
            terms = self.model.prediction_loss(
                frames,
                actions,
                horizon=self.cfg.training.obs_pred_horizon,
                teacher_force=self.cfg.training.obs_pred_teacher_force,
            )
        self.scaler.scale(terms["loss"]).backward()
        action_grad = self.model.action_path_grad_norm()
        self.scaler.unscale_(self.optimizer)
        max_norm = self.cfg.training.grad_clip if self.cfg.training.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        return {
            "loss": float(terms["loss"].detach().item()),
            "action_path_grad_norm": action_grad,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "grad_norm": float(grad_norm),
        }

    @torch.no_grad()
    def _obs_pred_evaluate_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        meter = _AverageMeter()
        for frames, actions in loader:
            terms = self.model.prediction_loss(
                frames.to(self.device),
                actions.to(self.device),
                horizon=self.cfg.training.obs_pred_horizon,
                teacher_force=True,
            )
            meter.update({"loss": float(terms["loss"].detach().item())}, batch_size=int(frames.shape[0]))
        return meter.mean()

    @torch.no_grad()
    def _obs_pred_evaluate_transitions(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        meter = _AverageMeter()
        for batch in loader:
            x_t, a_t, x_tp1, _a_prime, _x_prime, _cf_valid = _unpack_transition_batch(batch)
            frames = torch.stack([x_t, x_tp1], dim=1).to(self.device)
            terms = self.model.prediction_loss(
                frames,
                a_t.unsqueeze(1).to(self.device),
                horizon=1,
                teacher_force=True,
            )
            meter.update({"loss": float(terms["loss"].detach().item())}, batch_size=int(x_t.shape[0]))
        return meter.mean()

    def _fit_inverse_probe(self) -> None:
        """Fit the inverse head on a frozen encoder. Representation weights do not move."""
        from src.models.inverse_dynamics import InverseDynamicsHead, inverse_dynamics_loss

        if not self.cfg.data.normalize_actions:
            raise ValueError("the inverse probe requires normalized actions.")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.inverse_head = InverseDynamicsHead(
            self.cfg.model.effective_latent_dim,
            self.cfg.model.action_dim,
            self.cfg.model.predictor_embed_dim,
        ).to(self.device)
        optimizer = AdamW(
            self.inverse_head.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        for epoch in range(1, self.cfg.training.inverse_probe_epochs + 1):
            meter = _AverageMeter()
            for batch in self.train_loader:
                x_t, a_t, x_tp1 = (tensor.to(self.device) for tensor in batch[:3])
                with torch.no_grad():
                    z_t = self.model.encode(x_t)
                    z_tp1 = self.model.encode(x_tp1)
                inverse = inverse_dynamics_loss(
                    self.inverse_head(z_t, z_tp1),
                    a_t,
                    loss_type=self.cfg.training.inverse_loss_type,
                )
                optimizer.zero_grad(set_to_none=True)
                inverse["loss"].backward()
                optimizer.step()
                stats = {
                    "inverse": float(inverse["loss"].detach().item()),
                    "inverse_center": float(inverse["center"].detach().item()),
                    "inverse_force": float(inverse["force"].detach().item()),
                }
                for index, value in enumerate(inverse["per_coordinate"].detach()):
                    stats[f"inverse_coord_{index}"] = float(value.item())
                meter.update(stats, batch_size=int(x_t.shape[0]))
            self.history.append({"probe_epoch": epoch, **{f"probe_{key}": value for key, value in meter.mean().items()}})
        self.save_checkpoint(self.output_dir / "inverse_probe.pt", self.cfg.training.epochs, {"inverse_probe": True})

    def gnn_training_step(
        self,
        pos_t: torch.Tensor,
        vel_t: torch.Tensor,
        a_t: torch.Tensor,
        pos_tp1: torch.Tensor,
        vel_tp1: torch.Tensor,
    ) -> dict[str, float]:
        self.model.train()
        pos_t = pos_t.to(self.device, non_blocking=True)
        vel_t = vel_t.to(self.device, non_blocking=True)
        a_t = a_t.to(self.device, non_blocking=True)
        pos_tp1 = pos_tp1.to(self.device, non_blocking=True)
        vel_tp1 = vel_tp1.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.device, self.use_amp):
            loss, terms = self._gnn_objective(pos_t, vel_t, a_t, pos_tp1, vel_tp1)
        self.scaler.scale(loss).backward()
        params = list(self.model.trainable_parameters())
        self.scaler.unscale_(self.optimizer)
        max_norm = self.cfg.training.grad_clip if self.cfg.training.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        metrics = self._step_metrics(terms)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["grad_norm"] = float(grad_norm)
        metrics["ema_param_l2"] = float("nan")
        return metrics

    def _gnn_evaluate_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        meter = _AverageMeter()
        with torch.no_grad():
            for pos_t, vel_t, a_t, pos_tp1, vel_tp1 in loader:
                pos_t = pos_t.to(self.device, non_blocking=True)
                vel_t = vel_t.to(self.device, non_blocking=True)
                a_t = a_t.to(self.device, non_blocking=True)
                pos_tp1 = pos_tp1.to(self.device, non_blocking=True)
                vel_tp1 = vel_tp1.to(self.device, non_blocking=True)
                with _autocast(self.device, self.use_amp):
                    _, terms = self._gnn_objective(pos_t, vel_t, a_t, pos_tp1, vel_tp1)
                meter.update(self._step_metrics(terms), batch_size=pos_t.shape[0])
        return meter.mean()

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

    @torch.no_grad()
    def _ema_param_l2(self) -> float:
        """L2 distance between online encoder and EMA target parameters."""
        if not hasattr(self.model, "target_encoder"):
            return float("nan")
        total = 0.0
        for online, target in zip(
            self.model.encoder.parameters(), self.model.target_encoder.parameters(), strict=True
        ):
            total += float((online.detach() - target.detach()).pow(2).sum().item())
        return math.sqrt(total)

    def _log(self, epoch: int, step: int, metrics: dict[str, float], split: str) -> None:
        payload = {
            "run": self.run_name,
            "split": split,
            "epoch": epoch,
            "step": step,
            **metrics,
        }
        LOGGER.info(
            "[%s] epoch %d step %d  loss=%.4f  inv=%.4f  var=%.4f  cov=%.4f  ema_l2=%.4f  gn=%.3f  action_gn=%.3f  lr=%.2e",
            self.run_name,
            epoch,
            step,
            metrics.get("loss", float("nan")),
            metrics.get("invariance", float("nan")),
            metrics.get("variance", float("nan")),
            metrics.get("covariance", float("nan")),
            metrics.get("ema_param_l2", float("nan")),
            metrics.get("grad_norm", float("nan")),
            metrics.get("action_cond_grad_norm", float("nan")),
            metrics.get("lr", self.optimizer.param_groups[0]["lr"]),
        )
        self.logger.log(payload, step=step)

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
        if self.inverse_head is not None:
            payload["inverse_head"] = self.inverse_head.state_dict()
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
        try:
            for epoch in range(1, train_cfg.epochs + 1):
                epoch_meter = _AverageMeter()
                for step, batch in enumerate(self.train_loader, start=1):
                    if train_cfg.loss_type == "gnn":
                        metrics = self.gnn_training_step(*batch)
                        batch_n = int(batch[0].shape[0])
                    elif train_cfg.loss_type == "rssm":
                        frames, actions = batch
                        metrics = self.rssm_training_step(frames, actions)
                        batch_n = int(frames.shape[0])
                    elif train_cfg.loss_type == "latent_tf":
                        frames, actions = batch
                        metrics = self.latent_tf_training_step(frames, actions)
                        batch_n = int(frames.shape[0])
                    elif train_cfg.loss_type == "obs_pred" and train_cfg.obs_pred_horizon > 1:
                        frames, actions = batch
                        metrics = self.obs_pred_training_step(frames, actions)
                        batch_n = int(frames.shape[0])
                    elif train_cfg.loss_type == "obs_pred":
                        x_t, a_t, x_tp1, _a_prime, _x_prime, _cf_valid = _unpack_transition_batch(batch)
                        frames = torch.stack([x_t, x_tp1], dim=1)
                        metrics = self.obs_pred_training_step(frames, a_t.unsqueeze(1))
                        batch_n = int(x_t.shape[0])
                    else:
                        x_t, a_t, x_tp1, a_prime, x_prime, cf_valid = _unpack_transition_batch(batch)
                        metrics = self.training_step(x_t, a_t, x_tp1, a_prime, x_prime, cf_valid)
                        batch_n = int(x_t.shape[0])
                    epoch_meter.update(metrics, batch_size=batch_n)
                    if step % train_cfg.log_interval == 0 or step == 1:
                        self._log(epoch, self.global_step, metrics, split="train_step")
                epoch_stats = {f"train_{k}": v for k, v in epoch_meter.mean().items()}
                epoch_stats["ema_param_l2"] = self._ema_param_l2()
                if self.val_loader is not None:
                    if train_cfg.loss_type == "gnn":
                        val_stats = self._gnn_evaluate_loader(self.val_loader)
                    elif train_cfg.loss_type == "rssm":
                        val_stats = self._rssm_evaluate_loader(self.val_loader)
                    elif train_cfg.loss_type == "latent_tf":
                        val_stats = self._latent_tf_evaluate_loader(self.val_loader)
                    elif train_cfg.loss_type == "obs_pred" and train_cfg.obs_pred_horizon > 1:
                        val_stats = self._obs_pred_evaluate_loader(self.val_loader)
                    elif train_cfg.loss_type == "obs_pred":
                        val_stats = self._obs_pred_evaluate_transitions(self.val_loader)
                    else:
                        val_stats = self.evaluate_loader(self.val_loader)
                    epoch_stats.update({f"val_{k}": v for k, v in val_stats.items()})
                    val_loss = val_stats.get("loss", float("inf"))
                    if val_loss < best_val:
                        best_val = val_loss
                        self.save_checkpoint(
                            self.output_dir / "best.pt", epoch, {"best_val": best_val}
                        )
                epoch_log = {
                    "loss": epoch_stats.get("train_loss", float("nan")),
                    "invariance": epoch_stats.get("train_invariance", float("nan")),
                    "variance": epoch_stats.get("train_variance", float("nan")),
                    "covariance": epoch_stats.get("train_covariance", float("nan")),
                    "grad_norm": epoch_stats.get("train_grad_norm", float("nan")),
                    "action_cond_grad_norm": epoch_stats.get("train_action_cond_grad_norm", float("nan")),
                    "ema_param_l2": epoch_stats["ema_param_l2"],
                    "lr": epoch_stats.get("train_lr", float(self.optimizer.param_groups[0]["lr"])),
                }
                for key, value in epoch_stats.items():
                    if key.startswith("train_inverse") or key.startswith("train_cf_") or key.startswith("train_mask_"):
                        epoch_log[key.removeprefix("train_")] = value
                for key in ("val_loss", "val_invariance", "val_variance", "val_covariance"):
                    if key in epoch_stats:
                        epoch_log[key] = epoch_stats[key]
                self._log(epoch, self.global_step, epoch_log, split="train_epoch")
                self.history.append({"epoch": epoch, **epoch_stats})
                if epoch % train_cfg.ckpt_interval == 0 or epoch == train_cfg.epochs:
                    self.save_checkpoint(self.output_dir / "last.pt", epoch)
            if train_cfg.inverse_mode == "probe":
                self._fit_inverse_probe()
        finally:
            self.logger.close()
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


def peek_checkpoint_epoch(path: Path) -> int:
    """Read ``epoch`` from a checkpoint without constructing the model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("epoch") or 0)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate AC-JEPA.")
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "train_and_eval"),
        default="train_and_eval",
        help="train, eval a checkpoint, or run the 5-epoch smoke + archival suite.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML/JSON ExperimentConfig (e.g. configs/workshop_vicreg.yaml). Overrides the smoke defaults.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs. Default 5 for the smoke schedule; ignored unless set when --config is used.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-png", default="outputs/metrics.png")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    parser.add_argument("--ckpt", default=None, help="Checkpoint for --mode eval (VICReg run).")
    parser.add_argument("--baseline-ckpt", default=None, help="Optional MSE-baseline checkpoint.")
    parser.add_argument(
        "--loss-type",
        choices=("vicreg", "vicreg_no_cov", "mse", "smooth_l1", "cosine", "pixel_ae", "gnn"),
        default=None,
    )
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument("--skip-baseline", action="store_true", help="Do not train the MSE ablation.")
    parser.add_argument(
        "--skip-pixel-ae",
        action="store_true",
        help="Do not train the pixel-space autoencoder baseline (workshop --config only).",
    )
    parser.add_argument(
        "--no-action",
        action="store_true",
        help="Drop a_t from the predictor (condition_on_action=False).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the output run directory name under training.output_dir.",
    )
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
    if args.config:
        cfg = load_experiment_config(args.config)
        workshop = True
    else:
        cfg = build_smoke_config(epochs=args.epochs or 5, output_dir=args.output_dir or "outputs")
        workshop = False
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.device:
        cfg.training.device = args.device
    if args.seed is not None:
        cfg.training.seed = int(args.seed)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
        cfg.training.warmup_epochs = resolve_warmup_epochs(
            cfg.training.epochs, cfg.training.warmup_epochs, cfg.training.warmup_ratio
        )
    if args.loss_type is not None:
        cfg.training.loss_type = args.loss_type  # type: ignore[assignment]
    if args.no_action:
        cfg.model.condition_on_action = False
    if args.num_workers is not None and args.dataset_path is None and cfg.data.dataset_path is None:
        cfg.training.num_workers = args.num_workers
    output_root = Path(cfg.training.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        run_archival_evaluation = _load_eval()

        device = resolve_device(args.device or cfg.training.device)
        ckpt = args.ckpt or str(output_root / "vicreg" / "last.pt")
        model, loaded_cfg = load_model_from_checkpoint(Path(ckpt), device)
        models: dict[str, WorldModel] = {"vicreg": model}
        if args.baseline_ckpt:
            baseline, _ = load_model_from_checkpoint(Path(args.baseline_ckpt), device)
            models["mse"] = baseline
        pixel_ckpt = output_root / "pixel_ae" / "last.pt"
        if pixel_ckpt.is_file() and "pixel_ae" not in models:
            models["pixel_ae"], _ = load_model_from_checkpoint(pixel_ckpt, device)
        dataset_path = args.dataset_path or loaded_cfg.data.dataset_path
        if dataset_path:
            loaded_cfg = apply_dataset_path(loaded_cfg, dataset_path, args.num_workers)
        dataset = build_dataset(loaded_cfg.data, split="all")
        report = run_archival_evaluation(
            models,
            dataset,
            device=device,
            metrics_png=Path(args.metrics_png),
            max_horizon=args.max_horizon,
        )
        return report

    dataset_path = args.dataset_path or cfg.data.dataset_path
    if dataset_path:
        cfg = apply_dataset_path(cfg, dataset_path, args.num_workers)
        if args.device:
            cfg.training.device = args.device
        if args.output_dir:
            cfg.training.output_dir = args.output_dir

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

    models = {}
    summaries: dict[str, Any] = {}

    def _record(name: str, loss_type: str) -> WorldModel:
        model, summary = _train_variant(cfg, loss_type, name, dataset)
        models[name] = model
        summaries[name] = summary
        return model

    if args.mode == "train":
        loss_type = args.loss_type or cfg.training.loss_type
        run_name = args.run_name or {"mse": "mse", "pixel_ae": "pixel_ae", "vicreg": "vicreg"}.get(
            loss_type, loss_type
        )
        if args.no_action and args.run_name is None:
            run_name = "no_action"
        _record(run_name, loss_type)
    elif args.mode == "train_and_eval":
        _record("vicreg", "vicreg")
        if not args.skip_baseline:
            _record("mse", "mse")
        if workshop and not args.skip_pixel_ae:
            _record("pixel_ae", "pixel_ae")

    report = {"training": summaries}
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
        (output_root / "run_report.json").write_text(
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
        LOGGER.info("Wrote %s and %s", args.metrics_png, output_root / "run_report.json")
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
