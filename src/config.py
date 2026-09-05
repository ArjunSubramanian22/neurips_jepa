"""Dataclass configurations for Action-Conditioned JEPA (AC-JEPA).

The three primary configs (`ModelConfig`, `TrainingConfig`, `DataConfig`)
are grouped by `ExperimentConfig` so a training entrypoint can construct,
validate, and serialize a full run with a single object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import torch


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the AC-JEPA latent world model.

    The online encoder maps observation frames ``x_t`` to a latent state.
    An action-conditioned predictor maps ``(z_t, a_t)`` to a forecast
    ``ẑ_{t+1}``. A target encoder (EMA copy of the online encoder) produces
    the regression target from ``x_{t+1}``.

    Attributes:
        in_channels: Observation tensor channel count ``C``.
        img_size: Square spatial resolution ``H = W`` of encoded frames.
        patch_size: ViT patch size. Must divide ``img_size``.
        embed_dim: Token / latent embedding width after the encoder stem.
        encoder_depth: Number of transformer (or residual) blocks in the encoder.
        encoder_num_heads: Self-attention heads in the encoder.
        mlp_ratio: MLP expansion ratio inside encoder / predictor blocks.
        action_dim: Dimensionality ``A_dim`` of the action vector ``a_t``.
        action_embed_dim: Hidden size of the action MLP before fusion.
        predictor_embed_dim: Internal width of the dynamics predictor.
        predictor_depth: Number of predictor blocks.
        predictor_num_heads: Self-attention heads in the predictor.
        latent_dim: Optional projection size for the JEPA latent (``0`` keeps
            ``embed_dim``).
        ema_momentum: EMA coefficient for the target encoder (closer to 1
            updates more slowly).
        dropout: Dropout on embeddings and attention.
        drop_path: Stochastic-depth rate in encoder blocks.
        backbone: Encoder family. ``"vit"`` is the default JEPA backbone;
            ``"cnn"`` is a lightweight convolutional alternative.
        fuse_mode: How actions enter the predictor (additive token, FiLM, or
            concatenated to the CLS / pooled state).
    """

    in_channels: int = 3
    img_size: int = 64
    patch_size: int = 8
    embed_dim: int = 256
    encoder_depth: int = 6
    encoder_num_heads: int = 8
    mlp_ratio: float = 4.0
    action_dim: int = 4
    action_embed_dim: int = 128
    predictor_embed_dim: int = 256
    predictor_depth: int = 4
    predictor_num_heads: int = 8
    latent_dim: int = 0
    ema_momentum: float = 0.996
    dropout: float = 0.0
    drop_path: float = 0.1
    backbone: Literal["vit", "cnn"] = "vit"
    fuse_mode: Literal["token", "film", "concat"] = "concat"

    def __post_init__(self) -> None:
        if self.img_size <= 0 or self.patch_size <= 0:
            raise ValueError("img_size and patch_size must be positive.")
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size={self.img_size} must be divisible by "
                f"patch_size={self.patch_size}."
            )
        if self.embed_dim % self.encoder_num_heads != 0:
            raise ValueError("embed_dim must be divisible by encoder_num_heads.")
        if self.predictor_embed_dim % self.predictor_num_heads != 0:
            raise ValueError(
                "predictor_embed_dim must be divisible by predictor_num_heads."
            )
        if not (0.0 <= self.ema_momentum <= 1.0):
            raise ValueError("ema_momentum must lie in [0, 1].")
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1.")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1.")

    @property
    def num_patches(self) -> int:
        """Number of non-overlapping patches along one spatial axis, squared."""
        axis = self.img_size // self.patch_size
        return axis * axis

    @property
    def effective_latent_dim(self) -> int:
        return self.embed_dim if self.latent_dim <= 0 else self.latent_dim


@dataclass
class TrainingConfig:
    """Optimization, loss, and runtime hyperparameters for AC-JEPA training.

    Attributes:
        batch_size: Transitions per optimization step.
        epochs: Full passes over the training split.
        lr: Peak AdamW learning rate (after warmup).
        min_lr: Cosine-schedule floor.
        weight_decay: AdamW decoupled weight decay.
        warmup_epochs: Linear warmup length.
        grad_clip: Global-norm clip. ``0`` disables clipping.
        beta1: AdamW first-moment coefficient.
        beta2: AdamW second-moment coefficient.
        loss_type: Latent prediction loss. ``vicreg`` is the AC-JEPA default;
            ``vicreg_no_cov`` drops the covariance term; ``mse`` is invariance
            only; ``pixel_ae`` trains a pixel-space autoencoder baseline.
        vicreg_sim_coeff: VICReg invariance (prediction error) weight.
        vicreg_std_coeff: VICReg variance hinge weight.
        vicreg_cov_coeff: VICReg off-diagonal covariance weight.
        amp: Enable automatic mixed precision on CUDA / MPS.
        seed: Global RNG seed for numpy, torch, and dataset generation.
        num_workers: ``DataLoader`` worker processes.
        log_interval: Steps between scalar logs.
        ckpt_interval: Epochs between checkpoint writes.
        output_dir: Run directory for logs and checkpoints.
        device: ``"auto"`` selects CUDA, then MPS, then CPU.
        ema_sync_every: Optimizer steps between target-encoder EMA updates.
    """

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    loss_type: Literal["smooth_l1", "mse", "cosine", "vicreg", "vicreg_no_cov", "pixel_ae"] = "vicreg"
    vicreg_sim_coeff: float = 25.0
    vicreg_std_coeff: float = 25.0
    vicreg_cov_coeff: float = 1.0
    amp: bool = True
    seed: int = 42
    num_workers: int = 0
    log_interval: int = 50
    ckpt_interval: int = 10
    output_dir: str = "runs/ac_jepa"
    device: str = "auto"
    ema_sync_every: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if self.lr <= 0 or self.min_lr < 0:
            raise ValueError("lr must be > 0 and min_lr must be >= 0.")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")

    def resolve_device(self) -> torch.device:
        """Map ``device`` to a concrete ``torch.device``."""
        return resolve_device(self.device)


@dataclass
class DataConfig:
    """Dataset, rendering, and mass-spring physics hyperparameters.

    When ``dataset_path`` is ``None``, :class:`~src.dataset.DeformableObjectDataset`
    synthesizes action-conditioned trajectories from a mass-spring lattice.
    Otherwise it loads pre-computed arrays (``.npz`` or ``.pt``).

    Action layout (minimum ``2 * spatial_dims`` components):

    * 2D: ``[contact_x, contact_y, force_x, force_y]``
    * 3D: ``[contact_x, contact_y, contact_z, force_x, force_y, force_z]``

    Extra trailing action dimensions, if any, are ignored by the simulator
    and reserved for downstream policies (e.g. gripper open/close).

    Attributes:
        dataset_path: Optional path to a trajectory archive. ``None`` synthesizes
            mass-spring rollouts. Otherwise ``DeformablePhysicsDataset`` loads
            HDF5 / NPY / NPZ dumps (Taichi, PlasticineLab, MuJoCo soft-body).
        spatial_dims: Particle-space dimensionality (2 or 3). Observations are
            always rasterized to ``(C, H, W)``.
        img_size: Square frame resolution ``H = W``.
        in_channels: Rendered observation channels (see dataset renderer).
        action_dim: Action vector length ``A_dim``.
        n_episodes: Number of independent simulated rollouts (synthetic mode).
        episode_length: Frames per episode ``T``. Yields ``T - 1`` pairs.
        n_grid: Particles along each spatial axis (``n_grid ** spatial_dims``
            total particles).
        dt: Wall-clock time between consecutive observation frames.
        substeps: Physics sub-steps per observation interval.
        stiffness: Hookean spring stiffness.
        damping: Linear velocity damping in ``[0, 1)``.
        mass: Uniform particle mass.
        gravity: Gravity along the last spatial axis (tabletop 2D uses 0).
        contact_radius: Gaussian radius for distributing an action force.
        action_gain: Multiplier applied to the force components of ``a_t``.
        gaussian_sigma: Renderer splat size in pixels.
        seed: RNG seed for synthetic generation.
        val_fraction: Fraction of episodes held out when ``split='val'``.
        pin_memory: Forwarded to ``DataLoader`` when CUDA is used.
        n_stiffness_classes: Discrete material classes sampled per episode.
            ``1`` keeps a constant spring stiffness; ``>= 2`` enables the
            archival linear-probe stiffness readout.
        chunk_length: Trajectory window ``T`` for physics dumps (frames
            ``x_{t:t+T}``). Capped at the native episode length.
        chunk_stride: Hop between consecutive windows.
        augment: Train-time vision augmentations (shared across a window).
        crop_scale_min: Lower bound on random-crop area fraction.
        color_jitter: Strength of brightness / contrast / saturation jitter.
        blur_prob: Probability of a shared Gaussian blur on the window.
        blur_sigma_max: Max Gaussian sigma in pixels.
        normalize_actions: Zero-mean / unit-std actions using train-split stats.
        prefetch_factor: ``DataLoader`` batches prefetched per worker.
    """

    dataset_path: str | None = None
    spatial_dims: int = 2
    img_size: int = 64
    in_channels: int = 3
    action_dim: int = 4
    n_episodes: int = 256
    episode_length: int = 8
    n_grid: int = 8
    dt: float = 0.02
    substeps: int = 4
    stiffness: float = 40.0
    damping: float = 0.12
    mass: float = 1.0
    gravity: float = 0.0
    contact_radius: float = 0.12
    action_gain: float = 8.0
    gaussian_sigma: float = 1.75
    seed: int = 0
    val_fraction: float = 0.1
    pin_memory: bool = True
    n_stiffness_classes: int = 3
    chunk_length: int = 16
    chunk_stride: int = 1
    augment: bool = True
    crop_scale_min: float = 0.75
    color_jitter: float = 0.4
    blur_prob: float = 0.25
    blur_sigma_max: float = 2.0
    normalize_actions: bool = True
    prefetch_factor: int = 2

    def __post_init__(self) -> None:
        if self.spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3.")
        if self.dataset_path is None:
            min_action = 2 * self.spatial_dims
            if self.action_dim < min_action:
                raise ValueError(
                    f"action_dim={self.action_dim} is too small for spatial_dims="
                    f"{self.spatial_dims}; need at least {min_action} "
                    "(contact position + force)."
                )
        elif self.action_dim < 1:
            raise ValueError("action_dim must be >= 1.")
        if self.img_size < 8:
            raise ValueError("img_size must be >= 8.")
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1.")
        if self.n_episodes < 1:
            raise ValueError("n_episodes must be >= 1.")
        if self.episode_length < 2:
            raise ValueError("episode_length must be >= 2 to form (x_t, x_{t+1}) pairs.")
        if self.n_grid < 2:
            raise ValueError("n_grid must be >= 2.")
        if self.substeps < 1:
            raise ValueError("substeps must be >= 1.")
        if self.mass <= 0 or self.stiffness < 0:
            raise ValueError("mass must be > 0 and stiffness must be >= 0.")
        if not (0.0 <= self.damping < 1.0):
            raise ValueError("damping must lie in [0, 1).")
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError("val_fraction must lie in [0, 1).")
        if self.n_stiffness_classes < 1:
            raise ValueError("n_stiffness_classes must be >= 1.")
        if self.chunk_length < 2:
            raise ValueError("chunk_length must be >= 2.")
        if self.chunk_stride < 1:
            raise ValueError("chunk_stride must be >= 1.")
        if not (0.0 < self.crop_scale_min <= 1.0):
            raise ValueError("crop_scale_min must lie in (0, 1].")
        if self.color_jitter < 0.0:
            raise ValueError("color_jitter must be >= 0.")
        if not (0.0 <= self.blur_prob <= 1.0):
            raise ValueError("blur_prob must lie in [0, 1].")
        if self.blur_sigma_max <= 0.0:
            raise ValueError("blur_sigma_max must be > 0.")
        if self.prefetch_factor < 2:
            raise ValueError("prefetch_factor must be >= 2 (PyTorch DataLoader requirement).")
        n_particles = self.n_particles
        if n_particles > 4096:
            raise ValueError(
                f"n_grid ** spatial_dims = {n_particles} exceeds 4096; "
                "reduce n_grid for the in-process simulator."
            )

    @property
    def n_particles(self) -> int:
        return int(self.n_grid**self.spatial_dims)

    @property
    def pairs_per_episode(self) -> int:
        return self.episode_length - 1

    @property
    def internal_dt(self) -> float:
        return self.dt / float(self.substeps)


@dataclass
class ExperimentConfig:
    """Bundled model / training / data config for a single AC-JEPA run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        validate_experiment(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        """Construct from a nested dict (e.g. parsed YAML / JSON)."""
        return cls(
            model=ModelConfig(**payload.get("model", {})),
            training=TrainingConfig(**payload.get("training", {})),
            data=DataConfig(**payload.get("data", {})),
        )


def resolve_device(device: str = "auto") -> torch.device:
    """Select CUDA, Apple MPS, or CPU.

    Args:
        device: Explicit device string, or ``"auto"`` for a capability check.

    Returns:
        A concrete ``torch.device``.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_experiment(cfg: ExperimentConfig) -> None:
    """Raise if model and data configs disagree on tensor shapes."""
    mismatches: list[str] = []
    if cfg.model.in_channels != cfg.data.in_channels:
        mismatches.append(
            f"in_channels: model={cfg.model.in_channels} vs data={cfg.data.in_channels}"
        )
    if cfg.model.img_size != cfg.data.img_size:
        mismatches.append(
            f"img_size: model={cfg.model.img_size} vs data={cfg.data.img_size}"
        )
    if cfg.model.action_dim != cfg.data.action_dim:
        mismatches.append(
            f"action_dim: model={cfg.model.action_dim} vs data={cfg.data.action_dim}"
        )
    if mismatches:
        raise ValueError(
            "ExperimentConfig fields are inconsistent:\n  - " + "\n  - ".join(mismatches)
        )


def seed_everything(seed: int) -> None:
    """Seed numpy (if imported) and torch RNGs for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def with_updates(cfg: ExperimentConfig, **overrides: Any) -> ExperimentConfig:
    """Return a copy of ``cfg`` with selected top-level dataclass replacements."""
    return replace(cfg, **overrides)
