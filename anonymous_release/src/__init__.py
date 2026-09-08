"""Action-Conditioned JEPA (AC-JEPA) for deformable object manipulation."""

from __future__ import annotations

from typing import Any

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig

__all__ = [
    "ACJEPA",
    "DataConfig",
    "DeformableObjectDataset",
    "DeformablePhysicsDataset",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "VICRegLoss",
    "build_dataloader",
    "build_dataset",
    "generate_dummy_batch",
    "vicreg_loss",
]

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DeformableObjectDataset": (".dataset", "DeformableObjectDataset"),
    "DeformablePhysicsDataset": (".dataset", "DeformablePhysicsDataset"),
    "build_dataloader": (".dataset", "build_dataloader"),
    "build_dataset": (".dataset", "build_dataset"),
    "generate_dummy_batch": (".dataset", "generate_dummy_batch"),
    "ACJEPA": (".models.ac_jepa", "ACJEPA"),
    "VICRegLoss": (".losses", "VICRegLoss"),
    "vicreg_loss": (".losses", "vicreg_loss"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose dataset / model symbols so module entrypoints stay clean."""
    if name in _LAZY_EXPORTS:
        module_name, attr = _LAZY_EXPORTS[name]
        from importlib import import_module

        module = import_module(module_name, package=__name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
