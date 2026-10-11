"""Write the small/base/large width sweep.

Only ``embed_dim`` and ``predictor_embed_dim`` change. Depth, heads, MLP
ratio, action embedding, and the training recipe stay at the workshop
values. Base width 128 is the architecture already trained under
``vicreg_action``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "experiments"

SCALES = (
    {"name": "small", "width": 64},
    {"name": "base", "width": 128},
    {"name": "large", "width": 256},
)
FIXED = {
    "encoder_depth": 4,
    "predictor_depth": 4,
    "encoder_num_heads": 4,
    "predictor_num_heads": 4,
    "mlp_ratio": 2.0,
    "action_embed_dim": 64,
    "drop_path": 0.05,
}
BASE_CHECKPOINTS = {
    42: "outputs/vicreg_nocrop/last.pt",
    43: "outputs/vicreg_nocrop_seed43/last.pt",
    44: "results/reviewer_strengthening/05_more_seeds/vicreg_action_s44/last.pt",
    45: "results/reviewer_strengthening/05_more_seeds/vicreg_action_s45/last.pt",
    46: None,
}
SEEDS = (42, 43, 44, 45, 46)


def _training() -> dict[str, Any]:
    return {
        "batch_size": 128,
        "epochs": 200,
        "lr": 0.0002,
        "min_lr": 0.000001,
        "weight_decay": 0.05,
        "warmup_epochs": 20,
        "warmup_ratio": 0.1,
        "grad_clip": 1.0,
        "loss_type": "vicreg",
        "vicreg_sim_coeff": 25.0,
        "vicreg_std_coeff": 25.0,
        "vicreg_cov_coeff": 1.0,
        "amp": True,
        "seed": 42,
        "num_workers": 2,
        "log_interval": 20,
        "ckpt_interval": 25,
        "output_dir": "outputs",
        "device": "auto",
        "ema_sync_every": 1,
    }


def _data() -> dict[str, Any]:
    return {
        "split_version": "legacy_v1",
        "dataset_path": "outputs/physics_mpm",
        "spatial_dims": 2,
        "img_size": 64,
        "in_channels": 3,
        "action_dim": 4,
        "n_episodes": 2000,
        "episode_length": 16,
        "n_stiffness_classes": 3,
        "chunk_length": 16,
        "chunk_stride": 1,
        "val_fraction": 0.1,
        "augment": True,
        "crop_scale_min": 1.0,
        "normalize_actions": True,
        "pin_memory": True,
    }


def _model(width: int) -> dict[str, Any]:
    return {
        "in_channels": 3,
        "img_size": 64,
        "patch_size": 8,
        "embed_dim": int(width),
        "encoder_depth": FIXED["encoder_depth"],
        "encoder_num_heads": FIXED["encoder_num_heads"],
        "mlp_ratio": FIXED["mlp_ratio"],
        "action_dim": 4,
        "action_embed_dim": FIXED["action_embed_dim"],
        "predictor_embed_dim": int(width),
        "predictor_depth": FIXED["predictor_depth"],
        "predictor_num_heads": FIXED["predictor_num_heads"],
        "latent_dim": 0,
        "ema_momentum": 0.996,
        "dropout": 0.0,
        "drop_path": FIXED["drop_path"],
        "backbone": "vit",
        "fuse_mode": "concat",
        "condition_on_action": True,
    }


def experiment_payload(name: str, width: int) -> dict[str, Any]:
    return {
        "experiment_id": f"capacity_{name}_v1",
        "split_version": "legacy_v1",
        "selection_role": "val",
        "select_on_test": False,
        "is_test_set": False,
        "scale": name,
        "varied_factor": "width",
        "width": int(width),
        "fixed": copy.deepcopy(FIXED),
        "model": _model(width),
        "training": _training(),
        "data": _data(),
    }


def seed_registry() -> dict[str, Any]:
    configurations = []
    for scale in SCALES:
        checkpoints = {seed: None for seed in SEEDS}
        if scale["name"] == "base":
            checkpoints.update(BASE_CHECKPOINTS)
        configurations.append(
            {
                "id": f"capacity_{scale['name']}",
                "objective": "vicreg",
                "condition_on_action": True,
                "width": scale["width"],
                "checkpoints": checkpoints,
            }
        )
    return {
        "seed_registry_id": "capacity_sweep_v1",
        "seeds": list(SEEDS),
        "minimum_complete_seeds": 5,
        "split_version": "legacy_v1",
        "same_seeds_as": "headline_v1",
        "headline_role": "legacy_validation",
        "is_test_set": False,
        "varied_factor": "width",
        "fixed": copy.deepcopy(FIXED),
        "important_scale": "base",
        "configurations": configurations,
    }


def protocol() -> dict[str, Any]:
    return {
        "experiment_id": "capacity_sweep_v1",
        "split_version": "legacy_v1",
        "select_on_test": False,
        "is_test_set": False,
        "varied_factor": "width",
        "scales": [{"name": scale["name"], "width": scale["width"]} for scale in SCALES],
        "fixed": copy.deepcopy(FIXED),
        "training_protocol": "configs/workshop_vicreg.yaml",
        "important_scale": "base",
        "important_scale_seeds_required": [42, 43, 44, 45],
        "notes": (
            "Encoder and predictor width move together. Depth, head count, MLP ratio, "
            "action embedding, and the training recipe do not. Base width 128 is the "
            "existing vicreg_action checkpoint. Small and large are not trained."
        ),
    }


def write_configs(directory: Path | None = None) -> list[Path]:
    destination = directory or OUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for scale in SCALES:
        path = destination / f"capacity_{scale['name']}_v1.yaml"
        path.write_text(yaml.safe_dump(experiment_payload(scale["name"], scale["width"]), sort_keys=False), encoding="utf-8")
        written.append(path)
    protocol_path = destination / "capacity_sweep_v1.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol(), sort_keys=False), encoding="utf-8")
    written.append(protocol_path)
    seed_path = (destination.parent / "seeds" / "capacity_sweep_v1.yaml") if directory is None else destination / "capacity_sweep_seeds_v1.yaml"
    if directory is None:
        seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(yaml.safe_dump(seed_registry(), sort_keys=False), encoding="utf-8")
    written.append(seed_path)
    return written


def main() -> None:
    for path in write_configs():
        print(path)


if __name__ == "__main__":
    main()
