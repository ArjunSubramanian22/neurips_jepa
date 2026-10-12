"""Score arm checkpoints with the shared fork audit. Does not train."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import build_dataset
from src.envs.arm_audit import audit_task
from src.registry import to_jsonable
from src.train import load_model_from_checkpoint

TASKS = ("push_object", "fold_clear", "cover_post")
CHECKPOINTS = {
    "jepa": ROOT / "outputs" / "mujoco_arm_runs" / "jepa" / "last.pt",
    "rssm": ROOT / "outputs" / "mujoco_arm_runs" / "rssm" / "last.pt",
    "latent_tf": ROOT / "outputs" / "mujoco_arm_runs" / "latent_tf" / "last.pt",
}


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _represent(model, mean: torch.Tensor, std: torch.Tensor):
    loss_type = type(model).__name__

    def call(action: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        device = mean.device
        video = torch.as_tensor(frames, dtype=torch.float32, device=device).unsqueeze(0)
        horizon = int(video.shape[1] - 1)
        raw = torch.as_tensor(action, dtype=torch.float32, device=device).view(1, -1)
        norm = (raw - mean) / std
        acts = norm.view(1, 1, -1).expand(1, horizon, -1).contiguous()
        with torch.no_grad():
            if loss_type == "CompactRSSM":
                predicted, target = model.paired_open_loop(video, acts, 0, horizon)
                pred = predicted[0, -1]
                tgt = target[0, -1]
            elif loss_type == "CausalLatentDynamics":
                latent = model.encode(video[:, 0])
                pred = model.rollout(latent, acts)[0, -1]
                tgt = model.encode(video[:, -1], target=True)[0]
            else:
                latent = model.encode(video[:, 0])
                pred = model.rollout(latent, acts)[0, -1]
                tgt = model.encode(video[:, -1], target=True)[0]
        return pred.detach().cpu().numpy(), tgt.detach().cpu().numpy()

    return call


def _loss_row(history_path: Path) -> dict:
    if not history_path.is_file():
        return {"train_loss": None, "val_loss": None}
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload.get("history", payload) if isinstance(payload, dict) else payload
    last = history[-1] if history else {}
    return {
        "epoch": last.get("epoch"),
        "train_loss": _finite(last.get("train_loss", last.get("loss"))),
        "val_loss": _finite(last.get("val_loss")),
    }


def main() -> None:
    device = torch.device("cpu")
    rows = []
    for name, path in CHECKPOINTS.items():
        if not path.is_file():
            rows.append({"model": name, "status": "missing_checkpoint", "path": str(path)})
            continue
        model, cfg = load_model_from_checkpoint(path, device)
        dataset = build_dataset(cfg.data, split="train")
        mean = dataset.action_mean.to(device).float()
        std = dataset.action_std.to(device).float()
        represent = _represent(model, mean, std)
        tasks = {task: audit_task(task, seed=0, represent=represent) for task in TASKS}
        rows.append(
            {
                "model": name,
                "architecture": type(model).__name__,
                "checkpoint": str(path.relative_to(ROOT)),
                "status": "measured",
                "loss": _loss_row(path.parent / "history.json"),
                "tasks": tasks,
            }
        )
    out = ROOT / "experiments" / "runs" / "mujoco_arm_v1" / "model_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable({"models": rows, "is_test_set": False}), indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
