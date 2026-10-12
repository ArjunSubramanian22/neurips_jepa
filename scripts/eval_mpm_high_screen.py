#!/usr/bin/env python3
"""Validation-only force-reversal screen for one high-excitation checkpoint.

Uses the fresh-pool validation block only. Episodes at or after the test
start are refused. This is not a frozen test evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import (
    MLSMPM2D,
    apply_action_frames,
    render_rgb,
    replay_episode,
)
from src.config import DataConfig
from src.dataset import DeformablePhysicsDataset
from src.environments import environment_from_dict
from src.metrics.cascade import observation_visibility, physical_response
from src.metrics.counterfactual_direction import directional_contrast
from src.registry import RunRecorder, to_jsonable
from src.splits import fresh_pool_v2_split
from src.train import load_model_from_checkpoint

EXPERIMENT_ID = "mpm_high_jepa_s42_val_screen_v1"
CHECKPOINT = "outputs/mpm_high_runs/jepa_s42/last.pt"
N_EPISODES = 24


def main() -> None:
    spec = environment_from_dict(
        yaml.safe_load((ROOT / "configs/environments/mpm_high_excitation.yaml").read_text(encoding="utf-8"))
    )
    split = fresh_pool_v2_split(1800)
    val_ids = list(range(split.val[0], split.val[0] + N_EPISODES))
    if val_ids[-1] >= split.test[0] or val_ids[0] < split.val[0]:
        raise RuntimeError("screen episodes are not inside the validation block.")
    cfg = DataConfig(
        dataset_path=str(ROOT / "outputs/physics_mpm_high"),
        img_size=64,
        augment=False,
        split_version="fresh_pool_v2",
        normalize_actions=True,
    )
    train = DeformablePhysicsDataset(cfg, split="train")
    mean = train.action_mean.numpy().astype(np.float32)
    std = train.action_std.numpy().astype(np.float32)
    device = torch.device("cpu")
    model, _loaded = load_model_from_checkpoint(ROOT / CHECKPOINT, device)
    model.eval()
    dump = np.load(ROOT / "outputs/physics_mpm_high/observations.npy", mmap_mode="r")
    rows = []
    deltas_pred = []
    deltas_tgt = []
    affected = []
    with torch.no_grad():
        for episode in val_ids:
            replay = replay_episode(
                episode,
                dump_seed=0,
                episode_length=16,
                mpm=spec.mpm,
                hold=int(spec.command_hold),
            )
            if not np.array_equal(replay["frames"], dump[episode]):
                raise RuntimeError(f"replay does not match the dump at episode {episode}.")
            action = replay["actions"][0].astype(np.float32)
            reversed_action = action.copy()
            reversed_action[2:] *= np.float32(-1.0)
            sim = MLSMPM2D(spec.mpm)
            sim.restore(replay["snapshots"][0])
            apply_action_frames(sim, reversed_action, 1)
            reversed_frame = render_rgb(sim.x, sim.v, spec.mpm.img_size, spec.mpm.render_sigma)
            true_frame = replay["frames"][1]
            image = observation_visibility(true_frame, reversed_frame)
            physical = physical_response(replay["snapshots"][1]["x"], sim.x, replay["snapshots"][1]["v"], sim.v)
            def _norm(vector: np.ndarray) -> torch.Tensor:
                raw = torch.as_tensor(vector, dtype=torch.float32, device=device).view(1, -1)
                return (raw - torch.as_tensor(mean, device=device)) / torch.as_tensor(std, device=device)

            def _encode(frame: np.ndarray, target: bool) -> torch.Tensor:
                tensor = torch.as_tensor(frame, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                return model.encode(tensor, target=target)

            z0 = _encode(replay["frames"][0], False)
            pred_true = model.predictor(z0, _norm(action))
            pred_rev = model.predictor(z0, _norm(reversed_action))
            tgt_true = _encode(true_frame, True)
            tgt_rev = _encode(reversed_frame, True)
            d_pred = (pred_rev - pred_true).squeeze(0).cpu().numpy()
            d_tgt = (tgt_rev - tgt_true).squeeze(0).cpu().numpy()
            one_step = float(torch.mean((pred_true - tgt_true) ** 2).item())
            deltas_pred.append(d_pred)
            deltas_tgt.append(d_tgt)
            affected.append(1.0 - float(image["fraction_identical"]))
            rows.append(
                {
                    "episode": int(episode),
                    "one_step_mse": one_step,
                    "observation_mse": image["mse"],
                    "fraction_identical": image["fraction_identical"],
                    "rms_pos": physical["rms_pos"],
                    "split_role": "validation",
                    "is_test_set": False,
                }
            )
    contrast = directional_contrast(
        np.stack(deltas_pred),
        np.stack(deltas_tgt),
        fraction_affected=np.asarray(affected),
    )
    ok = np.asarray(contrast["cosine_status"]) == "ok"
    cosines = np.asarray(contrast["cosine"], dtype=np.float64)
    defined = cosines[ok]
    summary = {
        "checkpoint": CHECKPOINT,
        "environment_id": spec.environment_id,
        "split_version": "fresh_pool_v2",
        "split_role": "validation",
        "is_test_set": False,
        "n_episodes": len(rows),
        "episode_start": int(val_ids[0]),
        "episode_stop": int(val_ids[-1] + 1),
        "test_start": int(split.test[0]),
        "horizon": 1,
        "rollout_mode": "open_loop",
        "intervention": "force_reversal",
        "one_step_mse_median": float(np.median([row["one_step_mse"] for row in rows])),
        "observation_mse_median": float(np.median([row["observation_mse"] for row in rows])),
        "fraction_identical_median": float(np.median([row["fraction_identical"] for row in rows])),
        "rms_pos_median": float(np.median([row["rms_pos"] for row in rows])),
        "n_cosine_defined": int(ok.sum()),
        "n_cosine_undefined": int((~ok).sum()),
        "cosine_median": None if defined.size == 0 else float(np.median(defined)),
        "saw_test": False,
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "checkpoint": CHECKPOINT,
        "n_episodes": N_EPISODES,
        "split_version": "fresh_pool_v2",
        "is_test_set": False,
    }
    with RunRecorder(
        EXPERIMENT_ID,
        repo_root=ROOT,
        config=config,
        command="python3.13 scripts/eval_mpm_high_screen.py",
        seed=42,
        model={"name": "vicreg_action", "checkpoint": CHECKPOINT},
        environment={"id": spec.environment_id, "dump": "outputs/physics_mpm_high"},
        dataset_split={
            "split_version": "fresh_pool_v2",
            "role": "validation",
            "episodes": [int(val_ids[0]), int(val_ids[-1] + 1)],
            "is_test_set": False,
        },
        checkpoint={"path": CHECKPOINT, "epoch": 200},
    ) as run:
        payload = {"summary": summary, "rows": rows}
        (run.run_dir / "screen.json").write_text(
            json.dumps(to_jsonable(payload), indent=2) + "\n",
            encoding="utf-8",
        )
        run.complete(summary)
    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
