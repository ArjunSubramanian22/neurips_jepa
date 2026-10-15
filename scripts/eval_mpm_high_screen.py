#!/usr/bin/env python3
"""Validation-only force-reversal screen for one high-excitation checkpoint.

Uses the fresh-pool validation block only. Episodes at or after the test
start are refused. This is not a frozen test evaluation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
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
from src.registry import RunRecorder, to_jsonable, utc_now
from src.splits import fresh_pool_v2_split
from src.train import load_model_from_checkpoint

N_EPISODES = 24
TEST_START = 1620


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _as_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(frame), dtype=torch.float32, device=device)
    return tensor.unsqueeze(0) / 255.0


def _norm_action(vector: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    raw = torch.as_tensor(vector, dtype=torch.float32, device=device).view(1, -1)
    center = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, -1)
    scale = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, -1)
    return (raw - center) / scale


def _rssm_contrast(model, frame0, true_frame, reversed_frame, action, reversed_action, mean, std, device):
    """One-step RSSM contrast from one shared filtered posterior.

    The branch state is observe(h0, z0, action=None, frame 0). Both the
    recorded and reversed predictions call imagine on that same state.
    Both targets call observe on that same state with the forked next frame.
    Eval mode stores the stochastic mean. The cosine vector is concat(h, mean).
    encode(final frame) is not used: it would restart from h0.
    """
    frame_now = _as_frame(frame0, device)
    deter0, stochastic0 = model.initial_state(1, device)
    deter, stochastic, _feature = model.observe(deter0, stochastic0, None, frame_now)
    branch_deter = deter.detach().clone()
    branch_stochastic = stochastic.detach().clone()
    recorded = _norm_action(action, mean, std, device)
    reversed_norm = _norm_action(reversed_action, mean, std, device)

    def _branch() -> tuple[torch.Tensor, torch.Tensor]:
        return branch_deter.detach().clone(), branch_stochastic.detach().clone()

    deter_rec, stochastic_rec = _branch()
    _d, _s, pred_rec = model.imagine(deter_rec, stochastic_rec, recorded)
    deter_rev, stochastic_rev = _branch()
    _d, _s, pred_rev = model.imagine(deter_rev, stochastic_rev, reversed_norm)
    deter_tgt, stochastic_tgt = _branch()
    _d, _s, tgt_rec = model.observe(deter_tgt, stochastic_tgt, recorded, _as_frame(true_frame, device))
    deter_tf, stochastic_tf = _branch()
    _d, _s, tgt_rev = model.observe(deter_tf, stochastic_tf, reversed_norm, _as_frame(reversed_frame, device))
    return (
        (pred_rev - pred_rec)[0].detach().cpu().numpy(),
        (tgt_rev - tgt_rec)[0].detach().cpu().numpy(),
        float(torch.mean((pred_rec - tgt_rec) ** 2).item()),
    )


def validation_episode_ids(n_episodes: int) -> list[int]:
    """First ``n_episodes`` of the fresh-pool validation block. Never the test suffix."""
    split = fresh_pool_v2_split(1800)
    if int(n_episodes) < 1:
        raise RuntimeError("n_episodes must be positive.")
    ids = list(range(split.val[0], split.val[0] + int(n_episodes)))
    if ids[-1] >= split.test[0] or ids[0] < split.val[0] or ids[-1] >= TEST_START:
        raise RuntimeError("screen episodes are not inside the validation block.")
    return ids


def assert_no_test_episodes(episode_ids: list[int]) -> None:
    if any(int(episode) >= TEST_START for episode in episode_ids):
        raise RuntimeError("refusing test episodes 1620-1799.")


def run_recorded_screen(args, episode_ids: list[int], *, allow_test: bool = False, record: bool = True) -> dict:
    """Score one checkpoint. Validation refuses the test suffix.

    ``allow_test=True`` is only for scripts/eval_mpm_high_test_once.py after
    the protocol freeze and the lock check. The validation CLI never sets it.
    """
    from dataclasses import asdict, is_dataclass

    ids = [int(episode) for episode in episode_ids]
    if allow_test:
        if ids != list(range(TEST_START, 1800)):
            raise RuntimeError("held-out scoring must read exactly episodes 1620-1799.")
        role = "test"
        is_test = True
    else:
        assert_no_test_episodes(ids)
        split_check = fresh_pool_v2_split(1800)
        if any(episode < split_check.val[0] or episode >= split_check.test[0] for episode in ids):
            raise RuntimeError("screen episodes are not inside the validation block.")
        role = "validation"
        is_test = False
    spec = environment_from_dict(
        yaml.safe_load((ROOT / "configs/environments/mpm_high_excitation.yaml").read_text(encoding="utf-8"))
    )
    split = fresh_pool_v2_split(1800)
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
    model, _loaded = load_model_from_checkpoint(ROOT / args.checkpoint, device)
    model.eval()
    dump = np.load(ROOT / "outputs/physics_mpm_high/observations.npy", mmap_mode="r")
    rows = []
    deltas_pred = []
    deltas_tgt = []
    affected = []
    with torch.no_grad():
        for episode in ids:
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

            if type(model).__name__ == "CompactRSSM":
                d_pred, d_tgt, one_step = _rssm_contrast(
                    model, replay["frames"][0], true_frame, reversed_frame, action, reversed_action, mean, std, device
                )
            else:
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
                    "split_role": role,
                    "is_test_set": is_test,
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
    for index, row in enumerate(rows):
        row["cosine"] = None if not bool(ok[index]) else float(cosines[index])
        row["cosine_status"] = str(contrast["cosine_status"][index])
        row["pred_contrast_norm"] = float(contrast["pred_norm"][index])
        row["target_contrast_norm"] = float(contrast["target_norm"][index])
        ratio = float(contrast["magnitude_ratio"][index])
        row["response_ratio"] = None if not np.isfinite(ratio) else ratio
        projection = contrast["signed_projection"][index]
        projection_ok = str(contrast["projection_status"][index]) == "ok"
        row["signed_projection"] = None if not projection_ok else float(projection)
    family = "rssm" if type(model).__name__ == "CompactRSSM" else "acjepa"
    model_config = asdict(model.cfg) if is_dataclass(getattr(model, "cfg", None)) else None
    ratios = [row["response_ratio"] for row in rows if row["response_ratio"] is not None]
    summary = {
        "checkpoint": args.checkpoint,
        "environment_id": spec.environment_id,
        "environment_config": "configs/environments/mpm_high_excitation.yaml",
        "action_gain": float(spec.mpm.action_gain),
        "damping": float(spec.mpm.damping),
        "command_hold": int(spec.command_hold),
        "model_config": model_config,
        "seed": None if args.seed is None else int(args.seed),
        "split_version": "fresh_pool_v2",
        "split_role": role,
        "is_test_set": is_test,
        "n_episodes": len(rows),
        "episode_ids": ids,
        "episode_start": int(ids[0]),
        "episode_stop": int(ids[-1] + 1),
        "timestamp": utc_now(),
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
        "median_pred_contrast_norm": float(np.median(contrast["pred_norm"])),
        "median_target_contrast_norm": float(np.median(contrast["target_norm"])),
        "median_response_ratio": None if not ratios else float(np.median(ratios)),
        "representation_family": family,
        "rssm_state": None
        if family != "rssm"
        else {
            "space": "concat(deterministic h, stochastic mean)",
            "branch_state": "filtered posterior after frame 0, action None, shared by both forks",
            "prediction": "one-step prior mean from that posterior under the recorded or reversed action",
            "target": "one-step filtered posterior of the forked frame, continuing the same posterior",
            "sampling": "eval mode uses the mean; no stochastic sample is drawn",
            "not_used": "encode(final frame) restarts h0 and is not the target",
        },
        "metric_version": "mpm_high_h1_v2",
        "git_commit": _git_commit(),
        "checkpoint_sha256": _sha256(ROOT / args.checkpoint),
        "saw_test": is_test,
    }
    payload = to_jsonable({"summary": summary, "rows": rows})
    if not record:
        return payload
    config = {
        "experiment_id": args.experiment_id,
        "checkpoint": args.checkpoint,
        "n_episodes": len(ids),
        "split_version": "fresh_pool_v2",
        "is_test_set": is_test,
        "environment_config": "configs/environments/mpm_high_excitation.yaml",
        "metric_version": "mpm_high_h1_v2",
    }
    with RunRecorder(
        args.experiment_id,
        repo_root=ROOT,
        config=config,
        command="python3.13 scripts/eval_mpm_high_screen.py",
        seed=0 if args.seed is None else int(args.seed),
        model={"name": args.model_name, "checkpoint": args.checkpoint, "config": model_config},
        environment={
            "id": spec.environment_id,
            "dump": "outputs/physics_mpm_high",
            "config": "configs/environments/mpm_high_excitation.yaml",
            "action_gain": float(spec.mpm.action_gain),
            "damping": float(spec.mpm.damping),
            "command_hold": int(spec.command_hold),
        },
        dataset_split={
            "split_version": "fresh_pool_v2",
            "role": role,
            "episodes": ids,
            "is_test_set": is_test,
        },
        checkpoint={"path": args.checkpoint, "sha256": summary["checkpoint_sha256"]},
    ) as run:
        (run.run_dir / "screen.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        run.complete(payload["summary"])
    print(json.dumps(payload["summary"], indent=2))
    return payload


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validation force-reversal screen. Does not touch the test suffix.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    ids = validation_episode_ids(int(args.n_episodes))
    run_recorded_screen(args, ids, allow_test=False, record=True)


if __name__ == "__main__":
    main()
