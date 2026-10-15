#!/usr/bin/env python3
"""Negative controls for the high-excitation RSSM horizon-1 screen.

Validation episodes 1440-1463 only. Does not read episodes 1620-1799.
The one-step deterministic state is a shared function of the branch state
and the action, so concat(h, z) cosine can be inflated. This audit measures
that coupling instead of assuming the published cosine is independent.
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

from scripts.generate_trajectories import MLSMPM2D, apply_action_frames, render_rgb, replay_episode
from src.config import DataConfig
from src.dataset import DeformablePhysicsDataset
from src.environments import environment_from_dict
from src.metrics.cascade import observation_visibility, physical_response
from src.metrics.counterfactual_direction import NORM_TAU, directional_contrast
from src.registry import utc_now
from src.splits import fresh_pool_v2_split
from src.train import load_model_from_checkpoint

OUT = ROOT / "experiments" / "runs" / "rssm_h1_audit"
TEST_START = 1620
EPISODES = list(range(1440, 1464))
SEEDS = (42, 43, 44, 45, 46)


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(frame), dtype=torch.float32, device=device).unsqueeze(0) / 255.0


def _action(vector: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    raw = torch.as_tensor(vector, dtype=torch.float32, device=device).view(1, -1)
    center = torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, -1)
    scale = torch.as_tensor(std, dtype=torch.float32, device=device).view(1, -1)
    return (raw - center) / scale


def _posterior(model, frame, device):
    deter0, stochastic0 = model.initial_state(1, device)
    deter, stochastic, _feature = model.observe(deter0, stochastic0, None, _frame(frame, device))
    return deter.detach().clone(), stochastic.detach().clone()


def _pair(model, branch_h, branch_z, action_a, action_b, frame_a, frame_b, device):
    """One-step prior and posterior from identical branch-state copies."""
    copies = [(branch_h.detach().clone(), branch_z.detach().clone()) for _ in range(4)]
    _h, _z, pred_a = model.imagine(*copies[0], action_a)
    _h, _z, pred_b = model.imagine(*copies[1], action_b)
    h_ta, z_ta, tgt_a = model.observe(*copies[2], action_a, _frame(frame_a, device))
    h_tb, z_tb, tgt_b = model.observe(*copies[3], action_b, _frame(frame_b, device))
    pred_h_a, pred_z_a = model.split_feature(pred_a)
    pred_h_b, pred_z_b = model.split_feature(pred_b)
    return {
        "pred": (pred_b - pred_a)[0].detach().cpu().numpy(),
        "tgt": (tgt_b - tgt_a)[0].detach().cpu().numpy(),
        "pred_h": (pred_h_b - pred_h_a)[0].detach().cpu().numpy(),
        "tgt_h": (h_tb - h_ta)[0].detach().cpu().numpy(),
        "pred_z": (pred_z_b - pred_z_a)[0].detach().cpu().numpy(),
        "tgt_z": (z_tb - z_ta)[0].detach().cpu().numpy(),
        "h_equal_a": bool(torch.equal(pred_h_a, h_ta)),
        "h_equal_b": bool(torch.equal(pred_h_b, h_tb)),
        "branch_copies_equal": bool(torch.equal(copies[0][0], copies[1][0]) and torch.equal(copies[0][1], copies[1][1])),
        "prior_vs_posterior": float(torch.mean((pred_z_a - z_ta) ** 2).item()),
    }


def _summary(pred: np.ndarray, target: np.ndarray) -> dict:
    contrast = directional_contrast(pred, target)
    ok = np.asarray(contrast["cosine_status"]) == "ok"
    cosines = np.asarray(contrast["cosine"], dtype=np.float64)
    defined = cosines[ok]
    ratios = np.asarray(contrast["magnitude_ratio"], dtype=np.float64)
    finite_ratios = ratios[np.isfinite(ratios)]
    return {
        "n": int(pred.shape[0]),
        "n_defined_cosine": int(ok.sum()),
        "median_cosine": None if defined.size == 0 else float(np.median(defined)),
        "median_pred_norm": float(np.median(contrast["pred_norm"])),
        "median_target_norm": float(np.median(contrast["target_norm"])),
        "median_response_ratio": None if finite_ratios.size == 0 else float(np.median(finite_ratios)),
        "norm_tau": NORM_TAU,
    }


def _physical_forks(spec) -> list[dict]:
    rows = []
    dump = np.load(ROOT / "outputs/physics_mpm_high/observations.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    for index, episode in enumerate(EPISODES):
        if episode >= TEST_START:
            raise RuntimeError("audit refused a test episode.")
        replay = replay_episode(episode, dump_seed=0, episode_length=16, mpm=spec.mpm, hold=int(spec.command_hold))
        if not np.array_equal(replay["frames"], dump[episode]):
            raise RuntimeError(f"replay does not match the dump at episode {episode}.")
        recorded = replay["actions"][0].astype(np.float32)
        reversed_action = recorded.copy()
        reversed_action[2:] *= np.float32(-1.0)
        sim = MLSMPM2D(spec.mpm)
        sim.restore(replay["snapshots"][0])
        apply_action_frames(sim, reversed_action, 1)
        random_action = rng.normal(size=recorded.shape).astype(np.float32)
        random_sim = MLSMPM2D(spec.mpm)
        random_sim.restore(replay["snapshots"][0])
        apply_action_frames(random_sim, random_action, 1)
        donor = replay  # placeholder replaced after the loop
        rows.append(
            {
                "episode": int(episode),
                "frame0": replay["frames"][0],
                "recorded_frame": replay["frames"][1],
                "reversed_frame": render_rgb(sim.x, sim.v, spec.mpm.img_size, spec.mpm.render_sigma),
                "random_frame": render_rgb(random_sim.x, random_sim.v, spec.mpm.img_size, spec.mpm.render_sigma),
                "recorded_action": recorded,
                "reversed_action": reversed_action,
                "random_action": random_action,
                "recorded_state": replay["snapshots"][1],
                "reversed_pos": np.array(sim.x, copy=True),
                "reversed_vel": np.array(sim.v, copy=True),
                "random_pos": np.array(random_sim.x, copy=True),
                "random_vel": np.array(random_sim.v, copy=True),
            }
        )
        del donor
    for index, row in enumerate(rows):
        donor = rows[(index + 1) % len(rows)]
        row["shuffled_action"] = donor["recorded_action"].copy()
    return rows


def _score_seed(model, forks, mean, std, device) -> dict:
    bundles = {name: [] for name in ("force", "same", "shuffle", "random")}
    shared_h = []
    copies_ok = []
    prior_gap = []
    physical = []
    for row in forks:
        branch_h, branch_z = _posterior(model, row["frame0"], device)
        recorded = _action(row["recorded_action"], mean, std, device)
        reversed_action = _action(row["reversed_action"], mean, std, device)
        shuffled = _action(row["shuffled_action"], mean, std, device)
        random_action = _action(row["random_action"], mean, std, device)
        force = _pair(
            model, branch_h, branch_z, recorded, reversed_action,
            row["recorded_frame"], row["reversed_frame"], device,
        )
        same = _pair(
            model, branch_h, branch_z, recorded, recorded,
            row["recorded_frame"], row["recorded_frame"], device,
        )
        shuffle = _pair(
            model, branch_h, branch_z, recorded, shuffled,
            row["recorded_frame"], row["recorded_frame"], device,
        )
        for key in ("tgt", "tgt_h", "tgt_z"):
            shuffle[key] = force[key]
        random = _pair(
            model, branch_h, branch_z, recorded, random_action,
            row["recorded_frame"], row["random_frame"], device,
        )
        bundles["force"].append(force)
        bundles["same"].append(same)
        bundles["shuffle"].append(shuffle)
        bundles["random"].append(random)
        shared_h.append(bool(force["h_equal_a"] and force["h_equal_b"]))
        copies_ok.append(bool(force["branch_copies_equal"]))
        prior_gap.append(float(force["prior_vs_posterior"]))
        image = observation_visibility(row["recorded_frame"], row["reversed_frame"])
        state = physical_response(
            row["recorded_state"]["x"], row["reversed_pos"], row["recorded_state"]["v"], row["reversed_vel"]
        )
        physical.append(
            {
                "episode": row["episode"],
                "action_l2": float(np.linalg.norm(row["reversed_action"] - row["recorded_action"])),
                "rms_pos": state["rms_pos"],
                "rms_vel": state["rms_vel"],
                "raster_mse": image["mse"],
                "fraction_identical": image["fraction_identical"],
            }
        )

    def stack(name: str, pred_key: str, tgt_key: str) -> dict:
        pred = np.stack([item[pred_key] for item in bundles[name]])
        target = np.stack([item[tgt_key] for item in bundles[name]])
        return _summary(pred, target)

    components = {}
    for label, pred_key, tgt_key in (
        ("deterministic_h", "pred_h", "tgt_h"),
        ("stochastic_mean_z", "pred_z", "tgt_z"),
        ("concat_h_z", "pred", "tgt"),
    ):
        components[label] = stack("force", pred_key, tgt_key)
    return {
        "components": components,
        "same_action_concat": stack("same", "pred", "tgt"),
        "same_action_z": stack("same", "pred_z", "tgt_z"),
        "shuffle_vs_original_target_concat": stack("shuffle", "pred", "tgt"),
        "shuffle_vs_original_target_z": stack("shuffle", "pred_z", "tgt_z"),
        "random_action_matched_target_z": stack("random", "pred_z", "tgt_z"),
        "fraction_h_bitwise_shared": float(np.mean(shared_h)),
        "fraction_branch_copies_equal": float(np.mean(copies_ok)),
        "median_recorded_prior_posterior_z_mse": float(np.median(prior_gap)),
        "physical": {
            "median_action_l2": float(np.median([row["action_l2"] for row in physical])),
            "min_action_l2": float(np.min([row["action_l2"] for row in physical])),
            "median_rms_pos": float(np.median([row["rms_pos"] for row in physical])),
            "min_rms_pos": float(np.min([row["rms_pos"] for row in physical])),
            "median_raster_mse": float(np.median([row["raster_mse"] for row in physical])),
            "median_fraction_identical": float(np.median([row["fraction_identical"] for row in physical])),
        },
    }


def _verdict(seeds: dict) -> dict:
    first = next(iter(seeds.values()))
    h_shared = all(row["fraction_h_bitwise_shared"] == 1.0 for row in seeds.values())
    copies = all(row["fraction_branch_copies_equal"] == 1.0 for row in seeds.values())
    same_pred = max(row["same_action_z"]["median_pred_norm"] for row in seeds.values())
    z_cosines = [row["components"]["stochastic_mean_z"]["median_cosine"] for row in seeds.values()]
    concat_cosines = [row["components"]["concat_h_z"]["median_cosine"] for row in seeds.values()]
    shuffle_z = [row["shuffle_vs_original_target_z"]["median_cosine"] for row in seeds.values()]
    leakage = bool(h_shared)
    return {
        "concat_h_is_algebraically_shared_at_h1": leakage,
        "branch_copies_match": copies,
        "same_action_z_pred_norm_max": same_pred,
        "concat_cosine_medians": concat_cosines,
        "z_cosine_medians": z_cosines,
        "shuffle_z_cosine_medians": shuffle_z,
        "primary_rssm_vector_after_audit": "stochastic_mean_z",
        "reason": (
            "One-step observe and imagine apply the same GRU update before the "
            "posterior sees the frame. concat(h, z) therefore shares delta_h exactly. "
            "The corrected predictor contrast is the stochastic mean only."
            if leakage
            else "No algebraic sharing of the deterministic state was detected."
        ),
        "continue_after_correction": True,
        "reference_seed_physical": first["physical"],
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# RSSM horizon-1 audit",
        "",
        "Validation episodes 1440-1463. Time 0. Force reversal unless a control says otherwise.",
        "Test episodes 1620-1799 were not read.",
        "",
        payload["verdict"]["reason"],
        "",
        f"Primary vector after this audit: `{payload['verdict']['primary_rssm_vector_after_audit']}`.",
        "",
    ]
    for seed, row in payload["seeds"].items():
        lines.append(f"## seed {seed}")
        lines.append("")
        for name, metrics in row["components"].items():
            lines.append(
                f"- {name}: cosine {metrics['median_cosine']} "
                f"pred {metrics['median_pred_norm']:.6g} "
                f"target {metrics['median_target_norm']:.6g} "
                f"ratio {metrics['median_response_ratio']}"
            )
        lines.append(
            f"- same-action z pred norm {row['same_action_z']['median_pred_norm']:.6g}; "
            f"target norm {row['same_action_z']['median_target_norm']:.6g}"
        )
        lines.append(
            f"- shuffle vs original target, z cosine {row['shuffle_vs_original_target_z']['median_cosine']}"
        )
        lines.append(
            f"- random action with its own physical target, z cosine {row['random_action_matched_target_z']['median_cosine']}"
        )
        lines.append(
            f"- bitwise shared h {row['fraction_h_bitwise_shared']}; "
            f"branch copies equal {row['fraction_branch_copies_equal']}; "
            f"recorded prior/posterior z MSE {row['median_recorded_prior_posterior_z_mse']:.6g}"
        )
        lines.append("")
    physical = payload["verdict"]["reference_seed_physical"]
    lines.extend(
        [
            "## Physical fork",
            "",
            f"- median action L2 {physical['median_action_l2']:.6g} (min {physical['min_action_l2']:.6g})",
            f"- median position RMS {physical['median_rms_pos']:.6g} (min {physical['min_rms_pos']:.6g})",
            f"- median raster MSE {physical['median_raster_mse']:.6g}",
            f"- median identical-raster fraction {physical['median_fraction_identical']:.6g}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if any(episode >= TEST_START for episode in EPISODES):
        raise SystemExit("refusing test episodes.")
    spec = environment_from_dict(
        yaml.safe_load((ROOT / "configs/environments/mpm_high_excitation.yaml").read_text(encoding="utf-8"))
    )
    split = fresh_pool_v2_split(1800)
    if EPISODES[0] < split.val[0] or EPISODES[-1] >= split.test[0]:
        raise SystemExit("audit episodes are outside validation.")
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
    forks = _physical_forks(spec)
    seeds = {}
    checkpoint_hashes = {}
    for seed in SEEDS:
        path = ROOT / f"outputs/mpm_high_runs/rssm_s{seed}/last.pt"
        model, _loaded = load_model_from_checkpoint(path, device)
        model.eval()
        with torch.no_grad():
            seeds[str(seed)] = _score_seed(model, forks, mean, std, device)
        checkpoint_hashes[str(seed)] = _sha256(path)
        del model
    payload = {
        "metric_version": "rssm_h1_audit_v1",
        "timestamp": utc_now(),
        "git_commit": _git(),
        "environment_config": "configs/environments/mpm_high_excitation.yaml",
        "action_gain": float(spec.mpm.action_gain),
        "damping": float(spec.mpm.damping),
        "command_hold": int(spec.command_hold),
        "split": "validation",
        "episode_ids": EPISODES,
        "branch_time": 0,
        "horizon": 1,
        "intervention": "force_reversal",
        "checkpoint_sha256": checkpoint_hashes,
        "test_episodes_read": False,
        "seeds": seeds,
        "verdict": _verdict(seeds),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUT / "audit.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], indent=2))


if __name__ == "__main__":
    main()
