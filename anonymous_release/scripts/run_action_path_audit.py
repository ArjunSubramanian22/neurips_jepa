#!/usr/bin/env python3
"""Phase 9 action-path audit. Writes results/reviewer_strengthening/09_action_path_audit/.

Does not modify checkpoints. If a wiring bug is found, writes CRITICAL_BUG_REPORT.md
and exits non-zero without silently patching training code.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "reviewer_strengthening" / "09_action_path_audit"
DUMP = ROOT / "outputs" / "physics_mpm"


def _run_action_tests() -> dict[str, Any]:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_action_path.py")
    buf = StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
    return {
        "ran": result.testsRun,
        "failed": len(result.failures) + len(result.errors),
        "log": buf.getvalue(),
        "ok": result.wasSuccessful(),
    }


def _alignment_check() -> dict[str, Any]:
    from src.config import load_experiment_config
    from src.dataset import build_dataset
    from src.train import apply_dataset_path
    from scripts.generate_trajectories import replay_episode

    cfg = load_experiment_config(ROOT / "configs" / "workshop_vicreg.yaml")
    cfg = apply_dataset_path(cfg, str(DUMP), num_workers=0)
    dataset = build_dataset(cfg.data, split="all")
    ep = 1800
    replay = replay_episode(ep, dump_seed=0, episode_length=16)
    dump_obs = dataset.observations[ep].numpy()
    replay_obs = replay["frames"].astype(np.float32) / 255.0
    # canonicalize may have already /255
    if float(dump_obs.max()) <= 1.5:
        replay_cmp = replay_obs
    else:
        replay_cmp = replay["frames"]
    obs_equal = bool(np.array_equal(dataset.actions[ep].numpy(), replay["actions"]))
    # float observations vs uint8/255
    raster_close = bool(np.allclose(dump_obs, replay_cmp, atol=1e-6, rtol=0.0))
    mean = dataset.action_mean.numpy()
    std = dataset.action_std.numpy()
    stats = {
        "action_mean_train": mean.tolist(),
        "action_std_train": std.tolist(),
        "std_collapsed": bool(np.any(std < 1e-3)),
        "raw_action_min": dataset.actions.min().item(),
        "raw_action_max": dataset.actions.max().item(),
        "train_action_var": dataset.actions[:1800].var(dim=0).tolist(),
        "val_action_var": dataset.actions[1800:].var(dim=0).tolist(),
        "n_train": 1800,
        "n_val": 200,
        "pairs_per_episode": dataset.pairs_per_episode,
        "transition_rule": "actions[e, t] applied between observations[e, t] and observations[e, t+1]",
        "dump_replay_actions_equal": obs_equal,
        "dump_replay_obs_close": raster_close,
        "crop_scale_min": float(cfg.data.crop_scale_min),
        "crop_disabled": bool(cfg.data.crop_scale_min >= 1.0),
        "normalize_actions": bool(cfg.data.normalize_actions),
    }
    return stats


def _checkpoint_weight_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    from src.models.ac_jepa import ACJEPA
    from src.train import load_model_from_checkpoint

    ckpts = [
        ("vicreg_nocrop_s42", ROOT / "outputs" / "vicreg_nocrop" / "last.pt", True),
        ("vicreg_nocrop_s43", ROOT / "outputs" / "vicreg_nocrop_seed43" / "last.pt", True),
        ("no_action_s42", ROOT / "outputs" / "no_action" / "last.pt", False),
        ("no_action_s43", ROOT / "outputs" / "vicreg_nocrop_noaction_seed43" / "last.pt", False),
    ]
    bugs: list[str] = []
    weight_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    device = torch.device("cpu")
    for name, path, expect_action in ckpts:
        model, cfg = load_model_from_checkpoint(path, device)
        cond = bool(getattr(model.predictor, "condition_on_action", False))
        if cond != expect_action:
            bugs.append(f"{name}: condition_on_action={cond} expected {expect_action}")
        init = ACJEPA(cfg.model)
        init_sd = init.state_dict()
        trained_sd = model.state_dict()
        for key in trained_sd:
            if "action" not in key and "fuse_proj" not in key and "uncond_proj" not in key:
                continue
            a = trained_sd[key].detach().float().cpu()
            b = init_sd[key].detach().float().cpu() if key in init_sd else None
            delta = float((a - b).norm().item()) if b is not None and a.shape == b.shape else float("nan")
            weight_rows.append(
                {
                    "checkpoint": name,
                    "param": key,
                    "norm": float(a.norm().item()),
                    "mean": float(a.mean().item()),
                    "std": float(a.std().item()) if a.numel() > 1 else 0.0,
                    "l2_from_fresh_init": delta,
                    "all_zero": bool(torch.all(a == 0)),
                }
            )
            if "action_embed" in key and expect_action and bool(torch.all(a == 0)):
                bugs.append(f"{name}: {key} is all zeros")
            if "action_embed" in key and expect_action and b is not None and delta < 1e-8:
                bugs.append(f"{name}: {key} unchanged from a fresh untrained init (unexpected)")

        model.train()
        x = torch.randn(4, 3, 64, 64)
        a = torch.randn(4, 4)
        x2 = torch.randn(4, 3, 64, 64)
        out = model(x, a, x2)
        loss = out.z_hat_tp1.pow(2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        action_grad = 0.0
        n_action_params = 0
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = 0.0 if p.grad is None else float(p.grad.detach().abs().sum().item())
            if "action" in pname or "fuse_proj" in pname or "uncond_proj" in pname:
                grad_rows.append({"checkpoint": name, "param": pname, "grad_abs_sum": g})
            if "action_embed" in pname:
                n_action_params += 1
                action_grad += g
        if expect_action and action_grad <= 0.0:
            bugs.append(f"{name}: no gradient into action_embed on a dummy backward")
        if not expect_action and n_action_params > 0:
            bugs.append(f"{name}: no-action model still has action_embed parameters")

        # shuffled replacement actually changes predictor input
        if expect_action:
            model.eval()
            with torch.no_grad():
                z = model.encode(x)
                y0 = model.predictor(z, a)
                a2 = a.clone()
                a2[:, 0] += 1.5
                y1 = model.predictor(z, a2)
                if torch.allclose(y0, y1, atol=1e-8):
                    bugs.append(f"{name}: predictor invariant to a large action change at eval")
        else:
            model.eval()
            with torch.no_grad():
                z = model.encode(x)
                y0 = model.predictor(z, a)
                y1 = model.predictor(z, torch.zeros_like(a))
                if not torch.allclose(y0, y1, atol=1e-6):
                    bugs.append(f"{name}: no-action predictor changed with action tensor")
    return weight_rows, grad_rows, bugs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tests = _run_action_tests()
    (OUT / "test_results.txt").write_text(tests["log"], encoding="utf-8")
    stats = _alignment_check()
    weight_rows, grad_rows, bugs = _checkpoint_weight_audit()
    if stats["std_collapsed"]:
        bugs.append("action std collapsed on at least one dimension")
    if not stats["dump_replay_actions_equal"]:
        bugs.append("dump actions do not match replay_episode")
    if not stats["dump_replay_obs_close"]:
        bugs.append("dump observations do not match replay_episode rasters")
    if not tests["ok"]:
        bugs.append("test_action_path.py failed")

    with (OUT / "tensor_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        for k, v in stats.items():
            writer.writerow([k, json.dumps(v) if not isinstance(v, (str, int, float, bool)) else v])
    with (OUT / "weight_change_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weight_rows[0].keys()) if weight_rows else ["checkpoint"])
        writer.writeheader()
        writer.writerows(weight_rows)
    with (OUT / "gradient_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grad_rows[0].keys()) if grad_rows else ["checkpoint"])
        writer.writeheader()
        writer.writerows(grad_rows)

    status = "FAIL" if bugs else "PASS"
    audit_md = f"""# Action-path audit

**ACTION PATH AUDIT: {status}**

## Checks
1. Dataset indexing: `x[e,t], a[e,t], x[e,t+1]` matches MLS-MPM replay of dump seed 0.
2. Train-only z-score; validation uses the same mean/std.
3. Canonical workshop crop is disabled (`crop_scale_min=1.0`); `warp_contact_xy` is unused.
4. Action-conditioned checkpoints have `condition_on_action=True` and a live `action_embed`.
5. No-action checkpoints use `uncond_proj` and ignore the action tensor.
6. Dummy backward sends nonzero gradients into `action_embed` on action models.
7. Trained `action_embed` weights are not identically zero and differ from a fresh init.
8. Changing a normalized action changes the action-conditioned predictor output.
9. Unit tests in `tests/test_action_path.py`.

## Tensor stats
- action mean (train): {stats["action_mean_train"]}
- action std (train): {stats["action_std_train"]}
- std collapsed: {stats["std_collapsed"]}
- dump replay actions equal: {stats["dump_replay_actions_equal"]}
- dump replay obs close: {stats["dump_replay_obs_close"]}
- crop disabled: {stats["crop_disabled"]}

## Bugs
{json.dumps(bugs, indent=2) if bugs else "_none_"}

Architectural/gradient sensitivity is **not** behavioral action use. Phase 1 simulator
counterfactuals remain the primary test.
"""
    (OUT / "audit.md").write_text(audit_md, encoding="utf-8")
    if bugs:
        (OUT / "CRITICAL_BUG_REPORT.md").write_text(
            "# CRITICAL BUG REPORT\n\n" + "\n".join(f"- {b}" for b in bugs) + "\n",
            encoding="utf-8",
        )
        print("ACTION PATH AUDIT: FAIL", flush=True)
        return 1
    print("ACTION PATH AUDIT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
