#!/usr/bin/env python3
"""Canonical legacy-validation evaluator matching the revised-paper zip script.

Differences from the zip `evaluate_completed_runs.py`, all documented:
- Default data root is `outputs/physics_mpm` (original dump), not a Linux replay.
- Adds per-horizon rank/top-1 at k=1,3,5,10 (paper Table 3 used pooled 1-10).
- Writes `results/final_checkpoint_results.csv`.
- `replay_provenance.json` is optional; local dump hashes are recorded.
- Loads checkpoints with `weights_only=False` so config dicts survive.

This script is ``legacy_v1``. Probes train on episodes 0:1800. Rollout MSE /
VN-MSE / ranks use episodes 1800:2000. Those episodes are legacy validation,
not a ``cvpr_v1`` test set. No training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rank_slice(rank: dict) -> dict:
    return {
        "participation_ratio": rank["participation_ratio"],
        "eig_frac_top1": rank["eig_frac_top1"],
        "n_vectors": rank["n_vectors"],
        "latent_dim": rank["latent_dim"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs" / "physics_mpm")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "final_evaluation")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "final_checkpoint_results.csv")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--directional",
        action="store_true",
        help="Run legacy-validation directional counterfactuals and return.",
    )
    parser.add_argument(
        "--future-ranking",
        action="store_true",
        help="Run legacy-validation future ranking and return.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional checkpoint names to evaluate. Default: all ten no-crop last.pt runs.",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if args.directional:
        from scripts.run_counterfactual_direction import main as directional_main

        directional_main()
        return
    if args.future_ranking:
        from scripts.run_future_ranking import main as ranking_main

        ranking_main()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    from src.config import ModelConfig
    from src.eval import contact_classes, effective_rank, linear_probe_over_horizon
    from src.eval_protocol import assert_checkpoint_eval_pairing, reporting_slice
    from src.models.ac_jepa import ACJEPA
    from src.splits import get_split

    assert_checkpoint_eval_pairing("legacy_v1", "legacy_v1")
    legacy = get_split("legacy_v1")
    report = reporting_slice(legacy)
    if report["is_test_set"] or report["role"] != "legacy_validation":
        raise RuntimeError("evaluate_final.py must not label legacy validation as a test set.")
    n_train = len(legacy.train_ids)
    if legacy.train_ids != tuple(range(n_train)) or legacy.val_ids != tuple(range(n_train, legacy.n_episodes)):
        raise RuntimeError("legacy_v1 ids are no longer the contiguous tail this script slices.")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    obs = np.load(args.data / "observations.npy", mmap_mode="r")
    raw = torch.from_numpy(np.load(args.data / "actions.npy"))
    mean = raw[:n_train].reshape(-1, 4).mean(0)
    std = raw[:n_train].reshape(-1, 4).std(0, unbiased=False).clamp_min(1e-6)
    actions = (raw - mean) / std
    label = {
        name: torch.from_numpy(np.load(args.data / f"{name}_labels.npy"))
        for name in ("geometry", "stiffness")
    }
    label["contact"] = contact_classes(raw[:, :10])
    paths = [
        ("vicreg_action_s42", "outputs/vicreg_nocrop/last.pt"),
        ("vicreg_action_s43", "outputs/vicreg_nocrop_seed43/last.pt"),
        ("vicreg_noaction_s42", "outputs/no_action/last.pt"),
        ("vicreg_noaction_s43", "outputs/vicreg_nocrop_noaction_seed43/last.pt"),
    ]
    paths += [
        (f"vicreg_{act}_s{seed}", f"results/reviewer_strengthening/05_more_seeds/vicreg_{act}_s{seed}/last.pt")
        for seed in (44, 45)
        for act in ("action", "noaction")
    ]
    paths += [
        (f"mse_action_s{seed}", f"results/reviewer_strengthening/06_corrected_objective/mse_action_s{seed}/last.pt")
        for seed in (42, 43)
    ]
    if args.only:
        wanted = set(args.only)
        unknown = wanted.difference(name for name, _rel in paths)
        if unknown:
            raise SystemExit(f"Unknown checkpoint names: {sorted(unknown)}")
        paths = [(name, rel) for name, rel in paths if name in wanted]
    permutations = []
    for t in range(15):
        rng = np.random.default_rng(1009 * t)
        perm = rng.permutation(200)
        stuck = np.where(perm == np.arange(200))[0]
        if len(stuck) == 1:
            j = 0 if stuck[0] != 0 else 1
            perm[stuck[0]], perm[j] = perm[j], perm[stuck[0]]
        elif len(stuck) > 1:
            perm[stuck] = np.roll(perm[stuck], 1)
        permutations.append(torch.from_numpy(perm))

    @torch.no_grad()
    def encode(model, frames, target=False):
        out = []
        for start in range(0, len(frames), 64):
            x = torch.from_numpy(np.array(frames[start : start + 64], copy=True)).float() / 255.0
            out.append(model.encode(x, target=target))
        return torch.cat(out)

    provenance_path = args.data / "replay_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        scope = "recorded-action replay; legacy_v1 validation episodes 1800:2000; not a test set"
    else:
        provenance = {
            "method": "Original dump observations.npy (not the Linux replay tensor).",
            "hashes": {
                path.name: _sha256(path)
                for path in sorted(args.data.glob("*.npy"))
            },
        }
        scope = "original dump observations; legacy_v1 validation episodes 1800:2000; not a test set"

    rows = []
    csv_rows = []
    for name, rel in paths:
        dest = args.out / f"{name}.json"
        if dest.exists() and not args.force:
            row = json.loads(dest.read_text(encoding="utf-8"))
            rows.append(row)
            continue
        t0 = time.time()
        ckpt = repo / rel
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        assert payload["epoch"] == 200 and cfg["data"]["crop_scale_min"] == 1.0
        model = ACJEPA(ModelConfig(**cfg["model"]))
        model.load_state_dict(payload["model"])
        model.eval()
        print("Evaluating", name, flush=True)
        z0 = encode(model, obs[:, 0])
        zv = encode(model, obs[n_train:, :15].reshape(-1, 3, 64, 64)).reshape(200, 15, 128)
        zt = encode(model, obs[n_train:, 1:].reshape(-1, 3, 64, 64), True).reshape(200, 15, 128)
        with torch.no_grad():
            pred = model.rollout(z0, actions[:, :10])
            pv = pred[n_train:]
            tv = zt[:, :10]
            err = (pv - tv).square().mean(-1)
            per = err.mean(0).tolist()
            var = tv.var(0, unbiased=False).mean(-1)
            vn = err.mean(0) / var
            true = model.predictor(zv.reshape(-1, 128), actions[n_train:].reshape(-1, 4)).reshape(200, 15, 128)
            alt_a = torch.stack([actions[n_train:][perm, t] for t, perm in enumerate(permutations)], 1)
            alt = model.predictor(zv.reshape(-1, 128), alt_a.reshape(-1, 4)).reshape(200, 15, 128)
            neut = model.predictor(zv.reshape(-1, 128), torch.zeros_like(alt_a).reshape(-1, 4)).reshape(200, 15, 128)
            err_true = (true - zt).square().mean(-1)
            err_alt = (alt - zt).square().mean(-1)
            pred_mse = (true - alt).square().mean(-1)
            err_neut = (neut - zt).square().mean(-1)
            cf = {}
            for subset, idx in (("all", list(range(15))), ("change", [3, 6, 9, 12]), ("t0", [0])):
                e = err_true[:, idx].double()
                s = err_alt[:, idx].double()
                cf[subset] = {
                    "n_episodes": 200,
                    "n_transitions": 200 * len(idx),
                    "true_mse": e.mean().item(),
                    "shuffled_mse": s.mean().item(),
                    "shuffle_error_change_pct": 100 * (s.mean() / e.mean() - 1).item(),
                    "pred_mse_mean": pred_mse[:, idx].mean().item(),
                    "pred_mse_median": pred_mse[:, idx].median().item(),
                    "neutral_error_change_pct": 100 * (err_neut[:, idx].double().mean() / e.mean() - 1).item(),
                }
            rank = {"pred": effective_rank(pv), "target": effective_rank(tv)}
            per_horizon_rank = {}
            for k in (1, 3, 5, 10):
                per_horizon_rank[str(k)] = {
                    "pred": _rank_slice(effective_rank(pv[:, k - 1])),
                    "target": _rank_slice(effective_rank(tv[:, k - 1])),
                }
        probes = {}
        for probe_name, n_classes in (("geometry", 2), ("stiffness", 3), ("contact", 4)):
            torch.manual_seed(0)
            probes[probe_name] = linear_probe_over_horizon(
                pred[:n_train],
                pred[n_train:],
                label[probe_name][:n_train],
                label[probe_name][n_train:],
                n_classes,
                epochs=80,
                lr=0.01,
                device=torch.device("cpu"),
            )
        np.savez_compressed(
            args.out / f"{name}_per_episode.npz",
            rollout_mse=err.numpy(),
            true_mse=err_true.numpy(),
            shuffle_mse=err_alt.numpy(),
            pred_delta=pred_mse.numpy(),
            neutral_mse=err_neut.numpy(),
        )
        row = {
            "name": name,
            "checkpoint": rel,
            "checkpoint_sha256": _sha256(ckpt),
            "split_version": "legacy_v1",
            "headline_role": "legacy_validation",
            "is_test_set": False,
            "epoch": 200,
            "config": cfg,
            "evaluation_scope": scope,
            "rollout_mse": per,
            "vn_mse": vn.tolist(),
            "target_variance": var.tolist(),
            "effective_rank": {"pred": rank["pred"], "target": rank["target"]},
            "per_horizon_rank": per_horizon_rank,
            "counterfactual": cf,
            "probes": probes,
            "seconds": time.time() - t0,
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
        dest.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        print(
            name,
            "MSE",
            [round(per[k], 6) for k in (0, 4, 9)],
            "rank@10",
            round(per_horizon_rank["10"]["pred"]["participation_ratio"], 3),
            "CF%",
            cf["change"]["shuffle_error_change_pct"],
            "seconds",
            round(row["seconds"], 1),
            flush=True,
        )

    (args.out / "completed_evaluation.json").write_text(
        json.dumps({"data_provenance": provenance, "runs": rows}, indent=2),
        encoding="utf-8",
    )

    def mean_acc(probe: dict) -> float:
        return float(np.mean(probe["val_acc"]))

    for row in rows:
        name = row["name"]
        objective = "mse" if name.startswith("mse_") else "vicreg"
        action = "noaction" not in name
        seed = int(name.rsplit("_s", 1)[1])
        phr = row["per_horizon_rank"]
        csv_rows.append(
            {
                "model": name,
                "objective": objective,
                "condition_on_action": action,
                "seed": seed,
                "MSE@1": row["rollout_mse"][0],
                "MSE@5": row["rollout_mse"][4],
                "MSE@10": row["rollout_mse"][9],
                "VN-MSE@1": row["vn_mse"][0],
                "VN-MSE@5": row["vn_mse"][4],
                "VN-MSE@10": row["vn_mse"][9],
                "pred_rank_pooled": row["effective_rank"]["pred"]["participation_ratio"],
                "target_rank_pooled": row["effective_rank"]["target"]["participation_ratio"],
                "pred_top1_pooled": row["effective_rank"]["pred"]["eig_frac_top1"],
                "target_top1_pooled": row["effective_rank"]["target"]["eig_frac_top1"],
                "pred_rank@1": phr["1"]["pred"]["participation_ratio"],
                "pred_rank@3": phr["3"]["pred"]["participation_ratio"],
                "pred_rank@5": phr["5"]["pred"]["participation_ratio"],
                "pred_rank@10": phr["10"]["pred"]["participation_ratio"],
                "target_rank@1": phr["1"]["target"]["participation_ratio"],
                "target_rank@3": phr["3"]["target"]["participation_ratio"],
                "target_rank@5": phr["5"]["target"]["participation_ratio"],
                "target_rank@10": phr["10"]["target"]["participation_ratio"],
                "pred_top1@1": phr["1"]["pred"]["eig_frac_top1"],
                "pred_top1@3": phr["3"]["pred"]["eig_frac_top1"],
                "pred_top1@5": phr["5"]["pred"]["eig_frac_top1"],
                "pred_top1@10": phr["10"]["pred"]["eig_frac_top1"],
                "target_top1@1": phr["1"]["target"]["eig_frac_top1"],
                "target_top1@3": phr["3"]["target"]["eig_frac_top1"],
                "target_top1@5": phr["5"]["target"]["eig_frac_top1"],
                "target_top1@10": phr["10"]["target"]["eig_frac_top1"],
                "geometry_probe": mean_acc(row["probes"]["geometry"]),
                "stiffness_probe": mean_acc(row["probes"]["stiffness"]),
                "contact_probe": mean_acc(row["probes"]["contact"]),
                "shuffle_change_pct": row["counterfactual"]["change"]["shuffle_error_change_pct"],
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "evaluation_scope": row["evaluation_scope"],
            }
        )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print("wrote", args.csv, flush=True)


if __name__ == "__main__":
    main()
