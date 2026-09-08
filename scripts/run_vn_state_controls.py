#!/usr/bin/env python3
"""Variance-normalized action contrast and matched state-perturbation control.

Uses original dump observations and the original shuffled-same-time forks
(seeds 42/43 VICReg action last.pt). No training.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.eval import effective_rank
from src.models.ac_jepa import ACJEPA

DUMP = ROOT / "outputs" / "physics_mpm"
CF = ROOT / "results" / "reviewer_strengthening" / "01_simulator_counterfactual"
OUT = ROOT / "results" / "audit_phase2"
NEAR_ZERO = 1e-8
CHECKPOINTS = (
    ("vicreg_nocrop_s42", ROOT / "outputs" / "vicreg_nocrop" / "last.pt"),
    ("vicreg_nocrop_s43", ROOT / "outputs" / "vicreg_nocrop_seed43" / "last.pt"),
)


def derange(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    stuck = np.where(perm == np.arange(n))[0]
    if stuck.size == 1:
        j = 0 if stuck[0] != 0 else 1
        perm[stuck[0]], perm[j] = perm[j], perm[stuck[0]]
    elif stuck.size > 1:
        perm[stuck] = np.roll(perm[stuck], 1)
    return perm


def load_model(path: Path, device: torch.device) -> ACJEPA:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch") or 0) != 200:
        raise RuntimeError(f"{path} epoch != 200")
    cfg = payload["config"]["model"]
    model = ACJEPA(ModelConfig(**cfg))
    model.load_state_dict(payload["model"])
    model.eval()
    return model.to(device)


@torch.no_grad()
def encode(model: ACJEPA, frames: np.ndarray, device: torch.device, *, target: bool) -> torch.Tensor:
    out = []
    for start in range(0, len(frames), 64):
        x = torch.from_numpy(np.array(frames[start : start + 64], copy=True)).float() / 255.0
        out.append(model.encode(x.to(device), target=target).cpu())
    return torch.cat(out, dim=0)


def summarize(vals: np.ndarray) -> dict[str, float]:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(2)
    obs = np.load(DUMP / "observations.npy", mmap_mode="r")
    raw = torch.from_numpy(np.load(DUMP / "actions.npy"))
    mean = raw[:1800].reshape(-1, 4).mean(0)
    std = raw[:1800].reshape(-1, 4).std(0, unbiased=False).clamp_min(1e-6)
    actions = (raw - mean) / std
    val_obs = np.array(obs[1800:2000])
    val_a = actions[1800:2000]
    perms = [torch.from_numpy(derange(200, np.random.default_rng(1009 * t))) for t in range(15)]
    alt_a = torch.stack([val_a[pr, t] for t, pr in enumerate(perms)], dim=1)

    phys = [
        r
        for r in csv.DictReader((CF / "physical_divergence_per_example.csv").open())
        if r["intervention"] == "shuffled_same_time" and r["horizon"] == "1"
    ]
    raster = {(int(r["episode"]), int(r["t"])): float(r["mse_full"]) for r in phys}
    tgt_csv = list(csv.DictReader((CF / "target_latent_divergence_per_example.csv").open()))
    pred_csv = list(csv.DictReader((CF / "predictor_divergence_per_example.csv").open()))

    report: dict[str, object] = {
        "n_episodes": 200,
        "n_transitions": 3000,
        "near_zero": NEAR_ZERO,
        "delta_direction": "isotropic Gaussian, seed 0, L2 matched to sqrt(d * D_tgt)",
        "checkpoints": {},
    }
    rows = []

    for name, path in CHECKPOINTS:
        print("controls", name, flush=True)
        model = load_model(path, device)
        d_dim = int(model.latent_dim)
        z_t = encode(model, val_obs[:, :15].reshape(-1, 3, 64, 64), device, target=False).reshape(200, 15, -1)
        z_tgt_true = encode(model, val_obs[:, 1:].reshape(-1, 3, 64, 64), device, target=True).reshape(200, 15, -1)
        z_flat = z_t.reshape(-1, d_dim)
        a_flat = val_a.reshape(-1, 4)
        alt_flat = alt_a.reshape(-1, 4)
        with torch.no_grad():
            pred_true = model.predictor(z_flat, a_flat)
            pred_alt = model.predictor(z_flat, alt_flat)
            d_pred = (pred_true - pred_alt).square().mean(dim=-1).cpu().numpy()
        tgt_lookup = {
            (int(r["episode"]), int(r["t"])): float(r["tgt_mse"])
            for r in tgt_csv
            if r["checkpoint"] == name and r["intervention"] == "shuffled_same_time" and r["horizon"] == "1"
        }
        pred_lookup = {
            (int(r["episode"]), int(r["t"])): float(r["pred_mse"])
            for r in pred_csv
            if r["checkpoint"] == name and r["intervention"] == "shuffled_same_time"
        }
        d_tgt = np.array([tgt_lookup[(1800 + i, t)] for i in range(200) for t in range(15)], dtype=np.float64)
        d_pred_archived = np.array(
            [pred_lookup[(1800 + i, t)] for i in range(200) for t in range(15)], dtype=np.float64
        )
        pred_var = float(pred_true.var(dim=0, unbiased=False).mean().item())
        tgt_var = float(z_tgt_true.reshape(-1, d_dim).var(dim=0, unbiased=False).mean().item())
        online_var = float(z_flat.var(dim=0, unbiased=False).mean().item())
        d_pred_n = d_pred / max(pred_var, 1e-12)
        d_tgt_n = d_tgt / max(tgt_var, 1e-12)

        with torch.no_grad():
            rng = torch.Generator().manual_seed(0)
            u = torch.randn(z_flat.shape, generator=rng)
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            delta_l2 = torch.from_numpy(np.sqrt(d_dim * np.maximum(d_tgt, 0.0))).float()
            pred_state = model.predictor(z_flat + u * delta_l2[:, None], a_flat)
            d_state = (pred_true - pred_state).square().mean(dim=-1).cpu().numpy()

            included = d_tgt > NEAR_ZERO
            med_l2 = float(np.median(np.sqrt(d_dim * d_tgt[included]))) if included.any() else 0.0
            pred_state_med = model.predictor(z_flat + u * med_l2, a_flat)
            d_state_med = (pred_true - pred_state_med).square().mean(dim=-1).cpu().numpy()

        mse_full = np.array([raster[(1800 + i, t)] for i in range(200) for t in range(15)])
        nonzero = mse_full[mse_full > 0]
        hi_cut = float(np.quantile(nonzero, 0.67)) if nonzero.size else 1.0
        high_vis = mse_full >= hi_cut

        def pack(mask: np.ndarray) -> dict[str, object]:
            dp, dt, ds, dsm, dpn, dtn = (
                d_pred[mask],
                d_tgt[mask],
                d_state[mask],
                d_state_med[mask],
                d_pred_n[mask],
                d_tgt_n[mask],
            )
            ratio_raw = np.divide(dp, dt, out=np.full_like(dp, np.nan), where=dt > NEAR_ZERO)
            ratio_n = np.divide(dpn, dtn, out=np.full_like(dpn, np.nan), where=dtn > NEAR_ZERO)
            state_over_action = np.divide(ds, dp, out=np.full_like(ds, np.nan), where=dp > 1e-16)
            return {
                "n": int(mask.sum()),
                "D_pred": summarize(dp),
                "D_tgt": summarize(dt),
                "D_pred_norm": summarize(dpn),
                "D_tgt_norm": summarize(dtn),
                "D_pred_over_D_tgt": summarize(ratio_raw),
                "D_pred_norm_over_D_tgt_norm": summarize(ratio_n),
                "D_state_per_example": summarize(ds),
                "D_state_median_matched": summarize(dsm),
                "D_state_over_D_action": summarize(state_over_action),
                "frac_Dpred_lt_Dtgt": float(np.mean(dp < dt)) if mask.any() else float("nan"),
                "frac_Dpred_norm_lt_Dtgt_norm": float(np.mean(dpn < dtn)) if mask.any() else float("nan"),
                "frac_Dstate_gt_Daction": float(np.mean(ds > dp)) if mask.any() else float("nan"),
            }

        recon_err = float(np.max(np.abs(d_pred - d_pred_archived)))
        payload = {
            "pred_mean_var": pred_var,
            "target_mean_var": tgt_var,
            "online_mean_var": online_var,
            "pred_rank_pooled_transitions": effective_rank(pred_true)["participation_ratio"],
            "target_rank_val_futures": effective_rank(z_tgt_true.reshape(-1, d_dim))["participation_ratio"],
            "max_abs_Dpred_recompute_vs_archive": recon_err,
            "median_matched_delta_l2": med_l2,
            "all": pack(np.ones(3000, dtype=bool)),
            "D_tgt_gt_1e-8": pack(included),
            "identical_raster": pack(mse_full == 0.0),
            "high_visibility": pack(high_vis),
        }
        report["checkpoints"][name] = payload
        for split, key in (
            ("all", "all"),
            ("dtgt_gt_1e-8", "D_tgt_gt_1e-8"),
            ("identical_raster", "identical_raster"),
            ("high_visibility", "high_visibility"),
        ):
            block = payload[key]
            rows.append(
                {
                    "checkpoint": name,
                    "split": split,
                    "n": block["n"],
                    "median_D_pred": block["D_pred"]["median"],
                    "median_D_tgt": block["D_tgt"]["median"],
                    "median_D_pred_norm": block["D_pred_norm"]["median"],
                    "median_D_tgt_norm": block["D_tgt_norm"]["median"],
                    "median_raw_ratio": block["D_pred_over_D_tgt"]["median"],
                    "median_norm_ratio": block["D_pred_norm_over_D_tgt_norm"]["median"],
                    "median_D_state": block["D_state_per_example"]["median"],
                    "median_D_state_medmatched": block["D_state_median_matched"]["median"],
                    "median_D_state_over_D_action": block["D_state_over_D_action"]["median"],
                    "frac_Dstate_gt_Daction": block["frac_Dstate_gt_Daction"],
                    "pred_mean_var": pred_var,
                    "target_mean_var": tgt_var,
                }
            )
        del model
        print(name, "pred_var", pred_var, "tgt_var", tgt_var, "recon_max", recon_err, flush=True)

    (OUT / "vn_state_perturbation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "vn_state_perturbation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT / "vn_state_perturbation.csv", flush=True)


if __name__ == "__main__":
    main()
