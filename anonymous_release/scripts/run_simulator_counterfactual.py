#!/usr/bin/env python3
"""Ground-truth MLS-MPM counterfactuals vs frozen JEPA (Phase 1).

Writes only under results/reviewer_strengthening/01_simulator_counterfactual/.
Does not modify checkpoints or the physics dump.

Protocol: results/reviewer_strengthening/01_simulator_counterfactual/protocol.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import (  # noqa: E402
    MLSMPM2D,
    apply_action_frames,
    render_rgb,
    replay_episode,
)
from src.train import load_model_from_checkpoint  # noqa: E402

OUT = ROOT / "results" / "reviewer_strengthening" / "01_simulator_counterfactual"
HOLD = 3
HORIZONS = (1, 3, 5)
CHANGE_T = (3, 6, 9, 12)
SHUFFLE_SEED = 0
NEAR_ZERO = 1e-8
PRIMARY = ("true", "shuffled_same_time", "neutral")
SECONDARY = ("force_reversed", "force_rotated", "force_scaled", "contact_perturbed")
ALL_KINDS = PRIMARY + SECONDARY
CHECKPOINTS = (
    ("vicreg_nocrop_s42", ROOT / "outputs" / "vicreg_nocrop" / "last.pt"),
    ("vicreg_nocrop_s43", ROOT / "outputs" / "vicreg_nocrop_seed43" / "last.pt"),
)


def _command_for_step(k: int, t: int, alt: np.ndarray, recorded: np.ndarray) -> np.ndarray:
    remaining_hold = HOLD - (int(t) % HOLD)
    if k <= remaining_hold:
        return np.asarray(alt, dtype=np.float32)
    t_cmd = int(t) + (k - 1)
    if t_cmd < recorded.shape[0]:
        return recorded[t_cmd]
    return recorded[-1]


def roll_horizons(
    snap0: dict[str, Any],
    t: int,
    alt: np.ndarray,
    recorded: np.ndarray,
    img_size: int,
    render_sigma: float,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[int, tuple[dict[str, Any], np.ndarray]]:
    sim = MLSMPM2D()
    sim.restore(snap0)
    out: dict[int, tuple[dict[str, Any], np.ndarray]] = {}
    max_h = max(horizons)
    for k in range(1, max_h + 1):
        if t + k >= 16:
            break
        apply_action_frames(sim, _command_for_step(k, t, alt, recorded), 1)
        if k in horizons:
            out[k] = (
                sim.snapshot(),
                render_rgb(sim.x, sim.v, img_size, render_sigma),
            )
    return out


def phys_metrics(true_snap: dict, alt_snap: dict, p_mass: float) -> dict[str, float]:
    dx = alt_snap["x"] - true_snap["x"]
    dv = alt_snap["v"] - true_snap["v"]
    dist = np.linalg.norm(dx, axis=-1)
    speed_d = np.linalg.norm(dv, axis=-1)
    ke_t = 0.5 * p_mass * float(np.sum(true_snap["v"] ** 2))
    ke_a = 0.5 * p_mass * float(np.sum(alt_snap["v"] ** 2))
    com_t = true_snap["x"].mean(axis=0)
    com_a = alt_snap["x"].mean(axis=0)
    return {
        "rms_pos": float(np.sqrt(np.mean(dist ** 2))),
        "mean_pos": float(np.mean(dist)),
        "max_pos": float(np.max(dist)),
        "rms_vel": float(np.sqrt(np.mean(speed_d ** 2))),
        "d_ke": float(ke_a - ke_t),
        "com_dist": float(np.linalg.norm(com_a - com_t)),
    }


def raster_mse(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    fa = a.astype(np.float32) / 255.0
    fb = b.astype(np.float32) / 255.0
    err = (fa - fb) ** 2
    return {
        "mse_density": float(err[0].mean()),
        "mse_velocity": float(err[1:].mean()),
        "mse_full": float(err.mean()),
    }


def uint8_to_nchw(frames: np.ndarray) -> Tensor:
    ten = torch.from_numpy(np.asarray(frames, dtype=np.float32) / 255.0)
    if ten.ndim == 3:
        ten = ten.unsqueeze(0)
    return ten


def derange(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    stuck = np.where(perm == np.arange(n))[0]
    if stuck.size == 1:
        j = 0 if stuck[0] != 0 else 1
        perm[stuck[0]], perm[j] = perm[j], perm[stuck[0]]
    elif stuck.size > 1:
        perm[stuck] = np.roll(perm[stuck], 1)
    return perm


def cosine_mse(a: Tensor, b: Tensor) -> tuple[float, float]:
    mse = float((a - b).pow(2).mean().item())
    an = torch.nn.functional.normalize(a.flatten(1), dim=-1)
    bn = torch.nn.functional.normalize(b.flatten(1), dim=-1)
    cos = float((an * bn).sum(dim=-1).mean().item())
    return mse, 1.0 - cos


def make_alt(kind: str, action: np.ndarray, *, radius: float) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).copy()
    if kind == "true":
        return a
    if kind == "force_reversed":
        a[2:4] *= -1.0
        return a
    if kind == "force_rotated":
        fx, fy = float(a[2]), float(a[3])
        a[2], a[3] = -fy, fx
        return a
    if kind == "force_scaled":
        a[2:4] *= 1.5
        return a
    if kind == "contact_perturbed":
        a[0] = float(np.clip(a[0] + radius, 0.08, 0.92))
        return a
    raise KeyError(kind)


@torch.no_grad()
def encode_many(model, frames: np.ndarray, device: torch.device, *, target: bool, batch: int) -> Tensor:
    x = uint8_to_nchw(frames)
    chunks = []
    for start in range(0, x.shape[0], batch):
        sl = x[start : start + batch].to(device)
        chunks.append(model.encode(sl, target=target).cpu())
    return torch.cat(chunks, dim=0)


def _median(vals: list[float]) -> float | None:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return None
    return float(np.median(finite))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="outputs/physics_mpm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-val-episodes", type=int, default=200)
    parser.add_argument("--encode-batch", type=int, default=32)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    dump = Path(args.dataset_path)
    raw_actions = np.load(dump / "actions.npy")
    n_episodes = int(raw_actions.shape[0])
    n_val_full = max(1, int(round(n_episodes * 0.1)))
    n_train = n_episodes - n_val_full
    n_val = min(int(args.max_val_episodes), n_val_full)
    train_acts = raw_actions[:n_train].reshape(-1, raw_actions.shape[-1]).astype(np.float64)
    mean = train_acts.mean(axis=0).astype(np.float32)
    std = np.maximum(train_acts.std(axis=0, ddof=0), 1e-6).astype(np.float32)
    radius = 0.10
    p_mass = MLSMPM2D().p_mass
    img_size = int(MLSMPM2D().cfg.img_size)
    render_sigma = float(MLSMPM2D().cfg.render_sigma)

    models: dict[str, Any] = {}
    for name, ckpt in CHECKPOINTS:
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        model, loaded = load_model_from_checkpoint(ckpt, device)
        model.eval()
        models[name] = model
        epoch = int(torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch") or 0)
        if epoch != 200:
            raise RuntimeError(f"{ckpt} epoch={epoch}; protocol requires final-epoch last.pt")
        if not bool(getattr(model.predictor, "condition_on_action", False)):
            raise RuntimeError(f"{ckpt} is not action-conditioned")
        del loaded

    model_names = list(models.keys())
    val0 = n_train
    perms = {t: derange(n_val, np.random.default_rng(SHUFFLE_SEED + 1009 * t)) for t in range(15)}

    pair_fields = [
        "episode", "t", "intervention", "family", "command_change", "t0", "effective_contact",
        "true_cx", "true_cy", "true_fx", "true_fy",
        "alt_cx", "alt_cy", "alt_fx", "alt_fy",
        "true_cx_n", "true_cy_n", "true_fx_n", "true_fy_n",
        "alt_cx_n", "alt_cy_n", "alt_fx_n", "alt_fy_n",
    ]
    phys_fields = [
        "episode", "t", "horizon", "intervention", "family", "command_change", "t0",
        "effective_contact", "min_particle_distance", "force_norm", "weight_participation",
        "rms_pos", "mean_pos", "max_pos", "rms_vel", "d_ke", "com_dist",
        "mse_density", "mse_velocity", "mse_full",
    ]
    tgt_fields = [
        "episode", "t", "horizon", "intervention", "family", "command_change", "t0",
        "effective_contact", "checkpoint", "tgt_mse", "tgt_cosdist",
        "ratio_excluded_near_zero",
    ]
    pred_fields = [
        "episode", "t", "intervention", "family", "command_change", "t0",
        "effective_contact", "checkpoint",
        "pred_mse", "pred_cosdist", "pred_rel_l2",
        "err_true_to_tgt_true_h1", "err_alt_to_tgt_true_h1", "err_alt_to_tgt_alt_h1",
    ]
    contact_fields = [
        "episode", "t", "command_change", "t0", "effective_contact",
        "min_particle_distance", "raw_weight_sum", "max_normalized_weight",
        "weight_participation", "force_norm", "contact_radius",
        "true_cx", "true_cy", "true_fx", "true_fy",
    ]

    pair_path = OUT / "counterfactual_pairs.csv"
    phys_path = OUT / "physical_divergence_per_example.csv"
    tgt_path = OUT / "target_latent_divergence_per_example.csv"
    pred_path = OUT / "predictor_divergence_per_example.csv"
    contact_path = OUT / "contact_effectiveness.csv"

    pair_f = pair_path.open("w", newline="", encoding="utf-8")
    phys_f = phys_path.open("w", newline="", encoding="utf-8")
    tgt_f = tgt_path.open("w", newline="", encoding="utf-8")
    pred_f = pred_path.open("w", newline="", encoding="utf-8")
    contact_f = contact_path.open("w", newline="", encoding="utf-8")
    pair_w = csv.DictWriter(pair_f, fieldnames=pair_fields)
    phys_w = csv.DictWriter(phys_f, fieldnames=phys_fields)
    tgt_w = csv.DictWriter(tgt_f, fieldnames=tgt_fields)
    pred_w = csv.DictWriter(pred_f, fieldnames=pred_fields)
    contact_w = csv.DictWriter(contact_f, fieldnames=contact_fields)
    pair_w.writeheader()
    phys_w.writeheader()
    tgt_w.writeheader()
    pred_w.writeheader()
    contact_w.writeheader()

    n_phys_rows = 0
    n_change = 0
    n_eff = 0
    n_trans = 0

    try:
        for local_i, ep in enumerate(range(val0, val0 + n_val)):
            replay = replay_episode(int(ep), dump_seed=0, episode_length=16)
            recorded = replay["actions"]
            snapshots = replay["snapshots"]
            frames_true = replay["frames"]
            dumped = raw_actions[ep]
            if not np.allclose(recorded, dumped, rtol=0.0, atol=0.0):
                raise RuntimeError(f"replay actions diverge from dump at episode {ep}")
            print(f"cf episode {local_i + 1}/{n_val} ep={ep}", flush=True)

            online_z = {
                name: encode_many(model, frames_true[:15], device, target=False, batch=args.encode_batch)
                for name, model in models.items()
            }

            raster_keys: list[tuple[int, int, str]] = []
            raster_stack: list[np.ndarray] = []
            true_snaps_h: dict[tuple[int, int], dict[str, Any]] = {}
            alt_snaps_h: dict[tuple[int, int, str], dict[str, Any]] = {}
            alts_by_t: dict[int, dict[str, np.ndarray]] = {}
            diags: dict[int, dict[str, float]] = {}

            for t in range(15):
                snap0 = snapshots[t]
                sim0 = MLSMPM2D()
                sim0.restore(snap0)
                diag = sim0.contact_diagnostics(recorded[t])
                diags[t] = diag
                true_a = recorded[t]
                shuffled_a = raw_actions[val0 + int(perms[t][local_i]), t]
                alts = {
                    "true": true_a,
                    "shuffled_same_time": shuffled_a,
                    "neutral": mean.copy(),
                    "force_reversed": make_alt("force_reversed", true_a, radius=radius),
                    "force_rotated": make_alt("force_rotated", true_a, radius=radius),
                    "force_scaled": make_alt("force_scaled", true_a, radius=radius),
                    "contact_perturbed": make_alt("contact_perturbed", true_a, radius=radius),
                }
                alts_by_t[t] = alts
                n_trans += 1
                if t in CHANGE_T:
                    n_change += 1
                if diag["effective_contact"] > 0.5:
                    n_eff += 1
                contact_w.writerow(
                    {
                        "episode": int(ep),
                        "t": int(t),
                        "command_change": int(t in CHANGE_T),
                        "t0": int(t == 0),
                        "effective_contact": int(diag["effective_contact"] > 0.5),
                        "min_particle_distance": diag["min_particle_distance"],
                        "raw_weight_sum": diag["raw_weight_sum"],
                        "max_normalized_weight": diag["max_normalized_weight"],
                        "weight_participation": diag["weight_participation"],
                        "force_norm": diag["force_norm"],
                        "contact_radius": diag["contact_radius"],
                        "true_cx": float(true_a[0]),
                        "true_cy": float(true_a[1]),
                        "true_fx": float(true_a[2]),
                        "true_fy": float(true_a[3]),
                    }
                )
                true_n = (true_a - mean) / std
                for kind, alt_a in alts.items():
                    alt_n = (np.asarray(alt_a, dtype=np.float32) - mean) / std
                    pair_w.writerow(
                        {
                            "episode": int(ep),
                            "t": int(t),
                            "intervention": kind,
                            "family": "primary" if kind in PRIMARY else "secondary",
                            "command_change": int(t in CHANGE_T),
                            "t0": int(t == 0),
                            "effective_contact": int(diag["effective_contact"] > 0.5),
                            "true_cx": float(true_a[0]),
                            "true_cy": float(true_a[1]),
                            "true_fx": float(true_a[2]),
                            "true_fy": float(true_a[3]),
                            "alt_cx": float(alt_a[0]),
                            "alt_cy": float(alt_a[1]),
                            "alt_fx": float(alt_a[2]),
                            "alt_fy": float(alt_a[3]),
                            "true_cx_n": float(true_n[0]),
                            "true_cy_n": float(true_n[1]),
                            "true_fx_n": float(true_n[2]),
                            "true_fy_n": float(true_n[3]),
                            "alt_cx_n": float(alt_n[0]),
                            "alt_cy_n": float(alt_n[1]),
                            "alt_fx_n": float(alt_n[2]),
                            "alt_fy_n": float(alt_n[3]),
                        }
                    )
                true_roll = roll_horizons(snap0, t, true_a, recorded, img_size, render_sigma)
                for h, (snap, rast) in true_roll.items():
                    true_snaps_h[(t, h)] = snap
                    raster_keys.append((t, h, "true"))
                    raster_stack.append(rast)
                for kind, alt_a in alts.items():
                    if kind == "true":
                        continue
                    alt_roll = roll_horizons(snap0, t, alt_a, recorded, img_size, render_sigma)
                    for h, (snap, rast) in alt_roll.items():
                        alt_snaps_h[(t, h, kind)] = snap
                        raster_keys.append((t, h, kind))
                        raster_stack.append(rast)

            rasters = np.stack(raster_stack, axis=0)
            tgt_z = {
                name: encode_many(model, rasters, device, target=True, batch=args.encode_batch)
                for name, model in models.items()
            }
            key_index = {key: i for i, key in enumerate(raster_keys)}

            for t in range(15):
                diag = diags[t]
                alts = alts_by_t[t]
                true_a = alts["true"]
                true_n = torch.tensor((true_a - mean) / std, dtype=torch.float32).unsqueeze(0)
                preds_true: dict[str, Tensor] = {}
                for name, model in models.items():
                    z_t = online_z[name][t : t + 1]
                    with torch.no_grad():
                        preds_true[name] = model.predictor(z_t.to(device), true_n.to(device)).cpu()

                h1_tgt_true = {}
                h1_pred_alt: dict[tuple[str, str], Tensor] = {}
                h1_tgt_alt: dict[tuple[str, str], Tensor] = {}
                if (t, 1, "true") in key_index:
                    for name in model_names:
                        h1_tgt_true[name] = tgt_z[name][key_index[(t, 1, "true")] : key_index[(t, 1, "true")] + 1]

                for kind, alt_a in alts.items():
                    alt_n_np = (np.asarray(alt_a, dtype=np.float32) - mean) / std
                    alt_n = torch.tensor(alt_n_np, dtype=torch.float32).unsqueeze(0)
                    for name, model in models.items():
                        z_t = online_z[name][t : t + 1]
                        with torch.no_grad():
                            pred_alt = model.predictor(z_t.to(device), alt_n.to(device)).cpu()
                        pred_t = preds_true[name]
                        pmse, pcos = cosine_mse(pred_t, pred_alt)
                        rel = float(
                            (pred_t - pred_alt).norm() / (pred_t.norm().clamp_min(1e-12))
                        )
                        err_true = float("nan")
                        err_alt_true = float("nan")
                        err_alt_alt = float("nan")
                        if name in h1_tgt_true:
                            err_true = float((pred_t - h1_tgt_true[name]).pow(2).mean().item())
                            err_alt_true = float((pred_alt - h1_tgt_true[name]).pow(2).mean().item())
                        if kind != "true" and (t, 1, kind) in key_index:
                            tgt_alt = tgt_z[name][key_index[(t, 1, kind)] : key_index[(t, 1, kind)] + 1]
                            err_alt_alt = float((pred_alt - tgt_alt).pow(2).mean().item())
                            h1_tgt_alt[(name, kind)] = tgt_alt
                        elif kind == "true" and name in h1_tgt_true:
                            err_alt_alt = err_true
                        h1_pred_alt[(name, kind)] = pred_alt
                        pred_w.writerow(
                            {
                                "episode": int(ep),
                                "t": int(t),
                                "intervention": kind,
                                "family": "primary" if kind in PRIMARY else "secondary",
                                "command_change": int(t in CHANGE_T),
                                "t0": int(t == 0),
                                "effective_contact": int(diag["effective_contact"] > 0.5),
                                "checkpoint": name,
                                "pred_mse": pmse,
                                "pred_cosdist": pcos,
                                "pred_rel_l2": rel,
                                "err_true_to_tgt_true_h1": err_true,
                                "err_alt_to_tgt_true_h1": err_alt_true,
                                "err_alt_to_tgt_alt_h1": err_alt_alt,
                            }
                        )

                for kind, alt_a in alts.items():
                    for h in HORIZONS:
                        if t + h >= 16:
                            continue
                        if kind == "true":
                            alt_snap = true_snaps_h[(t, h)]
                            rast_alt = raster_stack[key_index[(t, h, "true")]]
                        else:
                            alt_snap = alt_snaps_h[(t, h, kind)]
                            rast_alt = raster_stack[key_index[(t, h, kind)]]
                        true_snap = true_snaps_h[(t, h)]
                        rast_true = raster_stack[key_index[(t, h, "true")]]
                        phys = phys_metrics(true_snap, alt_snap, p_mass)
                        rast = raster_mse(rast_true, rast_alt)
                        phys_w.writerow(
                            {
                                "episode": int(ep),
                                "t": int(t),
                                "horizon": int(h),
                                "intervention": kind,
                                "family": "primary" if kind in PRIMARY else "secondary",
                                "command_change": int(t in CHANGE_T),
                                "t0": int(t == 0),
                                "effective_contact": int(diag["effective_contact"] > 0.5),
                                "min_particle_distance": diag["min_particle_distance"],
                                "force_norm": diag["force_norm"],
                                "weight_participation": diag["weight_participation"],
                                **phys,
                                **rast,
                            }
                        )
                        n_phys_rows += 1
                        i_true = key_index[(t, h, "true")]
                        i_alt = key_index[(t, h, kind)]
                        for name in model_names:
                            tmse, tcos = cosine_mse(
                                tgt_z[name][i_true : i_true + 1],
                                tgt_z[name][i_alt : i_alt + 1],
                            )
                            excluded = int(tmse <= NEAR_ZERO)
                            tgt_w.writerow(
                                {
                                    "episode": int(ep),
                                    "t": int(t),
                                    "horizon": int(h),
                                    "intervention": kind,
                                    "family": "primary" if kind in PRIMARY else "secondary",
                                    "command_change": int(t in CHANGE_T),
                                    "t0": int(t == 0),
                                    "effective_contact": int(diag["effective_contact"] > 0.5),
                                    "checkpoint": name,
                                    "tgt_mse": tmse,
                                    "tgt_cosdist": tcos,
                                    "ratio_excluded_near_zero": excluded,
                                }
                            )
            for handle in (pair_f, phys_f, tgt_f, pred_f, contact_f):
                handle.flush()
    finally:
        pair_f.close()
        phys_f.close()
        tgt_f.close()
        pred_f.close()
        contact_f.close()

    phys_rows = list(csv.DictReader(phys_path.open(encoding="utf-8")))
    tgt_rows = list(csv.DictReader(tgt_path.open(encoding="utf-8")))
    pred_rows = list(csv.DictReader(pred_path.open(encoding="utf-8")))

    def filt(rows: list[dict[str, str]], **kw: Any) -> list[dict[str, str]]:
        out = rows
        for k, v in kw.items():
            out = [r for r in out if r[k] == str(v)]
        return out

    summary: dict[str, Any] = {
        "n_val_episodes": n_val,
        "n_transitions": n_trans,
        "n_command_change_transitions": n_change,
        "n_effective_contact_transitions": n_eff,
        "n_physical_rows": n_phys_rows,
        "wallclock_seconds": time.time() - t0,
        "device": str(device),
        "encode_batch": int(args.encode_batch),
        "near_zero_denominator": NEAR_ZERO,
        "hold": HOLD,
        "horizons": list(HORIZONS),
        "shuffle_seed": SHUFFLE_SEED,
        "checkpoints": [name for name, _ in CHECKPOINTS],
        "median_rms_pos_h1": {},
        "median_tgt_mse_h1": {},
        "median_pred_mse": {},
        "median_capture_ratio_h1": {},
        "ratio_excluded_h1": {},
        "median_raster_mse_full_h1": {},
        "median_rms_vel_h1": {},
        "fraction_identical_uint8_raster_h1": {},
    }
    for kind in ALL_KINDS:
        if kind == "true":
            continue
        summary["median_rms_pos_h1"][kind] = {
            "all": _median([float(r["rms_pos"]) for r in filt(phys_rows, intervention=kind, horizon=1)]),
            "command_change": _median(
                [float(r["rms_pos"]) for r in filt(phys_rows, intervention=kind, horizon=1, command_change=1)]
            ),
            "effective_contact": _median(
                [float(r["rms_pos"]) for r in filt(phys_rows, intervention=kind, horizon=1, effective_contact=1)]
            ),
        }
        summary["median_raster_mse_full_h1"][kind] = _median(
            [float(r["mse_full"]) for r in filt(phys_rows, intervention=kind, horizon=1)]
        )
        h1 = filt(phys_rows, intervention=kind, horizon=1)
        summary["median_rms_vel_h1"][kind] = _median([float(r["rms_vel"]) for r in h1])
        summary["fraction_identical_uint8_raster_h1"][kind] = (
            float(np.mean([float(r["mse_full"]) == 0.0 for r in h1])) if h1 else None
        )
        for name in model_names:
            summary["median_tgt_mse_h1"].setdefault(kind, {})[name] = _median(
                [float(r["tgt_mse"]) for r in filt(tgt_rows, intervention=kind, horizon=1, checkpoint=name)]
            )
            summary["median_pred_mse"].setdefault(kind, {})[name] = _median(
                [float(r["pred_mse"]) for r in filt(pred_rows, intervention=kind, checkpoint=name)]
            )
            tgt_h1 = filt(tgt_rows, intervention=kind, horizon=1, checkpoint=name)
            pred_h = filt(pred_rows, intervention=kind, checkpoint=name)
            pred_key = {(int(r["episode"]), int(r["t"])): float(r["pred_mse"]) for r in pred_h}
            ratios = []
            excluded = 0
            for r in tgt_h1:
                tmse = float(r["tgt_mse"])
                if tmse <= NEAR_ZERO:
                    excluded += 1
                    continue
                pmse = pred_key.get((int(r["episode"]), int(r["t"])))
                if pmse is None:
                    continue
                ratios.append(pmse / tmse)
            summary["median_capture_ratio_h1"].setdefault(kind, {})[name] = _median(ratios)
            summary["ratio_excluded_h1"].setdefault(kind, {})[name] = {
                "n_excluded": excluded,
                "n_total": len(tgt_h1),
            }

    (OUT / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    agg_csv = OUT / "aggregate_metrics.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "intervention", "checkpoint", "subset", "value"])
        writer.writerow(["n_transitions", "", "", "all", n_trans])
        writer.writerow(["n_command_change", "", "", "command_change", n_change])
        writer.writerow(["n_effective_contact", "", "", "effective_contact", n_eff])
        for kind, payload in summary["median_rms_pos_h1"].items():
            for subset, val in payload.items():
                writer.writerow(["median_rms_pos_h1", kind, "", subset, val])
        for kind, by_ckpt in summary["median_tgt_mse_h1"].items():
            for ckpt, val in by_ckpt.items():
                writer.writerow(["median_tgt_mse_h1", kind, ckpt, "all", val])
        for kind, by_ckpt in summary["median_pred_mse"].items():
            for ckpt, val in by_ckpt.items():
                writer.writerow(["median_pred_mse", kind, ckpt, "all", val])
        for kind, by_ckpt in summary["median_capture_ratio_h1"].items():
            for ckpt, val in by_ckpt.items():
                writer.writerow(["median_capture_ratio_h1", kind, ckpt, "all", val])

    print(json.dumps(summary, indent=2), flush=True)
    shuf_phys = summary["median_rms_pos_h1"]["shuffled_same_time"]["all"]
    neut_phys = summary["median_rms_pos_h1"]["neutral"]["all"]
    print("\nSIMULATOR COUNTERFACTUAL SUMMARY", flush=True)
    print(f"- N transitions: {n_trans}", flush=True)
    print(f"- N action-change transitions: {n_change}", flush=True)
    print(f"- N effective-contact transitions: {n_eff}", flush=True)
    print(f"- median true-vs-shuffled physical divergence (rms_pos h=1): {shuf_phys}", flush=True)
    print(f"- median true-vs-shuffled physical divergence (rms_vel h=1): {summary['median_rms_vel_h1']['shuffled_same_time']}", flush=True)
    print(f"- fraction identical uint8 rasters shuffled h=1: {summary['fraction_identical_uint8_raster_h1']['shuffled_same_time']}", flush=True)
    print(f"- median true-vs-neutral physical divergence (rms_pos h=1): {neut_phys}", flush=True)
    for name in model_names:
        print(
            f"- target-latent true-vs-shuffled ({name}): "
            f"{summary['median_tgt_mse_h1']['shuffled_same_time'][name]}",
            flush=True,
        )
        print(
            f"- predictor true-vs-shuffled ({name}): "
            f"{summary['median_pred_mse']['shuffled_same_time'][name]}",
            flush=True,
        )
        print(
            f"- predictor/target sensitivity ratio ({name}): "
            f"{summary['median_capture_ratio_h1']['shuffled_same_time'][name]}",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    main()
