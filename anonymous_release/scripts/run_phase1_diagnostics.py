#!/usr/bin/env python3
"""Cheap Phase-1 diagnostics. Does not retrain. Writes results/audit_phase1/."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import (  # noqa: E402
    MLSMPM2D,
    MPMConfig,
    apply_action_frames,
    render_rgb,
    replay_episode,
)

OUT = ROOT / "results" / "audit_phase1"
DUMP = ROOT / "outputs" / "physics_mpm"
HOLD = 3
N_STATES = 16
VAL0 = 1800
EPS_C = 0.05
EPS_F = 0.20


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def _render_float(sim: MLSMPM2D) -> np.ndarray:
    """Pre-uint8 float raster in [0,1], duplicating render_rgb math."""
    from scripts.generate_trajectories import render_rgb as _  # noqa: F401

    h = int(sim.cfg.img_size)
    sigma = float(sim.cfg.render_sigma)
    positions = sim.x
    velocities = sim.v
    density = np.zeros((h, h), dtype=np.float32)
    mom_x = np.zeros((h, h), dtype=np.float32)
    mom_y = np.zeros((h, h), dtype=np.float32)
    px = positions[:, 0] * np.float32(h - 1)
    py = positions[:, 1] * np.float32(h - 1)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    inv_two_s2 = 1.0 / (2.0 * sigma * sigma + 1e-8)
    ix = np.rint(px).astype(np.int32)
    iy = np.rint(py).astype(np.int32)
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    for dy in range(-radius, radius + 1):
        yy = np.clip(iy + dy, 0, h - 1)
        for dx in range(-radius, radius + 1):
            xx = np.clip(ix + dx, 0, h - 1)
            w = np.exp(-(np.float32(dx * dx + dy * dy)) * inv_two_s2)
            np.add.at(density, (yy, xx), w)
            np.add.at(mom_x, (yy, xx), w * vx)
            np.add.at(mom_y, (yy, xx), w * vy)
    mass = density + 1e-8
    ch0 = density / (density.max() + 1e-8)
    ch1 = 0.5 + 0.5 * np.tanh(mom_x / mass)
    ch2 = 0.5 + 0.5 * np.tanh(mom_y / mass)
    return np.stack([ch0, ch1, ch2], axis=0)


def _roll(sim: MLSMPM2D, action: np.ndarray, n_frames: int) -> None:
    apply_action_frames(sim, action, n_frames)


def actuator_and_damping() -> dict:
    rows = []
    damp_rows = []
    rng_eps = np.array([EPS_C, EPS_C, EPS_F, EPS_F], dtype=np.float32)
    for i, ep in enumerate(range(VAL0, VAL0 + N_STATES)):
        rec = replay_episode(int(ep), dump_seed=0, episode_length=16)
        t = 3
        snap = rec["snapshots"][t]
        a0 = rec["actions"][t].astype(np.float32)
        diameter = float(np.linalg.norm(rec["snapshots"][t]["x"].max(0) - rec["snapshots"][t]["x"].min(0)))
        spacing = 0.026
        for h in (1, 3, 5):
            base = MLSMPM2D()
            base.restore(snap)
            _roll(base, a0, h)
            x_b, v_b = base.x.copy(), base.v.copy()
            rf_b = _render_float(base)
            ru_b = render_rgb(base.x, base.v, 64, base.cfg.render_sigma)
            for dim, name in enumerate(("cx", "cy", "fx", "fy")):
                a1 = a0.copy()
                a1[dim] = float(np.clip(a1[dim] + rng_eps[dim], 0.08 if dim < 2 else -3.0, 0.92 if dim < 2 else 3.0))
                alt = MLSMPM2D()
                alt.restore(snap)
                _roll(alt, a1, h)
                rf = _render_float(alt)
                ru = render_rgb(alt.x, alt.v, 64, alt.cfg.render_sigma)
                dpos = _rms(alt.x, x_b)
                dvel = _rms(alt.v, v_b)
                ident = bool(np.array_equal(ru, ru_b))
                rows.append(
                    {
                        "episode": ep,
                        "t": t,
                        "horizon": h,
                        "dim": name,
                        "eps": float(rng_eps[dim]),
                        "rms_pos": dpos,
                        "rms_vel": dvel,
                        "rms_pos_pixels": dpos * 63.0,
                        "rms_pos_over_spacing": dpos / spacing,
                        "rms_pos_over_diameter": dpos / max(diameter, 1e-8),
                        "float_raster_mse": float(np.mean((rf - rf_b) ** 2)),
                        "uint8_raster_mse": float(np.mean((ru.astype(np.float32) / 255.0 - ru_b.astype(np.float32) / 255.0) ** 2)),
                        "uint8_identical": int(ident),
                        "sens_pos": dpos / float(rng_eps[dim]),
                        "sens_vel": dvel / float(rng_eps[dim]),
                    }
                )
        # impulse: apply recorded force for 1 frame, then zero
        for damp in (0.08, 0.03, 0.01):
            cfg = MPMConfig(damping=damp)
            s0 = MLSMPM2D(cfg)
            s0.restore(snap)
            a_imp = a0.copy()
            a_zero = a0.copy()
            a_zero[2:4] = 0.0
            apply_action_frames(s0, a_imp, 1)
            v1 = float(np.sqrt(np.mean(s0.v ** 2)))
            x1 = s0.x.copy()
            v_hist = [v1]
            d_hist = [0.0]
            for k in range(2, 6):
                apply_action_frames(s0, a_zero, 1)
                v_hist.append(float(np.sqrt(np.mean(s0.v ** 2))))
                d_hist.append(_rms(s0.x, x1))
            damp_rows.append(
                {
                    "episode": ep,
                    "damping": damp,
                    "mult_per_substep": 1.0 - damp,
                    "mult_per_obs": (1.0 - damp) ** 8,
                    "v_after_1": v_hist[0],
                    "v_after_2": v_hist[1],
                    "v_after_3": v_hist[2],
                    "v_after_5": v_hist[4],
                    "disp_from_postimpulse_h5": d_hist[4],
                }
            )
        print(f"diag state {i+1}/{N_STATES}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "actuator_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "damping_impulse.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(damp_rows[0].keys()))
        w.writeheader()
        w.writerows(damp_rows)

    def med(kind, h, field):
        vals = [r[field] for r in rows if r["dim"] == kind and r["horizon"] == h]
        return float(np.median(vals))

    summary = {
        "n_states": N_STATES,
        "eps_center": EPS_C,
        "eps_force": EPS_F,
        "damping_math": {
            "cfg_damping": 0.08,
            "velocity_multiplier_per_substep": 0.92,
            "substeps": 8,
            "0.92**8": float(0.92 ** 8),
        },
        "median_sens_pos_h1": {d: med(d, 1, "sens_pos") for d in ("cx", "cy", "fx", "fy")},
        "median_sens_vel_h1": {d: med(d, 1, "sens_vel") for d in ("cx", "cy", "fx", "fy")},
        "median_rms_pos_h1": {d: med(d, 1, "rms_pos") for d in ("cx", "cy", "fx", "fy")},
        "median_rms_vel_h1": {d: med(d, 1, "rms_vel") for d in ("cx", "cy", "fx", "fy")},
        "fraction_uint8_identical_h1": {
            d: float(np.mean([r["uint8_identical"] for r in rows if r["dim"] == d and r["horizon"] == 1]))
            for d in ("cx", "cy", "fx", "fy")
        },
        "force_vs_center_sens_pos_ratio_h1": (
            0.5 * (med("fx", 1, "sens_pos") + med("fy", 1, "sens_pos"))
            / max(1e-18, 0.5 * (med("cx", 1, "sens_pos") + med("cy", 1, "sens_pos")))
        ),
        "damping_median_v_after": {},
    }
    for damp in (0.08, 0.03, 0.01):
        sub = [r for r in damp_rows if r["damping"] == damp]
        summary["damping_median_v_after"][str(damp)] = {
            "h1": float(np.median([r["v_after_1"] for r in sub])),
            "h2": float(np.median([r["v_after_2"] for r in sub])),
            "h3": float(np.median([r["v_after_3"] for r in sub])),
            "h5": float(np.median([r["v_after_5"] for r in sub])),
            "retention_h5_over_h1": float(
                np.median([r["v_after_5"] / max(r["v_after_1"], 1e-12) for r in sub])
            ),
        }
    (OUT / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def replay_diff(n_ep: int = 20) -> dict:
    obs = np.load(DUMP / "observations.npy", mmap_mode="r")
    acts = np.load(DUMP / "actions.npy")
    n_frames = 0
    n_frames_diff = 0
    n_pix_diff = 0
    n_pix = 0
    max_abs = 0
    abs_sum = 0.0
    per_ch = np.zeros(3, dtype=np.int64)
    deltas = []
    for ep in range(VAL0, VAL0 + n_ep):
        rec = replay_episode(int(ep), dump_seed=0, episode_length=16)
        dump_f = np.asarray(obs[ep])
        rep_f = rec["frames"]
        n_frames += 16
        if not np.array_equal(dump_f, rep_f):
            n_frames_diff += int(np.any(dump_f != rep_f, axis=(1, 2, 3)).sum())
        d = np.abs(dump_f.astype(np.int16) - rep_f.astype(np.int16))
        n_pix += d.size
        n_pix_diff += int((d > 0).sum())
        max_abs = max(max_abs, int(d.max()))
        abs_sum += float(d.sum())
        for c in range(3):
            per_ch[c] += int((d[:, c] > 0).sum())
        if not np.array_equal(acts[ep], rec["actions"]):
            raise RuntimeError(f"action mismatch episode {ep}")
        deltas.append(float(d.mean()))
        print(f"replay diff ep {ep}", flush=True)
    report = {
        "n_episodes": n_ep,
        "n_frames": n_frames,
        "n_frames_any_pixel_diff": n_frames_diff,
        "n_pixels": n_pix,
        "n_pixels_diff": n_pix_diff,
        "frac_pixels_diff": n_pix_diff / n_pix,
        "max_abs_uint8": max_abs,
        "mean_abs_uint8": abs_sum / n_pix,
        "median_frame_mean_abs": float(np.median(deltas)),
        "per_channel_pixels_diff": per_ch.tolist(),
        "actions_all_equal": True,
        "note": "Subset of val episodes. Full dump hash is in baseline_reproduction.json.",
    }
    (OUT / "replay_diff_subset.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def visibility_bins() -> dict:
    phys = list(csv.DictReader((ROOT / "results/reviewer_strengthening/01_simulator_counterfactual/physical_divergence_per_example.csv").open()))
    tgt = list(csv.DictReader((ROOT / "results/reviewer_strengthening/01_simulator_counterfactual/target_latent_divergence_per_example.csv").open()))
    pred = list(csv.DictReader((ROOT / "results/reviewer_strengthening/01_simulator_counterfactual/predictor_divergence_per_example.csv").open()))
    tgt_h1 = {
        (int(r["episode"]), int(r["t"]), r["checkpoint"]): float(r["tgt_mse"])
        for r in tgt
        if r["intervention"] == "shuffled_same_time" and r["horizon"] == "1"
    }
    pred_d = {
        (int(r["episode"]), int(r["t"]), r["checkpoint"]): float(r["pred_mse"])
        for r in pred
        if r["intervention"] == "shuffled_same_time"
    }
    rows = [r for r in phys if r["intervention"] == "shuffled_same_time" and r["horizon"] == "1"]
    mse = np.array([float(r["mse_full"]) for r in rows])
    ident = mse == 0.0
    nonzero = mse[mse > 0]
    edges = np.quantile(nonzero, [0.0, 0.33, 0.67, 1.0]) if nonzero.size else np.array([0, 1])
    def bin_of(v: float) -> str:
        if v == 0.0:
            return "identical_raster"
        if v <= edges[1]:
            return "low_nonzero_raster"
        if v <= edges[2]:
            return "mid_raster"
        return "high_raster"
    ckpt = "vicreg_nocrop_s42"
    out_rows = []
    for r in rows:
        b = bin_of(float(r["mse_full"]))
        key = (int(r["episode"]), int(r["t"]), ckpt)
        dt = tgt_h1.get(key, float("nan"))
        dp = pred_d.get(key, float("nan"))
        out_rows.append(
            {
                "bin": b,
                "rms_pos": float(r["rms_pos"]),
                "rms_vel": float(r["rms_vel"]),
                "mse_full": float(r["mse_full"]),
                "d_tgt": dt,
                "d_pred": dp,
            }
        )
    summary = {"n": len(out_rows), "frac_identical": float(np.mean(ident)), "bins": {}}
    for b in ("identical_raster", "low_nonzero_raster", "mid_raster", "high_raster"):
        sub = [x for x in out_rows if x["bin"] == b]
        if not sub:
            continue
        dt = np.array([x["d_tgt"] for x in sub])
        dp = np.array([x["d_pred"] for x in sub])
        summary["bins"][b] = {
            "n": len(sub),
            "median_rms_pos": float(np.median([x["rms_pos"] for x in sub])),
            "median_rms_vel": float(np.median([x["rms_vel"] for x in sub])),
            "median_raster_mse": float(np.median([x["mse_full"] for x in sub])),
            "median_d_tgt": float(np.median(dt)),
            "median_d_pred": float(np.median(dp)),
            "frac_dpred_lt_dtgt": float(np.mean(dp < dt)) if np.isfinite(dt).all() else None,
        }
    top = sorted(out_rows, key=lambda x: x["mse_full"], reverse=True)[: max(1, len(out_rows) // 10)]
    dt = np.array([x["d_tgt"] for x in top])
    dp = np.array([x["d_pred"] for x in top])
    summary["top10pct_visible"] = {
        "n": len(top),
        "median_d_tgt": float(np.median(dt)),
        "median_d_pred": float(np.median(dp)),
        "frac_dpred_lt_dtgt": float(np.mean(dp < dt)),
    }
    pairs = list(csv.DictReader((ROOT / "results/reviewer_strengthening/01_simulator_counterfactual/counterfactual_pairs.csv").open()))
    pair_map = {
        (int(r["episode"]), int(r["t"]), r["intervention"]): r
        for r in pairs
    }
    tgt_rows = [
        r
        for r in tgt
        if r["intervention"] == "shuffled_same_time" and r["horizon"] == "1" and r["checkpoint"] == ckpt
    ]
    excl = {"n": 0, "rms_pos": [], "raster": [], "d_pred": [], "action_l2": [], "identical": 0}
    incl = {"n": 0, "rms_pos": [], "raster": [], "d_pred": [], "action_l2": [], "identical": 0}
    phys_lookup = {(int(r["episode"]), int(r["t"])): r for r in rows}
    for r in tgt_rows:
        ep, t = int(r["episode"]), int(r["t"])
        p = phys_lookup[(ep, t)]
        pr = pair_map[(ep, t, "shuffled_same_time")]
        al2 = float(np.sqrt(sum((float(pr[f"true_{k}_n"]) - float(pr[f"alt_{k}_n"])) ** 2 for k in ("cx", "cy", "fx", "fy"))))
        bucket = excl if float(r["tgt_mse"]) <= 1e-8 else incl
        bucket["n"] += 1
        bucket["rms_pos"].append(float(p["rms_pos"]))
        bucket["raster"].append(float(p["mse_full"]))
        bucket["d_pred"].append(pred_d[(ep, t, ckpt)])
        bucket["action_l2"].append(al2)
        bucket["identical"] += int(float(p["mse_full"]) == 0.0)

    def _summ(b):
        return {
            "n": b["n"],
            "frac_of_3000": b["n"] / 3000.0,
            "median_rms_pos": float(np.median(b["rms_pos"])),
            "median_raster_mse": float(np.median(b["raster"])),
            "median_d_pred": float(np.median(b["d_pred"])),
            "median_action_l2_normalized": float(np.median(b["action_l2"])),
            "frac_identical_raster": b["identical"] / max(b["n"], 1),
        }

    summary["dtgt_threshold_1e-8_s42"] = {"excluded": _summ(excl), "included": _summ(incl)}
    (OUT / "visibility_stratification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def checkpoint_rows() -> list[dict]:
    specs = [
        ("outputs/vicreg_nocrop/last.pt", "vicreg", True, 42, 1.0, True, "paper_and_fork"),
        ("outputs/vicreg_nocrop_seed43/last.pt", "vicreg", True, 43, 1.0, True, "paper_and_fork"),
        ("outputs/no_action/last.pt", "vicreg", False, 42, 1.0, True, "paper"),
        ("outputs/vicreg_nocrop_noaction_seed43/last.pt", "vicreg", False, 43, 1.0, True, "paper"),
        ("results/reviewer_strengthening/05_more_seeds/vicreg_action_s44/last.pt", "vicreg", True, 44, 1.0, True, "paper"),
        ("results/reviewer_strengthening/05_more_seeds/vicreg_action_s45/last.pt", "vicreg", True, 45, 1.0, True, "paper"),
        ("results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s44/last.pt", "vicreg", False, 44, 1.0, True, "paper"),
        ("results/reviewer_strengthening/05_more_seeds/vicreg_noaction_s45/last.pt", "vicreg", False, 45, 1.0, True, "paper"),
        ("results/reviewer_strengthening/06_corrected_objective/mse_action_s42/last.pt", "mse", True, 42, 1.0, True, "paper"),
        ("results/reviewer_strengthening/06_corrected_objective/mse_action_s43/last.pt", "mse", True, 43, 1.0, True, "paper"),
        ("outputs/vicreg/last.pt", "vicreg", True, 42, 0.75, False, "obsolete_cropped"),
        ("outputs/mse/last.pt", "mse", True, 42, 0.75, False, "obsolete_cropped"),
    ]
    import torch
    from src.config import ExperimentConfig

    rows = []
    for rel, obj, cond, seed, crop, used, note in specs:
        p = ROOT / rel
        exists = p.is_file()
        epoch = None
        crop_ckpt = crop
        cond_ckpt = cond
        obj_ckpt = obj
        if exists:
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
                epoch = int(payload.get("epoch") or 0)
                cfg = ExperimentConfig.from_dict(payload["config"])
                crop_ckpt = float(cfg.data.crop_scale_min)
                cond_ckpt = bool(cfg.model.condition_on_action)
                obj_ckpt = str(cfg.training.loss_type)
                del payload
            except Exception as exc:  # noqa: BLE001
                print("checkpoint inspect failed", rel, exc, flush=True)
        rows.append(
            {
                "path": rel,
                "exists": exists,
                "objective": obj_ckpt,
                "condition_on_action": cond_ckpt,
                "seed": seed,
                "crop_scale_min": crop_ckpt,
                "epoch": epoch,
                "sha256": _sha256(p) if exists else "",
                "bytes": p.stat().st_size if exists else 0,
                "used_in_revised_paper": used,
                "notes": note,
            }
        )
        print("hashed", rel, "epoch", epoch, "crop", crop_ckpt, flush=True)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("== checkpoints ==", flush=True)
    ck = checkpoint_rows()
    with (OUT / "checkpoint_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ck[0].keys()))
        w.writeheader()
        w.writerows(ck)
    print("== actuator/damping ==", flush=True)
    a = actuator_and_damping()
    print("== replay subset ==", flush=True)
    r = replay_diff(20)
    print("== visibility ==", flush=True)
    v = visibility_bins()
    print(json.dumps({"actuator": a, "replay": r, "visibility": v}, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
