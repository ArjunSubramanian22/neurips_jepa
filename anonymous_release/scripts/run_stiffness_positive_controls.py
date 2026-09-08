#!/usr/bin/env python3
"""Stiffness/geometry positive controls: particle state, float raster, uint8 raster.

No JEPA weights. Split 0:1800 / 1800:2000. Linear and MLP probes match Appendix F
(AdamW lr=1e-2, wd=1e-4, 80 full-batch steps, seed 0).
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_trajectories import replay_episode

DUMP = ROOT / "outputs" / "physics_mpm"
OUT = ROOT / "results" / "audit_phase2"
TIMES = (0, 3, 5, 7, 9, 12, 15)
HIST = (0, 3, 5, 7, 9, 12, 15)
MAX_N = 256
PROBE_SEED = 0
EPOCHS = 80
LR = 1e-2


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean().item())


def fit_linear(xtr, ytr, xva, yva, n_classes: int) -> float:
    torch.manual_seed(PROBE_SEED)
    probe = nn.Linear(xtr.shape[1], n_classes)
    opt = AdamW(probe.parameters(), lr=LR, weight_decay=1e-4)
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xva_t, yva_t = torch.from_numpy(xva), torch.from_numpy(yva)
    for _ in range(EPOCHS):
        probe.train()
        opt.zero_grad()
        nn.functional.cross_entropy(probe(xtr_t), ytr_t).backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        return _acc(probe(xva_t), yva_t)


def fit_mlp(xtr, ytr, xva, yva, n_classes: int) -> float:
    torch.manual_seed(PROBE_SEED)
    probe = MLP(xtr.shape[1], n_classes)
    opt = AdamW(probe.parameters(), lr=LR, weight_decay=1e-4)
    xtr_t, ytr_t = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xva_t, yva_t = torch.from_numpy(xva), torch.from_numpy(yva)
    for _ in range(EPOCHS):
        probe.train()
        opt.zero_grad()
        nn.functional.cross_entropy(probe(xtr_t), ytr_t).backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        return _acc(probe(xva_t), yva_t)


def render_float(positions: np.ndarray, velocities: np.ndarray, img_size: int = 64, sigma: float = 1.35) -> np.ndarray:
    """Same splat as ``render_rgb`` but returns pre-uint8 float channels in [0, 1]."""
    h = int(img_size)
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
    return np.clip(np.stack([ch0, ch1, ch2], axis=0), 0.0, 1.0).astype(np.float32)


def pack_mech(snap: dict, max_n: int = MAX_N) -> np.ndarray:
    n = int(snap["x"].shape[0])
    n_use = min(n, max_n)
    feat = np.zeros((max_n, 12), dtype=np.float32)
    feat[:n_use, 0:2] = snap["x"][:n_use]
    feat[:n_use, 2:4] = snap["v"][:n_use]
    feat[:n_use, 4:8] = snap["C"][:n_use].reshape(n_use, 4)
    feat[:n_use, 8:12] = snap["F"][:n_use].reshape(n_use, 4)
    extra = np.array([float(n), float(snap["mu"]), float(snap["lam"])], dtype=np.float32)
    return np.concatenate([feat.reshape(-1), extra], axis=0)


def _episode_features(episode: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    rec = replay_episode(int(episode), dump_seed=0, episode_length=16)
    snaps = rec["snapshots"]
    mech = np.stack([pack_mech(snaps[t]) for t in TIMES], axis=0)
    priv = np.array([float(snaps[0]["mu"]), float(snaps[0]["lam"]), float(snaps[0]["x"].shape[0])], dtype=np.float32)
    flt = np.stack([render_float(snaps[t]["x"], snaps[t]["v"]) for t in TIMES], axis=0)
    return int(episode), mech, priv, flt


def majority_acc(ytr: np.ndarray, yva: np.ndarray) -> float:
    counts = np.bincount(ytr, minlength=int(ytr.max()) + 1)
    pred = int(counts.argmax())
    return float((yva == pred).mean())


def shuffled_acc(xtr, ytr, xva, yva, n_classes: int) -> float:
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(ytr)
    return fit_linear(xtr, y_shuf, xva, yva, n_classes)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y_s = np.load(DUMP / "stiffness_labels.npy")
    y_g = np.load(DUMP / "geometry_labels.npy")
    obs = np.load(DUMP / "observations.npy", mmap_mode="r")
    n_train = 1800
    print("replaying 2000 episodes for particle/float features", flush=True)
    mech = np.zeros((2000, len(TIMES), MAX_N * 12 + 3), dtype=np.float32)
    priv = np.zeros((2000, 3), dtype=np.float32)
    flt = np.zeros((2000, len(TIMES), 3, 64, 64), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, (ep, m, p, f) in enumerate(pool.map(_episode_features, range(2000), chunksize=20)):
            mech[ep] = m
            priv[ep] = p
            flt[ep] = f
            if i % 200 == 0:
                print(f"stiffness replay {i}/2000", flush=True)
    np.savez_compressed(OUT / "stiffness_features_meta.npz", times=np.array(TIMES), priv=priv)

    uint8 = np.stack([np.array(obs[:, t], dtype=np.float32) / 255.0 for t in TIMES], axis=1)
    rows = []

    def add_rows(family: str, feat: np.ndarray, t_label: int) -> None:
        xtr, xva = feat[:n_train], feat[n_train:]
        for task, y in (("stiffness", y_s), ("geometry", y_g)):
            ncl = int(y.max()) + 1
            ytr, yva = y[:n_train], y[n_train:]
            rows.append(
                {
                    "family": family,
                    "task": task,
                    "t": t_label,
                    "probe": "majority",
                    "val_acc": majority_acc(ytr, yva),
                    "d_in": int(feat.shape[1]),
                }
            )
            rows.append(
                {
                    "family": family,
                    "task": task,
                    "t": t_label,
                    "probe": "linear",
                    "val_acc": fit_linear(xtr, ytr, xva, yva, ncl),
                    "d_in": int(feat.shape[1]),
                }
            )
            rows.append(
                {
                    "family": family,
                    "task": task,
                    "t": t_label,
                    "probe": "mlp",
                    "val_acc": fit_mlp(xtr, ytr, xva, yva, ncl),
                    "d_in": int(feat.shape[1]),
                }
            )
            if task == "stiffness" and t_label in (0, 15) and family in {
                "privileged_mulam",
                "mechanical",
                "float_full",
                "uint8_full",
            }:
                rows.append(
                    {
                        "family": family,
                        "task": task,
                        "t": t_label,
                        "probe": "shuffled_label_linear",
                        "val_acc": shuffled_acc(xtr, ytr, xva, yva, ncl),
                        "d_in": int(feat.shape[1]),
                    }
                )

    add_rows("privileged_mulam", priv[:, :2], -1)
    add_rows("privileged_mulam_n", priv, -1)
    for i, t in enumerate(TIMES):
        add_rows("mechanical", mech[:, i, :-3], t)
        add_rows("mechanical_plus_material", mech[:, i], t)
        add_rows("float_full", flt[:, i].reshape(2000, -1), t)
        add_rows("uint8_full", uint8[:, i].reshape(2000, -1), t)
        print("probed t", t, flush=True)

    def hist_index(times: tuple[int, ...], arr: np.ndarray) -> np.ndarray:
        idx = [TIMES.index(t) for t in times]
        return np.concatenate([arr[:, j].reshape(2000, -1) for j in idx], axis=1)

    add_rows("mechanical_history", hist_index(HIST, mech[:, :, :-3]), -1)
    add_rows("float_history", hist_index(HIST, flt), -1)
    add_rows("uint8_history", hist_index(HIST, uint8), -1)

    with (OUT / "stiffness_positive_controls.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for r in rows:
        if r["probe"] != "linear":
            continue
        key = f"{r['family']}/{r['task']}/t{r['t']}"
        summary[key] = r["val_acc"]
    (OUT / "stiffness_positive_controls_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("stiffness controls done", json.dumps({k: summary[k] for k in list(summary)[:12]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
