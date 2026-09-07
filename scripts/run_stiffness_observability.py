#!/usr/bin/env python3
"""Phase 7 stiffness/geometry observability probes (frozen latents + raw rasters).

Predeclared. No architecture search. Seeds fixed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "reviewer_strengthening" / "07_stiffness_observability"
DUMP = ROOT / "outputs" / "physics_mpm"
CKPTS = (
    ("vicreg_nocrop_s42", ROOT / "outputs" / "vicreg_nocrop" / "last.pt"),
    ("vicreg_nocrop_s43", ROOT / "outputs" / "vicreg_nocrop_seed43" / "last.pt"),
)
TIMES = (0, 3, 5, 7, 9, 12, 15)
PROBE_SEED = 0
MLP_EPOCHS = 80
MLP_LR = 1e-2


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean().item())


def fit_linear(xtr, ytr, xva, yva, n_classes: int) -> float:
    torch.manual_seed(PROBE_SEED)
    probe = nn.Linear(xtr.shape[1], n_classes)
    opt = AdamW(probe.parameters(), lr=MLP_LR, weight_decay=1e-4)
    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(ytr)
    xva_t = torch.from_numpy(xva)
    yva_t = torch.from_numpy(yva)
    for _ in range(MLP_EPOCHS):
        probe.train()
        opt.zero_grad()
        loss = nn.functional.cross_entropy(probe(xtr_t), ytr_t)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        return _acc(probe(xva_t), yva_t)


def fit_mlp(xtr, ytr, xva, yva, n_classes: int) -> float:
    torch.manual_seed(PROBE_SEED)
    probe = MLP(xtr.shape[1], n_classes)
    opt = AdamW(probe.parameters(), lr=MLP_LR, weight_decay=1e-4)
    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(ytr)
    xva_t = torch.from_numpy(xva)
    yva_t = torch.from_numpy(yva)
    for _ in range(MLP_EPOCHS):
        probe.train()
        opt.zero_grad()
        loss = nn.functional.cross_entropy(probe(xtr_t), ytr_t)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        return _acc(probe(xva_t), yva_t)


def downsample(frames: np.ndarray, size: int = 16) -> np.ndarray:
    t = torch.from_numpy(frames)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="area")
    return t.numpy().reshape(t.shape[0], -1)


def main() -> None:
    from src.config import load_experiment_config
    from src.dataset import build_dataset
    from src.train import apply_dataset_path, load_model_from_checkpoint

    OUT.mkdir(parents=True, exist_ok=True)
    protocol = (
        "# Probe protocol\n\n"
        "Split: episode 0-1799 train / 1800-1999 val. Frozen representations.\n"
        "Linear: nn.Linear, AdamW lr=1e-2, wd=1e-4, 80 epochs, seed 0.\n"
        "MLP:  d->128->C ReLU, same optimizer/epochs/seed. No search.\n"
        "Raw raster: area-downsample to 16x16, flatten, same probes.\n"
        "History: concat selected times; mean pool; [z_t, z_t-z_0].\n"
    )
    (OUT / "probe_protocol.md").write_text(protocol, encoding="utf-8")

    econf = load_experiment_config(ROOT / "configs" / "workshop_vicreg.yaml")
    econf = apply_dataset_path(econf, str(DUMP), num_workers=0)
    dataset = build_dataset(econf.data, split="all")
    n_train, n_val = dataset.split_sizes()
    y_s = dataset.stiffness_labels.numpy()
    y_g = dataset.geometry_labels.numpy() if dataset.geometry_labels is not None else None
    obs = dataset.observations.numpy()
    rows = []
    device = torch.device("cpu")

    # raw probes
    for t in TIMES:
        x = downsample(obs[:, t])
        xtr, xva = x[:n_train], x[n_train:]
        for task, y in (("stiffness", y_s), ("geometry", y_g)):
            if y is None:
                continue
            ncl = int(y.max()) + 1
            rows.append({
                "family": "raw_single",
                "checkpoint": "raw_obs",
                "task": task,
                "features": f"raw_t{t}_ds16",
                "probe": "linear",
                "t": t,
                "val_acc": fit_linear(xtr, y[:n_train], xva, y[n_train:], ncl),
            })
            rows.append({
                "family": "raw_single",
                "checkpoint": "raw_obs",
                "task": task,
                "features": f"raw_t{t}_ds16",
                "probe": "mlp",
                "t": t,
                "val_acc": fit_mlp(xtr, y[:n_train], xva, y[n_train:], ncl),
            })
    seq_t = (0, 3, 6, 9, 12, 15)
    xseq = np.concatenate([downsample(obs[:, t]) for t in seq_t], axis=1)
    for task, y in (("stiffness", y_s), ("geometry", y_g)):
        if y is None:
            continue
        ncl = int(y.max()) + 1
        rows.append({
            "family": "raw_history",
            "checkpoint": "raw_obs",
            "task": task,
            "features": "raw_t" + "_".join(str(t) for t in seq_t) + "_ds16",
            "probe": "linear",
            "t": -1,
            "val_acc": fit_linear(xseq[:n_train], y[:n_train], xseq[n_train:], y[n_train:], ncl),
        })
        rows.append({
            "family": "raw_history",
            "checkpoint": "raw_obs",
            "task": task,
            "features": "raw_t" + "_".join(str(t) for t in seq_t) + "_ds16",
            "probe": "mlp",
            "t": -1,
            "val_acc": fit_mlp(xseq[:n_train], y[:n_train], xseq[n_train:], y[n_train:], ncl),
        })

    for ckpt_name, path in CKPTS:
        model, _ = load_model_from_checkpoint(path, device)
        model.eval()
        zs = []
        with torch.no_grad():
            for start in range(0, obs.shape[0], 8):
                sl = torch.from_numpy(obs[start : start + 8])
                b, t, c, h, w = sl.shape
                z = model.encode(sl.reshape(b * t, c, h, w).to(device), target=True).cpu()
                zs.append(z.reshape(b, t, -1).numpy())
        z = np.concatenate(zs, axis=0)
        for t in TIMES:
            feat = z[:, t]
            for task, y in (("stiffness", y_s), ("geometry", y_g)):
                if y is None:
                    continue
                ncl = int(y.max()) + 1
                for probe, fn in (("linear", fit_linear), ("mlp", fit_mlp)):
                    rows.append({
                        "family": "jepa_single",
                        "checkpoint": ckpt_name,
                        "task": task,
                        "features": f"z_t{t}",
                        "probe": probe,
                        "t": t,
                        "val_acc": fn(feat[:n_train], y[:n_train], feat[n_train:], y[n_train:], ncl),
                    })
        for spec, feat in (
            ("concat_0_3", np.concatenate([z[:, 0], z[:, 3]], 1)),
            ("concat_0_3_6", np.concatenate([z[:, 0], z[:, 3], z[:, 6]], 1)),
            ("mean_pool", z.mean(axis=1)),
            ("final_diff_t15", np.concatenate([z[:, 15], z[:, 15] - z[:, 0]], 1)),
        ):
            for task, y in (("stiffness", y_s), ("geometry", y_g)):
                if y is None:
                    continue
                ncl = int(y.max()) + 1
                for probe, fn in (("linear", fit_linear), ("mlp", fit_mlp)):
                    rows.append({
                        "family": "jepa_history",
                        "checkpoint": ckpt_name,
                        "task": task,
                        "features": spec,
                        "probe": probe,
                        "t": -1,
                        "val_acc": fn(feat[:n_train], y[:n_train], feat[n_train:], y[n_train:], ncl),
                    })
        print(f"probes done {ckpt_name}", flush=True)

    with (OUT / "per_model_probe_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([r for r in rows if r["family"].startswith("jepa_single")])
    with (OUT / "temporal_probe_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([r for r in rows if "history" in r["family"]])
    with (OUT / "raw_probe_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([r for r in rows if r["family"].startswith("raw")])
    (OUT / "all_probe_results.csv").write_text("")  # placeholder filled below
    with (OUT / "all_probe_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def pick(family, task, probe=None):
        cand = [r for r in rows if r["family"] == family and r["task"] == task]
        if probe:
            cand = [r for r in cand if r["probe"] == probe]
        return cand

    stiff_lin = pick("jepa_single", "stiffness", "linear")
    (OUT / "summary.md").write_text(
        "# Stiffness observability\n\n"
        f"Chance stiffness = 1/3. Chance geometry = 1/2.\n\n"
        f"JEPA linear single-frame stiffness n={len(stiff_lin)} "
        f"mean_val={np.mean([r['val_acc'] for r in stiff_lin]) if stiff_lin else 'NA'}\n"
        "See CSVs for MLP, history, and raw-raster controls.\n",
        encoding="utf-8",
    )
    print("PHASE 7 probes complete", flush=True)


if __name__ == "__main__":
    main()
