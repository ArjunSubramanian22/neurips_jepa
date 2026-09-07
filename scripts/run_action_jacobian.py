#!/usr/bin/env python3
"""Phase 4 Jacobian / finite-difference / weight diagnostics (CPU).

Architectural/gradient sensitivity is descriptive; the simulator intervention
remains the primary behavioral test.
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

OUT = ROOT / "results" / "reviewer_strengthening" / "04_action_jacobian"
DUMP = ROOT / "outputs" / "physics_mpm"
CHECKPOINTS = (
    ("vicreg_nocrop_s42", ROOT / "outputs" / "vicreg_nocrop" / "last.pt"),
    ("vicreg_nocrop_s43", ROOT / "outputs" / "vicreg_nocrop_seed43" / "last.pt"),
)
EPSILONS = (0.01, 0.05, 0.1)
N_SAMPLES = 64
SAMPLE_SEED = 0


def main() -> None:
    from src.config import load_experiment_config
    from src.dataset import build_dataset
    from src.train import apply_dataset_path, load_model_from_checkpoint

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    cfg = load_experiment_config(ROOT / "configs" / "workshop_vicreg.yaml")
    cfg = apply_dataset_path(cfg, str(DUMP), num_workers=0)
    dataset = build_dataset(cfg.data, split="val")
    rng = np.random.default_rng(SAMPLE_SEED)
    n = min(N_SAMPLES, len(dataset))
    idxs = rng.choice(len(dataset), size=n, replace=False)
    jac_rows = []
    fd_rows = []
    weight_payload = {}
    for ckpt_name, path in CHECKPOINTS:
        model, _ = load_model_from_checkpoint(path, device)
        model.eval()
        pred = model.predictor
        ae = pred.action_embed
        weight_payload[ckpt_name] = {
            "action_embed.0.weight_fro": float(ae[0].weight.detach().norm().item()),
            "action_embed.2.weight_fro": float(ae[2].weight.detach().norm().item()),
            "note": "Weight norms are descriptive only.",
        }
        if hasattr(pred, "fuse_proj"):
            w = pred.fuse_proj.weight.detach()
            d_z = model.latent_dim
            d_a = int(model.cfg.action_embed_dim)
            weight_payload[ckpt_name]["fuse_in_dim"] = int(w.shape[1])
            weight_payload[ckpt_name]["fuse_latent_cols_fro"] = float(w[:, :d_z].norm().item())
            weight_payload[ckpt_name]["fuse_action_cols_fro"] = float(w[:, d_z : d_z + d_a].norm().item())
        for i, idx in enumerate(idxs):
            x_t, a_t, _ = dataset[int(idx)]
            x_t = x_t.unsqueeze(0)
            a_t = a_t.unsqueeze(0).requires_grad_(True)
            z = model.encode(x_t)
            y = pred(z, a_t)
            # Jacobian dy/da via autograd on each output dim is expensive; use
            # J^T J diagonal via per-action-dim unit perturbations of the scalar ||y||^2.
            grads = []
            for d in range(4):
                a = a_t.detach().clone().requires_grad_(True)
                y_d = pred(z, a)
                g = torch.autograd.grad(y_d.pow(2).mean(), a, retain_graph=False)[0]
                grads.append(g.detach().cpu().numpy().reshape(-1))
            gmat = np.stack(grads, axis=0)
            fro = float(np.linalg.norm(gmat))
            per = np.linalg.norm(gmat, axis=0)
            jac_rows.append(
                {
                    "checkpoint": ckpt_name,
                    "sample": int(i),
                    "dataset_index": int(idx),
                    "frobenius_proxy": fro,
                    "cx": float(per[0]),
                    "cy": float(per[1]),
                    "fx": float(per[2]),
                    "fy": float(per[3]),
                }
            )
            with torch.no_grad():
                y0 = pred(z, a_t.detach())
                for eps in EPSILONS:
                    for d, name in enumerate(("cx", "cy", "fx", "fy")):
                        a = a_t.detach().clone()
                        a[0, d] = a[0, d] + eps
                        y1 = pred(z, a)
                        fd_rows.append(
                            {
                                "checkpoint": ckpt_name,
                                "sample": int(i),
                                "eps": eps,
                                "dim": name,
                                "mse": float((y0 - y1).pow(2).mean().item()),
                                "rel_l2": float((y0 - y1).norm() / (y0.norm() + 1e-12)),
                            }
                        )
        print(f"jacobian done {ckpt_name}", flush=True)

    with (OUT / "jacobian_per_example.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(jac_rows[0].keys()))
        writer.writeheader()
        writer.writerows(jac_rows)
    with (OUT / "finite_difference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fd_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fd_rows)
    (OUT / "weight_diagnostics.json").write_text(json.dumps(weight_payload, indent=2), encoding="utf-8")
    (OUT / "summary.md").write_text(
        "# Action Jacobian diagnostics\n\n"
        "Architectural/gradient sensitivity is descriptive; the simulator intervention "
        "remains the primary behavioral test.\n\n"
        f"N samples: {N_SAMPLES} validation transitions, seed {SAMPLE_SEED}.\n"
        "The reported Jacobian proxy is the stacked per-dimension gradient of mean-squared "
        "predictor output, not the full D×4 Jacobian Frobenius norm.\n",
        encoding="utf-8",
    )
    print("PHASE 4 jacobian complete", flush=True)


if __name__ == "__main__":
    main()
