#!/usr/bin/env python3
"""Phase 8: action persistence and incremental linear information for z_{t+1}."""

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

OUT = ROOT / "results" / "reviewer_strengthening" / "08_action_identifiability"
DUMP = ROOT / "outputs" / "physics_mpm"
CKPT = ROOT / "outputs" / "vicreg_nocrop" / "last.pt"
HOLD = 3


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    d = x.shape[1]
    xtx = x.T @ x + lam * np.eye(d, dtype=np.float64)
    xty = x.T @ y
    return np.linalg.solve(xtx, xty)


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((y - yhat) ** 2))


def main() -> None:
    from src.train import load_model_from_checkpoint

    OUT.mkdir(parents=True, exist_ok=True)
    actions = np.load(DUMP / "actions.npy")
    n_ep, n_t, _ = actions.shape
    n_val = 200
    n_train = n_ep - n_val
    persist = []
    for e in range(n_ep):
        for t in range(1, n_t):
            persist.append(float(np.allclose(actions[e, t], actions[e, t - 1])))
    persist_frac = float(np.mean(persist))
    change_frac = 1.0 - persist_frac

    device = torch.device("cpu")
    model, cfg = load_model_from_checkpoint(CKPT, device)
    model.eval()
    from src.config import load_experiment_config
    from src.dataset import build_dataset
    from src.train import apply_dataset_path

    econf = load_experiment_config(ROOT / "configs" / "workshop_vicreg.yaml")
    econf = apply_dataset_path(econf, str(DUMP), num_workers=0)
    dataset = build_dataset(econf.data, split="all")
    obs = dataset.observations
    act_n = dataset.model_actions.numpy()
    zs = []
    with torch.no_grad():
        for start in range(0, obs.shape[0], 8):
            sl = obs[start : start + 8]
            b, t, c, h, w = sl.shape
            flat = sl.reshape(b * t, c, h, w)
            z = model.encode(flat.to(device), target=True).cpu()
            zs.append(z.reshape(b, t, -1).numpy())
    z = np.concatenate(zs, axis=0)
    rows = []
    coeff_rows = []
    for split, sl in (("train", slice(0, n_train)), ("val", slice(n_train, n_ep))):
        z_t, z_tp1, a = [], [], []
        change = []
        for e in range(sl.start, sl.stop):
            for t in range(n_t):
                z_t.append(z[e, t])
                z_tp1.append(z[e, t + 1])
                a.append(act_n[e, t])
                change.append(int(t in (3, 6, 9, 12)))
        z_t = np.asarray(z_t, dtype=np.float64)
        z_tp1 = np.asarray(z_tp1, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        change = np.asarray(change)
        if split == "train":
            w_z = ridge_fit(np.concatenate([z_t, np.ones((z_t.shape[0], 1))], 1), z_tp1)
            w_a = ridge_fit(np.concatenate([a, np.ones((a.shape[0], 1))], 1), z_tp1)
            w_za = ridge_fit(np.concatenate([z_t, a, np.ones((z_t.shape[0], 1))], 1), z_tp1)
            np.savez(OUT / "ridge_weights.npz", w_z=w_z, w_a=w_a, w_za=w_za)
        else:
            data = np.load(OUT / "ridge_weights.npz")
            w_z, w_a, w_za = data["w_z"], data["w_a"], data["w_za"]
        pred_z = np.concatenate([z_t, np.ones((z_t.shape[0], 1))], 1) @ w_z
        pred_a = np.concatenate([a, np.ones((a.shape[0], 1))], 1) @ w_a
        pred_za = np.concatenate([z_t, a, np.ones((z_t.shape[0], 1))], 1) @ w_za
        for name, pred in (("z_only", pred_z), ("a_only", pred_a), ("z_and_a", pred_za)):
            rows.append({"split": split, "subset": "all", "model": name, "mse": mse(z_tp1, pred)})
            mask = change == 1
            rows.append({"split": split, "subset": "command_change", "model": name, "mse": mse(z_tp1[mask], pred[mask])})
            rows.append({"split": split, "subset": "held_action", "model": name, "mse": mse(z_tp1[~mask], pred[~mask])})
        if split == "train":
            coeff_rows.append({"matrix": "w_za", "n_rows": int(w_za.shape[0]), "n_cols": int(w_za.shape[1]),
                               "action_block_fro": float(np.linalg.norm(w_za[z_t.shape[1]: z_t.shape[1] + 4]))})
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "subset", "model", "mse"])
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "model_coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coeff_rows[0].keys()) if coeff_rows else ["matrix"])
        writer.writeheader()
        writer.writerows(coeff_rows)
    val_all = {r["model"]: r["mse"] for r in rows if r["split"] == "val" and r["subset"] == "all"}
    (OUT / "summary.md").write_text(
        "# Action identifiability\n\n"
        f"Fraction of transitions with action == previous command: {persist_frac:.6f}\n"
        f"Complement (command changes, including t=0 first sample): {change_frac:.6f}\n\n"
        f"Val ridge MSE z_only={val_all.get('z_only')} a_only={val_all.get('a_only')} "
        f"z_and_a={val_all.get('z_and_a')}\n"
        "Incremental value of action ≈ z_only - z_and_a on frozen target latents. "
        "Linear ridge is a diagnostic, not an optimal dynamical model.\n",
        encoding="utf-8",
    )
    print(json.dumps({"persist_frac": persist_frac, "val": val_all}, indent=2), flush=True)


if __name__ == "__main__":
    main()
