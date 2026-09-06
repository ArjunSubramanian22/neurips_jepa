#!/usr/bin/env python3
"""Matched seed-42/43 eval for corrected no-crop VICReg (action + no-action).

Reuses ``evaluate_model`` / ``rollout_predictions`` from ``src/eval.py`` — the
same functions that wrote ``outputs/mpm_action_ablation.json``. Reports
``last.pt`` (final epoch), never ``best.pt``.

Writes ``outputs/corrected_seed43_audit.json`` only. Does not touch seed-42
run directories, ``probe_diagnostics.json``, or ``paper/``.

Variance-normalized MSE is not implemented in ``src/eval.py``; those fields
are recorded as null.

Action-counterfactual protocol (no prior script in this repo; this is a
minimal wrapper around ``rollout_predictions``):
* true / shuffled / neutral one-step latent MSE
* shuffle = a single episode permutation of ``model_actions`` (preserves the
  action marginal and the hold-3 temporal structure)
* neutral = zeros in z-scored action space (training-split mean action)
* change-boundary one-step MSE at t in {3, 6, 9, 12} (``t % hold == 0`` after
  the initial sample; see ``scripts/generate_trajectories.py``)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import seed_everything
from src.dataset import build_dataset
from src.eval import evaluate_model, effective_rank, horizon_mse, rollout_predictions
from src.train import apply_dataset_path, load_model_from_checkpoint, peek_checkpoint_epoch

LOGGER = logging.getLogger("ac_jepa.seed43_audit")
HORIZONS = (1, 5, 10)
CHANGE_TIMES = (3, 6, 9, 12)
SHUFFLE_SEED = 0

RUNS = (
    {
        "key": "vicreg_nocrop_s42",
        "label": "corrected no-crop + action",
        "seed": 42,
        "ckpt": Path("outputs/vicreg_nocrop/last.pt"),
        "expect_action": True,
    },
    {
        "key": "vicreg_nocrop_s43",
        "label": "corrected no-crop + action",
        "seed": 43,
        "ckpt": Path("outputs/vicreg_nocrop_seed43/last.pt"),
        "expect_action": True,
    },
    {
        "key": "vicreg_nocrop_noaction_s42",
        "label": "corrected no-crop + no-action",
        "seed": 42,
        "ckpt": Path("outputs/no_action/last.pt"),
        "expect_action": False,
    },
    {
        "key": "vicreg_nocrop_noaction_s43",
        "label": "corrected no-crop + no-action",
        "seed": 43,
        "ckpt": Path("outputs/vicreg_nocrop_noaction_seed43/last.pt"),
        "expect_action": False,
    },
)


def _mean(values: list[float] | None) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _mse_at(horizons: list[int], mse: list[float], k: int) -> float:
    return float(mse[horizons.index(k)])


def _pct_change(base: float, other: float) -> float:
    if abs(base) < 1e-12:
        return float("nan")
    return 100.0 * (other - base) / base


def _mean_half_range(a: float, b: float) -> dict[str, float]:
    return {"mean": 0.5 * (a + b), "half_range": 0.5 * abs(a - b)}


def _one_step(
    model: Any,
    observations: Tensor,
    actions: Tensor,
    device: torch.device,
    start_frame: int,
    batch_size: int = 8,
) -> tuple[Tensor, Tensor, float]:
    z_hat, z_tgt = rollout_predictions(
        model,
        observations,
        actions,
        max_horizon=1,
        device=device,
        batch_size=batch_size,
        start_frame=start_frame,
    )
    mse = float(horizon_mse(z_hat, z_tgt)[0])
    return z_hat, z_tgt, mse


def action_counterfactual_audit(
    model: Any,
    observations: Tensor,
    actions: Tensor,
    device: torch.device,
    *,
    batch_size: int = 8,
    shuffle_seed: int = SHUFFLE_SEED,
    change_times: tuple[int, ...] = CHANGE_TIMES,
) -> dict[str, Any]:
    """True / shuffled / neutral one-step audit on the existing rollout op."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(shuffle_seed))
    perm = torch.randperm(actions.shape[0], generator=generator)
    shuffled = actions[perm]
    neutral = torch.zeros_like(actions)

    z_true, _, mse_true = _one_step(model, observations, actions, device, 0, batch_size)
    z_shuf, _, mse_shuf = _one_step(model, observations, shuffled, device, 0, batch_size)
    _, _, mse_neut = _one_step(model, observations, neutral, device, 0, batch_size)
    pred_pred = float((z_true - z_shuf).pow(2).mean().item())

    change_rows: list[dict[str, Any]] = []
    for t in change_times:
        if t >= int(observations.shape[1]) - 1:
            continue
        zt, _, mt = _one_step(model, observations, actions, device, t, batch_size)
        zs, _, ms = _one_step(model, observations, shuffled, device, t, batch_size)
        _, _, mn = _one_step(model, observations, neutral, device, t, batch_size)
        change_rows.append(
            {
                "t": int(t),
                "true_mse": mt,
                "shuffled_mse": ms,
                "neutral_mse": mn,
                "shuffle_pct_change": _pct_change(mt, ms),
                "neutral_pct_change": _pct_change(mt, mn),
                "pred_to_pred_mse": float((zt - zs).pow(2).mean().item()),
            }
        )

    def _avg(key: str) -> float | None:
        vals = [float(row[key]) for row in change_rows]
        return None if not vals else float(sum(vals) / len(vals))

    return {
        "protocol": {
            "shuffle": "episode permutation of model_actions; one shared perm for all t",
            "neutral": "zeros in z-scored action space (training-split mean)",
            "one_step": "rollout_predictions start_frame=t, max_horizon=1",
            "change_times": list(change_times),
            "shuffle_seed": int(shuffle_seed),
            "prior_script": None,
            "note": (
                "No counterfactual-action helper existed in src/eval.py or paper macros. "
                "This wraps rollout_predictions only."
            ),
        },
        "all_t0_one_step": {
            "true_mse": mse_true,
            "shuffled_mse": mse_shuf,
            "neutral_mse": mse_neut,
            "shuffle_pct_change": _pct_change(mse_true, mse_shuf),
            "neutral_pct_change": _pct_change(mse_true, mse_neut),
            "pred_to_pred_mse": pred_pred,
        },
        "change_boundaries": change_rows,
        "change_boundary_mean": {
            "true_mse": _avg("true_mse"),
            "shuffled_mse": _avg("shuffled_mse"),
            "neutral_mse": _avg("neutral_mse"),
            "shuffle_pct_change": _avg("shuffle_pct_change"),
            "neutral_pct_change": _avg("neutral_pct_change"),
            "pred_to_pred_mse": _avg("pred_to_pred_mse"),
        },
    }


def summarize_eval(payload: dict[str, Any], counterfactual: dict[str, Any] | None) -> dict[str, Any]:
    horizons = [int(h) for h in payload["horizons"]]
    mse = [float(v) for v in payload["rollout_mse"]]
    rank = payload["effective_rank"]
    all_cf = None if counterfactual is None else counterfactual["all_t0_one_step"]
    ch_cf = None if counterfactual is None else counterfactual["change_boundary_mean"]
    return {
        "mse_k": {str(k): _mse_at(horizons, mse, k) for k in HORIZONS},
        "variance_normalized_mse_k": {str(k): None for k in HORIZONS},
        "effective_rank_z_target": rank["z_target"]["participation_ratio"],
        "effective_rank_z_hat": rank["z_hat"]["participation_ratio"],
        "eig_frac_top1_z_hat": rank["z_hat"]["eig_frac_top1"],
        "eig_frac_top1_z_target": rank["z_target"]["eig_frac_top1"],
        "geometry_val_mean": _mean(payload.get("geometry_probe_acc")),
        "stiffness_val_mean": _mean(payload.get("stiffness_probe_acc")),
        "contact_val_mean": _mean(payload.get("contact_probe_acc")),
        "shuffle_delta_all": None if all_cf is None else all_cf["shuffle_pct_change"],
        "shuffle_delta_change": None if ch_cf is None else ch_cf["shuffle_pct_change"],
        "neutral_delta_all": None if all_cf is None else all_cf["neutral_pct_change"],
        "pred_to_pred_mse_all": None if all_cf is None else all_cf["pred_to_pred_mse"],
    }


def eval_checkpoint(
    spec: dict[str, Any],
    dataset_path: str,
    device: torch.device,
    *,
    probe_epochs: int,
    batch_size: int,
    run_counterfactual: bool,
) -> dict[str, Any]:
    ckpt = Path(spec["ckpt"])
    epoch = peek_checkpoint_epoch(ckpt)
    if epoch < 200:
        raise RuntimeError(f"{ckpt} epoch={epoch}; need final-epoch 200 last.pt")
    seed_everything(0)
    model, cfg = load_model_from_checkpoint(ckpt, device)
    cond = bool(getattr(model.predictor, "condition_on_action", True))
    if cond != bool(spec["expect_action"]):
        raise RuntimeError(
            f"{ckpt}: condition_on_action={cond}, expected {spec['expect_action']}"
        )
    cfg = apply_dataset_path(cfg, dataset_path, num_workers=0)
    cfg.training.device = str(device)
    dataset = build_dataset(cfg.data, split="all")
    payload = evaluate_model(
        model,
        dataset,
        device,
        max_horizon=10,
        probe_epochs=probe_epochs,
        batch_size=batch_size,
        start_frames=(0,),
    )
    counterfactual = None
    if run_counterfactual:
        counterfactual = action_counterfactual_audit(
            model,
            dataset.observations,
            dataset.model_actions,
            device,
            batch_size=batch_size,
        )
    summary = summarize_eval(payload, counterfactual)
    return {
        "key": spec["key"],
        "label": spec["label"],
        "seed": spec["seed"],
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_epoch": epoch,
        "condition_on_action": cond,
        "config": cfg.to_dict(),
        "n_train": payload["n_train"],
        "n_val": payload["n_val"],
        "start_frame": payload["start_frame"],
        "summary": summary,
        "horizons": payload["horizons"],
        "rollout_mse": payload["rollout_mse"],
        "effective_rank": payload["effective_rank"],
        "geometry_probe_acc": payload.get("geometry_probe_acc"),
        "stiffness_probe_acc": payload.get("stiffness_probe_acc"),
        "contact_probe_acc": payload.get("contact_probe_acc"),
        "counterfactual": counterfactual,
        "variance_normalized_mse_supported": False,
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def print_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Model | Seed | MSE@1 | MSE@5 | MSE@10 | Pred. eff. rank | Top-1 | "
        "Contact | Stiffness | Geometry | Shuffle Δ all | Shuffle Δ change |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        s = row["summary"]
        lines.append(
            "| {label} | {seed} | {m1} | {m5} | {m10} | {rank} | {top1} | {c} | {st} | {g} | {d_all} | {d_ch} |".format(
                label=row["label"],
                seed=row["seed"],
                m1=_fmt(s["mse_k"]["1"], 4),
                m5=_fmt(s["mse_k"]["5"], 3),
                m10=_fmt(s["mse_k"]["10"], 3),
                rank=_fmt(s["effective_rank_z_hat"], 2),
                top1=_fmt(s["eig_frac_top1_z_hat"], 3),
                c=_fmt(s["contact_val_mean"], 3),
                st=_fmt(s["stiffness_val_mean"], 3),
                g=_fmt(s["geometry_val_mean"], 3),
                d_all=_fmt(s["shuffle_delta_all"], 2),
                d_ch=_fmt(s["shuffle_delta_change"], 2),
            )
        )
    text = "\n".join(lines)
    print(text, flush=True)
    return text


def two_seed_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["label"], []).append(row)
    out: dict[str, Any] = {}
    metric_keys = (
        ("mse_1", lambda s: s["mse_k"]["1"]),
        ("mse_5", lambda s: s["mse_k"]["5"]),
        ("mse_10", lambda s: s["mse_k"]["10"]),
        ("pred_eff_rank", lambda s: s["effective_rank_z_hat"]),
        ("target_eff_rank", lambda s: s["effective_rank_z_target"]),
        ("top1", lambda s: s["eig_frac_top1_z_hat"]),
        ("contact", lambda s: s["contact_val_mean"]),
        ("stiffness", lambda s: s["stiffness_val_mean"]),
        ("geometry", lambda s: s["geometry_val_mean"]),
        ("shuffle_delta_all", lambda s: s["shuffle_delta_all"]),
        ("shuffle_delta_change", lambda s: s["shuffle_delta_change"]),
    )
    for label, family in by_family.items():
        if len(family) != 2:
            continue
        a, b = family[0]["summary"], family[1]["summary"]
        out[label] = {
            key: _mean_half_range(float(fn(a)), float(fn(b)))
            for key, fn in metric_keys
            if fn(a) is not None and fn(b) is not None
        }
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="outputs/physics_mpm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/corrected_seed43_audit.json")
    parser.add_argument("--probe-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    for spec in RUNS:
        LOGGER.info("eval %s %s", spec["key"], spec["ckpt"])
        rows.append(
            eval_checkpoint(
                spec,
                args.dataset_path,
                device,
                probe_epochs=args.probe_epochs,
                batch_size=args.batch_size,
                run_counterfactual=True,
            )
        )
    table = print_table(rows)
    aggregates = two_seed_aggregates(rows)
    document = {
        "eval_device": str(device),
        "dataset_path": args.dataset_path,
        "checkpoint_policy": "last.pt final epoch, not best.pt",
        "mse_scope": (
            "evaluate_model averages rollout MSE over all episodes (train+val), "
            "matching outputs/mpm_action_ablation.json; rank and probes use val"
        ),
        "variance_normalized_mse": "not implemented in src/eval.py",
        "table_markdown": table,
        "two_seed_mean_half_range": aggregates,
        "runs": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    LOGGER.info("wrote %s", out)
    print("AUDIT_COMPLETE", out, flush=True)
    return document


if __name__ == "__main__":
    main()
