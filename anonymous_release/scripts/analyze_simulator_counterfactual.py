#!/usr/bin/env python3
"""Phase 2–3 analysis + candidate figures from Phase 1 CSVs.

Does not re-simulate. Does not overwrite Phase 1 raw CSVs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "reviewer_strengthening" / ".mplconfig"))

CF = ROOT / "results" / "reviewer_strengthening" / "01_simulator_counterfactual"
DIST = ROOT / "results" / "reviewer_strengthening" / "02_action_distance"
BOOT = ROOT / "results" / "reviewer_strengthening" / "03_episode_bootstrap"
FIGS = ROOT / "results" / "reviewer_strengthening" / "paper_figures"
BOOT_SEED = 0
N_BOOT = 5000
NEAR_ZERO = 1e-8


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx.astype(np.float64)
    ry = ry.astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom <= 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _angle_deg(fx0: float, fy0: float, fx1: float, fy1: float) -> float:
    n0 = math.hypot(fx0, fy0)
    n1 = math.hypot(fx1, fy1)
    if n0 < 1e-12 or n1 < 1e-12:
        return float("nan")
    cos = max(-1.0, min(1.0, (fx0 * fx1 + fy0 * fy1) / (n0 * n1)))
    return math.degrees(math.acos(cos))


def action_distance(pairs: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    DIST.mkdir(parents=True, exist_ok=True)
    primary = [r for r in pairs if r["intervention"] == "shuffled_same_time"]
    rows: list[dict[str, Any]] = []
    for r in primary:
        dcx = _f(r, "alt_cx") - _f(r, "true_cx")
        dcy = _f(r, "alt_cy") - _f(r, "true_cy")
        dfx = _f(r, "alt_fx") - _f(r, "true_fx")
        dfy = _f(r, "alt_fy") - _f(r, "true_fy")
        contact = math.hypot(dcx, dcy)
        force = math.hypot(dfx, dfy)
        mag0 = math.hypot(_f(r, "true_fx"), _f(r, "true_fy"))
        mag1 = math.hypot(_f(r, "alt_fx"), _f(r, "alt_fy"))
        mag_abs = abs(mag1 - mag0)
        mag_rel = mag_abs / mag0 if mag0 > 1e-12 else float("nan")
        ang = _angle_deg(_f(r, "true_fx"), _f(r, "true_fy"), _f(r, "alt_fx"), _f(r, "alt_fy"))
        nrm = math.sqrt(
            sum(
                (_f(r, f"alt_{k}_n") - _f(r, f"true_{k}_n")) ** 2
                for k in ("cx", "cy", "fx", "fy")
            )
        )
        rows.append(
            {
                "episode": int(r["episode"]),
                "t": int(r["t"]),
                "command_change": int(r["command_change"]),
                "effective_contact": int(r["effective_contact"]),
                "contact_euclid": contact,
                "force_euclid": force,
                "force_mag_abs": mag_abs,
                "force_mag_rel": mag_rel,
                "force_angle_deg": ang,
                "force_cosine": math.nan if math.isnan(ang) else math.cos(math.radians(ang)),
                "action_l2_normalized": nrm,
            }
        )
    _write_csv(DIST / "action_pair_metrics.csv", rows)

    dist_rows = []
    true_only = [r for r in pairs if r["intervention"] == "true"]
    arr = np.array(
        [[_f(r, k) for k in ("true_cx", "true_cy", "true_fx", "true_fy")] for r in true_only],
        dtype=np.float64,
    )
    names = ["cx", "cy", "fx", "fy"]
    for i, name in enumerate(names):
        dist_rows.append(
            {
                "quantity": name,
                "mean": float(arr[:, i].mean()),
                "std": float(arr[:, i].std()),
                "q10": float(np.quantile(arr[:, i], 0.1)),
                "q50": float(np.quantile(arr[:, i], 0.5)),
                "q90": float(np.quantile(arr[:, i], 0.9)),
            }
        )
    mags = np.hypot(arr[:, 2], arr[:, 3])
    dist_rows.append(
        {
            "quantity": "force_magnitude",
            "mean": float(mags.mean()),
            "std": float(mags.std()),
            "q10": float(np.quantile(mags, 0.1)),
            "q50": float(np.quantile(mags, 0.5)),
            "q90": float(np.quantile(mags, 0.9)),
        }
    )
    _write_csv(DIST / "action_distribution.csv", dist_rows)
    return rows, {"n_pairs": len(rows)}


def merge_sensitivity(
    dist_rows: list[dict[str, Any]],
    phys: list[dict[str, str]],
    tgt: list[dict[str, str]],
    pred: list[dict[str, str]],
) -> list[dict[str, Any]]:
    phys_h1 = {
        (int(r["episode"]), int(r["t"])): r
        for r in phys
        if r["intervention"] == "shuffled_same_time" and r["horizon"] == "1"
    }
    tgt_h1: dict[tuple[int, int, str], dict[str, str]] = {}
    for r in tgt:
        if r["intervention"] == "shuffled_same_time" and r["horizon"] == "1":
            tgt_h1[(int(r["episode"]), int(r["t"]), r["checkpoint"])] = r
    pred_d: dict[tuple[int, int, str], dict[str, str]] = {}
    for r in pred:
        if r["intervention"] == "shuffled_same_time":
            pred_d[(int(r["episode"]), int(r["t"]), r["checkpoint"])] = r
    merged: list[dict[str, Any]] = []
    ckpts = sorted({r["checkpoint"] for r in pred if r["intervention"] == "shuffled_same_time"})
    for row in dist_rows:
        key = (row["episode"], row["t"])
        p = phys_h1.get(key)
        if p is None:
            continue
        for ckpt in ckpts:
            trow = tgt_h1.get((row["episode"], row["t"], ckpt))
            prow = pred_d.get((row["episode"], row["t"], ckpt))
            if trow is None or prow is None:
                continue
            tmse = float(trow["tgt_mse"])
            pmse = float(prow["pred_mse"])
            ratio = float("nan") if tmse <= NEAR_ZERO else pmse / tmse
            merged.append(
                {
                    **row,
                    "checkpoint": ckpt,
                    "rms_pos": float(p["rms_pos"]),
                    "mse_full": float(p["mse_full"]),
                    "tgt_mse": tmse,
                    "pred_mse": pmse,
                    "capture_ratio": ratio,
                    "ratio_excluded": int(tmse <= NEAR_ZERO),
                }
            )
    return merged


def binned_and_corr(merged: list[dict[str, Any]]) -> None:
    # quintiles on action_l2_normalized computed once on unique (episode,t), then applied
    unique = {}
    for r in merged:
        unique[(r["episode"], r["t"])] = r["action_l2_normalized"]
    vals = np.array(list(unique.values()), dtype=np.float64)
    edges = np.quantile(vals, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0] = min(edges[0], vals.min()) - 1e-12
    edges[-1] = max(edges[-1], vals.max()) + 1e-12

    def bin_id(x: float) -> int:
        return int(np.searchsorted(edges, x, side="right") - 1)

    binned_rows = []
    corr_rows = []
    ckpts = sorted({r["checkpoint"] for r in merged})
    for ckpt in ckpts:
        sub = [r for r in merged if r["checkpoint"] == ckpt]
        x = np.array([r["action_l2_normalized"] for r in sub])
        for yname in ("rms_pos", "tgt_mse", "pred_mse", "capture_ratio"):
            y = np.array([r[yname] for r in sub], dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            corr_rows.append(
                {
                    "checkpoint": ckpt,
                    "x": "action_l2_normalized",
                    "y": yname,
                    "spearman": _spearman(x[mask], y[mask]),
                    "n": int(mask.sum()),
                }
            )
        for b in range(5):
            chunk = [r for r in sub if bin_id(r["action_l2_normalized"]) == b]
            if not chunk:
                continue
            binned_rows.append(
                {
                    "checkpoint": ckpt,
                    "quintile": b + 1,
                    "n": len(chunk),
                    "action_l2_median": float(np.median([r["action_l2_normalized"] for r in chunk])),
                    "rms_pos_median": float(np.median([r["rms_pos"] for r in chunk])),
                    "tgt_mse_median": float(np.median([r["tgt_mse"] for r in chunk])),
                    "pred_mse_median": float(np.median([r["pred_mse"] for r in chunk])),
                    "edge_lo": float(edges[b]),
                    "edge_hi": float(edges[b + 1]),
                }
            )
    _write_csv(DIST / "binned_sensitivity.csv", binned_rows)
    _write_csv(DIST / "correlations.csv", corr_rows)
    (DIST / "summary.md").write_text(
        "# Action-distance summary\n\nQuintiles were defined on normalized action L2 "
        "**before** inspecting predictor trends.\n\n"
        + json.dumps({"n_merged": len(merged), "correlations": corr_rows[:12]}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def bootstrap(merged: list[dict[str, Any]], phys: list[dict[str, str]]) -> None:
    BOOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    per_ep_rows: list[dict[str, Any]] = []
    ckpts = sorted({r["checkpoint"] for r in merged})
    episodes = sorted({r["episode"] for r in merged})

    def subset_mask(r: dict[str, Any], mode: str) -> bool:
        if mode == "all":
            return True
        if mode == "command_change":
            return int(r["command_change"]) == 1
        if mode == "effective_contact":
            return int(r["effective_contact"]) == 1
        if mode == "effective_contact_command_change":
            return int(r["effective_contact"]) == 1 and int(r["command_change"]) == 1
        return False

    # per-episode means
    for ep in episodes:
        for ckpt in ckpts:
            for mode in ("all", "command_change", "effective_contact", "effective_contact_command_change"):
                rows = [r for r in merged if r["episode"] == ep and r["checkpoint"] == ckpt and subset_mask(r, mode)]
                if not rows:
                    continue
                per_ep_rows.append(
                    {
                        "episode": ep,
                        "checkpoint": ckpt,
                        "subset": mode,
                        "n": len(rows),
                        "rms_pos": float(np.mean([r["rms_pos"] for r in rows])),
                        "tgt_mse": float(np.mean([r["tgt_mse"] for r in rows])),
                        "pred_mse": float(np.mean([r["pred_mse"] for r in rows])),
                        "capture_ratio": float(
                            np.nanmean([r["capture_ratio"] for r in rows if r["ratio_excluded"] == 0])
                        ),
                    }
                )
    _write_csv(BOOT / "per_episode_metrics.csv", per_ep_rows)

    summary_rows = []
    payload: dict[str, Any] = {"n_boot": N_BOOT, "seed": BOOT_SEED, "unit": "episode"}
    for ckpt in ckpts:
        for mode in ("all", "command_change", "effective_contact", "effective_contact_command_change"):
            ep_rows = [r for r in per_ep_rows if r["checkpoint"] == ckpt and r["subset"] == mode]
            if len(ep_rows) < 2:
                continue
            eps = np.array([r["episode"] for r in ep_rows])
            metrics = {
                "rms_pos": np.array([r["rms_pos"] for r in ep_rows], dtype=np.float64),
                "tgt_mse": np.array([r["tgt_mse"] for r in ep_rows], dtype=np.float64),
                "pred_mse": np.array([r["pred_mse"] for r in ep_rows], dtype=np.float64),
                "capture_ratio": np.array([r["capture_ratio"] for r in ep_rows], dtype=np.float64),
            }
            n = len(ep_rows)
            for mname, arr in metrics.items():
                stats = []
                finite = np.isfinite(arr)
                if finite.sum() < 2:
                    continue
                for _ in range(N_BOOT):
                    idx = rng.integers(0, n, size=n)
                    stats.append(float(np.nanmean(arr[idx])))
                lo, hi = np.quantile(stats, [0.025, 0.975])
                row = {
                    "checkpoint": ckpt,
                    "subset": mode,
                    "metric": mname,
                    "n_episodes": n,
                    "point": float(np.nanmean(arr)),
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                }
                summary_rows.append(row)
                payload.setdefault(ckpt, {}).setdefault(mode, {})[mname] = {
                    k: row[k] for k in ("n_episodes", "point", "ci95_lo", "ci95_hi")
                }
    _write_csv(BOOT / "bootstrap_summary.csv", summary_rows)
    (BOOT / "bootstrap_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (BOOT / "bootstrap_protocol.md").write_text(
        f"# Episode bootstrap\n\nResamples: {N_BOOT}\nSeed: {BOOT_SEED}\n"
        "Unit: validation episode (transitions within an episode are not treated as independent).\n"
        "Statistic: mean of per-episode means.\n"
        "CI: percentile 95%.\n"
        "Do not interpret a tiny CI around a tiny effect as practical action dependence.\n",
        encoding="utf-8",
    )


def figures(merged: list[dict[str, Any]], phys: list[dict[str, str]], pred: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    CF.mkdir(parents=True, exist_ok=True)
    DIST.mkdir(parents=True, exist_ok=True)
    BOOT.mkdir(parents=True, exist_ok=True)

    def save(fig, stem: str, extra: Path | None = None) -> None:
        fig.tight_layout()
        for folder in (FIGS, extra):
            if folder is None:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / f"{stem}.pdf")
            fig.savefig(folder / f"{stem}.png", dpi=200)
        plt.close(fig)

    kinds_primary = ["shuffled_same_time", "neutral"]
    kinds_secondary = ["force_reversed", "force_rotated", "force_scaled", "contact_perturbed"]
    ckpts = sorted({r["checkpoint"] for r in merged})

    # Panel figure physical vs model
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    phys_h1 = [r for r in phys if r["horizon"] == "1" and r["intervention"] != "true"]
    labels = kinds_primary + kinds_secondary
    data = []
    for k in labels:
        data.append([float(r["rms_pos"]) for r in phys_h1 if r["intervention"] == k])
    axes[0, 0].boxplot(data, tick_labels=["shuf", "neut", "rev", "rot", "×1.5", "cx+r"], showfliers=False)
    axes[0, 0].set_ylabel("RMS particle position Δ")
    axes[0, 0].set_title("A. Physical divergence (h=1)")
    axes[0, 0].axhline(0.0, color="0.6", lw=0.5)

    # target latent: use merged tgt for shuffle only plus we need other kinds from tgt file...
    # Use phys-joined? We only merged shuffle. Plot shuffle/neutral from pred+ we'll skip tgt others if needed.
    tgt_path = CF / "target_latent_divergence_per_example.csv"
    tgt = _load(tgt_path)
    tgt_h1 = [r for r in tgt if r["horizon"] == "1" and r["intervention"] != "true"]
    ckpt0 = ckpts[0] if ckpts else "vicreg_nocrop_s42"
    data_t = []
    for k in labels:
        data_t.append(
            [float(r["tgt_mse"]) for r in tgt_h1 if r["intervention"] == k and r["checkpoint"] == ckpt0]
        )
    axes[0, 1].boxplot(data_t, tick_labels=["shuf", "neut", "rev", "rot", "×1.5", "cx+r"], showfliers=False)
    axes[0, 1].set_ylabel(f"Target latent MSE ({ckpt0})")
    axes[0, 1].set_title("B. Target-encoder future Δ")

    pred_rows = [r for r in pred if r["intervention"] != "true" and r["checkpoint"] == ckpt0]
    data_p = []
    for k in labels:
        data_p.append([float(r["pred_mse"]) for r in pred_rows if r["intervention"] == k])
    axes[1, 0].boxplot(data_p, tick_labels=["shuf", "neut", "rev", "rot", "×1.5", "cx+r"], showfliers=False)
    axes[1, 0].set_ylabel(f"Predictor MSE ({ckpt0})")
    axes[1, 0].set_title("C. Predictor true-vs-alt Δ")

    sub = [r for r in merged if r["checkpoint"] == ckpt0 and r["ratio_excluded"] == 0]
    axes[1, 1].scatter(
        [r["tgt_mse"] for r in sub],
        [r["pred_mse"] for r in sub],
        s=6,
        alpha=0.25,
        c=["C0" if r["effective_contact"] else "C1" for r in sub],
    )
    axes[1, 1].set_xlabel("Target future MSE (true vs shuffled)")
    axes[1, 1].set_ylabel("Predictor MSE (true vs shuffled)")
    axes[1, 1].set_title("D. Capture: pred Δ vs target Δ")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    save(fig, "figure_physical_vs_model_action_sensitivity", CF)

    # horizon figure
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for kind, ls in (("shuffled_same_time", "-"), ("neutral", "--")):
        xs, ys = [], []
        for h in (1, 3, 5):
            vals = [float(r["rms_pos"]) for r in phys if r["intervention"] == kind and r["horizon"] == str(h)]
            xs.append(h)
            ys.append(float(np.median(vals)) if vals else math.nan)
        ax.plot(xs, ys, ls, marker="o", label=kind.replace("_", " "))
    ax.set_xlabel("horizon (observation steps)")
    ax.set_ylabel("median RMS particle Δ")
    ax.set_title("Physical divergence vs horizon")
    ax.legend(fontsize=7)
    save(fig, "figure_counterfactual_horizon", CF)

    # contact stratification
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    cats = ["no/weak", "effective"]
    for i, kind in enumerate(("shuffled_same_time", "neutral")):
        vals = []
        for flag in (0, 1):
            chunk = [
                float(r["rms_pos"])
                for r in phys
                if r["intervention"] == kind and r["horizon"] == "1" and int(r["effective_contact"]) == flag
            ]
            vals.append(float(np.median(chunk)) if chunk else math.nan)
        ax.plot([0, 1], vals, marker="o", label=kind.replace("_", " "))
    ax.set_xticks([0, 1], cats)
    ax.set_ylabel("median RMS particle Δ (h=1)")
    ax.set_title("Contact stratification")
    ax.legend(fontsize=7)
    save(fig, "figure_contact_stratification", CF)

    # action distance vs response
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    sub = [r for r in merged if r["checkpoint"] == ckpt0]
    axes[0].scatter([r["action_l2_normalized"] for r in sub], [r["rms_pos"] for r in sub], s=5, alpha=0.2)
    axes[0].set_xlabel("normalized action L2")
    axes[0].set_ylabel("physical RMS Δ")
    axes[1].scatter([r["action_l2_normalized"] for r in sub], [r["tgt_mse"] for r in sub], s=5, alpha=0.2)
    axes[1].set_xlabel("normalized action L2")
    axes[1].set_ylabel("target latent MSE")
    axes[2].scatter([r["action_l2_normalized"] for r in sub], [r["pred_mse"] for r in sub], s=5, alpha=0.2)
    axes[2].set_xlabel("normalized action L2")
    axes[2].set_ylabel("predictor MSE")
    fig.suptitle(f"Action distance vs response ({ckpt0})", fontsize=9)
    save(fig, "figure_action_distance_vs_response", DIST)

    # bootstrap CI figure
    boot_csv = BOOT / "bootstrap_summary.csv"
    if boot_csv.is_file() and boot_csv.stat().st_size > 0:
        boot_rows = _load(boot_csv)
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        y = 0
        labels_y = []
        for row in boot_rows:
            if row["metric"] != "pred_mse":
                continue
            if row["subset"] not in {"all", "command_change", "effective_contact"}:
                continue
            point = float(row["point"])
            lo = float(row["ci95_lo"])
            hi = float(row["ci95_hi"])
            ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", capsize=3)
            labels_y.append(f"{row['checkpoint'].split('_')[-1]} {row['subset']}")
            y += 1
        ax.set_yticks(range(len(labels_y)), labels_y, fontsize=7)
        ax.set_xlabel("episode-mean predictor MSE (95% CI)")
        ax.set_title("Bootstrap CIs (episode unit)")
        save(fig, "figure_counterfactual_ci", BOOT)

    captions = FIGS / "captions.md"
    captions.write_text(
        "# Candidate figure captions\n\n"
        "**figure_physical_vs_model_action_sensitivity**: Same-state MLS-MPM forks. "
        "A: particle RMS position change under alternate actions. "
        "B: EMA target-encoder MSE between true and counterfactual futures. "
        "C: predictor output MSE under true vs alternate *normalized* actions, same z_t. "
        "D: predictor Δ vs target Δ for shuffled pairs (log-log). "
        "Primary interventions are shuffled-same-time and neutral; others are OOD diagnostics "
        "and are not pooled.\n\n"
        "**figure_counterfactual_horizon**: Median physical RMS vs horizon for primary interventions.\n\n"
        "**figure_contact_stratification**: Effective contact is min-distance ≤ 2 contact_radius, "
        "predeclared from the normalized Gaussian actuator.\n\n"
        "**figure_action_distance_vs_response**: Normalized four-D action L2 vs physical / target / predictor Δ.\n\n"
        "**figure_counterfactual_ci**: Percentile bootstrap CIs; resampling unit is the validation episode.\n",
        encoding="utf-8",
    )


def tex_tables(summary: dict[str, Any]) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Intervention & median RMS pos $\Delta$ ($h{=}1$) & median target MSE & median predictor MSE \\",
        r"\midrule",
    ]
    # filled later if keys exist
    (CF / "table_simulator_counterfactual.tex").write_text(
        "% Auto-generated candidate table. Not manuscript text.\n"
        + "\n".join(lines)
        + "\n% See aggregate_metrics.json for exact values.\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )
    (DIST / "table_action_distance.tex").write_text(
        "% See action_pair_metrics.csv / binned_sensitivity.csv / correlations.csv\n",
        encoding="utf-8",
    )


def statistical_summary(agg: dict[str, Any], merged: list[dict[str, Any]]) -> None:
    shuf = agg.get("median_rms_pos_h1", {}).get("shuffled_same_time", {})
    neut = agg.get("median_rms_pos_h1", {}).get("neutral", {})
    supported = []
    not_supported = []
    phys = shuf.get("all")
    if phys is not None and phys > 1e-6:
        supported.append(
            "The simulator state diverges under shuffled-same-time actions at h=1 "
            f"(median RMS particle position Δ = {phys})."
        )
    else:
        not_supported.append("Material physical response to shuffled actions at h=1.")
    for ckpt, pred in agg.get("median_pred_mse", {}).get("shuffled_same_time", {}).items():
        tgt = agg.get("median_tgt_mse_h1", {}).get("shuffled_same_time", {}).get(ckpt)
        ratio = agg.get("median_capture_ratio_h1", {}).get("shuffled_same_time", {}).get(ckpt)
        if pred is not None and tgt is not None and pred < 0.01 * max(tgt, 1e-12):
            supported.append(
                f"{ckpt}: predictor shuffled Δ ({pred}) is much smaller than target-future Δ ({tgt}); "
                f"median capture ratio {ratio}."
            )
        elif pred is not None and tgt is not None:
            not_supported.append(
                f"{ckpt}: predictor/target relationship is not a near-zero capture ratio "
                f"(pred={pred}, tgt={tgt}, ratio={ratio})."
            )
    (CF / "statistical_summary.md").write_text(
        "# Statistical summary (Phase 1)\n\n"
        f"N transitions: {agg.get('n_transitions')}\n"
        f"N command-change: {agg.get('n_command_change_transitions')}\n"
        f"N effective-contact: {agg.get('n_effective_contact_transitions')}\n\n"
        "## Conclusion supported\n\n"
        + "\n".join(f"- {s}" for s in supported)
        + "\n\n## Conclusion NOT supported\n\n"
        + ("\n".join(f"- {s}" for s in not_supported) if not_supported else "- (none listed from these medians)")
        + "\n\nSecondary OOD interventions are diagnostic and are not pooled with TRUE/SHUFFLED/NEUTRAL.\n",
        encoding="utf-8",
    )


def main() -> int:
    phys_path = CF / "physical_divergence_per_example.csv"
    if not phys_path.is_file():
        print("BLOCKED: Phase 1 CSVs missing", flush=True)
        return 2
    pairs = _load(CF / "counterfactual_pairs.csv")
    phys = _load(phys_path)
    tgt = _load(CF / "target_latent_divergence_per_example.csv")
    pred = _load(CF / "predictor_divergence_per_example.csv")
    agg = json.loads((CF / "aggregate_metrics.json").read_text(encoding="utf-8"))
    dist_rows, _ = action_distance(pairs)
    merged = merge_sensitivity(dist_rows, phys, tgt, pred)
    binned_and_corr(merged)
    bootstrap(merged, phys)
    figures(merged, phys, pred)
    tex_tables(agg)
    statistical_summary(agg, merged)
    print("PHASE 2/3 analysis complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
