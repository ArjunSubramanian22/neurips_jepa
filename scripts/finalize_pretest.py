#!/usr/bin/env python3
"""Write the pre-test findings and freeze files from completed validation outputs.

Does not read episodes 1620-1799 and does not launch the test script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MULTI = ROOT / "experiments" / "runs" / "mpm_high_multih"
FIG = ROOT / "experiments" / "runs" / "mpm_high_figures"


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
            else:
                try:
                    if key in {"model", "regime", "protocol", "seeds", "bottleneck", "cosine_status", "majority_bottleneck"}:
                        continue
                    if "." in value or "e" in value.lower():
                        row[key] = float(value)
                    else:
                        row[key] = int(value)
                except ValueError:
                    pass
    return rows


def _git() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _seed_lines(table, regime, model, protocol, horizon, key) -> str:
    chosen = [
        row for row in table
        if row["regime"] == regime and row["model"] == model and row["protocol"] == protocol and int(row["horizon"]) == horizon
    ]
    return ", ".join(f"{row['seed']}={_fmt(row[key])}" for row in chosen)


def _plot_planning(summary: list[dict], seed_table: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {"vicreg_action": "#1b4f72", "mse_action": "#b9770e", "rssm": "#196f3d", "vicreg_noaction": "#7b241c"}
    for model, color in colors.items():
        for horizon, marker in ((5, "o"), (10, "s")):
            xs, ys = [], []
            for row in summary:
                if row["model"] == model and int(row["horizon"]) == horizon:
                    xs.append(int(row["seed"]))
                    ys.append(float(row["top1"]))
            if xs:
                ax.scatter(xs, ys, color=color, marker=marker, label=f"{model} H{horizon}")
    ax.axhline(0.125, color="#999999", linewidth=0.8, label="chance 1/8")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Top-1 recovery")
    ax.set_title("Planning on validation episodes 1440-1463")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "figure5_planning.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for model, color in colors.items():
        xs, ys = [], []
        for row in summary:
            if row["model"] != model or int(row["horizon"]) != 5:
                continue
            cosine_row = [
                item for item in seed_table
                if item["regime"] == "mpm_high" and item["model"] == model and int(item["seed"]) == int(row["seed"])
                and item["protocol"] == "open_loop" and int(item["horizon"]) == 1
            ]
            if cosine_row and cosine_row[0]["cosine"] is not None:
                xs.append(float(cosine_row[0]["cosine"]))
                ys.append(float(row["top1"]))
                ax.scatter(xs[-1], ys[-1], color=color, label=model if len(xs) == 1 else None)
        if len(xs) >= 2:
            ax.plot(xs, ys, color=color, alpha=0.4)
    ax.set_xlabel("H1 open-loop cosine")
    ax.set_ylabel("H5 planning top-1")
    ax.set_title("Alignment versus planning, one point per seed")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "figure6_alignment_vs_planning.png", dpi=160)
    plt.close(fig)


def _plot_arm(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = {"jepa": "#1b4f72", "rssm": "#196f3d", "latent_tf": "#6c3483"}
    for model, color in colors.items():
        for seed in (42, 43, 44, 45, 46):
            xs, ys = [], []
            for horizon in (1, 7):
                group = [row for row in rows if row["model"] == model and int(row["seed"]) == seed and int(row["horizon"]) == horizon]
                values = [row["cosine"] for row in group if row["cosine"] is not None]
                if values:
                    xs.append(horizon)
                    ys.append(float(np.median(values)))
            if xs:
                ax.plot(xs, ys, color=color, marker="o", alpha=0.8, label=model if seed == 42 else None)
    ax.set_xticks([1, 7])
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Median cosine over 8 validation episodes")
    ax.set_title("Arm validation, reflected joint targets")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "figure7_arm.png", dpi=160)
    plt.close(fig)


def main() -> None:
    required = [
        MULTI / "seed_summary.csv",
        MULTI / "legacy_vs_high.csv",
        ROOT / "experiments/runs/mpm_high_bottleneck/bottleneck_summary.csv",
        ROOT / "experiments/runs/mpm_high_planning/seed_summary.csv",
        ROOT / "experiments/runs/arm_counterfactual/rows.csv",
        ROOT / "experiments/runs/rssm_h1_audit/audit.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing pre-test outputs: " + ", ".join(missing))
    table = _read(MULTI / "seed_summary.csv")
    headline = _read(MULTI / "legacy_vs_high.csv")
    planning = _read(ROOT / "experiments/runs/mpm_high_planning/seed_summary.csv")
    arm = _read(ROOT / "experiments/runs/arm_counterfactual/rows.csv")
    audit = json.loads((ROOT / "experiments/runs/rssm_h1_audit/audit.json").read_text(encoding="utf-8"))
    visibility = json.loads((MULTI / "visibility.json").read_text(encoding="utf-8"))
    _plot_planning(planning, table)
    _plot_arm(arm)
    git = _git()
    findings = _findings(table, headline, planning, arm, audit, visibility, git)
    (ROOT / "experiments/PRETEST_FINDINGS.md").write_text(findings, encoding="utf-8")
    freeze_md = ROOT / "experiments/MPM_HIGH_PROTOCOL_FREEZE.md"
    freeze_json = ROOT / "experiments/mpm_high_protocol_freeze.json"
    freeze_json.write_text(json.dumps(_freeze_payload(git), indent=2) + "\n", encoding="utf-8")
    freeze_md.write_text(_freeze_md(git), encoding="utf-8")
    hashes = {"markdown_sha256": _sha(freeze_md), "json_sha256": _sha(freeze_json), "git_commit": git}
    (ROOT / "experiments/runs/mpm_high_protocol_freeze_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    # Re-hash after the hash file is separate. The two freeze documents are already final.
    print("READY_FOR_FINAL_TEST")
    print("TEST EPISODES 1620-1799 REMAIN UNOPENED")
    print("PROTOCOL HASH:", hashes["markdown_sha256"])
    print("JSON HASH:", hashes["json_sha256"])
    print("GIT COMMIT:", git)


def _findings(table, headline, planning, arm, audit, visibility, git) -> str:
    lines = [
        "# Pre-test findings",
        "",
        f"Git commit `{git}`. High-excitation test episodes 1620-1799 were not scored.",
        "",
        "## RSSM audit",
        "",
        audit["verdict"]["reason"],
        f"Concat cosines: {audit['verdict']['concat_cosine_medians']}.",
        f"Stochastic-mean cosines: {audit['verdict']['z_cosine_medians']}.",
        f"Shuffle-versus-original-target stochastic cosines: {audit['verdict']['shuffle_z_cosine_medians']}.",
        "The published H1 screen files were not rewritten. Later RSSM rows use the stochastic mean.",
        "",
        "## Horizon results, high excitation, open loop",
        "",
    ]
    for model in ("vicreg_action", "vicreg_noaction", "mse_action", "rssm"):
        for horizon in (1, 5, 10):
            lines.append(f"- {model} H{horizon} cosine: {_seed_lines(table, 'mpm_high', model, 'open_loop', horizon, 'cosine')}")
            lines.append(f"- {model} H{horizon} projection: {_seed_lines(table, 'mpm_high', model, 'open_loop', horizon, 'projection')}")
            lines.append(f"- {model} H{horizon} response ratio: {_seed_lines(table, 'mpm_high', model, 'open_loop', horizon, 'response_ratio')}")
        lines.append("")
    lines.extend(["## Open loop versus teacher forced", ""])
    for model in ("vicreg_action", "mse_action", "rssm", "vicreg_noaction"):
        for horizon in (1, 5, 10):
            lines.append(
                f"- {model} H{horizon} teacher-forced cosine: {_seed_lines(table, 'mpm_high', model, 'teacher_forced', horizon, 'cosine')}"
            )
        lines.append("")
    lines.extend([
        "For a one-step feedforward predictor, open loop and teacher forcing use the same state, so H=1 matches.",
        "Teacher-forced no-action contrast at H>1 can be nonzero because the two forks have different observed states. That is not action-argument sensitivity. The same-state action norm is the action-pathway check.",
        "",
        "## Legacy versus high excitation",
        "",
        "These are two controlled simulator regimes, not a randomized treatment.",
        "",
    ])
    for row in headline:
        lines.append(
            f"- {row['regime']} H{row['horizon']}: physical {_fmt(row['physical_effect'])}, "
            f"raster {_fmt(row['observation_visibility'])}, target {_fmt(row['target_contrast'])}, "
            f"predictor {_fmt(row['predictor_contrast'])}, cosine {_fmt(row['cosine'])}, "
            f"ratio {_fmt(row['response_ratio'])}"
        )
    lines.extend(["", "## Planning", ""])
    for row in planning:
        lines.append(
            f"- {row['model']} seed {row['seed']} H{row['horizon']}: top1 {_fmt(row['top1'])} "
            f"top3 {_fmt(row['top3'])} regret {_fmt(row['mean_regret'])}"
        )
    lines.extend(["", "## Arm", "", "Reflected joint-position targets on episodes 24-31. Not an MPM force reversal.", ""])
    for model in ("jepa", "rssm", "latent_tf"):
        for horizon in (1, 7):
            values = []
            for seed in (42, 43, 44, 45, 46):
                group = [row for row in arm if row["model"] == model and int(row["seed"]) == seed and int(row["horizon"]) == horizon]
                defined = [row["cosine"] for row in group if row["cosine"] is not None]
                values.append(None if not defined else float(np.median(defined)))
            lines.append(f"- {model} H{horizon} episode-median cosines by seed: {values}")
    lines.extend([
        "",
        "## Visibility",
        "",
        f"Top-third raster-MSE threshold, fit once on the physical forks: {visibility['threshold']}.",
        "Seed-level rank correlations are in experiments/runs/mpm_high_multih/visibility.json.",
        "",
        "## Limitations",
        "",
        "- The original H1 screen used 24 episodes at t=0. The multi-horizon table uses 180 validation episodes and six branch times, aggregated within episode.",
        "- JEPA seeds 42 and 43 and no-action seed 42 still lack episode-level bootstrap intervals in the old 24-episode screen. The new table bootstraps episodes.",
        "- RSSM concat cosine at H=1 is not an independent measurement. The stochastic mean is the reported RSSM cosine.",
        "- Legacy checkpoints do not include seed 46 or an RSSM. The regime table uses JEPA action seeds 42-45.",
        "- Legacy validation was used to choose historical checkpoints. It is a regime comparison, not a sealed test.",
        "- Planning uses 24 validation episodes and eight training actions. Five seeds is a small replicate for the alignment-versus-planning correlation.",
        "- The arm has eight validation episodes and no held-out test. No sealed arm dump was generated.",
        "- No-action teacher-forced multi-step contrast is not a pure action-null.",
        "",
    ])
    return "\n".join(lines)


def _freeze_payload(git: str) -> dict:
    return {
        "git_commit": git,
        "metric_version": "mpm_multih_v1_rssm_z",
        "hyperparameter_selection": False,
        "episodes": list(range(1620, 1800)),
        "validation_episodes": list(range(1440, 1620)),
        "branch_times": [0, 1, 2, 3, 4, 5],
        "horizons": [1, 5, 10],
        "intervention": "force reversal of action components 2:",
        "rssm_primary_vector": "stochastic_mean",
        "bottleneck_thresholds": {"phys_tau": 1e-5, "norm_tau": 1e-8, "success_cosine": 0.5, "success_ratio": 0.2},
        "undefined_cosine": "null, never zero",
        "aggregation": "median over branch times within episode, then bootstrap episodes, then report seed medians",
        "bootstrap_resamples": 10000,
        "test_scored": False,
        "statement": "High-excitation test episodes 1620-1799 have not been scored.",
        "models": ["vicreg_action", "vicreg_noaction", "mse_action", "rssm"],
        "seeds": [42, 43, 44, 45, 46],
        "protocols": ["open_loop", "teacher_forced"],
        "planning_library_episodes": [0, 80, 160, 240, 320, 400, 480, 560],
        "planning_validation_episodes": list(range(1440, 1464)),
        "planning_horizons": [5, 10],
    }


def _freeze_md(git: str) -> str:
    return "\n".join([
        "# High-excitation protocol freeze",
        "",
        f"Git commit `{git}`.",
        "",
        "High-excitation test episodes 1620–1799 have not been scored.",
        "",
        "Validation episodes are 1440-1619. Branch times are 0, 1, 2, 3, 4, and 5. Horizons are 1, 5, and 10.",
        "The intervention reverses action components 2: for the whole horizon.",
        "Open loop rolls the model state forward. Teacher forcing refreshes the state from the physical fork.",
        "RSSM cosine uses the stochastic mean. The deterministic state is reported separately because it is shared at one step.",
        "Undefined cosine stays null.",
        "Branch times are aggregated inside each episode before the episode bootstrap. Seeds are the replication unit.",
        "Bottleneck thresholds: position RMS 1e-5, target norm 1e-8, success cosine 0.5, success response ratio 0.2.",
        "Visibility uses one physical-fork threshold: all forks, non-identical rasters, and the top third of nonzero raster MSE.",
        "Planning uses eight actions from training episodes 0, 80, 160, 240, 320, 400, 480, and 560, on validation episodes 1440-1463.",
        "No architecture was selected on the test set. FiLM, gate, cross-attention, and alignment training stayed deferred.",
        "",
    ])


if __name__ == "__main__":
    main()
