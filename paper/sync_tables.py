#!/usr/bin/env python3
"""Emit LaTeX fragments + macros from generated artifacts.

Run from repo root or paper/:

    python paper/sync_tables.py

Writes paper/generated/*.tex. The workshop source \\input{}s those files so
table cells cannot drift from outputs/*.csv and outputs/*.json.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DEST = Path(__file__).resolve().parent / "generated"

BASELINE_HORIZONS = (1, 5, 10)
ABLATION_HORIZONS = (1, 5, 10, 20)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _fmt_mse(value: float) -> str:
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _fmt_acc_pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _completed_epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    best = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("split") == "train_epoch":
                best = max(best, int(rec["epoch"]))
    return best


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _add_rank_macros(values: dict[str, str], probes: dict) -> None:
    models = probes.get("models", {})
    for name, prefix in (("vicreg_physics_mpm", "Vicreg"), ("mse_physics_mpm", "Mse")):
        blob = (models.get(name) or {}).get("effective_rank") or {}
        for view, tag in (("z_target", "Tgt"), ("z_hat", "Hat")):
            pr = (blob.get(view) or {}).get("participation_ratio")
            key = f"EffRank{tag}{prefix}"
            values[key] = f"{float(pr):.2f}" if pr is not None else "NA"
            dim = (blob.get(view) or {}).get("latent_dim")
            values[f"EffRankDim{tag}{prefix}"] = str(int(dim)) if dim is not None else values.get("ZTLatentDim", "NA")


def _add_twoseed_macros(values: dict[str, str]) -> None:
    path = OUT / "ablation_twoseed.json"
    horizon_word = {1: "One", 5: "Five", 10: "Ten"}
    if not path.is_file():
        for prefix in ("TSVicreg", "TSMse"):
            for k, word in horizon_word.items():
                values[f"{prefix}K{word}Mean"] = "NA"
                values[f"{prefix}K{word}Range"] = "NA"
        return
    blob = json.loads(path.read_text(encoding="utf-8"))
    mapping = {"full_vicreg": "TSVicreg", "mse": "TSMse"}
    for key, prefix in mapping.items():
        by_h = ((blob.get("variants") or {}).get(key) or {}).get("by_horizon") or {}
        for k, word in horizon_word.items():
            cell = by_h.get(str(k)) or {}
            mean = cell.get("rollout_mse_mean")
            rng = cell.get("rollout_mse_range")
            values[f"{prefix}K{word}Mean"] = _fmt_mse(float(mean)) if mean is not None else "NA"
            values[f"{prefix}K{word}Range"] = _fmt_mse(float(rng)) if rng is not None else "NA"


def _macro(name: str, body: str) -> str:
    return rf"\newcommand{{\{name}}}{{{body}}}"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_macros() -> dict[str, str]:
    meta = json.loads((OUT / "physics_mpm" / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "dataset_report.json").read_text(encoding="utf-8"))
    probes = json.loads((OUT / "probe_diagnostics.json").read_text(encoding="utf-8"))
    physics = probes["models"]["vicreg_physics_mpm"]
    z_target = physics["z_target"]

    baseline_rows = load_csv_rows(OUT / "baseline_comparison.csv")
    by_method: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    labels: dict[str, str] = {}
    n_train = n_val = None
    for row in baseline_rows:
        method = row["method"]
        horizon = int(row["horizon"])
        labels[method] = row["method_label"]
        by_method[method][horizon] = {
            "rollout_mse": float(row["rollout_mse"]),
            "stiffness": float(row["stiffness_probe_acc"]),
            "contact": float(row["contact_probe_acc"]),
        }
        n_train = int(float(row["n_train"]))
        n_val = int(float(row["n_val"]))

    ablation_rows = load_csv_rows(OUT / "ablation_results.csv")
    abl: dict[str, dict[int, float]] = defaultdict(dict)
    abl_n_val = None
    for row in ablation_rows:
        abl[row["variant"]][int(row["horizon"])] = float(row["mse"])
        abl_n_val = int(float(row["n_val"]))
    abl_hist = json.loads((OUT / "ablation" / "full_vicreg" / "history.json").read_text(encoding="utf-8"))
    values_epochs_ablation = int(abl_hist.get("epochs", 1))
    values_device_ablation = str(abl_hist.get("device", "unknown"))

    caption = (OUT / "figures" / "failure_case_caption.txt").read_text(encoding="utf-8")
    fail_ep = re.search(r"Episode (\d+)", caption)
    fail_mse = re.search(r"k=10 \(([0-9.]+)", caption)

    history = json.loads((OUT / "baselines" / "ac_jepa_vicreg" / "history.json").read_text(encoding="utf-8"))
    baseline_epochs = int(history.get("epochs", 8))
    baseline_device = str(history.get("device", "cpu"))

    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    smoke_n_val = int(metrics["vicreg"]["n_val"])
    smoke_n_train = int(metrics["vicreg"]["n_train"])
    note = probes.get("metrics_json_smoke", {}).get("note", "")
    se_match = re.search(r"SE≈([0-9.]+)", note)

    workshop = {}
    yaml_path = ROOT / "configs" / "workshop_vicreg.yaml"
    if yaml is not None and yaml_path.is_file():
        workshop = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    w_model = workshop.get("model", {})
    w_train = workshop.get("training", {})
    w_vicreg_sim = w_train.get("vicreg_sim_coeff", 25.0)
    w_vicreg_std = w_train.get("vicreg_std_coeff", 25.0)
    w_vicreg_cov = w_train.get("vicreg_cov_coeff", 1.0)
    w_tau = w_model.get("ema_momentum", 0.996)
    w_patch = w_model.get("patch_size", 8)
    w_epochs = w_train.get("epochs", 200)
    w_embed = w_model.get("embed_dim", 128)

    readme = (OUT / "figures" / "README.txt").read_text(encoding="utf-8")
    showcase = re.findall(r"(\d+) \(E=", readme)

    stiff_mean = _mean([float(x) for x in physics["stiffness_probe_acc"]])
    contact_mean = _mean([float(x) for x in physics["contact_probe_acc"]])
    stiff_tgt_mean = _mean([float(x) for x in physics["stiffness_target_probe_acc"]])
    stiff_train_mean = _mean([float(x) for x in physics["stiffness_probe_train_acc"]])
    r2_mean = _mean([float(x) for x in physics["stiffness_reg_val_r2"]])

    values: dict[str, str] = {
        "NEpisodes": str(int(report["n_episodes"])),
        "EpisodeLength": str(int(report["episode_length"])),
        "ImgSize": str(int(report["img_size"])),
        "NTransitions": f"{int(report['n_transitions']):,}".replace(",", r"{,}"),
        "DumpSeed": str(int(meta["seed"])),
        "DumpWorkers": str(int(meta["workers"])),
        "DumpBackend": str(meta["backend"]),
        "NStiffZero": str(int(report["stiffness_counts"][0])),
        "NStiffOne": str(int(report["stiffness_counts"][1])),
        "NStiffTwo": str(int(report["stiffness_counts"][2])),
        "StiffELow": f"{float(meta['stiffness_E'][0]):.0f}",
        "StiffEMid": f"{float(meta['stiffness_E'][1]):.0f}",
        "StiffEHigh": f"{float(meta['stiffness_E'][2]):.0f}",
        "NTrainPhysics": str(int(physics["n_train"])),
        "NValPhysics": str(int(physics["n_val"])),
        "ChanceStiffness": f"{float(physics['chance_stiffness']):.3f}",
        "ChanceStiffnessPct": f"{100.0 * float(physics['chance_stiffness']):.1f}",
        "ChanceContact": f"{float(physics['chance_contact']):.3f}",
        "ChanceContactPct": f"{100.0 * float(physics['chance_contact']):.1f}",
        "ProbeStiffValMean": f"{stiff_mean:.3f}",
        "ProbeContactValMean": f"{contact_mean:.3f}",
        "ProbeStiffTargetMean": f"{stiff_tgt_mean:.3f}",
        "ProbeStiffTrainMean": f"{stiff_train_mean:.3f}",
        "ProbeStiffValRTwo": f"{r2_mean:.3f}",
        "ZTNormStd": f"{float(z_target['norm_std']):.1e}",
        "ZTLatentDim": str(int(z_target["latent_dim"])),
        "ZTDimsBelowGamma": str(int(z_target["n_dims_std_below_gamma"])),
        "ZTHingeBinding": "true" if z_target["variance_hinge_binding"] else "false",
        "BaselineNTrain": str(n_train),
        "BaselineNVal": str(n_val),
        "BaselineEpochs": str(baseline_epochs),
        "BaselineDevice": baseline_device,
        "AblationNVal": str(abl_n_val),
        "AblationEpochs": str(values_epochs_ablation),
        "AblationDevice": values_device_ablation,
        "FailureEpisode": fail_ep.group(1) if fail_ep else "?",
        "FailureMseKten": fail_mse.group(1) if fail_mse else "?",
        "WorkshopMultiseedExists": "true" if (OUT / "ablation_multiseed.json").is_file() else "false",
        "SmokeNVal": str(smoke_n_val),
        "SmokeNTrain": str(smoke_n_train),
        "SmokeProbeSE": se_match.group(1) if se_match else "0.14",
        "WorkshopEpochs": str(int(w_epochs)),
        "PatchSize": str(int(w_patch)),
        "EmaTau": str(w_tau),
        "VicregSim": f"{float(w_vicreg_sim):.0f}",
        "VicregStd": f"{float(w_vicreg_std):.0f}",
        "VicregCov": f"{float(w_vicreg_cov):.0f}",
        "VicregGamma": "1",
        "EmbedDimYaml": str(int(w_embed)),
        "ActionDim": str(int(meta["action_dim"])),
        "NStiffClasses": str(int(meta["n_stiffness_classes"])),
        "NGeometries": str(len(meta["geometries"])),
        "ShowcaseEpA": showcase[0] if len(showcase) > 0 else "1800",
        "ShowcaseEpB": showcase[1] if len(showcase) > 1 else "1801",
        "ShowcaseEpC": showcase[2] if len(showcase) > 2 else "1802",
        "TrainSeedYaml": str(int(w_train.get("seed", 42))),
        "SeedB": "43",
        "NSeeds": "2",
        "VicregCompletedEpochs": str(_completed_epochs(OUT / "vicreg" / "metrics.jsonl")),
        "MseCompletedEpochs": str(_completed_epochs(OUT / "mse" / "metrics.jsonl")),
        "GitHead": _git_head(),
        "TwoSeedExists": "true" if (OUT / "ablation_twoseed.json").is_file() else "false",
    }
    _add_rank_macros(values, probes)
    _add_twoseed_macros(values)

    method_macro = {
        "ac_jepa_vicreg": "Vicreg",
        "ac_jepa_no_cov": "Nocov",
        "ac_jepa_mse": "Mse",
        "pixel_ae": "Pixae",
        "no_action": "Noact",
        "particle_gnn": "Gnn",
    }
    horizon_word = {1: "One", 5: "Five", 10: "Ten", 20: "Twenty"}
    for method, short in method_macro.items():
        for horizon in BASELINE_HORIZONS:
            tag = horizon_word[horizon]
            values[f"BL{short}K{tag}Mse"] = _fmt_mse(by_method[method][horizon]["rollout_mse"])
            values[f"BL{short}K{tag}Stiff"] = _fmt_acc_pct(by_method[method][horizon]["stiffness"])
            values[f"BL{short}K{tag}Contact"] = _fmt_acc_pct(by_method[method][horizon]["contact"])

    variant_macro = {
        "full_vicreg": "Full",
        "variant_a_no_cov": "A",
        "variant_b_mse": "B",
        "variant_c_pixel_ae": "C",
    }
    for variant, short in variant_macro.items():
        for horizon in ABLATION_HORIZONS:
            tag = horizon_word[horizon]
            values[f"Abl{short}K{tag}Mse"] = f"{abl[variant][horizon]:.3f}"

    lines = [
        "% Auto-generated by paper/sync_tables.py — do not edit.",
        r"\makeatletter",
    ]
    for name, body in values.items():
        lines.append(_macro(name, body))
    lines.append(r"\makeatother")
    (DEST / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return values


def write_baseline_table() -> None:
    rows = load_csv_rows(OUT / "baseline_comparison.csv")
    by_method: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    labels: dict[str, str] = {}
    n_train = n_val = chance_s = chance_c = None
    order: list[str] = []
    for row in rows:
        method = row["method"]
        if method not in order:
            order.append(method)
        labels[method] = row["method_label"]
        by_method[method][int(row["horizon"])] = {
            "mse": float(row["rollout_mse"]),
            "stiff": float(row["stiffness_probe_acc"]),
            "contact": float(row["contact_probe_acc"]),
        }
        n_train = int(float(row["n_train"]))
        n_val = int(float(row["n_val"]))
        chance_s = float(row["chance_stiffness"])
        chance_c = float(row["chance_contact"])

    comparable = [m for m in order if m != "particle_gnn"]
    mse_at = {k: [by_method[m][k]["mse"] for m in comparable] for k in BASELINE_HORIZONS}

    def bold_min(method: str, horizon: int, value: float) -> str:
        if method == "particle_gnn":
            return _fmt_mse(value)
        best = min(mse_at[horizon])
        if abs(value - best) <= 1e-12:
            return rf"\textbf{{{_fmt_mse(value)}}}"
        return _fmt_mse(value)

    body = []
    for method in order:
        cells = [labels[method]]
        for k in BASELINE_HORIZONS:
            cells.append(bold_min(method, k, by_method[method][k]["mse"]))
        # probes at k=1 (same split for all methods)
        cells.append(_fmt_acc_pct(by_method[method][1]["stiff"]))
        cells.append(_fmt_acc_pct(by_method[method][1]["contact"]))
        body.append("    " + " & ".join(cells) + r" \\")

    history = json.loads((OUT / "baselines" / "ac_jepa_vicreg" / "history.json").read_text(encoding="utf-8"))
    caption = (
        r"Within-simulator baseline comparison on the mass-spring lattice "
        rf"($n_{{\mathrm{{train}}}}={n_train}$, $n_{{\mathrm{{val}}}}={n_val}$, "
        rf"{int(history.get('epochs', 8))} epochs, device \texttt{{{history.get('device', 'cpu')}}}). "
        r"Open-loop rollout MSE ($\downarrow$) at $k\in\{1,5,10\}$. "
        r"Linear-probe accuracy ($\uparrow$, \%) at $k{=}1$; chance is "
        rf"{100.0 * chance_s:.1f}\% stiffness / {100.0 * chance_c:.1f}\% contact. "
        r"Particle-GNN \texttt{rollout\_mse} is in pooled graph-readout space, "
        r"not the ViT latent --- magnitudes are not interchangeable across rows. "
        r"Source: \texttt{outputs/baseline\_comparison.csv}."
    )
    tex = "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \small",
            rf"  \caption{{{caption}}}",
            r"  \label{tab:baselines}",
            r"  \setlength{\tabcolsep}{3.5pt}",
            r"  \begin{tabular}{lccccc}",
            r"    \toprule",
            r"    Method & $k{=}1$ & $k{=}5$ & $k{=}10$ & Stiff.\ \% & Contact \% \\",
            r"    \midrule",
            *body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (DEST / "baseline_table.tex").write_text(tex, encoding="utf-8")


def _replace_caption(table: str, new_caption: str) -> str:
    token = r"\caption{"
    start = table.index(token) + len(token)
    depth = 1
    i = start
    while i < len(table) and depth:
        if table[i] == "{":
            depth += 1
        elif table[i] == "}":
            depth -= 1
        i += 1
    return table[: table.index(token)] + token + new_caption + "}" + table[i:]


def write_ablation_from_export() -> None:
    """Reuse cells from outputs/paper_tables.tex; replace the caption."""
    raw = (OUT / "paper_tables.tex").read_text(encoding="utf-8")
    match = re.search(r"\\begin\{table\}.*?\\end\{table\}", raw, flags=re.S)
    if not match:
        raise FileNotFoundError("No table environment in outputs/paper_tables.tex")
    table = match.group(0)
    n_val = load_csv_rows(OUT / "ablation_results.csv")[0]["n_val"]
    hist = json.loads((OUT / "ablation" / "full_vicreg" / "history.json").read_text(encoding="utf-8"))
    honest = (
        r"Single-seed dummy-scale ablation imported from "
        r"\texttt{outputs/paper\_tables.tex} (MSE cells from "
        r"\texttt{outputs/ablation\_results.csv}; probes/latency from a later live eval of the "
        rf"same checkpoints). $n_{{\mathrm{{val}}}}={n_val}$, "
        rf"{int(hist.get('epochs', 1))} training epoch, device \texttt{{{hist.get('device', 'unknown')}}}, "
        r"synthetic lattice --- \textbf{not} the workshop 200-epoch / 3-seed study. "
        r"Workshop mean\,$\pm$\,std JSON is missing "
        r"(\texttt{outputs/ablation\_multiseed.json} does not exist). "
        r"Do not read this table as a scaled result."
    )
    table = _replace_caption(table, honest)
    (DEST / "ablation_table.tex").write_text(table + "\n", encoding="utf-8")


def write_multiseed_or_flag() -> None:
    path = OUT / "ablation_multiseed.json"
    dest = DEST / "multiseed_table.tex"
    if path.is_file():
        sys.path.insert(0, str(ROOT))
        from src.utils.export_paper_artifacts import load_multiseed_summary, render_multiseed_table

        blob = json.loads(path.read_text(encoding="utf-8"))
        dest.write_text(
            render_multiseed_table(
                load_multiseed_summary(path),
                horizons=tuple(int(h) for h in blob.get("horizons", (1, 5, 10))),
                n_seeds=blob.get("n_seeds") or (len(blob["seeds"]) if blob.get("seeds") else None),
                n_val=blob.get("n_val"),
            ),
            encoding="utf-8",
        )
        return
    dest.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"  \centering",
                r"  \small",
                r"  \caption{Workshop-scale multi-seed ablation (mean\,$\pm$\,std over seeds "
                r"42/43/44 on the MLS-MPM dump) has \textbf{not been generated}: "
                r"\texttt{outputs/ablation\_multiseed.json} is absent. See \texttt{MAPPING.md}.}",
                r"  \label{tab:ac-jepa-ablation-multiseed}",
                r"  \begin{tabular}{l}",
                r"    \toprule",
                r"    No mean\,$\pm$\,std rows to include. \\",
                r"    \bottomrule",
                r"  \end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_twoseed_table() -> None:
    dest = DEST / "twoseed_table.tex"
    path = OUT / "ablation_twoseed.json"
    if not path.is_file():
        dest.write_text(
            "\n".join(
                [
                    r"\begin{table}[t]",
                    r"  \centering",
                    r"  \small",
                    r"  \caption{Workshop VICReg vs MSE on the MLS-MPM dump, two seeds "
                    r"(42, 43). Table pending \texttt{outputs/ablation\_twoseed.json}.}",
                    r"  \label{tab:twoseed}",
                    r"  \begin{tabular}{l}",
                    r"    \toprule",
                    r"    Two-seed range not written yet. \\",
                    r"    \bottomrule",
                    r"  \end{tabular}",
                    r"\end{table}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return
    blob = json.loads(path.read_text(encoding="utf-8"))
    labels = {"full_vicreg": "Full AC-JEPA (VICReg)", "mse": "MSE-only"}
    body = []
    for key, label in labels.items():
        by_h = (blob["variants"][key]["by_horizon"])
        cells = [label]
        for k in (1, 5, 10):
            cell = by_h[str(k)]
            cells.append(
                rf"${_fmt_mse(float(cell['rollout_mse_mean']))}{{\pm}}{_fmt_mse(float(cell['rollout_mse_range']))}$"
            )
        body.append("    " + " & ".join(cells) + r" \\")
    caption = (
        r"Workshop-scale VICReg vs MSE-only on the MLS-MPM dump. "
        r"Entries are mean\,$\pm$\,half-range over seeds 42 and 43 "
        r"($\frac{1}{2}|x_{42}-x_{43}|$). "
        r"This is a \textbf{2-seed range, not a sample standard deviation}. "
        r"Source: \texttt{outputs/ablation\_twoseed.json}."
    )
    dest.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"  \centering",
                r"  \small",
                rf"  \caption{{{caption}}}",
                r"  \label{tab:twoseed}",
                r"  \begin{tabular}{lccc}",
                r"    \toprule",
                r"    Method & $k{=}1$ & $k{=}5$ & $k{=}10$ \\",
                r"    \midrule",
                *body,
                r"    \bottomrule",
                r"  \end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_rank_finding() -> None:
    dest = DEST / "effective_rank_finding.tex"
    probes = json.loads((OUT / "probe_diagnostics.json").read_text(encoding="utf-8"))
    vic = ((probes.get("models") or {}).get("vicreg_physics_mpm") or {}).get("effective_rank")
    if not vic or "z_target" not in vic:
        dest.write_text(
            r"Effective-rank participation ratios will be filled from the final "
            r"\WorkshopEpochs{}-epoch probes (\texttt{effective\_rank} in "
            r"\texttt{outputs/probe\_diagnostics.json})."
            "\n",
            encoding="utf-8",
        )
        return
    pr = float(vic["z_target"]["participation_ratio"])
    dim = int(vic["z_target"]["latent_dim"])
    if pr < 5.0:
        verdict = (
            r"low relative to $D{=}$\ZTLatentDim{}: a few covariance eigenmodes "
            r"dominate, so the rising VICReg covariance term is a second collapse "
            r"mode, not benign correlation among full-rank features"
        )
    else:
        verdict = (
            r"not rank-1: per-dimension variance can still be below $\gamma$ while "
            r"several eigenmodes remain, i.e.\ correlated-but-not-fully-collapsed "
            r"rather than a single-direction collapse"
        )
    dest.write_text(
        (
            rf"On the physics val split, VICReg $z_{{\mathrm{{target}}}}$ participation "
            rf"ratio is \EffRankTgtVicreg{{}} (latent dim \ZTLatentDim{{}}). "
            rf"That is {verdict}. "
            rf"Predicted $z_{{\mathrm{{hat}}}}$ PR is \EffRankHatVicreg{{}}."
            + (
                r" MSE-only $z_{\mathrm{target}}$ PR is \EffRankTgtMse{}."
                if (probes.get("models") or {}).get("mse_physics_mpm", {}).get("effective_rank")
                else ""
            )
            + "\n"
        ),
        encoding="utf-8",
    )


def write_rollout_caption() -> None:
    n = _completed_epochs(OUT / "vicreg" / "metrics.jsonl")
    dest = DEST / "rollout_caption.tex"
    if n >= 200:
        text = (
            r"GT $|$ post-hoc-decoded VICReg $|$ Pixel-AE at $k\in\{1,5,10\}$ on the "
            r"MLS-MPM dump. VICReg is the finished seed-42 workshop checkpoint "
            r"(\texttt{outputs/vicreg/best.pt}, \WorkshopEpochs{} epochs). "
            r"The Pixel-AE row uses \texttt{outputs/pixel\_ae/} when that run exists; "
            r"otherwise a short qualitative fit. Source: \texttt{outputs/figures/}."
        )
    else:
        text = (
            r"GT $|$ post-hoc-decoded VICReg $|$ Pixel-AE at $k\in\{1,5,10\}$ on the "
            r"MLS-MPM dump. VICReg weights are \textbf{not} finished "
            r"(\VicregCompletedEpochs{} of \WorkshopEpochs{} epochs). "
            r"Source: \texttt{outputs/figures/}."
        )
    dest.write_text(text + "\n", encoding="utf-8")


def write_probe_table() -> None:
    probes = json.loads((OUT / "probe_diagnostics.json").read_text(encoding="utf-8"))
    physics = probes["models"]["vicreg_physics_mpm"]
    findings = physics["findings"]
    items = "\n".join(r"    \item " + _tex_escape(item) for item in findings)
    note = probes.get("checkpoint_note", {})
    if isinstance(note, dict):
        caveat = note.get("vicreg_physics_mpm")
        mse_note = note.get("mse_physics_mpm")
        if mse_note:
            caveat = f"{caveat}; {mse_note}" if caveat else mse_note
        if caveat is None:
            evaluated = note.get("evaluated")
            caveat = ", ".join(str(x) for x in evaluated) if evaluated else str(note)
    else:
        caveat = str(note)
    tex = "\n".join(
        [
            r"\begin{quote}",
            r"\small",
            r"Linear-probe findings copied from \texttt{outputs/probe\_diagnostics.json} "
            r"(\texttt{models.vicreg\_physics\_mpm.findings}):",
            r"\begin{itemize}\setlength{\itemsep}{2pt}\setlength{\parsep}{0pt}",
            items,
            r"\end{itemize}",
            r"Checkpoint caveat (same file, \texttt{checkpoint\_note}): "
            + _tex_escape(str(caveat)),
            r"\end{quote}",
            "",
        ]
    )
    (DEST / "probe_findings.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    write_macros()
    write_baseline_table()
    write_ablation_from_export()
    write_multiseed_or_flag()
    write_twoseed_table()
    write_rank_finding()
    write_rollout_caption()
    write_probe_table()
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
