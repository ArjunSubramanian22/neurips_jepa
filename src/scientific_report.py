"""Build report tables from saved experiment files.

Rows with NaN or an undefined cosine stay in the table. This module does not
run a significance test and does not label a result significant.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

SOURCES = {
    "visibility": "experiments/runs/rgb_visibility_v1/visibility_summary.json",
    "models": "analysis/metric_planning_models.csv",
    "correlations": "analysis/metric_planning_headline.json",
    "seeds": "experiments/runs/headline_seed_uncertainty_v1/seed_summaries.json",
    "planning": "experiments/runs/planning_benchmark_v1/aggregation.json",
    "capacity": "analysis/capacity_sweep_headline.json",
    "inverse": "experiments/runs/inverse_aux_v1/aggregation.json",
    "alignment": "experiments/runs/action_alignment_v1/aggregation.json",
    "target_mask": "experiments/runs/target_mask_v1/aggregation.json",
    "ledger": "experiments/ablations/v1/ledger.jsonl",
}

BENCHMARKS = (
    "experiments/runs/mujoco_arm_v1/benchmark_jepa.json",
    "experiments/runs/mujoco_arm_v1/benchmark_rssm.json",
    "experiments/runs/mujoco_arm_v1/benchmark_latent_tf.json",
    "experiments/runs/mujoco_arm_v1/benchmark_jepa_push.json",
    "experiments/runs/mujoco_arm_v1/benchmark_jepa_plan.json",
)

TABLES = (
    "environment_visibility",
    "model_rollout",
    "directional_counterfactual",
    "retrieval",
    "planning",
    "metric_planning_correlation",
    "seed_variability",
    "architecture_ablation",
    "training_loss_ablation",
    "resolution_scale",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_for_number(value: Any) -> tuple[Any, str]:
    if value is None or value == "":
        return None, "missing"
    if isinstance(value, str):
        if value.strip().lower() == "nan":
            return float("nan"), "nan"
        try:
            value = float(value)
        except ValueError:
            return value, "not_numeric"
    if isinstance(value, float) and math.isnan(value):
        return float("nan"), "nan"
    if isinstance(value, float) and math.isinf(value):
        return value, "nonfinite"
    return value, "ok"


def _row(**fields: Any) -> dict[str, Any]:
    fields.setdefault("uncertainty", None)
    fields.setdefault("uncertainty_status", "not_in_source")
    fields.setdefault("significance_test", "not_computed")
    fields.setdefault("is_test_set", False)
    return fields


def _visibility_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in summary.get("rows", []):
        value, status = _status_for_number(item.get("median_fraction_affected_pixels"))
        rows.append(
            _row(
                table="environment_visibility",
                source=SOURCES["visibility"],
                name=f"{item.get('environment_id')}:{item.get('pipeline')}",
                value=value,
                value_status=status,
                n=item.get("n"),
                metric="median_fraction_affected_pixels",
            )
        )
    return rows


def _model_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rollout, direction, retrieval = [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            for metric, bucket in (
                ("one_step_latent_mse", rollout),
                ("multi_step_latent_mse", rollout),
            ):
                value, status = _status_for_number(item.get(metric))
                bucket.append(
                    _row(
                        table="model_rollout",
                        source=SOURCES["models"],
                        name=item.get("model"),
                        family=item.get("family"),
                        seed=item.get("training_seed"),
                        metric=metric,
                        value=value,
                        value_status=status,
                        n=1,
                        n_planning_episodes=item.get("n_scored_planning"),
                    )
                )
            cosine_raw = item.get("directional_cosine")
            if cosine_raw is None or str(cosine_raw).strip() == "":
                value, status = None, "undefined_cosine"
            else:
                value, status = _status_for_number(cosine_raw)
                if status == "nan":
                    status = "nan"
            direction.append(
                _row(
                    table="directional_counterfactual",
                    source=SOURCES["models"],
                    name=item.get("model"),
                    family=item.get("family"),
                    seed=item.get("training_seed"),
                    metric="directional_cosine",
                    value=value,
                    value_status=status,
                    n=1,
                )
            )
            value, status = _status_for_number(item.get("retrieval_accuracy"))
            retrieval.append(
                _row(
                    table="retrieval",
                    source=SOURCES["models"],
                    name=item.get("model"),
                    family=item.get("family"),
                    seed=item.get("training_seed"),
                    metric="retrieval_accuracy",
                    value=value,
                    value_status=status,
                    n=1,
                )
            )
    return rollout, direction, retrieval


def _planning_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    block = payload.get("report_seeds", {})
    for name, item in block.items():
        if not isinstance(item, dict) or "success_rate" not in item:
            continue
        value, status = _status_for_number(item.get("final_distance_mean"))
        rows.append(
            _row(
                table="planning",
                source=SOURCES["planning"],
                name=name,
                metric="final_distance_mean",
                value=value,
                value_status=status,
                n=item.get("n_scored"),
                n_episodes=item.get("n_episodes"),
                success_rate=item.get("success_rate"),
                is_test_set=bool(payload.get("is_test_set")),
                uncertainty_status="not_in_source",
            )
        )
    return rows


def _correlation_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("evaluations", []):
        value, status = _status_for_number(item.get("spearman"))
        low, high = item.get("spearman_ci_low"), item.get("spearman_ci_high")
        uncertainty = None
        uncertainty_status = item.get("ci_status", "not_in_source")
        if low is not None and high is not None and status == "ok":
            uncertainty = f"[{low}, {high}]"
            uncertainty_status = "bootstrap_interval_from_source"
        rows.append(
            _row(
                table="metric_planning_correlation",
                source=SOURCES["correlations"],
                name=item.get("metric"),
                metric="spearman_vs_final_goal_distance",
                value=value,
                value_status=item.get("spearman_status") or status,
                n=item.get("n"),
                uncertainty=uncertainty,
                uncertainty_status=uncertainty_status,
                is_test_set=bool(payload.get("is_test_set")),
                significance_test="not_computed",
            )
        )
    return rows


def _seed_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for block in payload.get("blocks", []):
        metric = block.get("metric")
        mode = block.get("rollout_mode")
        if mode not in (None, "open_loop"):
            continue
        for configuration in block.get("configurations", []):
            aggregate = configuration.get("aggregate", {})
            value, status = _status_for_number(aggregate.get("mean"))
            if aggregate.get("n_complete", 0) == 0:
                status = "missing"
            interval = configuration.get("hierarchical", {}).get("training_seed_interval", {})
            rows.append(
                _row(
                    table="seed_variability",
                    source=SOURCES["seeds"],
                    name=configuration.get("id"),
                    metric=metric,
                    rollout_mode=mode,
                    value=value,
                    value_status=status,
                    n=aggregate.get("n_complete"),
                    n_required=aggregate.get("n_required"),
                    uncertainty=aggregate.get("std"),
                    uncertainty_status=interval.get("status", "std_of_complete_seeds"),
                    not_run_seeds=aggregate.get("not_run_seeds", []),
                    failed_seeds=aggregate.get("failed_seeds", []),
                )
            )
    return rows


def _capacity_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scale in payload.get("scales", []):
        metric = scale.get("metrics", {}).get("directional_alignment", {})
        value, status = _status_for_number(metric.get("mean"))
        if scale.get("status") != "trained" and value is None:
            status = str(scale.get("status") or "missing")
        rows.append(
            _row(
                table="resolution_scale",
                source=SOURCES["capacity"],
                name=scale.get("scale"),
                metric="directional_alignment",
                sweep="width",
                value=value,
                value_status=status,
                n=metric.get("n"),
                is_test_set=bool(payload.get("is_test_set")),
            )
        )
    return rows


def _loss_rows(payload: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    rows = []
    entries = payload.get("rows", [])
    for item in entries:
        name = item.get("name", item.get("key", item.get("weight")))
        rows.append(
            _row(
                table="training_loss_ablation",
                source=source,
                name=str(name),
                metric="status",
                value=None,
                value_status=str(item.get("status", "missing")),
                n=0 if item.get("status") != "COMPLETE" else None,
                is_test_set=bool(payload.get("select_weight_on_test") or payload.get("select_variant_on_test")),
            )
        )
    return rows


def _architecture_rows(ledger_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"conditioning_film", "conditioning_gate", "conditioning_cross_attn", "model_rssm", "model_transformer", "model_pixel"}
    rows = []
    if not ledger_rows:
        return rows
    latest = ledger_rows[-1]
    for job in latest.get("jobs", []):
        if job.get("template") not in wanted:
            continue
        rows.append(
            _row(
                table="architecture_ablation",
                source=SOURCES["ledger"],
                name=f"{job.get('template')}@{job.get('seed')}",
                metric="queue_status",
                value=None,
                value_status=str(job.get("status")),
                n=0 if job.get("status") != "COMPLETE" else 1,
            )
        )
    return rows


def _benchmark_rows(path: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    relative = str(path)
    for result in payload.get("results", []):
        direction = result.get("layers", {}).get("direction_metrics", {}).get("learned", {})
        cosine = direction.get("cosine")
        cosine_status = direction.get("status")
        if cosine_status == "undefined_zero_vector" or cosine is None and cosine_status != "ok":
            value, status = None, "undefined_cosine" if cosine_status == "undefined_zero_vector" or cosine is None else cosine_status
        else:
            value, status = _status_for_number(cosine)
        rows.append(
            _row(
                table="directional_counterfactual",
                source=relative,
                name=f"{payload.get('model')}:{result.get('task')}",
                metric="learned_cosine",
                value=value,
                value_status=status,
                n=1,
                seed=payload.get("seed"),
                is_test_set=bool(payload.get("is_test_set")),
            )
        )
        ranking = result.get("layers", {}).get("ranking_metrics", {}).get("learned", {})
        top1, top1_status = _status_for_number(ranking.get("top1"))
        if ranking.get("status") == "not_measured":
            top1, top1_status = None, "not_measured"
        rows.append(
            _row(
                table="retrieval",
                source=relative,
                name=f"{payload.get('model')}:{result.get('task')}",
                metric="learned_top1",
                value=top1,
                value_status=top1_status,
                n=ranking.get("n"),
                seed=payload.get("seed"),
                is_test_set=bool(payload.get("is_test_set")),
            )
        )
        learned = result.get("layers", {}).get("planning", {}).get("learned", {})
        chosen = learned.get("chosen")
        plan_status = learned.get("status")
        if plan_status in {"not_measured", "ambiguous_tie"}:
            value_status = str(plan_status)
        elif chosen:
            value_status = "ok"
        else:
            value_status = "missing"
        rows.append(
            _row(
                table="planning",
                source=relative,
                name=f"{payload.get('model')}:{result.get('task')}",
                metric="learned_chosen_action",
                value=chosen,
                value_status=value_status,
                n=1,
                seed=payload.get("seed"),
                is_test_set=bool(payload.get("is_test_set")),
            )
        )
    return rows


def collect_flags(rows: Iterable[Mapping[str, Any]], documents: Iterable[tuple[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Audit flags. Flagged rows remain in the metric tables."""
    flags = []
    material = list(rows)
    for row in material:
        if row.get("value_status") == "nan":
            flags.append({"flag": "nan", "table": row.get("table"), "name": row.get("name"), "metric": row.get("metric")})
        if row.get("value_status") == "undefined_cosine":
            flags.append({"flag": "undefined_cosine", "table": row.get("table"), "name": row.get("name"), "metric": row.get("metric")})
        for seed in row.get("not_run_seeds") or []:
            flags.append({"flag": "missing_seed", "table": row.get("table"), "name": row.get("name"), "metric": row.get("metric"), "seed": seed})
        for seed in row.get("failed_seeds") or []:
            flags.append({"flag": "failed_run", "table": row.get("table"), "name": row.get("name"), "metric": row.get("metric"), "seed": seed})
        if row.get("value_status") == "FAILED":
            flags.append({"flag": "failed_run", "table": row.get("table"), "name": row.get("name"), "status": "FAILED"})
        elif row.get("value_status") in {"NOT_RUN", "BLOCKED", "not_trained"}:
            flags.append({"flag": "not_run", "table": row.get("table"), "name": row.get("name"), "status": row.get("value_status")})
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for row in material:
        if row.get("table") != "seed_variability":
            continue
        key = (row.get("metric"), row.get("rollout_mode"))
        grouped.setdefault(key, []).append(row.get("n"))
    for key, counts in grouped.items():
        present = [count for count in counts if count is not None]
        if present and len(set(present)) > 1:
            flags.append({"flag": "unequal_sample_count", "table": "seed_variability", "name": key[0], "metric": key[0], "counts": present})
    seen: dict[tuple[Any, ...], list[str]] = {}
    for row in material:
        if row.get("value_status") != "ok":
            continue
        if not isinstance(row.get("value"), (int, float)):
            continue
        identity = (row.get("table"), row.get("metric"), row.get("value"))
        seen.setdefault(identity, []).append(str(row.get("name")))
    for identity, names in seen.items():
        unique = sorted(set(names))
        if len(unique) > 1 and identity[0] == "planning" and identity[1] == "final_distance_mean":
            flags.append({"flag": "suspicious_duplicate", "table": identity[0], "name": identity[1], "metric": identity[1], "value": identity[2], "names": unique})
    for source, document in documents:
        is_test = document.get("is_test_set")
        split = str(document.get("split") or document.get("split_version") or "")
        selected_on_test = document.get("select_on_test") or document.get("select_weight_on_test") or document.get("select_variant_on_test")
        if is_test is True and selected_on_test:
            flags.append({"flag": "train_test_contamination", "source": source, "reason": "selection used a test split"})
        if split == "test" and is_test is False:
            flags.append({"flag": "train_test_contamination", "source": source, "reason": "split is named test but is_test_set is false"})
    unique = []
    seen = set()
    for flag in flags:
        key = json.dumps(flag, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(flag)
    return unique


def build_report(root: Path) -> dict[str, Any]:
    root = Path(root)
    visibility = _visibility_rows(_load_json(root / SOURCES["visibility"])) if (root / SOURCES["visibility"]).is_file() else []
    rollout, direction, retrieval = ([], [], [])
    if (root / SOURCES["models"]).is_file():
        rollout, direction, retrieval = _model_rows(root / SOURCES["models"])
    planning = _planning_rows(_load_json(root / SOURCES["planning"])) if (root / SOURCES["planning"]).is_file() else []
    correlations = _correlation_rows(_load_json(root / SOURCES["correlations"])) if (root / SOURCES["correlations"]).is_file() else []
    seeds = _seed_rows(_load_json(root / SOURCES["seeds"])) if (root / SOURCES["seeds"]).is_file() else []
    capacity = _capacity_rows(_load_json(root / SOURCES["capacity"])) if (root / SOURCES["capacity"]).is_file() else []
    losses = []
    if (root / SOURCES["inverse"]).is_file():
        losses.extend(_loss_rows(_load_json(root / SOURCES["inverse"]), SOURCES["inverse"]))
    if (root / SOURCES["alignment"]).is_file():
        losses.extend(_loss_rows(_load_json(root / SOURCES["alignment"]), SOURCES["alignment"]))
    if (root / SOURCES["target_mask"]).is_file():
        losses.extend(_loss_rows(_load_json(root / SOURCES["target_mask"]), SOURCES["target_mask"]))
    ledger_rows = []
    ledger_path = root / SOURCES["ledger"]
    if ledger_path.is_file():
        ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    architecture = _architecture_rows(ledger_rows)
    documents = []
    for relative in BENCHMARKS:
        path = root / relative
        if not path.is_file():
            continue
        payload = _load_json(path)
        documents.append((relative, payload))
        extra = _benchmark_rows(Path(relative), payload)
        direction.extend(row for row in extra if row["table"] == "directional_counterfactual")
        retrieval.extend(row for row in extra if row["table"] == "retrieval")
        planning.extend(row for row in extra if row["table"] == "planning")
    for source_key in ("planning", "correlations", "capacity"):
        path = root / SOURCES[source_key]
        if path.is_file():
            documents.append((SOURCES[source_key], _load_json(path)))
    resolution = [row for row in visibility if ":rgb_" in str(row["name"]) or str(row["name"]).endswith(":physical_raster")]
    resolution.extend(capacity)
    tables = {
        "environment_visibility": visibility,
        "model_rollout": rollout,
        "directional_counterfactual": direction,
        "retrieval": retrieval,
        "planning": planning,
        "metric_planning_correlation": correlations,
        "seed_variability": seeds,
        "architecture_ablation": architecture,
        "training_loss_ablation": losses,
        "resolution_scale": resolution,
    }
    flat = [row for group in tables.values() for row in group]
    flags = collect_flags(flat, documents)
    return {"tables": tables, "flags": flags, "n_rows": len(flat), "n_flags": len(flags)}


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "table",
        "source",
        "name",
        "metric",
        "value",
        "value_status",
        "n",
        "uncertainty",
        "uncertainty_status",
        "significance_test",
        "is_test_set",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fieldnames})


def _tex(value: Any) -> str:
    return _cell(value).replace("_", "\\_").replace("%", "\\%")


def write_latex(rows: list[Mapping[str, Any]], path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}",
        "\\centering",
        "\\begin{tabular}{llrrl}",
        "name & metric & value & n & status \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row.get('name'))} & {_tex(row.get('metric'))} & {_tex(row.get('value'))} & {_tex(row.get('n'))} & {_tex(row.get('value_status'))} \\\\"
        )
    lines.extend([f"\\caption{{{_tex(caption)}}}", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _observed_sentence(table: str, rows: list[Mapping[str, Any]]) -> str:
    counted = [row for row in rows if row.get("n") not in (None, "")]
    n_values = sorted({str(row.get("n")) for row in counted})
    nan_rows = sum(row.get("value_status") == "nan" for row in rows)
    undefined = sum(row.get("value_status") == "undefined_cosine" for row in rows)
    return (
        f"{table} has {len(rows)} rows. "
        f"Sample sizes present: {', '.join(n_values) if n_values else 'none'}. "
        f"NaN values kept: {nan_rows}. Undefined cosines kept: {undefined}."
    )


def _interpretation(table: str) -> str:
    notes = {
        "environment_visibility": "Pixel differences describe the rendered forks. They do not say whether a learned model uses the action.",
        "model_rollout": "Latent MSE is on each checkpoint's own scale. A smaller number is not compared across objectives here.",
        "directional_counterfactual": "An undefined cosine is a zero difference vector, not a cosine of zero. Open-loop rows are the ones that do not read the true next frame.",
        "retrieval": "Retrieval is a hit rate on the scored rows stored in the source. Ties were not counted as hits by that source.",
        "planning": "Success and distance are the stored closed-loop outcomes. The report does not add a significance test.",
        "metric_planning_correlation": "Spearman values and bootstrap intervals are copied from the source file. An interval is not labeled significant by this report.",
        "seed_variability": "Means use completed seeds only where the source says so. Missing seeds stay listed and are not imputed.",
        "architecture_ablation": "Queue status is not a trained metric. An unrun conditioning or baseline has no accuracy to compare.",
        "training_loss_ablation": "Unrun loss variants stay in the table with status NOT_RUN. No weight or lambda was selected.",
        "resolution_scale": "RGB rows are observation differences. Width rows with n 0 were not trained, so no scale trend is identified.",
    }
    return notes.get(table, "This is a description of the saved file, not a new test.")


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Results",
        "",
        "Generated from saved experiment files. No significance test was added.",
        "",
        "Regenerate: `python3.13 scripts/generate_scientific_report.py`",
        "",
        f"Rows: {report['n_rows']}. Flags: {report['n_flags']}.",
        "",
        "## Flags",
        "",
    ]
    if not report["flags"]:
        lines.append("No flags.")
    else:
        for flag in report["flags"]:
            detail = flag.get("reason") or flag.get("names") or flag.get("counts") or flag.get("seed") or ""
            lines.append(f"- {flag.get('flag')}: {flag.get('name') or flag.get('source')} {detail}".rstrip())
    lines.append("")
    for table in TABLES:
        rows = report["tables"].get(table, [])
        lines.extend(
            [
                f"## {table}",
                "",
                "OBSERVED RESULT",
                "",
                _observed_sentence(table, rows),
                "",
                "POSSIBLE INTERPRETATION",
                "",
                _interpretation(table),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_audit(flags: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["flag", "table", "name", "metric", "source", "reason", "seed", "status", "value", "names"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for flag in flags:
            writer.writerow({key: _cell(flag.get(key)) for key in fieldnames})


def write_report(root: Path, output: Path) -> dict[str, Any]:
    report = build_report(root)
    output.mkdir(parents=True, exist_ok=True)
    flat = [row for table in TABLES for row in report["tables"].get(table, [])]
    write_csv(flat, output / "paper_tables.csv")
    write_audit(report["flags"], output / "audit_flags.csv")
    for table in TABLES:
        rows = report["tables"].get(table, [])
        write_csv(rows, output / "tables" / f"{table}.csv")
        write_latex(rows, output / "tables" / f"{table}.tex", table)
    write_markdown(report, output / "RESULTS.md")
    return report
