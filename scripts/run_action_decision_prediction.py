#!/usr/bin/env python3
"""Pilot: do intervention features predict physical candidate-ranking regret?

Reads saved validation artifacts only. Does not train, does not score episodes
1620-1799, and does not read the arm test. Outputs stay under the chosen
directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.action_decision_prediction import (
    BASELINE_FEATURES,
    BOOTSTRAP_DRAWS,
    DIAGNOSTIC_FEATURES,
    FIT_EPISODES,
    HOLDOUT_EPISODES,
    PRIMARY_HORIZON,
    PRIMARY_LIBRARY,
    PRIMARY_MODELS,
    TEST_EPISODE_MIN,
    assert_development_episodes,
    bootstrap_incremental_r2,
    decide,
    incremental_prediction,
    intervention_features,
    linear_redundancy,
    spearman,
)

PROBE_FEATURES = (
    "cosine",
    "response_ratio",
    "effect_nve",
    "effect_rmse",
    "log_ratio_distance",
    "orthogonal_energy",
    "parallel_gain",
)
FULL_FEATURES = BASELINE_FEATURES + DIAGNOSTIC_FEATURES
MODEL_DIRS = {"vicreg_action": "jepa", "mse_action": "mse"}


def _float_or_none(value: str) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def load_prediction_losses(data_root: Path) -> dict[tuple[str, int], dict[str, float]]:
    losses = {}
    for model, folder in MODEL_DIRS.items():
        for seed in range(42, 47):
            path = data_root / "outputs" / "mpm_high_runs" / f"{folder}_s{seed}" / "history.json"
            history = json.loads(path.read_text(encoding="utf-8"))["history"]
            last = history[-1]
            if model == "vicreg_action":
                prediction = float(last["val_invariance"])
            else:
                prediction = float(last["val_loss"])
            losses[(model, seed)] = {
                "validation_prediction_loss": prediction,
                "validation_objective": float(last["val_loss"]),
                "epoch": int(last["epoch"]),
            }
    return losses


def load_probe_medians(path: Path) -> tuple[dict[tuple, dict], dict[str, float]]:
    """Median open-loop force-reversal features. Test episodes are counted and dropped."""
    buckets: dict[tuple, list[dict[str, float]]] = defaultdict(list)
    seen: set[tuple] = set()
    ignored_test = 0
    ratio_gap = 0.0
    n_ratio = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode"])
            if episode >= TEST_EPISODE_MIN:
                ignored_test += 1
                continue
            if row["regime"] != "mpm_high" or row["protocol"] != "open_loop":
                continue
            if row["model"] not in PRIMARY_MODELS:
                continue
            if int(row["horizon"]) not in (5, 10):
                continue
            if not str(row["fork_id"]).endswith(":force_reversal"):
                continue
            if not 1440 <= episode <= 1463:
                continue
            if row["cosine_status"] != "ok":
                continue
            cosine = _float_or_none(row["cosine"])
            pred_norm = _float_or_none(row["pred_norm"])
            target_norm = _float_or_none(row["target_norm"])
            stored_ratio = _float_or_none(row["response_ratio"])
            if cosine is None or pred_norm is None or target_norm is None:
                continue
            features = intervention_features(cosine, pred_norm, target_norm)
            if features is None:
                continue
            if stored_ratio is not None and features["response_ratio"] > 0.0:
                ratio_gap = max(ratio_gap, abs(stored_ratio - features["response_ratio"]))
                n_ratio += 1
            key = (row["model"], int(row["seed"]), episode, int(row["horizon"]), int(row["branch_time"]))
            if key in seen:
                raise RuntimeError(f"duplicate probe row {key}.")
            seen.add(key)
            group = (row["model"], int(row["seed"]), episode, int(row["horizon"]))
            buckets[group].append(features)
    medians = {}
    branch_counts = []
    for key, group in buckets.items():
        branch_counts.append(len(group))
        medians[key] = {name: _median([item[name] for item in group]) for name in PROBE_FEATURES}
        medians[key]["n_branches"] = len(group)
    meta = {
        "test_rows_ignored": ignored_test,
        "max_abs_ratio_gap": ratio_gap,
        "n_ratio_compared": n_ratio,
        "n_groups": len(medians),
        "min_branches": 0 if not branch_counts else int(min(branch_counts)),
        "max_branches": 0 if not branch_counts else int(max(branch_counts)),
    }
    return medians, meta


def load_planning(path: Path) -> list[dict]:
    rows = []
    seen: set[tuple] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode"])
            assert_development_episodes([episode])
            if row["goal_information"] != "final_image":
                raise RuntimeError(f"episode {episode} goal information is {row['goal_information']}.")
            if row["model"] not in PRIMARY_MODELS:
                continue
            key = (row["model"], int(row["seed"]), episode, int(row["horizon"]), row["library"])
            if key in seen:
                raise RuntimeError(f"duplicate planning row {key}.")
            seen.add(key)
            regret = float(row["regret"])
            chosen = float(row["chosen_rms"])
            oracle = float(row["oracle_rms"])
            if abs(regret - (chosen - oracle)) > 1e-8:
                raise RuntimeError(f"regret identity failed for {key}.")
            rows.append(
                {
                    "model": row["model"],
                    "seed": int(row["seed"]),
                    "episode": episode,
                    "horizon": int(row["horizon"]),
                    "library": row["library"],
                    "regret": regret,
                    "chosen_rms": chosen,
                    "oracle_rms": oracle,
                    "latent_top1": float(row["latent_top1"]),
                    "n_candidates": int(row["n_candidates"]),
                }
            )
    return rows


def load_margins(path: Path) -> dict[tuple, dict[str, float]]:
    margins = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode"])
            if episode >= TEST_EPISODE_MIN:
                raise RuntimeError("margin file contains a test episode.")
            key = (episode, int(row["horizon"]), row["library"])
            margins[key] = {
                "pairwise_median": float(row["pairwise_median"]),
                "margin": float(row["margin"]),
                "close": float(row["close"]),
            }
    return margins


def build_rows(
    probes: dict[tuple, dict],
    planning: list[dict],
    margins: dict[tuple, dict[str, float]],
    losses: dict[tuple[str, int], dict[str, float]],
) -> list[dict]:
    built = []
    missing = []
    for item in planning:
        probe = probes.get((item["model"], item["seed"], item["episode"], item["horizon"]))
        margin = margins.get((item["episode"], item["horizon"], item["library"]))
        loss = losses.get((item["model"], item["seed"]))
        if probe is None or margin is None or loss is None:
            missing.append((item["model"], item["seed"], item["episode"], item["horizon"], item["library"]))
            continue
        if margin["pairwise_median"] <= 0.0:
            raise RuntimeError(f"non-positive pairwise median for episode {item['episode']}.")
        row = {
            **item,
            **probe,
            **margin,
            "validation_prediction_loss": loss["validation_prediction_loss"],
            "validation_objective": loss["validation_objective"],
            "checkpoint_epoch": loss["epoch"],
            "regret_over_spread": item["regret"] / margin["pairwise_median"],
            "split": "fit" if item["episode"] in FIT_EPISODES else "holdout",
        }
        built.append(row)
    if missing:
        raise RuntimeError(f"{len(missing)} planning rows had no probe, margin, or loss. First: {missing[:3]}.")
    return built


def _select(
    rows: list[dict],
    *,
    horizon: int,
    library: str,
    episodes: tuple[int, ...] | None = None,
    model: str | None = None,
) -> list[dict]:
    chosen = []
    for row in rows:
        if row["horizon"] != horizon or row["library"] != library:
            continue
        if episodes is not None and int(row["episode"]) not in episodes:
            continue
        if model is not None and row["model"] != model:
            continue
        chosen.append(row)
    return chosen


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key not in {"baseline_pred", "full_pred"}}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _score_block(
    train: list[dict],
    test: list[dict],
    target: str,
    extra_baseline: tuple[str, ...] = (),
    *,
    with_bootstrap: bool = True,
) -> dict:
    baseline = tuple(BASELINE_FEATURES) + tuple(extra_baseline)
    full = baseline + tuple(DIAGNOSTIC_FEATURES)
    scored = incremental_prediction(train, test, baseline, full, target)
    if with_bootstrap and scored["status"] == "ok":
        interval = bootstrap_incremental_r2(train, test, baseline, full, target, draws=BOOTSTRAP_DRAWS)
        scored["ci_low"] = interval["low"]
        scored["ci_high"] = interval["high"]
        scored["ci_status"] = interval["status"]
    else:
        scored["ci_low"] = None
        scored["ci_high"] = None
        scored["ci_status"] = "not_run"
    return scored


def _leave_one_seed(train: list[dict], test: list[dict], target: str) -> dict:
    values = []
    for seed in sorted({int(row["seed"]) for row in test}):
        kept = [row for row in test if int(row["seed"]) != seed]
        scored = incremental_prediction(train, kept, BASELINE_FEATURES, FULL_FEATURES, target)
        values.append({"left_out_seed": seed, "incremental_r2": scored.get("incremental_r2"), "n_test": len(kept)})
    finite = [float(item["incremental_r2"]) for item in values if item["incremental_r2"] is not None]
    return {"folds": values, "min": None if not finite else float(min(finite))}


def _checkpoint_table(train: list[dict], test: list[dict]) -> list[dict]:
    """Probe medians on the fit episodes, mean regret on the holdout episodes."""
    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in train:
        slot = groups[(row["model"], int(row["seed"]))]
        slot["__train__"].append(1.0)
        for name in PROBE_FEATURES:
            slot[name].append(float(row[name]))
        slot["validation_prediction_loss"].append(float(row["validation_prediction_loss"]))
    outcomes: dict[tuple, list[float]] = defaultdict(list)
    for row in test:
        outcomes[(row["model"], int(row["seed"]))].append(float(row["regret"]))
    table = []
    for key, slot in sorted(groups.items()):
        if key not in outcomes:
            continue
        record = {
            "model": key[0],
            "seed": key[1],
            "n_probe_episodes": len(slot["__train__"]),
            "n_outcome_episodes": len(outcomes[key]),
            "mean_holdout_regret": float(np.mean(outcomes[key])),
        }
        for name, values in slot.items():
            if name == "__train__":
                continue
            record[name] = _median(values)
        table.append(record)
    return table


def _univariate(rows: list[dict], features: tuple[str, ...], target: str) -> list[dict]:
    results = []
    for feature in features:
        scored = spearman([row[feature] for row in rows], [row[target] for row in rows])
        results.append({"feature": feature, "target": target, **scored})
    return results


def _family_indicator_r2(train: list[dict], test: list[dict], target: str) -> dict:
    """One-bit model: predict the training mean of each model family."""
    means = {}
    for model in PRIMARY_MODELS:
        taken = [float(row[target]) for row in train if row["model"] == model]
        means[model] = float(np.mean(taken))
    predicted = [means[row["model"]] for row in test]
    return {"status": "ok", "means": means, **r2_only(test, predicted, target)}


def r2_only(rows: list[dict], predicted: list[float], target: str) -> dict:
    from src.metrics.action_decision_prediction import r2_score

    score = r2_score([row[target] for row in rows], predicted)
    return {"r2": score["value"], "r2_status": score["status"], "n": score["n"]}


def _failure_cases(test: list[dict], scored: dict, limit: int = 8) -> list[dict]:
    if scored["status"] != "ok":
        return []
    cases = []
    for row, baseline, full in zip(test, scored["baseline_pred"], scored["full_pred"]):
        base_error = abs(float(baseline) - float(row["regret"]))
        full_error = abs(float(full) - float(row["regret"]))
        cases.append(
            {
                "episode": int(row["episode"]),
                "model": row["model"],
                "seed": int(row["seed"]),
                "regret": float(row["regret"]),
                "baseline_prediction": float(baseline),
                "full_prediction": float(full),
                "baseline_abs_error": base_error,
                "full_abs_error": full_error,
                "diagnostic_increased_error": full_error > base_error + 1e-12,
            }
        )
    cases.sort(key=lambda item: (-item["regret"], -item["full_abs_error"]))
    return cases[:limit]


def _per_group(test: list[dict], scored: dict, key: str) -> list[dict]:
    if scored["status"] != "ok":
        return []
    buckets: dict[object, list[int]] = defaultdict(list)
    for index, row in enumerate(test):
        buckets[row[key]].append(index)
    groups = []
    for name, indexes in sorted(buckets.items(), key=lambda item: str(item[0])):
        regret = np.asarray([float(test[index]["regret"]) for index in indexes])
        baseline = np.asarray([float(scored["baseline_pred"][index]) for index in indexes])
        full = np.asarray([float(scored["full_pred"][index]) for index in indexes])
        groups.append(
            {
                key: name,
                "n": len(indexes),
                "mean_regret": float(np.mean(regret)),
                "mean_baseline_prediction": float(np.mean(baseline)),
                "mean_full_prediction": float(np.mean(full)),
                "baseline_r2": r2_only(indexes_to_rows(test, indexes), baseline.tolist(), "regret")["r2"],
                "full_r2": r2_only(indexes_to_rows(test, indexes), full.tolist(), "regret")["r2"],
            }
        )
    return groups


def indexes_to_rows(rows: list[dict], indexes: list[int]) -> list[dict]:
    return [rows[index] for index in indexes]


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "split",
        "model",
        "seed",
        "episode",
        "horizon",
        "library",
        "regret",
        "regret_over_spread",
        "chosen_rms",
        "latent_top1",
        "pairwise_median",
        "margin",
        "validation_prediction_loss",
        "validation_objective",
        "checkpoint_epoch",
        "n_branches",
        *PROBE_FEATURES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_predictions(path: Path, test: list[dict], scored: dict) -> None:
    fields = ["episode", "model", "seed", "regret", "baseline_prediction", "full_prediction"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if scored["status"] != "ok":
            return
        for row, baseline, full in zip(test, scored["baseline_pred"], scored["full_pred"]):
            writer.writerow(
                {
                    "episode": row["episode"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "regret": row["regret"],
                    "baseline_prediction": float(baseline),
                    "full_prediction": float(full),
                }
            )


def _figures(out: Path, checks: dict, predictions: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = []
    points = []
    lows = []
    highs = []
    for name in ("primary", "within_vicreg_action", "within_mse_action", "difficulty", "normalized", "reverse"):
        block = checks[name]
        if block.get("incremental_r2") is None:
            continue
        names.append(name.replace("_", "\n"))
        points.append(float(block["incremental_r2"]))
        low = block.get("ci_low")
        high = block.get("ci_high")
        lows.append(0.0 if low is None else float(point_err_low(block)))
        highs.append(0.0 if high is None else float(point_err_high(block)))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    positions = np.arange(len(names))
    ax.bar(positions, points, color="#4c78a8", width=0.72)
    yerr = np.vstack([lows, highs]) if names else None
    if yerr is not None and np.any(yerr):
        ax.errorbar(positions, points, yerr=yerr, fmt="none", ecolor="#222222", capsize=3)
    ax.axhline(0.05, color="#b35806", linestyle="--", linewidth=1, label="0.05 go threshold")
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Holdout incremental R²")
    ax.set_title("Intervention features over the linear baseline")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "incremental_r2.png", dpi=140)
    plt.close(fig)

    if not predictions:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharex=True, sharey=True)
    regret = np.asarray([float(row["regret"]) for row in predictions])
    for axis, key, title in (
        (axes[0], "baseline_prediction", "Baseline linear model"),
        (axes[1], "full_prediction", "Baseline plus intervention features"),
    ):
        pred = np.asarray([float(row[key]) for row in predictions])
        for model, color in (("vicreg_action", "#4c78a8"), ("mse_action", "#f58518")):
            mask = np.asarray([row["model"] == model for row in predictions])
            axis.scatter(regret[mask], pred[mask], s=16, alpha=0.75, c=color, label=model)
        limit = float(max(np.max(regret), np.max(pred), 0.0))
        axis.plot([0.0, limit], [0.0, limit], color="#444444", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Holdout physical regret (particle RMS)")
        axis.set_ylabel("Predicted regret")
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "holdout_predictions.png", dpi=140)
    plt.close(fig)


def point_err_low(block: dict) -> float:
    if block.get("ci_low") is None:
        return 0.0
    return max(0.0, float(block["incremental_r2"]) - float(block["ci_low"]))


def point_err_high(block: dict) -> float:
    if block.get("ci_high") is None:
        return 0.0
    return max(0.0, float(block["ci_high"]) - float(block["incremental_r2"]))


def _strip_predictions(block: dict) -> dict:
    return {key: value for key, value in block.items() if key not in {"baseline_pred", "full_pred"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "runs" / "action_decision_prediction_pilot",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    data_root = args.data_root.resolve()
    out = args.out.resolve()
    if "arm_campaign" in str(out) or "test_once" in str(out):
        raise SystemExit("refusing to write into an arm test directory.")
    out.mkdir(parents=True, exist_ok=True)
    assert_development_episodes(FIT_EPISODES)
    assert_development_episodes(HOLDOUT_EPISODES)
    protocol = {
        "fit_episodes": list(FIT_EPISODES),
        "holdout_episodes": list(HOLDOUT_EPISODES),
        "horizon": PRIMARY_HORIZON,
        "library": PRIMARY_LIBRARY,
        "models": list(PRIMARY_MODELS),
        "baseline_features": list(BASELINE_FEATURES),
        "diagnostic_features": list(DIAGNOSTIC_FEATURES),
        "min_incremental_r2": 0.05,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "refuses_episodes_at_or_above": TEST_EPISODE_MIN,
        "outcome": "physical regret of matched final-image sequence ranking",
        "note": "Written before the fit. Sensitivities at horizon 5 and the constant library are not decision criteria.",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    losses = load_prediction_losses(data_root)
    probes, probe_meta = load_probe_medians(data_root / "experiments" / "runs" / "mpm_high_multih" / "branch_rows.csv")
    if probe_meta["max_abs_ratio_gap"] > 1e-4:
        raise RuntimeError(f"stored response ratio disagrees with pred/target by {probe_meta['max_abs_ratio_gap']}.")
    planning = load_planning(data_root / "experiments" / "runs" / "mpm_high_planning_matched" / "raw_results.csv")
    margins = load_margins(data_root / "experiments" / "runs" / "mpm_high_planning_separability" / "episode_margins.csv")
    rows = build_rows(probes, planning, margins, losses)
    primary_rows = _select(rows, horizon=PRIMARY_HORIZON, library=PRIMARY_LIBRARY)
    train = _select(primary_rows, horizon=PRIMARY_HORIZON, library=PRIMARY_LIBRARY, episodes=FIT_EPISODES)
    test = _select(primary_rows, horizon=PRIMARY_HORIZON, library=PRIMARY_LIBRARY, episodes=HOLDOUT_EPISODES)
    if len(train) + len(test) != len(primary_rows):
        raise RuntimeError("fit and holdout do not cover the primary rows.")

    primary = _score_block(train, test, "regret")
    within = {
        model: _score_block(
            _select(train, horizon=PRIMARY_HORIZON, library=PRIMARY_LIBRARY, model=model),
            _select(test, horizon=PRIMARY_HORIZON, library=PRIMARY_LIBRARY, model=model),
            "regret",
        )
        for model in PRIMARY_MODELS
    }
    difficulty = _score_block(train, test, "regret", extra_baseline=("pairwise_median",))
    normalized = _score_block(train, test, "regret_over_spread")
    reverse = _score_block(
        _select(primary_rows, horizon=PRIMARY_HORIZON, library=PRIMARY_LIBRARY, episodes=HOLDOUT_EPISODES),
        _select(primary_rows, horizon=PRIMARY_HORIZON, library=PRIMARY_LIBRARY, episodes=FIT_EPISODES),
        "regret",
    )
    seeds = _leave_one_seed(train, test, "regret")
    family = _family_indicator_r2(train, test, "regret")
    redundancy = [linear_redundancy(train, feature, BASELINE_FEATURES) for feature in DIAGNOSTIC_FEATURES]
    checkpoint_rows = _checkpoint_table(train, test)
    checkpoint_features = ("validation_prediction_loss", "cosine", "effect_nve", "effect_rmse", *DIAGNOSTIC_FEATURES)
    checkpoint_spearman = _univariate(checkpoint_rows, checkpoint_features, "mean_holdout_regret")
    checkpoint_within_family = {
        model: _univariate(
            [row for row in checkpoint_rows if row["model"] == model],
            checkpoint_features,
            "mean_holdout_regret",
        )
        for model in PRIMARY_MODELS
    }
    episode_spearman = {
        "fit": _univariate(train, ("validation_prediction_loss", "cosine", "effect_nve", "effect_rmse", *DIAGNOSTIC_FEATURES), "regret"),
        "holdout_descriptive": _univariate(
            test,
            ("validation_prediction_loss", "cosine", "effect_nve", "effect_rmse", *DIAGNOSTIC_FEATURES),
            "regret",
        ),
    }
    h5_train = _select(rows, horizon=5, library=PRIMARY_LIBRARY, episodes=FIT_EPISODES)
    h5_test = _select(rows, horizon=5, library=PRIMARY_LIBRARY, episodes=HOLDOUT_EPISODES)
    constant_train = _select(rows, horizon=PRIMARY_HORIZON, library="constant", episodes=FIT_EPISODES)
    constant_test = _select(rows, horizon=PRIMARY_HORIZON, library="constant", episodes=HOLDOUT_EPISODES)
    sensitivities = {
        "horizon_5_sequence": _strip_predictions(_score_block(h5_train, h5_test, "regret", with_bootstrap=False)),
        "horizon_10_constant": _strip_predictions(_score_block(constant_train, constant_test, "regret", with_bootstrap=False)),
    }
    report = {
        "n_rows": len(primary_rows),
        "primary": primary,
        "within_vicreg_action": within["vicreg_action"],
        "within_mse_action": within["mse_action"],
        "difficulty": difficulty,
        "normalized": normalized,
        "reverse": reverse,
        "leave_one_seed_min_incremental_r2": seeds["min"],
    }
    decision = decide({key: (_strip_predictions(value) if isinstance(value, dict) else value) for key, value in report.items()})
    failures = _failure_cases(test, primary)
    n_compared = len(test) if primary["status"] == "ok" else 0
    n_worse = 0
    if primary["status"] == "ok":
        n_worse = int(
            sum(
                abs(float(full) - float(row["regret"])) > abs(float(baseline) - float(row["regret"])) + 1e-12
                for row, baseline, full in zip(test, primary["baseline_pred"], primary["full_pred"])
            )
        )
    summary = {
        "decision": decision,
        "unit": "checkpoint-episode",
        "outcome": "physical endpoint regret of finite-candidate sequence ranking at horizon 10",
        "use_case": "offline probe before the candidate decision; not closed-loop control and not deployable without the simulator target encoder",
        "fit_episodes": list(FIT_EPISODES),
        "holdout_episodes": list(HOLDOUT_EPISODES),
        "models": list(PRIMARY_MODELS),
        "baseline_features": list(BASELINE_FEATURES),
        "diagnostic_features": list(DIAGNOSTIC_FEATURES),
        "probe": probe_meta,
        "n_primary_rows": len(primary_rows),
        "n_fit": len(train),
        "n_holdout": len(test),
        "mean_holdout_regret": float(np.mean([row["regret"] for row in test])),
        "family_indicator": family,
        "redundancy_on_fit_block": redundancy,
        "checks": {key: _strip_predictions(value) if isinstance(value, dict) else value for key, value in report.items()},
        "leave_one_seed": seeds,
        "checkpoint_spearman": checkpoint_spearman,
        "checkpoint_spearman_within_family": checkpoint_within_family,
        "checkpoint_within_family_note": "Descriptive only. Five checkpoints per family. Not an input to decide().",
        "episode_spearman": episode_spearman,
        "per_model": _per_group(test, primary, "model"),
        "per_seed": _per_group(test, primary, "seed"),
        "per_episode": _per_group(test, primary, "episode"),
        "n_holdout_rows_compared": n_compared,
        "n_holdout_rows_where_diagnostic_increased_abs_error": n_worse,
        "failure_cases": failures,
        "sensitivities_not_used_for_decision": sensitivities,
        "elapsed_seconds": time.perf_counter() - started,
        "test_episodes_scored": False,
    }
    _write_rows(out / "rows.csv", primary_rows)
    _write_predictions(out / "holdout_predictions.csv", test, primary)
    (out / "checkpoint_rows.json").write_text(json.dumps(_jsonable(checkpoint_rows), indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    with (out / "holdout_predictions.csv").open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    try:
        _figures(out, report, prediction_rows)
    except Exception as exc:  # noqa: BLE001 — figures must not discard a finished summary.
        (out / "figure_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "reasons": decision["reasons"], "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
