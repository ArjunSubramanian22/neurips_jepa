"""Counterfactual future ranking.

For one starting state and K candidate actions, ``pred[i]`` is the predicted
future under action i and ``target[j]`` is the encoded true future under
action j. Retrieval asks whether action i's prediction is closest to action
i's own future.

An observationally identical competitor is ambiguity, not a miss. That
includes futures whose particles differ while the rasters match. Those rows
stay in the result with a status and are left out of the accuracy denominator.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.stats import episode_bootstrap_mean

DISTANCE_KINDS = ("mse", "l2", "vn_mse", "cosine")
MODES = ("model", "shuffled_predictions", "oracle_targets", "constant_prediction")
CANDIDATE_ROLES = ("recorded", "shuffled", "force_reversed", "force_half")

STATUS_SCORED = "scored"
STATUS_AMBIGUOUS = "observationally_ambiguous"
STATUS_TIE = "distance_tie"
STATUS_UNDEFINED = "undefined_distance"
STATUS_SIM = "simulation_nonfinite"


def derangement(n_items: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation with no fixed point. Falls back to a cycle if rejection fails."""
    n = int(n_items)
    if n < 2:
        raise ValueError("a derangement needs at least 2 items.")
    identity = np.arange(n)
    for _ in range(100):
        perm = rng.permutation(n)
        if np.all(perm != identity):
            return np.asarray(perm, dtype=np.int64)
    return np.roll(identity, 1)


def candidate_actions(
    recorded: Any,
    shuffled: Any,
    *,
    force_half_scale: float = 0.5,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Recorded action, one shuffled valid action, force reversal, and force scaling.

    Force channels are the last two entries. A zero force makes the reversal
    and the scaled force equal the recorded action; callers still keep both
    rows and let the observation check mark the ambiguity.
    """
    base = np.asarray(recorded, dtype=np.float64).reshape(-1)
    other = np.asarray(shuffled, dtype=np.float64).reshape(-1)
    if base.shape != other.shape or base.size < 4:
        raise ValueError("recorded and shuffled actions must share a shape of at least 4.")
    if not np.isfinite(float(force_half_scale)):
        raise ValueError("force_half_scale must be finite.")
    reversed_force = base.copy()
    reversed_force[2:4] *= -1.0
    halved = base.copy()
    halved[2:4] *= float(force_half_scale)
    actions = np.stack([base, other, reversed_force, halved], axis=0)
    return actions, CANDIDATE_ROLES


def control_predictions(
    pred: Any,
    target: Any,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Model predictions plus chance, oracle, and action-invariant controls.

    Shuffled predictions are a random permutation of the predicted rows.
    Fixed points are allowed, so a perfect assignment is broken only by
    chance. The oracle copies the target embeddings. The constant control
    repeats the first prediction, so every query ranks the same list.
    """
    predicted = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 2 or predicted.shape[0] < 2:
        raise ValueError("pred and target must share shape (K, D) with K >= 2.")
    perm = rng.permutation(predicted.shape[0])
    return {
        "model": predicted,
        "shuffled_predictions": predicted[perm],
        "oracle_targets": truth.copy(),
        "constant_prediction": np.repeat(predicted[:1], predicted.shape[0], axis=0),
    }


def distance_matrix(
    pred: Any,
    target: Any,
    kind: str,
    *,
    ref_variance: float | None = None,
    norm_tau: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise distances from each prediction to each target.

    Returns ``(distances, defined)``. Cosine distance is ``1 - cosine`` and is
    undefined when either vector has norm at or below ``norm_tau``. A
    non-positive variance leaves every variance-normalized entry undefined.
    L2 and MSE are monotone for a fixed dimension, so they induce the same
    ranking; both are still returned so the margin can be read in either unit.
    """
    if kind not in DISTANCE_KINDS:
        raise ValueError(f"unknown distance {kind!r}.")
    predicted = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 2 or predicted.shape[0] < 2:
        raise ValueError("pred and target must share shape (K, D) with K >= 2.")
    if not np.isfinite(float(norm_tau)) or float(norm_tau) < 0.0:
        raise ValueError("norm_tau must be >= 0.")
    delta = predicted[:, None, :] - truth[None, :, :]
    finite = np.all(np.isfinite(predicted), axis=1)[:, None] & np.all(np.isfinite(truth), axis=1)[None, :]
    mse = np.mean(delta ** 2, axis=-1)
    if kind == "mse":
        return mse, finite
    if kind == "l2":
        return np.sqrt(np.sum(delta ** 2, axis=-1)), finite
    if kind == "vn_mse":
        ref = np.nan if ref_variance is None else float(ref_variance)
        if not np.isfinite(ref) or ref <= 0.0:
            return np.full(mse.shape, np.nan), np.zeros(mse.shape, dtype=bool)
        return mse / ref, finite
    pred_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(truth, axis=1)
    denom = pred_norm[:, None] * target_norm[None, :]
    usable = finite & (pred_norm[:, None] > float(norm_tau)) & (target_norm[None, :] > float(norm_tau))
    cosine = np.ones(mse.shape, dtype=np.float64)
    cosine[usable] = np.clip(
        np.sum(predicted[:, None, :] * truth[None, :, :], axis=-1)[usable] / denom[usable],
        -1.0,
        1.0,
    )
    distances = np.full(mse.shape, np.nan, dtype=np.float64)
    distances[usable] = 1.0 - cosine[usable]
    return distances, usable


def rank_candidates(
    pred: Any,
    target: Any,
    obs_equal: Any,
    physical_rms: Any,
    *,
    distance: str,
    ref_variance: float | None = None,
    topk: int = 2,
    norm_tau: float = 1e-8,
    physical_tau: float = 0.0,
    simulation_ok: bool = True,
) -> dict[str, Any]:
    """Score every query action. Ambiguous queries are returned, not dropped."""
    distances, defined = distance_matrix(
        pred, target, distance, ref_variance=ref_variance, norm_tau=norm_tau
    )
    equal = np.asarray(obs_equal, dtype=bool)
    rms = np.asarray(physical_rms, dtype=np.float64)
    k = int(distances.shape[0])
    if equal.shape != (k, k) or rms.shape != (k, k):
        raise ValueError("obs_equal and physical_rms must be (K, K).")
    if not np.all(np.diag(equal)):
        raise ValueError("obs_equal diagonal must be true.")
    if int(topk) < 1:
        raise ValueError("topk must be >= 1.")
    queries = [
        _score_query(
            index,
            distances,
            defined,
            equal,
            rms,
            topk=int(topk),
            physical_tau=float(physical_tau),
            simulation_ok=bool(simulation_ok),
        )
        for index in range(k)
    ]
    return {
        "distance": distance,
        "k": k,
        "topk": int(topk),
        "queries": queries,
        "summary": summarize_ranking(queries, n_candidates=k, topk=int(topk)),
    }


def _score_query(
    index: int,
    distances: np.ndarray,
    defined: np.ndarray,
    equal: np.ndarray,
    rms: np.ndarray,
    *,
    topk: int,
    physical_tau: float,
    simulation_ok: bool,
) -> dict[str, Any]:
    k = int(distances.shape[0])
    others = [j for j in range(k) if j != index]
    ambiguous_partners = [j for j in others if bool(equal[index, j])]
    physical_hidden = any(float(rms[index, j]) > physical_tau for j in ambiguous_partners)
    pair_ambiguous = len(ambiguous_partners)
    base = {
        "query_index": index,
        "physically_distinct_observationally_identical": bool(physical_hidden),
        "n_ambiguous_partners": pair_ambiguous,
        "top1": None,
        "topk": None,
        "reciprocal_rank": None,
        "rank": None,
        "margin": None,
        "correct_distance": None,
        "nearest_incorrect_distance": None,
        "pairwise_n": 0,
        "pairwise_n_ambiguous": pair_ambiguous,
        "pairwise_n_undefined": 0,
        "pairwise_n_correct": 0,
        "pairwise_n_tie": 0,
        "pairwise_n_incorrect": 0,
        "pairwise_rate": None,
    }
    if not simulation_ok:
        base["status"] = STATUS_SIM
        return base
    if ambiguous_partners:
        base["status"] = STATUS_AMBIGUOUS
        return base
    if not bool(defined[index].all()):
        undefined_partners = [j for j in others if not bool(defined[index, j])]
        base["status"] = STATUS_UNDEFINED
        base["pairwise_n_undefined"] = len(undefined_partners)
        return base
    row = distances[index]
    correct = float(row[index])
    n_closer = int(np.sum(row[others] < correct))
    n_tied = int(np.sum(row[others] == correct))
    rank = n_closer + n_tied + 1
    nearest = float(np.min(row[others]))
    correct_pairs = 0
    tie_pairs = 0
    incorrect_pairs = 0
    for partner in others:
        gap = float(row[partner])
        if correct < gap:
            correct_pairs += 1
        elif correct == gap:
            tie_pairs += 1
        else:
            incorrect_pairs += 1
    identifiable_pairs = correct_pairs + tie_pairs + incorrect_pairs
    status = STATUS_TIE if n_tied else STATUS_SCORED
    topk_value = None if k <= topk else float(rank <= topk)
    base.update(
        {
            "status": status,
            "top1": float(n_closer == 0 and n_tied == 0),
            "topk": topk_value,
            "reciprocal_rank": 1.0 / float(rank),
            "rank": int(rank),
            "margin": nearest - correct,
            "correct_distance": correct,
            "nearest_incorrect_distance": nearest,
            "pairwise_n": identifiable_pairs,
            "pairwise_n_correct": correct_pairs,
            "pairwise_n_tie": tie_pairs,
            "pairwise_n_incorrect": incorrect_pairs,
            "pairwise_rate": None if identifiable_pairs == 0 else float(correct_pairs / identifiable_pairs),
        }
    )
    return base


def summarize_ranking(
    queries: Sequence[Mapping[str, Any]],
    *,
    n_candidates: int,
    topk: int,
) -> dict[str, Any]:
    """Accuracy over scored queries. Ambiguous and undefined rows stay in the counts."""
    rows = list(queries)
    if int(n_candidates) < 2:
        raise ValueError("n_candidates must be >= 2.")
    scored_status = {STATUS_SCORED, STATUS_TIE}
    scored = [row for row in rows if row["status"] in scored_status]
    hits = [row for row in scored if row["top1"] == 1.0]
    topk_rows = [row for row in scored if row["topk"] is not None]
    topk_hits = [row for row in topk_rows if row["topk"] == 1.0]
    margins = np.asarray([float(row["margin"]) for row in scored if row["margin"] is not None], dtype=np.float64)
    ranks = np.asarray(
        [float(row["reciprocal_rank"]) for row in scored if row["reciprocal_rank"] is not None],
        dtype=np.float64,
    )
    pair_correct = int(sum(int(row["pairwise_n_correct"]) for row in rows))
    pair_tie = int(sum(int(row["pairwise_n_tie"]) for row in rows))
    pair_incorrect = int(sum(int(row["pairwise_n_incorrect"]) for row in rows))
    pair_ambiguous = int(sum(int(row["pairwise_n_ambiguous"]) for row in rows))
    pair_undefined = int(sum(int(row["pairwise_n_undefined"]) for row in rows))
    pair_scored = pair_correct + pair_tie + pair_incorrect
    summary = {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_hit": len(hits),
        "n_ambiguous": int(sum(row["status"] == STATUS_AMBIGUOUS for row in rows)),
        "n_physically_distinct_observationally_identical": int(
            sum(bool(row["physically_distinct_observationally_identical"]) for row in rows)
        ),
        "n_undefined": int(sum(row["status"] == STATUS_UNDEFINED for row in rows)),
        "n_simulation_nonfinite": int(sum(row["status"] == STATUS_SIM for row in rows)),
        "n_tie": int(sum(row["status"] == STATUS_TIE for row in rows)),
        "n_candidates": int(n_candidates),
        "chance_top1": 1.0 / float(n_candidates),
        "top1": None if not scored else float(len(hits) / len(scored)),
        "topk": None if not topk_rows else float(len(topk_hits) / len(topk_rows)),
        "topk_k": int(topk),
        "topk_applicable": bool(int(n_candidates) > int(topk)),
        "mean_reciprocal_rank": None if ranks.size == 0 else float(ranks.mean()),
        "margin_mean": None if margins.size == 0 else float(margins.mean()),
        "margin_median": None if margins.size == 0 else float(np.median(margins)),
        "pairwise_n_scored": pair_scored,
        "pairwise_n_correct": pair_correct,
        "pairwise_n_tie": pair_tie,
        "pairwise_n_incorrect": pair_incorrect,
        "pairwise_n_ambiguous": pair_ambiguous,
        "pairwise_n_undefined": pair_undefined,
        "pairwise_probability": None if pair_scored == 0 else float(pair_correct / pair_scored),
    }
    for percentile in (10, 90):
        summary[f"margin_p{percentile}"] = None if margins.size == 0 else float(np.percentile(margins, percentile))
    return summary


def visibility_bin(correct_observation_unique: bool, fraction_affected: float, *, high_visibility: float) -> str:
    """Ambiguous rasters stay in their own bin. They are not labeled low visibility."""
    if not bool(correct_observation_unique):
        return "ambiguous"
    if float(fraction_affected) >= float(high_visibility):
        return "high_visibility"
    return "low_visibility"


def physical_bin(physical_rms: float, *, small_rms: float) -> str:
    """Predeclared displacement bins. A zero RMS is exact agreement, not a small move."""
    rms = float(physical_rms)
    if rms <= 0.0:
        return "zero_displacement"
    if rms <= float(small_rms):
        return "small_displacement"
    return "large_displacement"


def bootstrap_top1(
    episode_ids: Sequence[int],
    top1: Sequence[float | None],
    statuses: Sequence[str],
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Equal-episode interval for top-1. Episodes with no scored query are listed, not imputed."""
    episodes = np.asarray(episode_ids)
    values = np.asarray([np.nan if value is None else float(value) for value in top1], dtype=np.float64)
    status = np.asarray(list(statuses), dtype=object)
    if not (episodes.shape[0] == values.shape[0] == status.shape[0]):
        raise ValueError("bootstrap columns must have equal length.")
    scored = np.isin(status, (STATUS_SCORED, STATUS_TIE)) & np.isfinite(values)
    means = []
    excluded = []
    for episode in np.unique(episodes).tolist():
        taken = values[(episodes == episode) & scored]
        if taken.size == 0:
            excluded.append(int(episode))
        else:
            means.append(float(taken.mean()))
    result: dict[str, Any] = {
        "n_episodes": int(np.unique(episodes).size),
        "n_episodes_with_scored": len(means),
        "n_episodes_without_scored": len(excluded),
        "excluded_episode_ids": excluded,
    }
    if not means:
        result["bootstrap"] = None
        return result
    result["bootstrap"] = episode_bootstrap_mean(np.asarray(means), n_resamples, rng)
    return result
