"""Rules for which episodes may train, select, and headline a result.

Normalization statistics are computed from training episodes only.
Model selection reads validation episodes only.
``cvpr_v1`` headline metrics read test episodes only.
``legacy_v1`` has no test split; its reporting slice is legacy validation
and must not be labeled as a test set.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .splits import VersionedEpisodeSplit, get_split


class ProtocolError(ValueError):
    """Raised when a split role is used for the wrong purpose."""


def _canonical(split_version: str) -> str:
    if split_version == "hist_v1_last_fraction_val":
        return "legacy_v1"
    return split_version


def train_action_statistics(actions: Any, split: VersionedEpisodeSplit) -> dict[str, Any]:
    """Mean and population std of actions on training episodes only.

    ``actions`` is ``(N, T, A)`` or ``(N, A)`` with one row per episode.
    Validation, test, and excluded episodes do not enter the moments.
    """
    if split.normalization_role != "train":
        raise ProtocolError("action normalization must use the train role.")
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] != split.n_episodes:
        raise ValueError(
            f"actions leading dimension must be n_episodes={split.n_episodes}, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("actions contain a non-finite value.")
    train_rows = array[np.asarray(split.train_ids, dtype=np.int64)]
    flat = train_rows.reshape(-1, array.shape[-1])
    if flat.shape[0] == 0:
        raise ValueError("train split has no action rows.")
    std = flat.std(axis=0)
    std = np.maximum(std, 1e-6)
    return {
        "mean": flat.mean(axis=0),
        "std": std,
        "split_version": split.split_version,
        "source_role": "train",
        "n_episodes": len(split.train_ids),
        "n_action_rows": int(flat.shape[0]),
        "uses_validation": False,
        "uses_test": False,
        "uses_excluded": False,
    }


def reporting_slice(split: VersionedEpisodeSplit) -> dict[str, Any]:
    """Headline episode ids and whether they are a test set."""
    if split.headline_role == "test":
        if not split.test_ids:
            raise ProtocolError(f"{split.split_version} has no test ids.")
        return {
            "split_version": split.split_version,
            "role": "test",
            "episode_ids": split.test_ids,
            "is_test_set": True,
        }
    if split.headline_role == "legacy_validation":
        return {
            "split_version": split.split_version,
            "role": "legacy_validation",
            "episode_ids": split.val_ids,
            "is_test_set": False,
        }
    raise ProtocolError(f"unknown headline role {split.headline_role!r}.")


def fit_visibility_threshold(
    values_by_episode: Mapping[int, float],
    split: VersionedEpisodeSplit,
    *,
    role: str = "train",
    quantile: float = 0.67,
) -> dict[str, Any]:
    """Quantile of a per-episode statistic on train or validation only.

    Test episodes and the excluded legacy reporting tail are rejected.
    """
    if role in {"test", "legacy_validation", "excluded"}:
        raise ProtocolError("visibility thresholds cannot be fit on reporting episodes.")
    source = "val" if role in {"val", "validation"} else role
    if source not in {"train", "val"}:
        raise ProtocolError(f"cannot fit a threshold on role {role!r}.")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie in (0, 1).")
    allowed = set(split.ids(source))
    keys = {int(episode) for episode in values_by_episode}
    if keys & set(split.test_ids):
        raise ProtocolError("visibility threshold saw test episodes.")
    if keys & set(split.excluded_ids):
        raise ProtocolError("visibility threshold saw excluded legacy-reporting episodes.")
    if not keys <= allowed:
        raise ProtocolError("visibility threshold episodes are outside the declared role.")
    if not keys:
        raise ValueError("visibility threshold received no episodes.")
    ordered = np.asarray([float(values_by_episode[episode]) for episode in split.ids(source) if episode in keys])
    if not np.all(np.isfinite(ordered)):
        raise ValueError("visibility values contain a non-finite value.")
    return {
        "threshold": float(np.quantile(ordered, float(quantile))),
        "role": source,
        "n_episodes": int(ordered.size),
        "quantile": float(quantile),
        "split_version": split.split_version,
        "saw_test": False,
    }


def assert_checkpoint_eval_pairing(checkpoint_split: str, eval_split: str) -> None:
    """Historical checkpoints stay on the legacy protocol."""
    left = _canonical(checkpoint_split)
    right = _canonical(eval_split)
    if left != right:
        raise ProtocolError(
            f"Refusing to evaluate a {left} checkpoint under {right}. "
            "Historical checkpoints are reported only under legacy_v1, and those "
            "numbers are legacy validation, not cvpr_v1 test numbers."
        )
    if left not in {"legacy_v1", "cvpr_v1"}:
        raise ProtocolError(f"Unknown split_version {checkpoint_split!r}.")


def assert_legacy_reproduction_config(config: Mapping[str, Any]) -> None:
    """The archived-table reproduction is validation, not a test result."""
    if config.get("split_version") != "legacy_v1":
        raise ProtocolError("held-out reproduction must declare split_version legacy_v1.")
    if config.get("headline_role") != "legacy_validation":
        raise ProtocolError("held-out reproduction headline_role must be legacy_validation.")
    if config.get("is_test_set") is not False:
        raise ProtocolError("held-out reproduction is not a test set.")


def load_named_split(split_version: str) -> VersionedEpisodeSplit:
    return get_split(split_version)
