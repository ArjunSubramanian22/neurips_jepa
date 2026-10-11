"""Episode splits for the historical dump and for future generated pools.

``legacy_v1`` (also ``hist_v1_last_fraction_val``) reproduces the loader rule
in ``src.dataset._episode_split_counts``: the tail fraction is validation, and
there is no test split. That tail was available while ``best.pt`` was
selected, so it is not an untouched test set.

``cvpr_v1`` is the study split for future experiments on the canonical
2000-episode dump. It is an exact 80/10/10 episode split of the legacy
training episodes ``[0, 1800)``. Episodes ``[1800, 2000)`` stay out of
``cvpr_v1`` because they were the legacy selection and reporting set.
Exact ids are in ``configs/protocols/cvpr_v1_episode_ids.json``.

``fresh_pool_v2`` is a predeclared three-way split for a newly generated
episode pool. It does not repartition episodes ``[0, 2000)`` of the
historical dump.
"""

from __future__ import annotations

from dataclasses import dataclass


HIST_V1_PROTOCOL_ID = "hist_v1_last_fraction_val"
LEGACY_V1_SPLIT = "legacy_v1"
CVPR_V1_SPLIT = "cvpr_v1"
FRESH_POOL_V2_PROTOCOL_ID = "fresh_pool_v2"
FRESH_POOL_V2_CYCLE = 6
FRESH_POOL_V2_MIN_EPISODES = 60
CVPR_V1_N_EPISODES = 2000
CVPR_V1_ELIGIBLE = 1800
CVPR_V1_CYCLE = 6
CVPR_V1_BLOCK_STRIDE = 10


@dataclass(frozen=True)
class EpisodeSplit:
    """Half-open episode intervals. ``test`` is ``None`` when undefined."""

    protocol_id: str
    n_episodes: int
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int] | None
    test_defined: bool
    notes: str

    def ids(self, name: str) -> set[int]:
        if name == "test" and not self.test_defined:
            return set()
        span = {"train": self.train, "val": self.val, "test": self.test}[name]
        if span is None:
            return set()
        start, stop = span
        return set(range(start, stop))

    def assert_disjoint(self) -> None:
        """Raise if any episode id is shared, or if a defined split leaves gaps."""
        train_ids = self.ids("train")
        val_ids = self.ids("val")
        test_ids = self.ids("test")
        if train_ids & val_ids:
            raise AssertionError("train and val episode ids overlap.")
        if train_ids & test_ids:
            raise AssertionError("train and test episode ids overlap.")
        if val_ids & test_ids:
            raise AssertionError("val and test episode ids overlap.")
        covered = train_ids | val_ids | test_ids
        expected = set(range(self.n_episodes))
        if self.test_defined:
            if covered != expected:
                raise AssertionError("train, val, and test do not partition the pool.")
        else:
            if test_ids:
                raise AssertionError("undefined test split must be empty.")
            if covered != expected:
                raise AssertionError("train and val do not partition the pool.")
        for name, span in ("train", self.train), ("val", self.val):
            _check_span(name, span, self.n_episodes)
        if self.test_defined:
            _check_span("test", self.test, self.n_episodes)


def _check_span(name: str, span: tuple[int, int] | None, n_episodes: int) -> None:
    if span is None:
        raise AssertionError(f"{name} span is missing.")
    start, stop = span
    if not (0 <= start < stop <= n_episodes):
        raise AssertionError(f"{name} span {(start, stop)} is outside [0, {n_episodes}).")


def historical_val_counts(n_episodes: int, val_fraction: float = 0.1) -> tuple[int, int]:
    """Return ``(n_train, n_val)`` using the historical loader rule."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie in (0, 1).")
    n_val = max(1, int(round(n_episodes * val_fraction))) if n_episodes > 1 else 0
    n_val = min(n_val, n_episodes - 1) if n_episodes > 1 else 0
    return n_episodes - n_val, n_val


def historical_split(n_episodes: int = 2000, val_fraction: float = 0.1) -> EpisodeSplit:
    """Tail-fraction validation split. No test ids are allocated."""
    n_train, n_val = historical_val_counts(n_episodes, val_fraction)
    return EpisodeSplit(
        protocol_id=HIST_V1_PROTOCOL_ID,
        n_episodes=n_episodes,
        train=(0, n_train),
        val=(n_train, n_episodes),
        test=None,
        test_defined=False,
        notes=(
            "The tail fraction is validation. Training writes best.pt from "
            "validation loss on this tail, and the revised paper evaluates "
            "episodes [1800, 2000) of the N=2000 dump. That interval is not "
            "an untouched test set."
        ),
    )


def fresh_pool_v2_split(n_episodes: int) -> EpisodeSplit:
    """Predeclared 80/10/10 split for a new pool.

    ``n_episodes`` must be a positive multiple of 60. Ten percent of that
    count is then a positive multiple of 6, the stiffness-class by geometry
    cycle used by ``scripts/generate_trajectories.py``. Train is the prefix,
    validation the middle block, and test the suffix.
    """
    if n_episodes < FRESH_POOL_V2_MIN_EPISODES or n_episodes % FRESH_POOL_V2_MIN_EPISODES != 0:
        raise ValueError(
            "fresh_pool_v2 requires n_episodes >= 60 and divisible by 60, "
            "so each split length is a positive multiple of the 6-episode "
            "class-geometry cycle and the val/test fractions are exactly 0.1."
        )
    n_val = n_episodes // 10
    n_test = n_episodes // 10
    n_train = n_episodes - n_val - n_test
    if min(n_train, n_val, n_test) % FRESH_POOL_V2_CYCLE != 0:
        raise AssertionError("split lengths broke the class-geometry cycle.")
    return EpisodeSplit(
        protocol_id=FRESH_POOL_V2_PROTOCOL_ID,
        n_episodes=n_episodes,
        train=(0, n_train),
        val=(n_train, n_train + n_val),
        test=(n_train + n_val, n_episodes),
        test_defined=True,
        notes=(
            "Applies only to a newly generated episode pool. Does not "
            "repartition historical dump episodes [0, 2000). The test suffix "
            "is closed to training, checkpoint selection, and hyperparameter "
            "choice. Index order is predeclared and is not a shuffle fit to "
            "outcomes."
        ),
    )


@dataclass(frozen=True)
class VersionedEpisodeSplit:
    """Explicit episode ids. One episode is one trajectory in this dump."""

    split_version: str
    n_episodes: int
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    excluded_ids: tuple[int, ...]
    headline_role: str
    selection_role: str
    normalization_role: str
    notes: str

    def ids(self, role: str) -> tuple[int, ...]:
        mapping = {
            "train": self.train_ids,
            "val": self.val_ids,
            "validation": self.val_ids,
            "test": self.test_ids,
            "excluded": self.excluded_ids,
        }
        if role not in mapping:
            raise KeyError(f"Unknown split role {role!r}.")
        return mapping[role]

    def assert_disjoint(self) -> None:
        """Raise if any episode is shared or any id falls outside the dump."""
        groups = {
            "train": set(self.train_ids),
            "val": set(self.val_ids),
            "test": set(self.test_ids),
            "excluded": set(self.excluded_ids),
        }
        names = tuple(groups)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = groups[left] & groups[right]
                if overlap:
                    raise AssertionError(f"{left} and {right} share episode ids.")
        covered = set().union(*groups.values())
        expected = set(range(self.n_episodes))
        if covered != expected:
            raise AssertionError("split roles do not partition the episode ids.")
        for role, ids in groups.items():
            if len(ids) != len(self.ids(role)):
                raise AssertionError(f"{role} contains a repeated episode id.")


def legacy_v1_split(n_episodes: int = CVPR_V1_N_EPISODES) -> VersionedEpisodeSplit:
    """Legacy tail validation. Identical to ``historical_split`` for N=2000.

    There is no test role. Reporting on this split is validation, not a test set.
    """
    if n_episodes != CVPR_V1_N_EPISODES:
        raise ValueError("legacy_v1 is defined only for the canonical 2000-episode dump.")
    n_train, n_val = historical_val_counts(n_episodes, 0.1)
    return VersionedEpisodeSplit(
        split_version=LEGACY_V1_SPLIT,
        n_episodes=n_episodes,
        train_ids=tuple(range(n_train)),
        val_ids=tuple(range(n_train, n_episodes)),
        test_ids=(),
        excluded_ids=(),
        headline_role="legacy_validation",
        selection_role="val",
        normalization_role="train",
        notes=(
            "Preserves episodes [0, 1800) for training and [1800, 2000) for "
            "validation. Published historical numbers on [1800, 2000) are "
            "legacy validation numbers. They are not cvpr_v1 test numbers."
        ),
    )


def _cvpr_v1_roles() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for episode in range(CVPR_V1_ELIGIBLE):
        slot = (episode // CVPR_V1_CYCLE) % CVPR_V1_BLOCK_STRIDE
        if slot == 8:
            val.append(episode)
        elif slot == 9:
            test.append(episode)
        else:
            train.append(episode)
    excluded = list(range(CVPR_V1_ELIGIBLE, CVPR_V1_N_EPISODES))
    return tuple(train), tuple(val), tuple(test), tuple(excluded)


def cvpr_v1_split() -> VersionedEpisodeSplit:
    """Exact 80/10/10 split of legacy-train episodes, with legacy val excluded.

    Eligible episodes are ``[0, 1800)``, already 300 class-geometry cycles of
    length 6. Block ``episode // 6`` assigns validation when ``block % 10 == 8``,
    test when ``block % 10 == 9``, and train otherwise. Episodes ``[1800, 2000)``
    are excluded. The assignment does not depend on measured outcomes.
    """
    train, val, test, excluded = _cvpr_v1_roles()
    if (len(train), len(val), len(test), len(excluded)) != (1440, 180, 180, 200):
        raise AssertionError("cvpr_v1 role sizes drifted from 1440/180/180/200.")
    return VersionedEpisodeSplit(
        split_version=CVPR_V1_SPLIT,
        n_episodes=CVPR_V1_N_EPISODES,
        train_ids=train,
        val_ids=val,
        test_ids=test,
        excluded_ids=excluded,
        headline_role="test",
        selection_role="val",
        normalization_role="train",
        notes=(
            "80/10/10 of episodes [0, 1800). Episodes [1800, 2000) are the "
            "legacy reporting set and are excluded. Historical checkpoints "
            "were trained on all of [0, 1800) and must not be scored on the "
            "cvpr_v1 test ids."
        ),
    )


def get_split(split_version: str) -> VersionedEpisodeSplit:
    """Return a frozen split. ``hist_v1_last_fraction_val`` aliases ``legacy_v1``."""
    if split_version in {LEGACY_V1_SPLIT, HIST_V1_PROTOCOL_ID}:
        return legacy_v1_split()
    if split_version == CVPR_V1_SPLIT:
        return cvpr_v1_split()
    raise KeyError(
        f"Unknown split_version {split_version!r}. Expected legacy_v1 or cvpr_v1."
    )
