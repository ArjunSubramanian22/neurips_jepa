"""Episode splits for the historical dump and for future generated pools.

``hist_v1`` reproduces the loader rule in ``src.dataset._episode_split_counts``:
the tail fraction is validation, and there is no test split. That tail was
available while ``best.pt`` was selected, so it is not an untouched test set.

``fresh_pool_v2`` is a predeclared three-way split for a newly generated
episode pool. It does not repartition episodes ``[0, 2000)`` of the
historical dump.
"""

from __future__ import annotations

from dataclasses import dataclass


HIST_V1_PROTOCOL_ID = "hist_v1_last_fraction_val"
FRESH_POOL_V2_PROTOCOL_ID = "fresh_pool_v2"
FRESH_POOL_V2_CYCLE = 6
FRESH_POOL_V2_MIN_EPISODES = 60


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
