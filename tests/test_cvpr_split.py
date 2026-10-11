"""cvpr_v1 ids are disjoint, frozen, and disjoint from the legacy reporting tail."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.splits import cvpr_v1_split, get_split, historical_split, legacy_v1_split

ID_PATH = ROOT / "configs" / "protocols" / "cvpr_v1_episode_ids.json"


def _margins(ids: tuple[int, ...]) -> tuple[dict[int, int], dict[int, int]]:
    classes = {0: 0, 1: 0, 2: 0}
    geometries = {0: 0, 1: 0}
    for episode in ids:
        classes[episode % 3] += 1
        geometries[(episode // 3) % 2] += 1
    return classes, geometries


class CvprSplitTests(unittest.TestCase):
    def test_roles_are_disjoint_and_cover_the_dump(self) -> None:
        split = cvpr_v1_split()
        split.assert_disjoint()
        self.assertEqual(len(split.train_ids), 1440)
        self.assertEqual(len(split.val_ids), 180)
        self.assertEqual(len(split.test_ids), 180)
        self.assertEqual(len(split.excluded_ids), 200)
        self.assertEqual(
            len(split.train_ids) + len(split.val_ids) + len(split.test_ids) + len(split.excluded_ids),
            2000,
        )

    def test_no_trajectory_overlap(self) -> None:
        split = cvpr_v1_split()
        trajectories = {
            "train": set(split.train_ids),
            "val": set(split.val_ids),
            "test": set(split.test_ids),
        }
        self.assertTrue(trajectories["train"].isdisjoint(trajectories["val"]))
        self.assertTrue(trajectories["train"].isdisjoint(trajectories["test"]))
        self.assertTrue(trajectories["val"].isdisjoint(trajectories["test"]))

    def test_legacy_reporting_episodes_are_excluded_from_cvpr(self) -> None:
        legacy = legacy_v1_split()
        cvpr = cvpr_v1_split()
        self.assertEqual(set(cvpr.excluded_ids), set(legacy.val_ids))
        self.assertTrue(set(cvpr.test_ids).isdisjoint(legacy.val_ids))
        self.assertTrue(set(cvpr.val_ids).isdisjoint(legacy.val_ids))
        self.assertTrue(set(cvpr.train_ids).isdisjoint(legacy.val_ids))
        self.assertTrue(set(cvpr.test_ids) <= set(legacy.train_ids))
        self.assertEqual(legacy.test_ids, ())
        self.assertEqual(legacy.headline_role, "legacy_validation")
        self.assertEqual(cvpr.headline_role, "test")

    def test_legacy_matches_the_historical_tail(self) -> None:
        legacy = get_split("legacy_v1")
        alias = get_split("hist_v1_last_fraction_val")
        historical = historical_split(2000, 0.1)
        self.assertEqual(legacy.train_ids, tuple(range(*historical.train)))
        self.assertEqual(legacy.val_ids, tuple(range(*historical.val)))
        self.assertEqual(alias.train_ids, legacy.train_ids)
        self.assertEqual(alias.val_ids, legacy.val_ids)
        legacy.assert_disjoint()

    def test_reconstruction_matches_the_saved_ids(self) -> None:
        first = cvpr_v1_split()
        second = cvpr_v1_split()
        self.assertEqual(first, second)
        saved = json.loads(ID_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved["split_version"], "cvpr_v1")
        self.assertEqual(tuple(saved["train_ids"]), first.train_ids)
        self.assertEqual(tuple(saved["val_ids"]), first.val_ids)
        self.assertEqual(tuple(saved["test_ids"]), first.test_ids)
        self.assertEqual(tuple(saved["excluded_ids"]), first.excluded_ids)

    def test_class_and_geometry_margins_match_inside_each_role(self) -> None:
        split = cvpr_v1_split()
        for role in ("train", "val", "test"):
            classes, geometries = _margins(split.ids(role))
            self.assertEqual(len(set(classes.values())), 1, role)
            self.assertEqual(len(set(geometries.values())), 1, role)
            self.assertEqual(sum(classes.values()) % 6, 0)

    def test_eligible_fraction_is_exact(self) -> None:
        split = cvpr_v1_split()
        eligible = 1800
        self.assertEqual(len(split.train_ids) / eligible, 0.8)
        self.assertEqual(len(split.val_ids) / eligible, 0.1)
        self.assertEqual(len(split.test_ids) / eligible, 0.1)


if __name__ == "__main__":
    unittest.main()
