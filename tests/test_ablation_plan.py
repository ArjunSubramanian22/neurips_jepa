"""The ablation plan is smaller than the product and does not overwrite a finished run."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ablation_plan import (
    BLOCKED,
    COMPLETE,
    NOT_RUN,
    PARTIAL,
    annotate_status,
    assert_output_available,
    cartesian_upper_bound,
    dry_run_summary,
    expand_jobs,
    format_dry_run,
    load_catalog,
    load_seed_registry,
    materialize_config,
    measured_reference_seconds,
)

CATALOG = ROOT / "configs" / "ablations" / "catalog_v1.yaml"


class AblationPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog(CATALOG)
        self.registry = load_seed_registry(ROOT / self.catalog["seed_registry"])
        self.reference = measured_reference_seconds(self.catalog, ROOT)
        self.jobs = annotate_status(expand_jobs(self.catalog, self.registry, reference_seconds=self.reference), ROOT)

    def test_the_plan_is_not_the_cartesian_product(self) -> None:
        product = cartesian_upper_bound(self.catalog, self.registry)
        self.assertGreater(product, 1000)
        self.assertLess(len(self.jobs), 40)
        self.assertLess(len(self.jobs), product / 50)

    def test_existing_headline_checkpoints_are_complete(self) -> None:
        done = {f"{job['template']}@{job['seed']}" for job in self.jobs if job["status"] == COMPLETE}
        self.assertIn("headline_jepa@42", done)
        self.assertIn("headline_noaction@45", done)
        self.assertIn("headline_mse@42", done)
        self.assertIn("arm_audit@0", done)
        missing = next(job for job in self.jobs if job["template"] == "headline_jepa" and job["seed"] == 46)
        self.assertEqual(missing["status"], NOT_RUN)
        self.assertNotIn("outputs/vicreg_nocrop", " ".join(missing["command"]))

    def test_blocked_jobs_are_not_commands(self) -> None:
        summary = dry_run_summary(self.jobs)
        blocked = [job for job in self.jobs if job["template"] in ("resolution_256", "loss_direction", "env_medium")]
        self.assertTrue(all(job["status"] == BLOCKED for job in blocked))
        launched = {item["job"] for item in summary["commands"]}
        self.assertNotIn("resolution_256@42", launched)
        self.assertNotIn("loss_direction@42", launched)
        self.assertIn("headline_jepa@46", launched)

    def test_a_child_waits_for_its_dependency(self) -> None:
        film = next(job for job in self.jobs if job["template"] == "conditioning_film")
        self.assertEqual(film["depends_on"], ["headline_jepa@42"])
        self.assertEqual(film["status"], NOT_RUN)
        self.assertIn("results/ablations/v1/runs", " ".join(film["command"]))

    def test_dry_run_lists_counts_and_commands(self) -> None:
        text = format_dry_run(dry_run_summary(self.jobs))
        self.assertIn(f"jobs {len(self.jobs)}", text)
        self.assertIn("headline_jepa@46", text)
        self.assertIn("bash scripts/train_reviewer_seed.sh 46", text)

    def test_partial_output_is_not_relaunched_and_existing_dirs_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {
                "template": "toy",
                "seed": 1,
                "version": "v1",
                "job_id": "abc",
                "output_dir": "runs/toy_s1",
                "feasible": True,
                "depends_on": [],
                "satisfied_by": None,
                "status": None,
                "command": ["echo", "train"],
            }
            partial = root / "runs" / "toy_s1"
            partial.mkdir(parents=True)
            (partial / "history.json").write_text("{}", encoding="utf-8")
            annotated = annotate_status([job], root)
            self.assertEqual(annotated[0]["status"], PARTIAL)
            with self.assertRaises(FileExistsError):
                assert_output_available(partial)

    def test_materialize_does_not_overwrite_a_config(self) -> None:
        film = next(job for job in self.jobs if job["template"] == "conditioning_film")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workshop = root / "configs"
            workshop.mkdir()
            (workshop / "workshop_vicreg.yaml").write_text("model:\n  fuse_mode: concat\ntraining: {}\n", encoding="utf-8")
            film = dict(film)
            film["base_config"] = "configs/workshop_vicreg.yaml"
            first = materialize_config(film, root)
            self.assertEqual(yaml_fuse(first), "film")
            with self.assertRaises(FileExistsError):
                materialize_config(film, root)

    def test_reference_seconds_come_from_the_measured_histories(self) -> None:
        self.assertGreater(self.reference, 9000)
        self.assertLess(self.reference, 12000)
        payload = json.loads((ROOT / "outputs/vicreg_nocrop/history.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(payload["seconds"], 9046.350118290982, places=3)


def yaml_fuse(path: Path) -> str:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))["model"]["fuse_mode"]


if __name__ == "__main__":
    unittest.main()
