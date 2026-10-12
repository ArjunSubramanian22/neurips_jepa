"""The queue records failures, retries, and does not resize a model."""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.queue import apply_fallback_batch, read_registry, run_queue

CPU = {"selected": "cpu", "cuda": False, "mps": False}
MPS = {"selected": "mps", "cuda": False, "mps": True}


def _job(job_id: str, command: list[str], **extra) -> dict:
    payload = {
        "id": job_id,
        "command": command,
        "device": "cpu",
        "exclusive": False,
        "max_transient_retries": 2,
        "adaptive_batch": False,
    }
    payload.update(extra)
    return payload


def _script(body: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(body)]


class QueueTests(unittest.TestCase):
    def test_dry_run_does_not_execute_or_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            result = run_queue(
                {"max_parallel": 1, "jobs": [_job("hello", _script("raise SystemExit(1)"))]},
                registry=registry,
                dry_run=True,
                git_commit="abc",
                devices=CPU,
            )
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["plan"][0]["action"], "run")
            self.assertFalse(registry.exists())
            self.assertEqual(result["failures"], [])

    def test_success_records_exit_and_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            result = run_queue(
                {"jobs": [_job("hello", _script("print('ok')"))]},
                registry=registry,
                git_commit="abc",
                devices=CPU,
            )
            self.assertEqual(result["failures"], [])
            rows = read_registry(registry)
            finished = [row for row in rows if row["status"] == "COMPLETE"]
            self.assertEqual(finished[-1]["exit_code"], 0)
            self.assertGreater(finished[-1]["elapsed_seconds"], 0)
            self.assertTrue(Path(finished[-1]["stdout_path"]).read_text(encoding="utf-8").strip() == "ok")

    def test_a_failure_stays_failed_and_is_not_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            job = _job("bad", _script("import sys; sys.stderr.write('nope\\n'); raise SystemExit(3)"))
            first = run_queue({"jobs": [job]}, registry=registry, git_commit="abc", devices=CPU)
            self.assertEqual(first["failures"][0]["exit_code"], 3)
            second = run_queue({"jobs": [job]}, registry=registry, git_commit="abc", devices=CPU)
            self.assertEqual(second["plan"][0]["action"], "skip")
            self.assertIn("failed previously", second["plan"][0]["reason"])
            failed = [row for row in read_registry(registry) if row["status"] == "FAILED"]
            self.assertEqual(len(failed), 1)

    def test_transient_failures_retry_then_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            counter = Path(tmp) / "count"
            script = (
                "import pathlib, sys\n"
                f"path = pathlib.Path({str(counter)!r})\n"
                "n = int(path.read_text()) if path.exists() else 0\n"
                "path.write_text(str(n + 1))\n"
                "raise SystemExit(0 if n >= 2 else 75)\n"
            )
            job = _job("flaky", [sys.executable, "-c", script], transient_exit_codes=[75], max_transient_retries=2)
            result = run_queue({"jobs": [job]}, registry=registry, git_commit="abc", devices=CPU)
            self.assertEqual(result["failures"], [])
            statuses = [row["status"] for row in read_registry(registry)]
            self.assertEqual(statuses.count("ATTEMPT_FAILED"), 2)
            self.assertEqual(statuses[-1], "COMPLETE")

    def test_retries_stop_and_the_failure_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            job = _job(
                "always",
                _script("import sys; sys.stderr.write('Resource temporarily unavailable\\n'); raise SystemExit(1)"),
                max_transient_retries=1,
            )
            result = run_queue({"jobs": [job]}, registry=registry, git_commit="abc", devices=CPU)
            self.assertEqual(len(result["failures"]), 1)
            statuses = [row["status"] for row in read_registry(registry)]
            self.assertEqual(statuses.count("ATTEMPT_FAILED"), 1)
            self.assertEqual(statuses[-1], "FAILED")

    def test_oom_without_adaptation_is_not_retried_and_architecture_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            job = _job(
                "oom",
                [sys.executable, "-c", "import sys; sys.stderr.write('CUDA out of memory embed-dim 128\\n'); raise SystemExit(1)", "--embed-dim", "128"],
                adaptive_batch=False,
            )
            result = run_queue({"jobs": [job]}, registry=registry, git_commit="abc", devices=CPU)
            self.assertTrue(result["failures"][0]["oom"])
            rows = [row for row in read_registry(registry) if row["status"] == "FAILED"]
            self.assertEqual(len(rows), 1)
            self.assertIn("--embed-dim", rows[0]["command"])
            self.assertEqual(rows[0]["command"].count("128"), 1)
            self.assertFalse(rows[0]["batch_adapted"])

    def test_oom_retries_once_with_the_configured_smaller_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            script = (
                "import sys\n"
                "batch = sys.argv[sys.argv.index('--batch-size') + 1]\n"
                "if batch != '4':\n"
                "    sys.stderr.write('CUDA out of memory\\n')\n"
                "    raise SystemExit(1)\n"
            )
            job = _job(
                "adapt",
                [sys.executable, "-c", script, "--batch-size", "32", "--embed-dim", "128"],
                adaptive_batch=True,
                batch_size=32,
                fallback_batch_size=4,
                batch_flag="--batch-size",
            )
            result = run_queue({"jobs": [job]}, registry=registry, git_commit="abc", devices=CPU)
            self.assertEqual(result["failures"], [])
            complete = [row for row in read_registry(registry) if row["status"] == "COMPLETE"][-1]
            self.assertTrue(complete["batch_adapted"])
            self.assertEqual(complete["batch_size"], 4)
            self.assertEqual(complete["command"][complete["command"].index("--embed-dim") + 1], "128")
            self.assertEqual(complete["command"][complete["command"].index("--batch-size") + 1], "4")

    def test_completed_job_is_skipped_only_when_config_and_git_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            job = _job("hello", _script("print('ok')"))
            run_queue({"jobs": [job]}, registry=registry, git_commit="abc", devices=CPU)
            again = run_queue({"jobs": [job]}, registry=registry, git_commit="abc", devices=CPU)
            self.assertEqual(again["plan"][0]["action"], "skip")
            changed = dict(job)
            changed["command"] = _script("print('other')")
            rerun = run_queue({"jobs": [changed]}, registry=registry, git_commit="abc", devices=CPU)
            self.assertEqual(rerun["plan"][0]["action"], "run")
            other_commit = run_queue({"jobs": [job]}, registry=registry, git_commit="def", devices=CPU)
            self.assertEqual(other_commit["plan"][0]["action"], "run")

    def test_accelerator_jobs_do_not_overlap_and_cpu_jobs_can(self) -> None:
        sleep = _script("import time; time.sleep(0.4)")
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            run_queue(
                {"max_parallel": 2, "jobs": [_job("a", sleep, device="mps"), _job("b", sleep, device="mps")]},
                registry=registry,
                git_commit="abc",
                devices=MPS,
            )
            rows = [row for row in read_registry(registry) if row["status"] == "COMPLETE"]
            first, second = sorted(rows, key=lambda row: row["started_unix"])
            self.assertGreaterEqual(second["started_unix"], first["ended_unix"])
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            run_queue(
                {"max_parallel": 2, "jobs": [_job("a", sleep), _job("b", sleep)]},
                registry=registry,
                git_commit="abc",
                devices=CPU,
            )
            rows = [row for row in read_registry(registry) if row["status"] == "COMPLETE"]
            first, second = sorted(rows, key=lambda row: row["started_unix"])
            self.assertLess(second["started_unix"], first["ended_unix"])

    def test_a_dead_running_job_is_interrupted_and_then_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.jsonl"
            registry.write_text(
                json.dumps(
                    {
                        "job_id": "hello",
                        "config_hash": "stale",
                        "git_commit": "abc",
                        "status": "RUNNING",
                        "pid": 999999,
                        "attempt": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = run_queue(
                {"jobs": [_job("hello", _script("print('again')"))]},
                registry=registry,
                git_commit="abc",
                devices=CPU,
                alive=lambda _pid: False,
            )
            self.assertEqual(result["failures"], [])
            statuses = [row["status"] for row in read_registry(registry)]
            self.assertEqual(statuses[1], "INTERRUPTED")
            self.assertEqual(statuses[-1], "COMPLETE")

    def test_fallback_batch_does_not_touch_other_flags(self) -> None:
        updated = apply_fallback_batch(["train", "--embed-dim", "128", "--batch-size", "32"], "--batch-size", 8)
        self.assertEqual(updated, ["train", "--embed-dim", "128", "--batch-size", "8"])


if __name__ == "__main__":
    unittest.main()
