"""Queue, screen, and cache guards for the high-excitation study.

These checks do not read episodes 1620-1799 and do not write a pair cache.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from scripts.eval_mpm_high_screen import (  # noqa: E402
    _rssm_contrast,
    assert_no_test_episodes,
    run_recorded_screen,
    validation_episode_ids,
)
from scripts.eval_mpm_high_test_once import main as test_once_main  # noqa: E402
from scripts.generate_counterfactual_pairs import main as pairs_main  # noqa: E402
from scripts.run_confirmatory_queue import planned_screen_commands  # noqa: E402
from scripts.summarize_mpm_high_h1 import _row_from_screen  # noqa: E402
from src.config import ModelConfig  # noqa: E402
from src.models.rssm import CompactRSSM  # noqa: E402


class HighExcitationProtocolTest(unittest.TestCase):
    def test_validation_ids_stop_before_the_test_suffix(self) -> None:
        ids = validation_episode_ids(24)
        self.assertEqual(ids[0], 1440)
        self.assertEqual(ids[-1], 1463)
        self.assertTrue(all(episode < 1620 for episode in ids))
        with self.assertRaises(RuntimeError):
            validation_episode_ids(181)

    def test_test_episodes_are_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_no_test_episodes([1619, 1620])
        with self.assertRaises(RuntimeError):
            run_recorded_screen(
                Namespace(checkpoint="unused", experiment_id="unused", model_name="unused", seed=42),
                [1620],
                allow_test=False,
            )
        with self.assertRaises(RuntimeError):
            run_recorded_screen(
                Namespace(checkpoint="unused", experiment_id="unused", model_name="unused", seed=42),
                [1620],
                allow_test=True,
            )

    def test_queue_does_not_call_the_test_runner(self) -> None:
        commands = planned_screen_commands()
        flat = " ".join(" ".join(command) for command in commands)
        self.assertNotIn("eval_mpm_high_test_once.py", flat)
        self.assertIn("summarize_mpm_high_h1.py", flat)
        self.assertNotIn("latent_tf", flat)
        source = (ROOT / "scripts" / "run_confirmatory_queue.py").read_text(encoding="utf-8")
        main = source.split("def main()", 1)[1]
        self.assertNotIn("jobs()", main)
        self.assertNotIn("src/train.py", main)

    def test_last_pt_without_history_is_not_finished(self) -> None:
        from scripts.run_confirmatory_queue import rssm_run_complete

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "last.pt").write_bytes(b"partial")
            self.assertFalse(rssm_run_complete(run))

    def test_held_out_runner_refuses_without_a_freeze(self) -> None:
        with self.assertRaises(SystemExit):
            test_once_main([])
        with self.assertRaises(SystemExit) as caught:
            test_once_main(["--i-understand-this-opens-the-test-set"])
        self.assertIn("PROTOCOL_FREEZE", str(caught.exception))

    def test_pair_generator_requires_the_environment_and_refuses_legacy_high(self) -> None:
        with self.assertRaises(SystemExit):
            pairs_main(["--episodes", "0", "--out", "outputs/counterfactual_pairs/should_not_exist.npz"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dry.json"
            pairs_main(
                [
                    "--env-config",
                    "configs/environments/mpm_high_excitation.yaml",
                    "--episodes",
                    "0",
                    "--out",
                    str(out),
                    "--dry-run",
                ]
            )
            self.assertFalse(out.exists())
            bad = {
                "environment_id": "mpm_high_excitation",
                "role": "high",
                "command_hold": 3,
                "damping": 0.08,
                "action_gain": 55.0,
            }
            path = Path(tmp) / "legacy_labeled_high.yaml"
            path.write_text(yaml.safe_dump(bad), encoding="utf-8")
            with self.assertRaises(SystemExit):
                pairs_main(
                    [
                        "--env-config",
                        str(path),
                        "--dump",
                        "outputs/physics_mpm_high",
                        "--episodes",
                        "0",
                        "--out",
                        str(Path(tmp) / "bad.npz"),
                        "--dry-run",
                    ]
                )

    def test_v1_screen_keeps_null_cosine_out_of_zero(self) -> None:
        payload = json.loads(
            (ROOT / "experiments/runs/mpm_high_noaction_s42_val_screen_v1/screen.json").read_text(encoding="utf-8")
        )
        row = _row_from_screen("vicreg_noaction", "no_action", 42, payload)
        self.assertIsNone(row["median_cosine"])
        self.assertEqual(row["n_defined_cosine"], 0)
        self.assertNotEqual(row["median_cosine"], 0)
        self.assertEqual(row["cosine_ci_status"], "not_in_source_rows")

    def test_rssm_contrast_reuses_one_posterior_and_does_not_encode_the_final_frame(self) -> None:
        cfg = ModelConfig(img_size=16, rssm_deter=8, rssm_stoch=4, rssm_embed=8, rssm_channels=4)
        model = CompactRSSM(cfg).eval()
        encode_calls = {"n": 0}
        original_encode = model.encode

        def encode(*args, **kwargs):
            encode_calls["n"] += 1
            return original_encode(*args, **kwargs)

        model.encode = encode
        imagine_inputs = []
        original_imagine = model.imagine

        def imagine(deter, stochastic, action):
            imagine_inputs.append(stochastic.detach().clone())
            return original_imagine(deter, stochastic, action)

        model.imagine = imagine
        observe_inputs = []
        original_observe = model.observe

        def observe(deter, stochastic, action, frame):
            observe_inputs.append(None if action is None else stochastic.detach().clone())
            return original_observe(deter, stochastic, action, frame)

        model.observe = observe
        generator = np.random.default_rng(0)
        frame0 = generator.integers(0, 255, size=(3, 16, 16), dtype=np.uint8)
        true_frame = generator.integers(0, 255, size=(3, 16, 16), dtype=np.uint8)
        reversed_frame = generator.integers(0, 255, size=(3, 16, 16), dtype=np.uint8)
        action = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        reversed_action = action.copy()
        reversed_action[2:] *= np.float32(-1.0)
        mean = np.zeros(4, dtype=np.float32)
        std = np.ones(4, dtype=np.float32)
        with torch.no_grad():
            delta_pred, delta_target, one_step = _rssm_contrast(
                model, frame0, true_frame, reversed_frame, action, reversed_action, mean, std, torch.device("cpu")
            )
        self.assertEqual(encode_calls["n"], 0)
        self.assertEqual(len(imagine_inputs), 2)
        self.assertTrue(torch.equal(imagine_inputs[0], imagine_inputs[1]))
        self.assertEqual(observe_inputs[0], None)
        self.assertTrue(torch.equal(observe_inputs[1], observe_inputs[2]))
        self.assertTrue(torch.equal(imagine_inputs[0], observe_inputs[1]))
        self.assertEqual(delta_pred.shape, delta_target.shape)
        self.assertEqual(delta_pred.shape[-1], cfg.rssm_deter + cfg.rssm_stoch)
        self.assertTrue(np.isfinite(one_step))

        blind = CompactRSSM(ModelConfig(img_size=16, rssm_deter=8, rssm_stoch=4, rssm_embed=8, rssm_channels=4, condition_on_action=False)).eval()
        with torch.no_grad():
            blind_pred, _blind_target, _blind_error = _rssm_contrast(
                blind, frame0, true_frame, reversed_frame, action, reversed_action, mean, std, torch.device("cpu")
            )
        self.assertLess(float(np.linalg.norm(blind_pred)), 1e-6)


if __name__ == "__main__":
    unittest.main()
