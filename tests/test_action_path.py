"""Action pathway wiring tests (Phase 9). Fail closed on silent no-ops."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig
from src.models.ac_jepa import ACJEPA, ActionPredictor
from src.models.capacity import capacity_report


def _cfg(**kwargs) -> ModelConfig:
    payload = dict(
        in_channels=3,
        img_size=32,
        patch_size=8,
        embed_dim=64,
        encoder_depth=2,
        encoder_num_heads=4,
        mlp_ratio=2.0,
        action_dim=4,
        action_embed_dim=32,
        predictor_embed_dim=64,
        predictor_depth=2,
        predictor_num_heads=4,
        latent_dim=0,
        ema_momentum=0.996,
        dropout=0.0,
        drop_path=0.0,
        backbone="vit",
        fuse_mode="concat",
        condition_on_action=True,
    )
    payload.update(kwargs)
    return ModelConfig(**payload)


class ActionPathTests(unittest.TestCase):
    def test_changing_action_changes_predictor_output_at_init(self) -> None:
        torch.manual_seed(0)
        pred = ActionPredictor(_cfg())
        z = torch.randn(4, 64)
        a0 = torch.zeros(4, 4)
        a1 = a0.clone()
        a1[:, 2] = 1.0
        y0 = pred(z, a0)
        y1 = pred(z, a1)
        self.assertFalse(torch.allclose(y0, y1, atol=1e-6))

    def test_action_embed_changes_with_action(self) -> None:
        torch.manual_seed(1)
        pred = ActionPredictor(_cfg())
        a0 = torch.zeros(2, 4)
        a1 = torch.tensor([[0.2, 0.3, -0.4, 0.5], [0.9, 0.1, 0.0, 0.0]])
        e0 = pred.action_embed(a0)
        e1 = pred.action_embed(a1)
        self.assertFalse(torch.allclose(e0, e1))

    def test_gradients_flow_into_action_embed(self) -> None:
        torch.manual_seed(2)
        model = ACJEPA(_cfg())
        x = torch.randn(3, 3, 32, 32)
        a = torch.randn(3, 4, requires_grad=True)
        x2 = torch.randn(3, 3, 32, 32)
        out = model(x, a, x2)
        loss = out.z_hat_tp1.pow(2).mean()
        loss.backward()
        grads = [p.grad for n, p in model.predictor.named_parameters() if "action_embed" in n]
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0.0 for g in grads))

    def test_no_action_ignores_action_tensor(self) -> None:
        torch.manual_seed(3)
        pred = ActionPredictor(_cfg(condition_on_action=False))
        z = torch.randn(5, 64)
        a0 = torch.zeros(5, 4)
        a1 = torch.randn(5, 4)
        self.assertTrue(torch.allclose(pred(z, a0), pred(z, a1)))
        self.assertFalse(hasattr(pred, "action_embed"))
        self.assertTrue(hasattr(pred, "uncond_proj"))

    def test_condition_flag_on_action_model(self) -> None:
        pred = ActionPredictor(_cfg(condition_on_action=True))
        self.assertTrue(pred.condition_on_action)
        self.assertTrue(hasattr(pred, "action_embed"))

    def test_every_conditioning_mode_changes_with_the_action(self) -> None:
        z = torch.randn(4, 64)
        a0 = torch.zeros(4, 4)
        a1 = a0.clone()
        a1[:, 2] = 1.0
        for mode in ("concat", "film", "gate", "cross_attn", "token"):
            torch.manual_seed(0)
            pred = ActionPredictor(_cfg(fuse_mode=mode))
            self.assertFalse(torch.allclose(pred(z, a0), pred(z, a1), atol=1e-6), msg=mode)

    def test_action_path_gradient_is_nonzero_for_every_mode(self) -> None:
        from src.models.ac_jepa import action_conditioning_grad_norm

        for mode in ("concat", "film", "gate", "cross_attn", "token"):
            torch.manual_seed(4)
            model = ACJEPA(_cfg(fuse_mode=mode))
            x = torch.randn(3, 3, 32, 32)
            a = torch.randn(3, 4)
            out = model(x, a, torch.randn(3, 3, 32, 32))
            out.z_hat_tp1.pow(2).mean().backward()
            self.assertGreater(action_conditioning_grad_norm(model.predictor), 0.0, msg=mode)

    def test_capacity_report_counts_action_path_separately(self) -> None:
        reports = []
        for mode in ("concat", "film", "gate", "cross_attn", "token"):
            report = capacity_report(ACJEPA(_cfg(fuse_mode=mode)))
            reports.append(report)
            self.assertGreater(report["trainable_parameters"], report["predictor_parameters"])
            self.assertGreater(report["action_path_parameters"], 0)
            self.assertGreater(report["predictor_forward_flops"], 0)
            self.assertFalse(report["flops_include_backward"])
        by_mode = {report["fuse_mode"]: report for report in reports}
        concat = by_mode["concat"]["trainable_parameters"]
        for mode in ("film", "gate"):
            self.assertLess(by_mode[mode]["trainable_parameters"] / concat, 1.05)
        self.assertLess(by_mode["cross_attn"]["trainable_parameters"] / concat, 1.15)
        self.assertLess(by_mode["token"]["trainable_parameters"] / concat, 1.5)


if __name__ == "__main__":
    unittest.main()
