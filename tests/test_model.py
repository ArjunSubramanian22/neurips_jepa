"""Sanity checks for the AC-JEPA encoder, predictor, EMA target, and VICReg loss."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DataConfig, ModelConfig
from src.dataset import generate_dummy_batch
from src.losses import (
    VICRegLoss,
    covariance_loss,
    invariance_loss,
    prediction_objective,
    variance_loss,
    vicreg_loss,
)
from src.models.ac_jepa import ACJEPA, ActionPredictor, ResNetEncoder, VisionTransformerEncoder
from src.models.pixel_ae import PixelSpaceAutoencoder


def _tiny_cfg(**overrides: object) -> ModelConfig:
    payload: dict[str, object] = {
        "in_channels": 3,
        "img_size": 32,
        "patch_size": 8,
        "embed_dim": 64,
        "encoder_depth": 2,
        "encoder_num_heads": 4,
        "mlp_ratio": 2.0,
        "action_dim": 4,
        "action_embed_dim": 32,
        "predictor_embed_dim": 64,
        "predictor_depth": 2,
        "predictor_num_heads": 4,
        "latent_dim": 0,
        "ema_momentum": 0.996,
        "dropout": 0.0,
        "drop_path": 0.0,
        "backbone": "vit",
        "fuse_mode": "concat",
    }
    payload.update(overrides)
    return ModelConfig(**payload)  # type: ignore[arg-type]


def _dummy_batch(
    batch_size: int = 4, cfg: ModelConfig | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    cfg = cfg if cfg is not None else _tiny_cfg()
    x_t = torch.randn(batch_size, cfg.in_channels, cfg.img_size, cfg.img_size)
    a_t = torch.randn(batch_size, cfg.action_dim)
    x_tp1 = torch.randn(batch_size, cfg.in_channels, cfg.img_size, cfg.img_size)
    return x_t, a_t, x_tp1


def _named_params_require_grad(module: torch.nn.Module, enabled: bool) -> None:
    for name, param in module.named_parameters():
        if param.requires_grad is not enabled:
            raise AssertionError(
                f"{name}: requires_grad={param.requires_grad}, expected {enabled}."
            )


class VICRegLossTests(unittest.TestCase):
    def test_invariance_zero_when_equal(self) -> None:
        z = torch.randn(16, 8)
        self.assertEqual(invariance_loss(z, z).item(), 0.0)

    def test_variance_hinge_penalizes_collapse(self) -> None:
        collapsed = torch.ones(32, 8)
        spread = torch.randn(32, 8) * 3.0
        self.assertGreater(variance_loss(collapsed).item(), 0.9)
        self.assertEqual(variance_loss(spread).item(), 0.0)

    def test_covariance_penalizes_duplicate_features(self) -> None:
        torch.manual_seed(0)
        independent = torch.randn(128, 8)
        duplicated = independent[:, :1].expand(-1, 8).contiguous()
        self.assertGreater(covariance_loss(duplicated).item(), covariance_loss(independent).item())
        self.assertGreaterEqual(covariance_loss(independent).item(), 0.0)

    def test_vicreg_weighted_sum_and_stopgrad_target(self) -> None:
        z_hat = torch.randn(8, 16, requires_grad=True)
        z_target = torch.randn(8, 16, requires_grad=True)
        loss, terms = vicreg_loss(z_hat, z_target.detach(), sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0)
        expected = 25.0 * terms["invariance"] + 25.0 * terms["variance"] + 1.0 * terms["covariance"]
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNotNone(z_hat.grad)
        self.assertIsNone(z_target.grad)

    def test_criterion_module_matches_functional(self) -> None:
        torch.manual_seed(1)
        z_hat = torch.randn(6, 10)
        z_target = torch.randn(6, 10)
        criterion = VICRegLoss(sim_coeff=2.0, std_coeff=3.0, cov_coeff=4.0)
        loss_mod, _ = criterion(z_hat, z_target)
        loss_fn, _ = vicreg_loss(z_hat, z_target, sim_coeff=2.0, std_coeff=3.0, cov_coeff=4.0)
        self.assertTrue(torch.allclose(loss_mod, loss_fn))

    def test_vicreg_no_cov_drops_covariance_from_gradient(self) -> None:
        torch.manual_seed(0)
        z_hat = torch.randn(8, 16, requires_grad=True)
        z_target = torch.randn(8, 16)
        loss, terms = prediction_objective(z_hat, z_target, loss_type="vicreg_no_cov")
        expected = 25.0 * terms["invariance"] + 25.0 * terms["variance"]
        self.assertTrue(torch.allclose(loss, expected))
        full, _ = prediction_objective(z_hat, z_target, loss_type="vicreg")
        self.assertFalse(torch.allclose(loss, full))


class PixelSpaceAutoencoderTests(unittest.TestCase):
    def test_decode_and_rollout_shapes(self) -> None:
        cfg = _tiny_cfg()
        model = PixelSpaceAutoencoder(cfg)
        x_t, a_t, x_tp1 = _dummy_batch(4, cfg)
        out = model(x_t, a_t, x_tp1)
        recon = model.decode(out.z_hat_tp1)
        self.assertEqual(tuple(recon.shape), tuple(x_t.shape))
        actions = torch.randn(4, 5, cfg.action_dim)
        rolled = model.rollout(out.z_t, actions)
        self.assertEqual(tuple(rolled.shape), (4, 5, cfg.effective_latent_dim))

    def test_decoder_receives_gradients(self) -> None:
        cfg = _tiny_cfg()
        model = PixelSpaceAutoencoder(cfg)
        model.train()
        x_t, a_t, x_tp1 = _dummy_batch(4, cfg)
        z_t = model.encode(x_t)
        z_hat = model.predictor(z_t, a_t)
        loss = torch.nn.functional.mse_loss(model.decode(z_hat), x_tp1)
        loss.backward()
        decoder_grads = [p.grad for p in model.decoder.parameters()]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in decoder_grads))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))


class ACJEPAModelTests(unittest.TestCase):
    def test_vit_forward_shapes(self) -> None:
        cfg = _tiny_cfg(backbone="vit")
        model = ACJEPA(cfg)
        x_t, a_t, x_tp1 = _dummy_batch(4, cfg)
        out = model(x_t, a_t, x_tp1)
        dim = cfg.effective_latent_dim
        self.assertEqual(tuple(out.z_t.shape), (4, dim))
        self.assertEqual(tuple(out.z_hat_tp1.shape), (4, dim))
        assert out.z_tp1 is not None
        self.assertEqual(tuple(out.z_tp1.shape), (4, dim))
        self.assertIsInstance(model.encoder, VisionTransformerEncoder)

    def test_resnet_forward_shapes(self) -> None:
        cfg = _tiny_cfg(backbone="cnn")
        model = ACJEPA(cfg)
        x_t, a_t, x_tp1 = _dummy_batch(4, cfg)
        out = model(x_t, a_t, x_tp1)
        dim = cfg.effective_latent_dim
        self.assertEqual(tuple(out.z_hat_tp1.shape), (4, dim))
        self.assertIsInstance(model.encoder, ResNetEncoder)
        self.assertIsInstance(model.target_encoder, ResNetEncoder)

    def test_predictor_fuse_modes(self) -> None:
        x_t, a_t, x_tp1 = _dummy_batch(3, _tiny_cfg())
        for fuse in ("concat", "film", "token"):
            model = ACJEPA(_tiny_cfg(fuse_mode=fuse))
            out = model(x_t, a_t, x_tp1)
            self.assertEqual(tuple(out.z_hat_tp1.shape), (3, 64), msg=fuse)
            self.assertIsInstance(model.predictor, ActionPredictor)

    def test_target_matches_online_at_init(self) -> None:
        model = ACJEPA(_tiny_cfg())
        for online, target in zip(
            model.encoder.parameters(), model.target_encoder.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(online, target))
            self.assertFalse(target.requires_grad)

    def test_gradients_skip_target_and_flow_online(self) -> None:
        cfg = _tiny_cfg()
        model = ACJEPA(cfg)
        model.train()
        x_t, a_t, x_tp1 = _dummy_batch(8, cfg)
        out = model(x_t, a_t, x_tp1)
        assert out.z_tp1 is not None
        self.assertTrue(out.z_t.requires_grad)
        self.assertTrue(out.z_hat_tp1.requires_grad)
        self.assertFalse(out.z_tp1.requires_grad)

        loss, _ = vicreg_loss(out.z_hat_tp1, out.z_tp1)
        loss.backward()

        _named_params_require_grad(model.target_encoder, enabled=False)
        for name, param in model.target_encoder.named_parameters():
            self.assertIsNone(param.grad, msg=name)

        online_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        pred_grads = [p.grad for p in model.predictor.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in online_grads))
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in pred_grads))
        self.assertTrue(all(g is not None for g in online_grads))
        self.assertTrue(all(g is not None for g in pred_grads))

    def test_ema_update_follows_momentum_rule(self) -> None:
        tau = 0.996
        model = ACJEPA(_tiny_cfg(ema_momentum=tau))
        with torch.no_grad():
            for param in model.encoder.parameters():
                param.add_(0.25)
        online_after = [p.detach().clone() for p in model.encoder.parameters()]
        target_before = [p.detach().clone() for p in model.target_encoder.parameters()]
        model.update_target_encoder()
        for online, before, after in zip(
            online_after, target_before, model.target_encoder.parameters(), strict=True
        ):
            expected = tau * before + (1.0 - tau) * online
            self.assertTrue(torch.allclose(after, expected, atol=1e-6, rtol=1e-5))

    def test_optimizer_step_does_not_move_target(self) -> None:
        cfg = _tiny_cfg()
        model = ACJEPA(cfg)
        model.train()
        opt = torch.optim.AdamW(list(model.trainable_parameters()), lr=1e-2)
        x_t, a_t, x_tp1 = _dummy_batch(8, cfg)
        out = model(x_t, a_t, x_tp1)
        assert out.z_tp1 is not None
        loss, _ = vicreg_loss(out.z_hat_tp1, out.z_tp1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        target_before = [p.detach().clone() for p in model.target_encoder.parameters()]
        opt.step()
        for before, after in zip(target_before, model.target_encoder.parameters(), strict=True):
            self.assertTrue(torch.equal(before, after))

    def test_target_encoder_stays_in_eval(self) -> None:
        model = ACJEPA(_tiny_cfg(backbone="cnn"))
        model.train()
        self.assertTrue(model.encoder.training)
        self.assertTrue(model.predictor.training)
        self.assertFalse(model.target_encoder.training)
        model.eval()
        self.assertFalse(model.target_encoder.training)

    def test_forward_without_future_frame(self) -> None:
        cfg = _tiny_cfg()
        model = ACJEPA(cfg)
        x_t, a_t, _ = _dummy_batch(2, cfg)
        out = model(x_t, a_t)
        self.assertIsNone(out.z_tp1)
        self.assertEqual(tuple(out.z_hat_tp1.shape), (2, cfg.effective_latent_dim))

    def test_forward_on_synthetic_dataset_batch(self) -> None:
        data_cfg = DataConfig(
            n_episodes=4,
            episode_length=3,
            img_size=32,
            n_grid=4,
            action_dim=4,
            in_channels=3,
            seed=0,
            val_fraction=0.0,
        )
        x_t, a_t, x_tp1 = generate_dummy_batch(4, data_cfg)
        model = ACJEPA(_tiny_cfg())
        out = model(x_t, a_t, x_tp1)
        loss, terms = vicreg_loss(out.z_hat_tp1, out.z_tp1)
        self.assertEqual(tuple(x_t.shape), (4, 3, 32, 32))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(terms["invariance"]))
        self.assertGreaterEqual(terms["variance"].item(), 0.0)
        self.assertGreaterEqual(terms["covariance"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
