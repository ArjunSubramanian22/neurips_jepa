"""VICReg loss for Action-Conditioned JEPA.

VICReg (Bardes, Ponce, LeCun, ICLR 2022) prevents representation collapse
without negative samples by combining three terms on a pair of embeddings
``(ẑ_{t+1}, z_{t+1})``:

* **Invariance** — MSE between the predictor output and the target latent.
* **Variance** — hinge that keeps per-feature batch std at least ``gamma``
  (default 1), blocking constant embeddings.
* **Covariance** — off-diagonal penalty on the batch covariance, decorrelating
  latent dimensions.

The target branch is expected to be stop-gradient (EMA encoder). Variance and
covariance are still evaluated on both tensors so the reported scale matches
the original two-view VICReg objective; gradients flow only through ``z_hat``.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VICRegTerms(TypedDict):
    """Unweighted constituent losses plus the combined scalar."""

    loss: Tensor
    invariance: Tensor
    variance: Tensor
    covariance: Tensor


def invariance_loss(z_hat: Tensor, z_target: Tensor) -> Tensor:
    """Mean squared error between predicted and target latents.

    Args:
        z_hat: Predictor output ``ẑ_{t+1}``, shape ``(B, D)``.
        z_target: Stop-gradient target ``z_{t+1}``, shape ``(B, D)``.

    Returns:
        Scalar MSE averaged over batch and feature dimensions.
    """
    return F.mse_loss(z_hat, z_target, reduction="mean")


def variance_loss(z: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """Hinge on per-feature standard deviation across the batch.

    Enforces ``std_j(z) >= gamma`` for every latent channel ``j``.

    Args:
        z: Embeddings of shape ``(B, D)``.
        gamma: Minimum allowed standard deviation (VICReg uses 1).
        eps: Ridge added inside the square root for numerical stability.

    Returns:
        Scalar ``mean_j ReLU(gamma - std_j)``.
    """
    if z.ndim != 2:
        raise ValueError(f"variance_loss expects (B, D) embeddings, got {tuple(z.shape)}.")
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z: Tensor) -> Tensor:
    """Off-diagonal squared covariance, normalized by feature count.

    Args:
        z: Embeddings of shape ``(B, D)``. ``B == 1`` returns zero (covariance
            is undefined for a single sample).

    Returns:
        Scalar ``(1/D) * sum_{i ≠ j} C_{ij}^2``.
    """
    if z.ndim != 2:
        raise ValueError(f"covariance_loss expects (B, D) embeddings, got {tuple(z.shape)}.")
    batch_size, dim = z.shape
    if batch_size < 2 or dim < 2:
        return z.new_zeros(())
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.transpose(0, 1) @ z_centered) / float(batch_size - 1)
    off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
    return off_diag / float(dim)


def vicreg_loss(
    z_hat: Tensor,
    z_target: Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[Tensor, VICRegTerms]:
    """Full VICReg objective between ``ẑ_{t+1}`` and ``z_{t+1}``.

    Args:
        z_hat: Online prediction, shape ``(B, D)``. Gradients flow here.
        z_target: EMA target latent, shape ``(B, D)``. Should be detached.
        sim_coeff: Weight ``λ`` on the invariance (MSE) term.
        std_coeff: Weight ``μ`` on the variance hinge (summed over both views).
        cov_coeff: Weight ``ν`` on the covariance penalty (summed over both views).
        gamma: Variance hinge threshold.
        eps: Numerical ridge for the standard-deviation estimate.

    Returns:
        ``(loss, terms)`` where ``loss`` is the weighted sum and ``terms``
        stores unweighted constituents for logging.
    """
    if z_hat.shape != z_target.shape:
        raise ValueError(
            f"z_hat and z_target shape mismatch: {tuple(z_hat.shape)} vs {tuple(z_target.shape)}."
        )
    if z_hat.ndim != 2:
        raise ValueError(f"VICReg expects (B, D) latents, got {tuple(z_hat.shape)}.")

    inv = invariance_loss(z_hat, z_target)
    var = variance_loss(z_hat, gamma=gamma, eps=eps) + variance_loss(
        z_target, gamma=gamma, eps=eps
    )
    cov = covariance_loss(z_hat) + covariance_loss(z_target)
    loss = sim_coeff * inv + std_coeff * var + cov_coeff * cov
    terms: VICRegTerms = {
        "loss": loss,
        "invariance": inv,
        "variance": var,
        "covariance": cov,
    }
    return loss, terms


def prediction_objective(
    z_hat: Tensor,
    z_target: Tensor,
    loss_type: str = "vicreg",
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[Tensor, VICRegTerms]:
    """Train-time objective with VICReg logging terms always populated.

    ``loss_type="vicreg"`` backprops the full weighted VICReg objective.
    ``"vicreg_no_cov"`` keeps invariance + variance (Variant A).
    ``"mse"`` is invariance only (Variant B). ``smooth_l1`` and ``cosine``
    similarly replace the invariance term while still reporting var/cov.
    """
    _, terms = vicreg_loss(
        z_hat,
        z_target,
        sim_coeff=sim_coeff,
        std_coeff=std_coeff,
        cov_coeff=cov_coeff,
        gamma=gamma,
        eps=eps,
    )
    if loss_type == "vicreg":
        return terms["loss"], terms
    if loss_type == "vicreg_no_cov":
        loss = sim_coeff * terms["invariance"] + std_coeff * terms["variance"]
        terms["loss"] = loss
        return loss, terms
    if loss_type == "mse":
        loss = terms["invariance"]
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(z_hat, z_target, reduction="mean")
        terms["invariance"] = loss
    elif loss_type == "cosine":
        loss = 1.0 - F.cosine_similarity(z_hat, z_target, dim=-1).mean()
        terms["invariance"] = loss
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'.")
    terms["loss"] = loss
    return loss, terms


class VICRegLoss(nn.Module):
    """Module wrapper around :func:`vicreg_loss` for use as a criterion."""

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    def forward(self, z_hat: Tensor, z_target: Tensor) -> tuple[Tensor, VICRegTerms]:
        return vicreg_loss(
            z_hat,
            z_target,
            sim_coeff=self.sim_coeff,
            std_coeff=self.std_coeff,
            cov_coeff=self.cov_coeff,
            gamma=self.gamma,
            eps=self.eps,
        )
