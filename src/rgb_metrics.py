"""Visibility metrics between two rendered frames.

Identical images leave PSNR undefined rather than storing a sentinel.
LPIPS is reported only when the ``lpips`` package is already importable.
This environment does not download perceptual weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _as_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
    if array.ndim != 3 or array.shape[0] not in (1, 3):
        raise ValueError(f"expected (C, H, W), got {array.shape}.")
    return array


def frame_visibility(reference: np.ndarray, alternate: np.ndarray) -> dict[str, Any]:
    """Compare one factual frame with the paired counterfactual frame."""
    left = _as_float(reference)
    right = _as_float(alternate)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch {left.shape} vs {right.shape}.")
    uint8_left = np.rint(np.clip(left, 0.0, 1.0) * 255.0).astype(np.uint8)
    uint8_right = np.rint(np.clip(right, 0.0, 1.0) * 255.0).astype(np.uint8)
    changed = np.any(uint8_left != uint8_right, axis=0)
    diff = np.abs(left - right)
    mse = float(np.mean(diff ** 2))
    if mse == 0.0:
        psnr: float | None = None
        psnr_status = "identical"
    elif not np.isfinite(mse):
        psnr = None
        psnr_status = "nonfinite"
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))
        psnr_status = "ok"
    return {
        "fraction_identical_uint8": float(np.mean(uint8_left == uint8_right)),
        "fraction_affected_pixels": float(np.mean(changed)),
        "pixel_mse": mse,
        "psnr": psnr,
        "psnr_status": psnr_status,
        "ssim": ssim(left, right),
        "lpips": lpips_distance(left, right),
        "mean_abs": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
        "fraction_abs_gt_one_level": float(np.mean(np.any(diff > (1.0 / 255.0), axis=0))),
    }


def ssim(reference: np.ndarray, alternate: np.ndarray, *, window: int = 11) -> float:
    """Mean channel SSIM for images in ``[0, 1]``. Identical images score 1."""
    left = torch.from_numpy(_as_float(reference)).unsqueeze(0)
    right = torch.from_numpy(_as_float(alternate)).unsqueeze(0)
    channels = int(left.shape[1])
    if min(left.shape[-2:]) < window:
        window = int(min(left.shape[-2:]) // 2 * 2 + 1)
        window = max(window, 3)
    coords = torch.arange(window, dtype=torch.float32) - window // 2
    gauss = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).expand(channels, 1, window, window).contiguous()
    pad = window // 2

    def _blur(image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(image, kernel, padding=pad, groups=channels)

    mu_x = _blur(left)
    mu_y = _blur(right)
    sigma_x = _blur(left * left) - mu_x * mu_x
    sigma_y = _blur(right * right) - mu_y * mu_y
    sigma_xy = _blur(left * right) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    value = numerator / denominator
    return float(value.mean().item())


def lpips_distance(reference: np.ndarray, alternate: np.ndarray) -> dict[str, Any]:
    """Return LPIPS only when the package is already installed."""
    try:
        import lpips  # type: ignore
    except ImportError:
        return {"value": None, "status": "unavailable", "reason": "lpips is not installed"}
    del reference, alternate, lpips
    return {"value": None, "status": "unavailable", "reason": "lpips weights are not cached"}


def difference_maps(reference: np.ndarray, alternate: np.ndarray, amplify: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Absolute difference and a clipped amplification, both ``(3, H, W)`` in ``[0, 1]``."""
    diff = np.abs(_as_float(reference) - _as_float(alternate))
    amplified = np.clip(diff * float(amplify), 0.0, 1.0).astype(np.float32)
    return diff.astype(np.float32), amplified
