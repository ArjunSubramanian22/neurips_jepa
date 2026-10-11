"""Deterministic RGB renderer. It does not replace the density/velocity raster.

``render_rgb`` in ``scripts/generate_trajectories.py`` remains the physical
observation. This module paints an appearance image from the same particle
positions. ``rgb_v1`` shading uses the silhouette, not the velocity channels,
and the object color does not depend on stiffness.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

RENDERER_VERSION = "rgb_v1"
SUPPORTED_RESOLUTIONS = (64, 128, 256)


@dataclass(frozen=True)
class RGBRenderConfig:
    """Versioned appearance and camera settings."""

    resolution: int = 64
    object_color: tuple[float, float, float] = (0.82, 0.32, 0.18)
    background_color: tuple[float, float, float] = (0.94, 0.93, 0.90)
    camera_center: tuple[float, float] = (0.5, 0.5)
    camera_scale: float = 1.0
    shading: bool = True
    light_dir: tuple[float, float, float] = (0.35, -0.45, 0.82)
    texture: bool = False
    texture_frequency: float = 18.0
    texture_amplitude: float = 0.12
    distractors: bool = False
    distractor_seed: int = 0
    n_distractors: int = 4
    splat_sigma_at_64: float = 1.35
    coverage_gain: float = 1.6
    version: str = RENDERER_VERSION

    def __post_init__(self) -> None:
        if int(self.resolution) not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"resolution must be one of {SUPPORTED_RESOLUTIONS}.")
        if float(self.camera_scale) <= 0.0:
            raise ValueError("camera_scale must be positive.")
        if float(self.splat_sigma_at_64) <= 0.0:
            raise ValueError("splat_sigma_at_64 must be positive.")
        if len(self.object_color) != 3 or len(self.background_color) != 3:
            raise ValueError("colors must have three channels.")


def preset(resolution: int, **overrides: Any) -> RGBRenderConfig:
    """Named resolution with the shared rgb_v1 appearance."""
    return RGBRenderConfig(resolution=int(resolution), **overrides)


def splat_sigma_px(cfg: RGBRenderConfig) -> float:
    """Keep the same world-space blur as the 64-pixel physical raster."""
    return float(cfg.splat_sigma_at_64) * float(cfg.resolution) / 64.0


def config_metadata(cfg: RGBRenderConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["renderer_version"] = RENDERER_VERSION
    payload["splat_sigma_px"] = splat_sigma_px(cfg)
    return payload


def _distractors(cfg: RGBRenderConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(cfg.distractor_seed))
    centers = rng.uniform(0.08, 0.92, size=(int(cfg.n_distractors), 2)).astype(np.float32)
    radii = rng.uniform(0.025, 0.055, size=(int(cfg.n_distractors),)).astype(np.float32)
    colors = rng.uniform(0.25, 0.75, size=(int(cfg.n_distractors), 3)).astype(np.float32)
    return centers, radii, colors


def _world_grid(cfg: RGBRenderConfig) -> tuple[np.ndarray, np.ndarray]:
    step = np.linspace(0.0, 1.0, int(cfg.resolution), dtype=np.float32)
    pixel_y, pixel_x = np.meshgrid(step, step, indexing="ij")
    world_x = (pixel_x - np.float32(0.5)) / np.float32(cfg.camera_scale) + np.float32(cfg.camera_center[0])
    world_y = (pixel_y - np.float32(0.5)) / np.float32(cfg.camera_scale) + np.float32(cfg.camera_center[1])
    return world_x, world_y


def _background(cfg: RGBRenderConfig) -> np.ndarray:
    color = np.asarray(cfg.background_color, dtype=np.float32)
    image = np.empty((3, int(cfg.resolution), int(cfg.resolution)), dtype=np.float32)
    image[:] = color[:, None, None]
    if not cfg.distractors:
        return image
    world_x, world_y = _world_grid(cfg)
    centers, radii, colors = _distractors(cfg)
    for index in range(centers.shape[0]):
        dist2 = (world_x - centers[index, 0]) ** 2 + (world_y - centers[index, 1]) ** 2
        alpha = np.exp(-dist2 / (np.float32(2.0) * radii[index] ** 2))
        image = image * (1.0 - alpha) + colors[index][:, None, None] * alpha
    return image.astype(np.float32)


def _shade(coverage: np.ndarray, cfg: RGBRenderConfig) -> np.ndarray:
    if not cfg.shading:
        return np.ones_like(coverage)
    grad_y, grad_x = np.gradient(coverage)
    light = np.asarray(cfg.light_dir, dtype=np.float64)
    light = light / np.linalg.norm(light)
    norm = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.0)
    shade = (-grad_x * light[0] - grad_y * light[1] + light[2]) / norm
    return np.clip(0.42 + 0.58 * shade, 0.0, 1.0).astype(np.float32)


def render_rgb_image(
    positions: np.ndarray,
    velocities: np.ndarray,
    cfg: RGBRenderConfig,
) -> np.ndarray:
    """Return float RGB ``(3, H, W)`` in ``[0, 1]``.

    ``velocities`` are part of the state signature but ``rgb_v1`` does not
    paint them. Two states with the same positions and different velocities
    therefore match.
    """
    del velocities  # rgb_v1 appearance is positional. Kept for the cache key.
    points = np.asarray(positions, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")
    height = int(cfg.resolution)
    coverage = np.zeros((height, height), dtype=np.float32)
    albedo = np.zeros((3, height, height), dtype=np.float32)
    if points.size:
        center = np.asarray(cfg.camera_center, dtype=np.float32)
        scale = np.float32(cfg.camera_scale)
        pixel = (points - center) * scale + np.float32(0.5)
        pixel = pixel * np.float32(height - 1)
        sigma = np.float32(splat_sigma_px(cfg))
        radius = max(1, int(np.ceil(3.0 * float(sigma))))
        inv_two_s2 = np.float32(1.0 / (2.0 * float(sigma) * float(sigma) + 1e-8))
        ix = np.rint(pixel[:, 0]).astype(np.int32)
        iy = np.rint(pixel[:, 1]).astype(np.int32)
        color = np.asarray(cfg.object_color, dtype=np.float32)
        if cfg.texture:
            phase = np.sin(points[:, 0] * np.float32(cfg.texture_frequency)) * np.sin(
                points[:, 1] * np.float32(cfg.texture_frequency)
            )
            color = color * (np.float32(1.0) + np.float32(cfg.texture_amplitude) * phase)[:, None]
        else:
            color = np.repeat(color[None, :], points.shape[0], axis=0)
        for dy in range(-radius, radius + 1):
            yy = np.clip(iy + dy, 0, height - 1)
            for dx in range(-radius, radius + 1):
                xx = np.clip(ix + dx, 0, height - 1)
                weight = np.exp(-np.float32(dx * dx + dy * dy) * inv_two_s2)
                np.add.at(coverage, (yy, xx), weight)
                np.add.at(albedo[0], (yy, xx), weight * color[:, 0])
                np.add.at(albedo[1], (yy, xx), weight * color[:, 1])
                np.add.at(albedo[2], (yy, xx), weight * color[:, 2])
    mass = coverage + np.float32(1e-8)
    painted = albedo / mass
    shade = _shade(coverage, cfg)
    painted = painted * shade
    alpha = 1.0 - np.exp(-coverage * np.float32(cfg.coverage_gain))
    background = _background(cfg)
    image = background * (1.0 - alpha) + painted * alpha
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _cache_key(cfg: RGBRenderConfig, positions: np.ndarray, velocities: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(config_metadata(cfg), sort_keys=True).encode("utf-8"))
    digest.update(np.ascontiguousarray(positions, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(velocities, dtype=np.float32).tobytes())
    return digest.hexdigest()


def render_rgb_cached(
    positions: np.ndarray,
    velocities: np.ndarray,
    cfg: RGBRenderConfig,
    cache_dir: Path,
) -> np.ndarray:
    """Read a cached frame or render and store it. The key includes renderer metadata."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "renderer_metadata.json"
    if not meta_path.is_file():
        meta_path.write_text(json.dumps(config_metadata(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    key = _cache_key(cfg, positions, velocities)
    path = cache_dir / f"{key}.npy"
    if path.is_file():
        return np.load(path)
    image = render_rgb_image(positions, velocities, cfg)
    np.save(path, image)
    return image
