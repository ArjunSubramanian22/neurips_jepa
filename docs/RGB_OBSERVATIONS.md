# RGB observations

The training observation is still the 3×64×64 density/velocity raster from `render_rgb` in `scripts/generate_trajectories.py`. `rgb_v1` is a separate appearance image of the same particle positions.

## Renderer

`src/rgb_render.py` paints float RGB in `[0, 1]`:

- fixed object color `(0.82, 0.32, 0.18)` and background `(0.94, 0.93, 0.90)`. Color does not depend on stiffness.
- camera center `(0.5, 0.5)` and scale `1`, which matches the physical raster's full-box framing
- Lambert-style shading from the coverage gradient and a fixed light
- optional world-space sine texture
- optional seeded background blobs

`rgb_v1` does not paint velocity. Two states with the same positions and different velocities produce the same RGB image. Action visibility in RGB has to come from a change in shape or position.

Splat width scales with resolution: `1.35 * resolution / 64`. Configurations:

- `configs/rgb/rgb_v1_64.yaml`
- `configs/rgb/rgb_v1_128.yaml`
- `configs/rgb/rgb_v1_256.yaml`

`render_rgb_cached` stores `renderer_metadata.json` and one `.npy` per state. The key hashes the metadata and the position and velocity bytes.

## Metrics

`frame_visibility` compares a factual frame with the paired counterfactual frame:

- exact uint8 equality and the fraction of pixels with any channel changed
- pixel MSE on `[0, 1]`
- PSNR, which is null with status `identical` when MSE is 0
- SSIM, implemented here (11×11 Gaussian, range 1). Identical frames score 1
- LPIPS is `unavailable`. The `lpips` package is not installed and no network weights are cached. The value is not filled in
- mean and max absolute difference, and the fraction of pixels whose absolute difference exceeds one uint8 level

## Visibility run

`rgb_visibility_v1` rolls a force reversal for three frames on disk and bar at Young's modulus 70, in `mpm_legacy` and `mpm_high_excitation`. No world model is trained. Each cell below is the median of 6 frames (2 shapes × 3 frames).

| Environment | Pipeline | Affected pixels | Pixel MSE | SSIM | Identical uint8 |
| --- | --- | --- | --- | --- | --- |
| legacy | physical raster | 0 | 0 | 1 | 1 |
| legacy | RGB 64 | 0 | 0 | 1 | 1 |
| legacy | RGB 128 | 0 | 0 | 1 | 1 |
| legacy | RGB 256 | 0 | 0 | 1 | 1 |
| high excitation | physical raster | 0.192 | 1.63e-3 | 0.928 | 0.834 |
| high excitation | RGB 64 | 0.124 | 1.93e-4 | 0.990 | 0.889 |
| high excitation | RGB 128 | 0.118 | 6.27e-5 | 0.997 | 0.901 |
| high excitation | RGB 256 | 0.120 | 3.21e-5 | 0.999 | 0.910 |

The legacy median is zero because at least half of those six frames are byte-identical. The maximum affected-pixel fraction on legacy is 0.069 for the physical raster and 0.020, 0.041, and 0.024 for RGB 64, 128, and 256. On the high-excitation fork, the physical raster changes a larger share of pixels than RGB at any of the three resolutions. RGB affected-pixel fractions stay near 0.12 from 64 to 256, while pixel MSE falls as unchanged background occupies more of the frame.

The qualitative pair is high-excitation disk, frame 2, chosen as the RGB-128 frame with the largest affected-pixel fraction:

- `figures/rgb_observations/rgb128_counterfactual_difference.png`
- `figures/rgb_observations/physical_counterfactual_difference.png`

Per-frame rows are `experiments/runs/rgb_visibility_v1/visibility.csv`. Rendered frames are under that run's `cache/`.

```bash
python3.13 experiments/rgb_observations/run_visibility.py
```

The experiment id is already complete. A second run will not overwrite it.

## What this does not say

These medians are not a trained model's action sensitivity. They are how much of a force reversal is visible in each observation. The legacy actuator often leaves both representations unchanged at uint8 precision. Where the high-excitation fork is visible, the density/velocity raster exposes more changed pixels than the RGB appearance image, including at 256.
