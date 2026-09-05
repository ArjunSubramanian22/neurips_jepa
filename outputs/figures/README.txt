VICReg columns use a post-hoc frozen-latent PixelDecoder; it is not part of AC-JEPA.
Pixel-AE source: short qualitative Pixel-AE fit (not the unfinished 200-epoch workshop run)
VICReg checkpoint snapshot: outputs/vicreg/best.pt
Horizons: k in {1, 5, 10}
Showcase episodes: 1800 (E=18), 1801 (E=70), 1802 (E=220), 1803 (E=18, geom=1)
Failure case: episode 1841 (worst val latent rollout MSE at k=10)
