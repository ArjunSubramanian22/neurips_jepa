# License / provenance audit

This file records **apparent** origins for inherited or third-party material
in the anonymous artifact. It does **not** grant a license and does **not**
claim compliance where the status is uncertain.

The project tree as shipped has **no root LICENSE** asserting terms for the
submission's own code. Redistribution beyond confidential review is therefore
not authorized by this document.

Manuscript LaTeX, NeurIPS style files, and the bibliography are **not**
included in this code/results snapshot.

## NumPy MLS-MPM simulator

- **Path:** `scripts/generate_trajectories.py` (`MLSMPM2D`)
- **Apparent origin:** In-repo vectorized NumPy implementation documented as a
  Taichi-style MLS-MPM following Hu et al. (2018) and the widely circulated
  88-line Taichi demo. Constitutive stress in code is fixed-corotated SVD
  (not neo-Hookean); `outputs/physics_mpm/metadata.json` still labels the dump
  neo-Hookean (stale metadata, scientific value unchanged).
- **License if established:** Taichi (`taichi-dev/taichi`) is commonly Apache
  2.0, but this tree does **not** vendor Taichi sources and does **not**
  include an Apache notice for an 88-line excerpt.
- **Attribution requirements:** Keep the Hu et al. 2018 credit in the
  simulator module docstring.
- **Redistribution:** UNRESOLVED — DO NOT CLAIM LICENSE COMPLIANCE for any
  possibly copyrighted expression inherited from demo code. The implementation
  here is a NumPy rewrite, not a paste of Taichi Python.
- **Evidence:** Module docstring; stress assembly around the SVD / polar
  rotation in `MLSMPM2D.substep`; absence of a third-party LICENSE file.
- **Uncertainty:** High. Treat as an algorithm reimplementation pending a
  line-by-line comparison to a specific licensed demo file, which was not
  performed for this snapshot.

## Particle-graph baseline

- **Path:** `src/models/particle_gnn.py`
- **Apparent origin:** Original code in this tree, in the GNS / DPI-Net
  *family* (encode-process-decode over the mass-spring lattice).
- **License if established:** Unspecified (project has no LICENSE).
- **Attribution requirements:** File docstring credits DPI-Net and GNS papers.
- **Redistribution:** UNRESOLVED — DO NOT CLAIM LICENSE COMPLIANCE for the
  project code as a whole. No evidence of copied Google / Stanford GNS
  sources in this file.
- **Evidence:** File docstring; no third-party copyright header.

## VICReg / JEPA losses and ViT predictor

- **Path:** `src/losses.py`, `src/models/ac_jepa.py`
- **Apparent origin:** Original code implementing a VICReg-*style* prediction
  regularizer (variance/covariance on the prediction and a detached target)
  and an action-conditioned JEPA. This is **not** a copy of the Facebook
  VICReg or I-JEPA repositories.
- **License if established:** Unspecified for this tree.
- **Attribution requirements:** Method names in comments refer to published
  algorithms, not submitting-author identity.
- **Redistribution:** UNRESOLVED — DO NOT CLAIM LICENSE COMPLIANCE
- **Evidence:** `src/losses.py` comments; no vendored `vicreg` package.

## Python / scientific stack (not vendored)

PyTorch, NumPy, Matplotlib, PyYAML, Pillow, scikit-learn, optional wandb.
These are declared in `requirements.txt` and are **not** copied into the
artifact. Their licenses apply when a reviewer installs them.

## Dump metadata

- **Path:** `outputs/physics_mpm/metadata.json`
- **Note:** Constitutive string still says neo-Hookean. Code path is
  fixed-corotated. This is a known stale label, not a license claim.

## Summary

| Component | Status |
| --- | --- |
| Project code LICENSE | Missing — do not claim a grant |
| MLS-MPM vs Taichi 88-line | UNRESOLVED — DO NOT CLAIM LICENSE COMPLIANCE |
| GNS / VICReg / JEPA implementations | Original-looking; project license unspecified |
| Manuscript / style / bibliography | Not shipped in this snapshot |
