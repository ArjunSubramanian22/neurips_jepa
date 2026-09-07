# Action-path audit

**ACTION PATH AUDIT: PASS**

## Checks
1. Dataset indexing: `x[e,t], a[e,t], x[e,t+1]` matches MLS-MPM replay of dump seed 0.
2. Train-only z-score; validation uses the same mean/std.
3. Canonical workshop crop is disabled (`crop_scale_min=1.0`); `warp_contact_xy` is unused.
4. Action-conditioned checkpoints have `condition_on_action=True` and a live `action_embed`.
5. No-action checkpoints use `uncond_proj` and ignore the action tensor.
6. Dummy backward sends nonzero gradients into `action_embed` on action models.
7. Trained `action_embed` weights are not identically zero and differ from a fresh init.
8. Changing a normalized action changes the action-conditioned predictor output.
9. Unit tests in `tests/test_action_path.py`.

## Tensor stats
- action mean (train): [0.5038278698921204, 0.4293104112148285, 0.00684838742017746, -0.006768804509192705]
- action std (train): [0.08210279047489166, 0.05712604150176048, 0.4015662670135498, 0.39795324206352234]
- std collapsed: False
- dump replay actions equal: True
- dump replay obs close: True
- crop disabled: True

## Bugs
_none_

Architectural/gradient sensitivity is **not** behavioral action use. Phase 1 simulator
counterfactuals remain the primary test.
