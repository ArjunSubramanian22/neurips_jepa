# PhysWorldAI audit matrix (Phase 0–2)

Date: 2026-09-07. No retraining. Phase 0–1 is the issue inventory. Phase 2 is the required non-training controls (VN contrast, state perturbation, per-horizon rank, stiffness hierarchy, full replay diff, canonical evaluator).

**Canonical sources**

| Role | Location |
| --- | --- |
| Original dump + 10 last.pt | this git workspace |
| Revised paper source/PDF/tables | Downloads zip `...Joint_Embedding_Predictive_World_Models__1_.zip` and attached `PhysWorldAI_Revised__2_.pdf` |
| Original counterfactuals | `results/reviewer_strengthening/01_simulator_counterfactual/` |
| Phase-1 measurements | `results/audit_phase1/` |
| Stale manuscript in git | `paper/main.tex` (old title; ignore for submission claims) |

**Final checkpoint set: VERIFIED**, not assumed.

| Condition | Seeds | Epoch | Crop | SHA match to paper registry |
| --- | --- | --- | --- | --- |
| VICReg + action | 42, 43, 44, 45 | 200 | 1.0 | yes |
| VICReg no-action | 42, 43, 44, 45 | 200 | 1.0 | yes |
| MSE-only + action | 42, 43 | 200 | 1.0 | yes |

See `CHECKPOINT_MANIFEST.csv`. Cropped `outputs/vicreg/last.pt` and `outputs/mse/last.pt` exist, are epoch 200 with `crop_scale_min=0.75`, and are **obsolete**.

---

## Phase 2 decision table (no retraining)

Controls used original dump observations (bitwise identical to local RNG replay; SHA `ee3fe0fe…`). Headline MSE@10 on this dump matches the paper’s Linux-replay Table 3 to printed precision.

| Finding | Survives? | Exact number | Paper implication | Retrain? | Abstract change? |
| --- | --- | --- | --- | --- | --- |
| Predictor attenuates action vs target (raw D) | Yes | High-vis median D_pred / D_tgt = **0.0051** (s42), **0.022** (s43); n=579 | Keep §5.2. Stronger with visibility stratification | No | No (already claimed) |
| Attenuation is only global shrinkage | **No** — shrinkage does not explain it | pred_var **0.510** > tgt_var **0.272** (s42). VN ratio **0.0027** (s42 high-vis), **0.012** (s43) | VN **strengthens** D_pred ≪ D_tgt. Report both raw and VN | No | Optional: say attenuation survives variance normalization |
| Predictor is dead / cannot move | **No** | Matched-L2 state perturbation: median D_state / D_action = **59×** (s42, D_tgt>1e-8, n=1727) and **190×** (high-vis). 86–97% of those pairs have D_state > D_action | Isolates **action-path under-use**. Add this control next to Fig 3 | No | **Yes, strengthen**: predictor responds to state, not to action |
| Observation bottleneck (identical rasters) | Yes | **41.5%** identical uint8; **0 / 393,216,000** local replay pixel diffs | Keep §5.1. Local dump is self-reproducing | No | No |
| Physical intervention is object-scale | **No** | Median RMS pos **1.19e-5** = **7.5e-4 px** | Keep subpixel wording; do not imply large motion | No | No |
| All four action dims equally effective | **No** | Force ≈ **2×** center at 1-std; total impulse independent of (cx,cy) | Qualify actuator in methods | No | No |
| MSE-only low error is a competitive model | **No** | MSE@10 **0.138** vs VICReg **1.156** (8.36×); pred rank@1 already **3.6–4.3** vs VICReg **2.4–9.2** | Frame as collapse / metric failure. Pooled rank overstates VICReg diversity (s42 **5.97@1** vs **12.27** pooled) | No | **Yes**: do not lead with 8.36× as an advantage; keep as disagreement + collapse |
| Target collapse ruled out | **No** | VICReg s43 target rank **2.12** at every k∈{1,3,5,10}; top-1 **0.67** | Table 3 must show target rank per horizon | No | Mention if discussing spectra |
| Stiffness absent from JEPA encoder | **Not supported** | Full-res uint8/float stiffness **33.0–33.5%** (chance **33.5%**) at t=0..15. Mechanical x,v,C,F **33–40%**. Privileged μ,λ MLP **100%**; linear μ,λ only **66.5%** (probe capacity). Geometry **100%** from particles and rasters | Rewrite as **not identifiable under this excitation**, not encoder discard | No | **Yes**: “stiffness is not recoverable from particles or rasters here,” not “the representation discards stiffness” |
| Action vs no-action advantage | **No** (null) | MSE@10 ordering still flips 42/44 vs 43/45 on original dump | Keep null | No | No |
| Need new simulator / aug-off / fresh seeds before submission edits | **No** | None of the six controls invalidated the three supported findings; they **tightened** them | Do not retrain. Edit claims/figures only | No | See rows above |

Retraining, aug-off, unused-seed eval, and actuator redesign remain **not required** for an honest paper.

---

## Issue matrix

| # | Reviewer concern | Verified? | Evidence | Code / data | Paper section | Severity | Fix (not yet applied) | Retrain? | Done? | Final interpretation this pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Normalized actuator makes (cx,cy) nearly inert | PARTIALLY VERIFIED | Total impulse is weight-normalized, so distant centers still inject the full scaled force (code + Eq. 8). Centers are **not** inert: 1-std comparable effects are ~2× larger for force than center, but center FD sensitivity per unit is nonzero. Secondary CF `contact_perturbed` (Δcx=+0.10) median RMS pos 4.64e-6 vs `force_reversed` 1.58e-5. Small FD at t=3, N=16 val states, h=1: median RMS pos cx 1.48e-6 / cy 0.91e-6 / fx 2.36e-6 / fy 2.36e-6. ~69–75% uint8-identical after those perturbations. | `scripts/generate_trajectories.py` `_action_force`; `results/audit_phase1/actuator_sensitivity.csv`; CF Table 5 | App A already discloses total-impulse normalization; main text still treats a 4-D action | High for interpretation; not a silent bug | State: location changes **distribution**, not total impulse. Do not claim four equally meaningful dimensions. Do **not** redesign actuator unless we open a new simulator condition. | No | Measurement done | Keep current checkpoints. Qualify action semantics. |
| 2 | 0.92^8 damping erases action memory | VERIFIED | Math: damping cfg=0.08 ⇒ multiplier 0.92 per substep; 0.92^8 = 0.51321887. Impulse then zero action (N=16): median RMS\|v\| retention h5/h1 = 0.163 (damp 0.08), 0.394 (0.03), 0.707 (0.01). Displacement after 4 further obs remains 1.6e-5 / 3.6e-5 / 5.6e-5. | `MLSMPM2D.substep`; `results/audit_phase1/damping_impulse.csv` | App A Eq. 8; §5.1 “actuation and damping” | High (environment-side) | Report impulse response. Do not call short-horizon disappearance a JEPA failure. | No | Measurement done | Environment removes much velocity in a few observations. |
| 3 | “Observation bottleneck” is just weak physics | PARTIALLY VERIFIED | Physical scale is tiny: shuffled h=1 median RMS pos 1.19e-5 = 7.5e-4 px = 4.6e-4 particle-spacings. **Category B exists:** 1246/3000 pairs have nonzero particle Δ and identical uint8 rasters. Of all 3000, 48.5% have density-identical / velocity-different rasters; only 9.9% change density. **Among visible bins, D_pred ≪ D_tgt still holds:** high-raster n=579, median D_tgt 4.13e-4 vs D_pred 2.14e-6 (97% D_pred<D_tgt). Top 10% raster n=300: D_tgt 1.19e-3 vs D_pred 1.95e-6. D_pred is almost flat across bins. | CF CSVs; `results/audit_phase1/visibility_stratification.json` | §5.1–5.2 uses all pairs + largest action quintile; **not** visibility-stratified | High | Add visibility-stratified figure (best use of the 8th page). Keep observation-bottleneck for identical-raster subset. Keep predictor-attenuation for visible subset. Do not pretend displacements are object-scale. | No | CSV analysis done | Both bottlenecks exist. Physical magnitude is still subpixel. |
| 4 | Stiffness near chance may be under-excitation, not encoder discard | **VERIFIED** | Full-res float and uint8 rasters: stiffness 33.0–33.5% at t=0,3,5,7,9,12,15 and history (majority 33.5%). Mechanical x,v,C,F: 33–40% (peak 39.5% linear at t=3). Privileged μ,λ: MLP 100%, linear 66.5%. Geometry 100% from particles and rasters (positive control). Shuffled-label ~33.5%. | `results/audit_phase2/stiffness_positive_controls.csv` | §5.5, App F | High | Rewrite: not identifiable under this excitation/observation. Do not blame the encoder. | No | **Yes** | Dataset/intervention under-excites stiffness. |
| 5 | D_pred < D_tgt is global shrinkage | **NOT SUPPORTED as the explanation** | pred_var 0.510 (s42) / 0.289 (s43) **exceeds** tgt_var 0.272 / 0.154. VN high-vis ratio 0.0027 / 0.012, smaller than raw. State perturbation with \|\|δ\|\|_2=sqrt(d D_tgt): D_state/D_action median 59× (s42 included) and 16.9× (s43). Recomputed D_pred matches archive to 1e-9. | `results/audit_phase2/vn_state_perturbation.csv` | §5.2 | High (now resolved) | Report VN contrasts + state-vs-action control. Keep attenuation claim. | No | **Yes** | Action-path under-use, not dead predictor or scale artifact. |
| 6 | MSE-only is collapsed, not an objective win | PARTIALLY VERIFIED | MSE pred rank 4.31/3.46; top-1 0.45/0.51. VICReg matched seeds 12.27/5.78. Paper already says error ≠ better physics. Abstract still leads with 8.36×. Target ranks for MSE are 4.49/3.70 — similar to pred — so MSE collapse is in **both** branches, not pred-only. | `data/final_run_summary.csv` | Abstract; §5.4 | Medium | Reframe §5.4 as collapse / metric-failure. Keep 8.36× only as a cautionary metric disagreement. | No | Numbers exist | Partial collapse / spectral concentration. Not a competitive world model. |
| 7 | Not standard VICReg | VERIFIED (code); paper mostly already fixed | `vicreg_loss` evaluates var/cov on z_hat **and** detached z_target; gradients only through z_hat (EMA stop-grad). This is VICReg-style prediction regularization, not two-view VICReg. | `src/losses.py` | §3 and App B already say this | Medium (naming) | Rename remaining “VICReg loss” comments/module docs. Keep main-text wording. | No | Paper OK | Do not call it standard VICReg. |
| 8 | Report target-encoder rank | VERIFIED as missing from Table 3; **now measured per horizon** | Pooled target ranks unchanged. Per-horizon target rank is **stable**: s43 **2.10–2.14** at k=1,3,5,10 (top-1 ~0.66–0.67). s42 ~4.61–4.66. MSE targets ~3.67–4.56. | `results/final_checkpoint_results.csv` | Table 3 | High | Add target rank @1/3/5/10. | No | **Yes** | Target collapse not ruled out, especially s43. |
| 9 | Photometric augs on physical channels | VERIFIED as implemented; ablation missing | `crop_scale_min=1` skips crop. Jitter 0.4 and blur p=0.25 apply brightness/contrast/saturation to density/vx/vy. YAML `augment: true`. No aug-off run. | `src/dataset.py` `ConsistentFrameAugment`; `configs/workshop_vicreg.yaml` | App B discloses; no ablation | Medium | One aug-off **pilot** seed after remaining cheap evals. Expand only if material. | Not yet | Not run | Do not claim augs are physically valid. |
| 10 | Rank must be per horizon | VERIFIED; **pooling inflates VICReg pred rank** | s42 pred rank **5.97@1 / 8.26@3 / 9.38@5 / 9.91@10** vs pooled **12.27**. s43 **2.44@1** vs pooled **5.78**. MSE already low at k=1 (**4.30 / 3.59**). | `results/final_checkpoint_results.csv`; `scripts/evaluate_final.py` | Eq. 3, Table 3 | High | Report rank@1,3,5,10. Pooled supplemental only. | No | **Yes** | Do not cite pooled rank as the primary geometry number. |
| 11 | Geometry is a positive control | PARTIALLY VERIFIED | Accuracies 99.75–100%. Paper §5.5 states recoverability without calling it a scientific finding. Could be more explicit (“pipeline sanity check”). | Table 4 | §5.5 | Low | One sentence: positive control. | No | Mostly OK | Do not oversell. |
| 12 | Contact below majority baseline | NOT SUPPORTED as a remaining paper defect | Paper already: contact 38.35–39.70% vs 40.1% training-majority; “do not establish useful control.” Uniform 25% is rejected. Empirical failure is real. | Table 4; App C class counts | §4.3, §5.5 | Low now | Keep; consider dropping contact from the main claim list. | No | Paper already fixed | Contact decoding failed the relevant baseline. |
| 13 | Probe selectivity / Hewitt & Liang | PARTIALLY VERIFIED | Majority + geometry exist. No shuffled-label control, no random-frozen representation, no Hewitt & Liang citation. Appendix F uses a fixed spec (good) but failed geometry MLP is reported (good). | App F | §4.3 | Medium | Add shuffled-label + cite probing literature after verification. | No | Incomplete | Null stiffness is not yet selectivity-controlled. |
| 14 | D_tgt≤1e-8 exclusions are selected | VERIFIED | s42: 1273/3000 (42.4%) excluded. Excluded: 97.9% identical raster, median RMS pos 8.69e-6, median action L2 2.31. Included: 0% identical, pos 1.45e-5, action L2 2.85. Paper already prefers unnormalized D and discloses n. Missing: excluded-vs-included table. | CF CSVs; `visibility_stratification.json` | App E | Medium | Show the split. Keep ratios secondary. | No | Analysis done | Conditional ratio is not representative of all pairs. |
| 15 | Mean±half-range looks like a CI | PARTIALLY VERIFIED | Text says “2-seed range, not a CI.” Fig 5 still uses error-bar *style*. | Fig 5 caption | §4.3, §5.4 | Medium | Plot both seed points; write “range across two training seeds.” | No | Partial | Do not change numbers. Change display. |
| 16 | Action vs no-action is a null | PARTIALLY VERIFIED | Ordering flips 42/44 vs 43/45. Paper states this. Still prints four-seed means 1.162 vs 1.116. Shuffle % also tiny. | Fig 4; §5.3 | §5.3 | Medium | Drop mean difference. Paired dots only. Sentence: no consistent advantage. | No | Almost fixed | Null result. |
| 17 | Over-precise percentages | VERIFIED | Table 4: +0.01702%, +0.01131%, +0.03912%, −0.00983%. Seed-to-seed variation is 0.05 percentage points. | Table 4 | §5.3 | Low | Round to ~0.02 / 0.01 / 0.04 / −0.01 or 3 decimals in percent. | No | Not done | Keep exact values in CSV. |
| 18 | Table 2 seven digits | VERIFIED | e.g. D_pred 3.183241 vs 3.183233. Agreement is the point; digits overstate independent precision. | Table 2 | App C | Low | 3–4 significant figures in the PDF. | No | Not done | Machine values stay in CSV. |
| 19 | Spearman IID p-value | PARTIALLY VERIFIED | Paper already: “descriptive across correlated transitions,” ρ=0.684. No episode-bootstrap CI. This pass: 5000 episode resamples, n_ep=200, n_trans=3000, median ρ=0.684, 95% [0.659, 0.707]. IID p=0 is invalid and unused by the paper. | `02_action_distance`; this pass | §5.2 | Medium | Report cluster CI. | No | CI computed | Correlation is real and not an IID artifact. |
| 20 | Mean vs median consistency | PARTIALLY VERIFIED | Physical Δ: medians + IQR. Latent MSE: means. Ratios: medians. Paper explains skew for physical Δ. No single aggregation table. | throughout | §4–5 | Low | One methods table: metric / unit / statistic / why. | No | Incomplete | Keep current statistics; document them. |
| 21 | Diff original vs replay observations | **VERIFIED local bitwise match**; Linux tensor still absent | Full dump: **2000 episodes, 32,000 frames, 393,216,000 pixels, 0 differ, 0 action mismatches**. Linux replay SHA remains different and unmeasured pixelwise. | `results/replay_diff_report.csv`; `results/audit_phase2/replay_diff_full.json` | App C | Medium now | Disclose local exact match vs cross-machine hash mismatch. | No | **Yes (local)** | This Mac reproduces the dump exactly. Do not claim Linux byte identity. |
| 22 | Distribute large data correctly | VERIFIED as missing | No `scripts/download_data.py`. Checklist Q5 is **No** (correct). 375MB npy must not be ordinary-git committed. | checklist.tex | App B, Q5 | High for release | Anonymous host + hash verify. Only then Q5=Yes. | No | Not done | Do not flip Q5. |
| 23 | License / constitutive provenance | VERIFIED as unresolved | No repo LICENSE. `metadata.json` still “neo-Hookean”; stress is fixed-corotated (paper corrected). Taichi 88-line heritage undocumented for redistribution. | `generate_trajectories.py`; metadata.json | App A | High for artifact | `LICENSE_AUDIT.md`. Do not claim compliance until done. | No | Not done | Paper constitutive label is OK; dump metadata is stale. |
| 24 | Title too long | PARTIALLY VERIFIED | Accurate but clause-heavy. Contribution is the audit protocol. | title | title | Low | Choose after remaining controls. Candidates below. | No | Not chosen | Wait for Issue 3 figure. |
| 25 | Abstract opens then retracts 8.36× | PARTIALLY VERIFIED | Problem→method→findings structure exists. Finding 3 still advertises 8.36× then says it is not better physics. | abstract | abstract | Medium | Move 8.36× out of the abstract or frame as collapse. | No | Not rewritten this pass | Strongest abstractable results: identical-raster fraction + visible-pair D_pred≪D_tgt. |
| 26 | Hedge–retraction rhythm | PARTIALLY VERIFIED | Calibrated, but every paragraph withdraws. Three findings in §1 are clearer. | throughout | §1, §6 | Low | State 2–3 supported claims; park caveats in §6. | No | Not rewritten | Supported: (i) identical rasters with nonzero particle Δ; (ii) D_pred≪D_tgt among visible pairs; (iii) action/no-action null + MSE spectral concentration. |
| 27 | One causal chain | PARTIALLY VERIFIED | Plausible: normalized actuator + damping + subpixel Δ + uint8/tanh + weak action grads + predictor attenuation. **Not asserted.** Visibility bins show predictor attenuation after observation is visible, so the chain is not *only* environment. | Phase 1 | §6 methodology framing | High if overclaimed | Keep methodology framing until state-perturbation + stiffness particle probe. | No | Chain not proven | Do not write the arrow chain as fact. |
| 28 | Related work misses eval literature | VERIFIED | Related work is JEPA / GNS / PhysTwin / PhysWorld / DeformMaster. No probing, collapse, counterfactual dynamics eval, or intervention tracing citations. | references.bib | §2 | Medium | Add verified citations only. | No | Not done | Evaluation-methodology paper needs evaluation citations. |
| 29 | NeurIPS main-track checklist | PARTIALLY VERIFIED | PhysWorldAI CFP (checked 2026-09-07): long papers ≤8 pages, NeurIPS 2026 style, double-blind. **No checklist requirement.** Checklist starts on PDF page 14 (after refs/appendix), so it does not consume the 8 main pages. | https://physworld-org.github.io/physworld.github.io/cfp/ | pages 14–16 | Low | Remove from workshop PDF if space/policy prefer; keep in source. | No | Present | Optional, not a main-text leak. |
| 30 | Unused 8th main page | VERIFIED | Main text is 7 pages; CFP allows 8. Strongest missing figure is visibility-stratified physical→raster→D_tgt→D_pred. | PDF | — | Medium | Use page 8 for that figure, not prose. | No | Not done | Priority control. |
| 31 | Figure readability | PARTIALLY VERIFIED | Fig 4 connects seeds 42–45 with lines on a non-continuous axis. Fig 3c is cramped. Print-size audit not completed this pass. | Figs 3,4,5,7 | §5 | Medium | Paired dots; no seed-axis lines. | No | Visual pass pending | Likely easy fix. |
| 32 | DeformMaster citation | NOT SUPPORTED | Verified: Li, Zhoujian Li, Ren Li, Jie Gu, Lei Lei, Jingmin Chen, Lei Sun, “DeformMaster: An Interactive Physics-Neural World Model for Deformable Objects from Videos,” arXiv:2605.09586, 2026. | arXiv / GitHub CAN-Lee/DeformMaster | §2 | None | Keep. | No | Verified | Citation is correct. |
| 33 | Compiled PDF layout | CANNOT DETERMINE | Zip SOURCE_AUDIT claims no overfull boxes / Type 3 fonts / 7 main pages. This pass did not recompile. | SOURCE_AUDIT.md | — | Medium | Compile and inspect after text edits. | No | Not inspected here | Do not trust unseen pages. |

---

## Phase 1 measurements (cheap, no retrain)

### Actuator (Issue 1)

N=16 validation episodes, t=3, hold recorded (fx,fy) or (cx,cy), perturb one coordinate (Δc=0.05, Δf=0.20).

Train-split action std: cx 0.082, cy 0.057, fx 0.402, fy 0.398.

Approximate 1-std position effect at h=1 (sensitivity × train std): cx ~2.4e-6, cy ~1.0e-6, fx ~4.7e-6, fy ~4.7e-6. Force dominates typical variation by about **2×**, not ∞. Centers are weak, not dead.

Existing protocol already measured a large center shift (`contact_perturbed`) and a force reverse; those secondary rows must stay unpooled (paper already does this).

### Damping (Issue 2)

| Damping | Per-obs velocity factor | Median \|v\| h=1 | h=5 | h5/h1 |
| --- | --- | --- | --- | --- |
| 0.08 (paper) | 0.513 | 1.36e-3 | 2.24e-4 | 0.163 |
| 0.03 | 0.784 | 1.88e-3 | 8.46e-4 | 0.394 |
| 0.01 | 0.923 | 2.17e-3 | 1.65e-3 | 0.707 |

### Visibility bins, shuffled h=1, seed 42 (Issue 3)

| Bin | n | med RMS pos | med D_tgt | med D_pred | frac D_pred<D_tgt |
| --- | --- | --- | --- | --- | --- |
| identical raster | 1246 | 8.56e-6 | 0 | 1.75e-6 | 0.00 |
| low nonzero raster | 583 | 1.24e-5 | 8.14e-6 | 1.87e-6 | 0.61 |
| mid raster | 592 | 1.47e-5 | 1.35e-4 | 2.01e-6 | 0.95 |
| high raster | 579 | 1.66e-5 | 4.13e-4 | 2.14e-6 | 0.97 |
| top 10% raster | 300 | 1.41e-5 | 1.19e-3 | 1.95e-6 | 0.95 |

### Local replay (Issue 21)

20 validation episodes, 3,932,160 channel-pixels: **zero** differences vs `observations.npy`. Actions match. This does **not** prove the Linux replay tensor matches.

---

## Required non-training controls — completed

1. VN D_pred / D_tgt — `results/audit_phase2/vn_state_perturbation.csv`
2. State vs action perturbation — same file
3. Per-horizon pred/target rank @ 1,3,5,10 — `results/final_checkpoint_results.csv`
4. Stiffness particle / float / uint8 — `results/audit_phase2/stiffness_positive_controls.csv` and `results/probe_results.csv`
5. Full 2000-episode replay diff — `results/replay_diff_report.csv`
6. Canonical evaluator — `scripts/evaluate_final.py` (val-only + VN-MSE + per-horizon rank; original dump)

`src/eval.py::evaluate_model` still pools train+val if called directly; do not use it for paper tables.

### Optional later (still no 10-model retrain)

7. One **aug-off VICReg action pilot** (Issue 9), only if we keep photometric augs in the method story.
8. **New unused-seed evaluation dump** (Phase 8), protocol frozen *before* any metric is looked at. Current val 1800–1999 is not untouched.

### Do not do unless (1–6) force it

9. Corrected actuator or lower-damping **new simulator** + matched retrain. That is a **separate experiment**. Do not mix with the ten checkpoints.

---

## Paper–code inconsistency (Phase 5)

`scripts/evaluate_final.py` is now the canonical val-only + VN-MSE + per-horizon-rank evaluator. `src/eval.py::horizon_mse` still averages all loaded episodes if you call `evaluate_model` on the full dataset; do not use that path for paper numbers.

Original-dump MSE@1/5/10 for all ten last.pt match the revised paper Table 3 to printed precision. SHA-256 values match `CHECKPOINT_MANIFEST.csv`.

`outputs/corrected_seed43_audit.json` still contains absolute home paths and is obsolete.

---

## Anonymity — VERIFIED unsafe for double-blind git upload

Hits include `outputs/corrected_seed43_audit.json`, `outputs/physics_smoke/vicreg/history.json`, `docs/_tmp_zip_enum.json`, `docs/_tmp_enumerate_zips.py`. Git history contains author name and institutional email. `scripts/run_anonymity_audit.py` only scans `results/`, so it can report PASS while the repo leaks.

Need: `anonymous_release/` snapshot + recursive `scripts/check_anonymity.py`. Not done this pass.

---

## Title candidates (Issue 24; do not pick until Issue 3 figure lands)

1. Counterfactual Action Audits for Physical Grounding in Joint-Embedding World Models
2. Tracing Physical Action Effects Through Joint-Embedding World Models
3. Observation Bottlenecks in Action-Conditioned Joint-Embedding Predictors
4. When Latent Error Is Not Physical Grounding: A Counterfactual Audit
5. Intervention Tracing for Joint-Embedding Predictive World Models
