# New experiment summary

## Phase 0 baseline audit
QUESTION: Do existing dump/checkpoints/tests match the brief?
METHOD: Hash tensors, load audit JSON, run 83 unit tests, record env without identity.
N: dump 2000×16; 4 corrected last.pt already on disk.
RESULT: PASS. SHA-256 matches. Trainable params 874368. Headline last.pt numbers match sanity checks to rounding.
INTERPRETATION: Pipeline is usable. `ablation_twoseed.json` is best.pt and must not be mixed.
LIMITATION: Did not re-execute 200-epoch eval.
PAPER VALUE: Trustworthy methods/data paragraph.

## Phase 9 action path
QUESTION: Is action neglected because of a wiring bug?
METHOD: Unit tests + checkpoint weight/grad/eval checks + dump alignment.
N: 4 last.pt files.
RESULT: PASS. No CRITICAL_BUG_REPORT.
INTERPRETATION: Behavioral neglect is not an unconnected action tensor.
LIMITATION: Wiring ≠ useful action use.
PAPER VALUE: Rules out a class of implementation objections.

## Phase 1A fork
QUESTION: Can we clone exact MPM state?
METHOD: snapshot/restore/clone tests; dump replay bitwise match.
RESULT: PASS.
PAPER VALUE: Ground-truth counterfactuals are well-defined.

## Phase 1B–H simulator counterfactual
QUESTION: Does the true system change when actions are replaced, and does the JEPA?
METHOD: Exact forks, primary TRUE/SHUFFLED/NEUTRAL, h=1,3,5, two last.pt.
N: 200 val episodes, 3000 transitions, 800 command-change, 3000/3000 effective-contact under the predeclared rule.
RESULT: shuffled h=1 median RMS pos 1.1906067811651155e-05, RMS vel 0.002385589526966214, 41.533% identical uint8 rasters; predictor median MSE 1.927e-06 (s42) / 2.588e-06 (s43); median capture 0.01647 / 0.06699 with 1273 / 1256 exclusions.
INTERPRETATION: Particles move (velocity); rasters often do not; predictor change is ~1e-6. Action neglect is real at the observation/predictor interface; it is not because shuffled actions are identical or because contacts miss.
LIMITATION: h=1 raster quantization; contact flag does not stratify.
PAPER VALUE: Central figure/table for action use vs physics.

## Phase 2 action distance
QUESTION: Are shuffled actions too similar?
METHOD: Euclidean contact/force, normalized L2, quintiles first, Spearman.
RESULT: Spearman L2 vs RMS pos 0.684; largest quintile still pred MSE ~5e-6 (s42) while tgt MSE ~9e-5.
INTERPRETATION: Action swaps are large; predictor still barely moves.
PAPER VALUE: Blocks “maybe shuffle is a no-op” reviews.

## Phase 3 episode bootstrap
QUESTION: Separate episode-sampling uncertainty from two training seeds.
METHOD: 5000 bootstrap, seed 0, episode unit.
RESULT: s42 all-transition mean pred MSE 3.183e-06, 95% CI [3.033e-06, 3.343e-06].
INTERPRETATION: The tiny predictor Δ is precisely estimated across episodes. That is not practical action dependence. Mean capture ratio CIs are invalid as typical values.
PAPER VALUE: Honest uncertainty language.

## Phase 5–7, 4, 8
NOT COMPLETE at package time (training seed 44 running; stiffness/jacobian/identifiability not executed).
