# Compute budget

Estimates use the measured one-epoch high-excitation JEPA step on this machine
(MPS, batch 128, 169 steps): 42.6 seconds per epoch, about 8520 seconds for
200 epochs. Legacy VICReg seeds 42–45 finished in 9046–11078 seconds, which
matches that scale. Arm runs are tens of seconds and are not on this clock.

Device for the in-progress dump model: Apple MPS. No CUDA.

## In progress

None. The high-excitation JEPA seed 42 run finished.

## Completed this wave

| model | environment | seed | epochs | device | seconds | notes |
| --- | --- | --- | --- | --- | --- | --- |
| vicreg_action | mpm_high | 42 | 200 | mps | 7701.1 | last.pt; val loss 25.65; action grad norm about 11 |

Validation screen, not the test suffix: 24 episodes, open-loop one-step force reversal. Cosine median 0.769 on 24/24 defined rows. Median raster fraction identical 0.855. Median position RMS 0.00135. See `experiments/runs/mpm_high_jepa_s42_val_screen_v1/screen.json`.

## Next

No-action seed 42 on the same dump. If its cosine is undefined or near zero while this checkpoint stays high, the action path is doing the work. If the no-action cosine is also high, the screen is not measuring action use.

## Already spent, first wave

Ten no-crop legacy checkpoints exist (VICReg action and no-action seeds 42–45,
MSE action seeds 42–43). Their wall times are in each `history.json`. They are
not re-trained.

## If the full matrix were launched

The confirmatory matrix has 68 cells. About 25 of the high-excitation latent
runs would each cost roughly 2.4 hours. That alone is about 60 hours. Five
missing legacy seeds add about 12 hours. Conditioning and mitigation screens
are extra. The arm five-seed block is cheap.

That full matrix does not fit in one sitting on this laptop. Order:

1. Finish high-excitation JEPA seed 42, then evaluate it before more seeds.
2. High-excitation no-action seed 42 as the control.
3. One other family (RSSM or latent transformer) seed 42 on the same dump.
4. Only then add seeds 43–46 for cells that still matter.
5. Do not open FiLM, gate, cross-attention, or the blocked alignment losses
   until those screens are the binding ambiguity.

Alignment direction, magnitude, and combined stay blocked until
`outputs/counterfactual_pairs/force_reversal_train.npz` exists for the
high-excitation dump. Inverse dynamics is unblocked but is not the first run.
