# Compute budget

Estimates use the measured one-epoch high-excitation JEPA step on this machine
(MPS, batch 128, 169 steps): 42.6 seconds per epoch, about 8520 seconds for
200 epochs. Legacy VICReg seeds 42–45 finished in 9046–11078 seconds, which
matches that scale. Arm runs are tens of seconds and are not on this clock.

Device for the in-progress dump model: Apple MPS. No CUDA.

## In progress

| model | environment | seed | epochs | device | status |
| --- | --- | --- | --- | --- | --- |
| vicreg_action | mpm_high | 42 | 200 | mps | running |

The one-epoch gate `outputs/mpm_high_runs/jepa_s42_smoke1` finished in 42.6s
with finite train loss 50.86, validation loss 45.26, and a nonzero action
gradient. That checkpoint is not a confirmatory result.

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
