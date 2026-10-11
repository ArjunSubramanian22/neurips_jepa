# Action conditioning

`concat` is the baseline. The action embedding is concatenated with the latent and mixed by residual MLP blocks. `film`, `gate`, `cross_attn`, and `token` use the same encoder, predictor depth, and widths. Counts below are one forward at batch size 1 on `configs/experiments/action_conditioning_v1.yaml`. Backward is not included. A training step counts two encoder forwards and one predictor forward.

| Mode | Trainable | Predictor | Action path | Predictor FLOPs | Versus concat |
| --- | --- | --- | --- | --- | --- |
| concat | 874368 | 310784 | 29312 | 614912 | 1.00 |
| film | 882816 | 319232 | 21248 | 631296 | 1.010 |
| gate | 882816 | 319232 | 21248 | 631296 | 1.010 |
| cross_attn | 915968 | 352384 | 54400 | 698880 | 1.048 |
| token | 1139968 | 576384 | 12928 | 1147392 | 1.304 |

`film` and `gate` match in these counts because each adds two action-sized linear maps and a latent projection, laid out differently. `cross_attn` builds four action tokens so the softmax is not a single key. `token` replaces the residual MLP mixer with transformer blocks, which is why its predictor is larger. That gap is the existing token mixer, not an extra hidden width.

The trainer logs `action_cond_grad_norm` after the prediction backward, before gradient clipping. It is the L2 norm of gradients in the action path.

No mode has been trained for this comparison. Rollout error, rank, sensitivity, cosine, retrieval, and planning are `not_measured`. `winner` is null. Architecture selection, when those runs exist, uses `cvpr_v1` validation and does not read the test split.

```bash
python3.13 scripts/report_action_conditioning_capacity.py
```
