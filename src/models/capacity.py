"""Parameter and forward-FLOP counts for action-conditioned predictors.

FLOPs count multiply-adds as two operations. They cover one predictor
forward at batch size 1, plus the encoder forwards in a training step.
Backward is not included. The encoder is shared across conditioning modes,
so mode-to-mode differences are the predictor terms.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .ac_jepa import ACJEPA, ActionPredictor


def _numel(parameters) -> int:
    return int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad))


def predictor_forward_flops(predictor: ActionPredictor, batch: int = 1) -> int:
    """Forward FLOPs of one predictor call, including attention matmuls."""
    total = 0
    for layer in predictor.modules():
        if isinstance(layer, nn.Linear):
            total += 2 * batch * int(layer.in_features) * int(layer.out_features)
    cfg = predictor.cfg
    heads = int(cfg.predictor_num_heads)
    head_dim = int(cfg.predictor_embed_dim) // heads
    if predictor.fuse_mode == "token" and predictor.condition_on_action:
        # QK^T and AV, sequence length 2, once per block. Projections are already counted.
        seq = 2
        total += cfg.predictor_depth * 4 * batch * heads * seq * seq * head_dim
    elif predictor.fuse_mode == "cross_attn" and predictor.condition_on_action:
        queries = 1
        keys = int(cfg.n_action_tokens)
        total += 4 * batch * heads * queries * keys * head_dim
    return int(total)


def encoder_forward_flops(model: ACJEPA, batch: int = 1) -> int:
    """Forward FLOPs of one online-encoder call. Patch embedding uses ``img_size``."""
    total = 0
    for layer in model.encoder.modules():
        if isinstance(layer, nn.Linear):
            total += 2 * batch * int(layer.in_features) * int(layer.out_features)
        elif isinstance(layer, nn.Conv2d):
            grid = int(model.cfg.img_size) // int(layer.stride[0])
            kernel = int(layer.kernel_size[0]) * int(layer.kernel_size[1])
            total += 2 * batch * grid * grid * int(layer.in_channels) * kernel * int(layer.out_channels)
    if model.cfg.backbone == "vit":
        heads = int(model.cfg.encoder_num_heads)
        head_dim = int(model.cfg.embed_dim) // heads
        seq = int(model.cfg.num_patches) + 1
        total += model.cfg.encoder_depth * 4 * batch * heads * seq * seq * head_dim
    return int(total)


def capacity_report(model: ACJEPA, batch: int = 1) -> dict[str, Any]:
    """Trainable counts and a forward-FLOP estimate. Backward is excluded."""
    predictor_flops = predictor_forward_flops(model.predictor, batch)
    encoder_flops = encoder_forward_flops(model, batch)
    return {
        "fuse_mode": model.cfg.fuse_mode,
        "condition_on_action": bool(model.cfg.condition_on_action),
        "trainable_parameters": _numel(model.trainable_parameters()),
        "predictor_parameters": _numel(model.predictor.parameters()),
        "action_path_parameters": _numel(param for _name, param in model.predictor.action_path_parameters()),
        "predictor_forward_flops": predictor_flops,
        "encoder_forward_flops": encoder_flops,
        "train_step_forward_flops": 2 * encoder_flops + predictor_flops,
        "flops_include_backward": False,
        "batch": int(batch),
    }
