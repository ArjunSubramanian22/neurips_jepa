"""Fixed linear and MLP probes for material identifiability.

Hyperparameters are constants. The train split supplies both the readout and
the feature standardization. Regime selection reads validation accuracy only.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.eval_protocol import ProtocolError

PROBE_SEED = 0
PROBE_EPOCHS = 80
PROBE_LR = 1e-2
PROBE_WEIGHT_DECAY = 1e-4
MLP_HIDDEN = 128


class SmallMLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = MLP_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class RasterProjector:
    """Fixed Gaussian projection of a full-resolution frame. The matrix ignores labels."""

    def __init__(self, height: int, width: int, channels: int, dim: int, seed: int) -> None:
        if dim < 1:
            raise ValueError("projection dim must be positive.")
        rng = np.random.default_rng(int(seed))
        matrix = rng.normal(size=(int(dim), int(channels * height * width))).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + np.float32(1e-8)
        self.matrix = matrix

    def apply(self, frame: np.ndarray) -> np.ndarray:
        flat = np.asarray(frame, dtype=np.float32).reshape(-1)
        if flat.size != self.matrix.shape[1]:
            raise ValueError(f"frame has {flat.size} values, projector expects {self.matrix.shape[1]}.")
        return self.matrix @ flat


def train_standardize(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Z-score from the training rows only."""
    mean = train.mean(axis=0)
    std = np.maximum(train.std(axis=0), 1e-6)
    scaled = tuple((array - mean) / std for array in (train, *others))
    return scaled


def _confusion(predicted: np.ndarray, labels: np.ndarray, n_classes: int) -> list[list[int]]:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, guess in zip(labels.tolist(), predicted.tolist(), strict=True):
        matrix[int(truth), int(guess)] += 1
    return matrix.tolist()


def _fit(
    kind: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    n_classes: int,
) -> dict[str, Any]:
    torch.manual_seed(PROBE_SEED)
    d_in = int(train_x.shape[1])
    model: nn.Module
    if kind == "linear":
        model = nn.Linear(d_in, n_classes)
    elif kind == "mlp":
        model = SmallMLP(d_in, n_classes)
    else:
        raise ValueError(f"Unknown probe {kind!r}.")
    optimizer = AdamW(model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    features = torch.from_numpy(np.ascontiguousarray(train_x))
    labels = torch.from_numpy(np.ascontiguousarray(train_y)).long()
    for _ in range(PROBE_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(features), labels).backward()
        optimizer.step()
    model.eval()
    out: dict[str, Any] = {"probe": kind, "d_in": d_in, "epochs": PROBE_EPOCHS}
    with torch.no_grad():
        for split_name, (features_np, labels_np) in eval_sets.items():
            logits = model(torch.from_numpy(np.ascontiguousarray(features_np)))
            predicted = logits.argmax(dim=-1).cpu().numpy()
            truth = np.asarray(labels_np)
            out[split_name] = {
                "accuracy": float((predicted == truth).mean()) if truth.size else None,
                "n": int(truth.size),
                "confusion": _confusion(predicted, truth, n_classes),
            }
    return out


def evaluate_probe(
    kind: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """Fit on train and score train, validation, and test."""
    train_s, val_s, test_s = train_standardize(train_x, val_x, test_x)
    return _fit(
        kind,
        train_s.astype(np.float32),
        train_y.astype(np.int64),
        {
            "train": (val_dtype(train_s), train_y.astype(np.int64)),
            "val": (val_dtype(val_s), val_y.astype(np.int64)),
            "test": (val_dtype(test_s), test_y.astype(np.int64)),
        },
        n_classes,
    )


def val_dtype(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def history_matrix(frames: np.ndarray, length: int) -> np.ndarray:
    """Concatenate the first ``length`` post-rest frames.

    ``frames`` is ``(N, T_post, D)``.
    """
    if length < 1 or length > frames.shape[1]:
        raise ValueError(f"history length {length} is outside 1..{frames.shape[1]}.")
    taken = frames[:, :length, :]
    return taken.reshape(taken.shape[0], -1)


def select_protocol(
    records: Sequence[Mapping[str, Any]],
    *,
    chance: float,
    margin: float,
    privileged_min: float,
    geometry_min: float,
) -> dict[str, Any]:
    """Choose a regime from validation accuracies.

    Records must not carry a test block. Among protocols that clear every
    validation bar, the highest uint8-history stiffness accuracy wins.
    """
    threshold = float(chance) + float(margin)
    admissible: list[Mapping[str, Any]] = []
    for record in records:
        if "test" in record:
            raise ProtocolError("regime selection cannot read test accuracy.")
        if not record.get("selectable", False) or not record.get("stable", False):
            continue
        val = record["val"]
        required = {
            "privileged_stiffness_linear": privileged_min,
            "geometry_uint8_linear": geometry_min,
            "geometry_mechanical_linear": geometry_min,
            "mechanical_history_stiffness_linear": threshold,
            "uint8_history_stiffness_linear": threshold,
        }
        if all(float(val[key]) >= float(floor) for key, floor in required.items()):
            admissible.append(record)
    if not admissible:
        return {"status": "NONE", "protocol_id": None, "threshold": threshold, "admissible": []}
    ordered = sorted(
        admissible,
        key=lambda record: (
            -float(record["val"]["uint8_history_stiffness_linear"]),
            -float(record["val"]["mechanical_history_stiffness_linear"]),
            str(record["protocol_id"]),
        ),
    )
    return {
        "status": "SELECTED",
        "protocol_id": ordered[0]["protocol_id"],
        "threshold": threshold,
        "admissible": [str(record["protocol_id"]) for record in ordered],
    }
