"""Layer-separated intervention metrics.

The five layers are stored in separate fields and are never reduced to one score:

1. physical response of the simulator state
2. observation visibility in the rendered raster
3. representation response of an encoder
4. predictor action response of an action-conditioned predictor
5. downstream decision utility

Latent distance matches the held-out evaluator: for vectors ``u, v``,

    D = mean_i (u_i - v_i)^2

Variance-normalized distance divides that scalar by a declared reference
variance (the mean per-dimension variance of a reference population). A
non-positive reference variance is left undefined. It is not replaced by zero
and it is not dropped without a count.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

LAYERS: tuple[str, ...] = (
    "physical",
    "observation",
    "representation",
    "predictor",
    "decision",
)

_MEASURED = frozenset({"measured", "ok"})


def not_measured(reason: str) -> dict[str, Any]:
    """Explicit placeholder. Callers must not omit a layer instead."""
    if not reason:
        raise ValueError("not_measured requires a reason.")
    return {"status": "not_measured", "reason": str(reason)}


def _flat(array: Any, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains a non-finite value.")
    return values


def action_delta(action_a: Any, action_b: Any) -> dict[str, Any]:
    """L2 distance between two action vectors.

    Identical actions return ``l2 == 0``.
    """
    left = _flat(action_a, "action_a")
    right = _flat(action_b, "action_b")
    if left.shape != right.shape:
        raise ValueError(
            f"action shapes differ: {left.shape} vs {right.shape}."
        )
    delta = left - right
    return {
        "l2": float(np.linalg.norm(delta)),
        "linf": float(np.max(np.abs(delta))),
        "n_dims": int(delta.size),
        "status": "measured",
    }


def physical_response(
    positions_a: Any,
    positions_b: Any,
    velocities_a: Any | None = None,
    velocities_b: Any | None = None,
) -> dict[str, Any]:
    """Particle-state response. Positions are ``(N, D)`` coordinates.

    ``rms_pos`` is the root-mean-square Euclidean displacement, matching
    ``phys_metrics`` in ``scripts/run_simulator_counterfactual.py``.
    """
    left = np.asarray(positions_a, dtype=np.float64)
    right = np.asarray(positions_b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            f"positions must share shape (N, D); got {left.shape} and {right.shape}."
        )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("positions contain a non-finite value.")
    dist = np.linalg.norm(right - left, axis=-1)
    out: dict[str, Any] = {
        "rms_pos": float(np.sqrt(np.mean(dist ** 2))),
        "mean_pos": float(np.mean(dist)),
        "max_pos": float(np.max(dist)),
        "n_particles": int(left.shape[0]),
        "status": "measured",
    }
    if (velocities_a is None) ^ (velocities_b is None):
        raise ValueError("velocities must be passed together.")
    if velocities_a is not None:
        va = np.asarray(velocities_a, dtype=np.float64)
        vb = np.asarray(velocities_b, dtype=np.float64)
        if va.shape != vb.shape or va.shape != left.shape:
            raise ValueError("velocities must match the position shape.")
        if not np.all(np.isfinite(va)) or not np.all(np.isfinite(vb)):
            raise ValueError("velocities contain a non-finite value.")
        speed = np.linalg.norm(vb - va, axis=-1)
        out["rms_vel"] = float(np.sqrt(np.mean(speed ** 2)))
    return out


def observation_visibility(obs_a: Any, obs_b: Any) -> dict[str, Any]:
    """Raster agreement. Integer frames are compared exactly and scaled by 255.

    ``fraction_identical`` is the fraction of elements that compare equal.
    ``mse`` is the mean squared difference after scaling integer frames to
    ``[0, 1]``. Float frames are not rescaled.
    """
    left = np.asarray(obs_a)
    right = np.asarray(obs_b)
    if left.shape != right.shape or left.size == 0:
        raise ValueError(
            f"observations must share a non-empty shape; got {left.shape} and {right.shape}."
        )
    identical = left == right
    n_identical = int(np.sum(identical))
    n_values = int(left.size)
    scale = 255.0 if np.issubdtype(left.dtype, np.integer) else 1.0
    err = (left.astype(np.float64) - right.astype(np.float64)) / scale
    if not np.all(np.isfinite(err)):
        raise ValueError("observations contain a non-finite value.")
    out: dict[str, Any] = {
        "n_values": n_values,
        "n_identical": n_identical,
        "fraction_identical": float(n_identical / n_values),
        "mse": float(np.mean(err ** 2)),
        "status": "measured",
    }
    if left.ndim == 3:
        spatial = tuple(range(1, err.ndim))
        out["mse_per_channel"] = [float(v) for v in np.mean(err ** 2, axis=spatial)]
    return out


def latent_contrast(z_a: Any, z_b: Any) -> dict[str, Any]:
    """Mean squared difference and L2 between two latent vectors.

    Identical vectors return ``mse == 0`` and ``l2 == 0``.
    """
    left = _flat(z_a, "z_a")
    right = _flat(z_b, "z_b")
    if left.shape != right.shape:
        raise ValueError(f"latent shapes differ: {left.shape} vs {right.shape}.")
    delta = left - right
    return {
        "mse": float(np.mean(delta ** 2)),
        "l2": float(np.linalg.norm(delta)),
        "n_dims": int(delta.size),
        "status": "measured",
    }


def cosine_alignment(vector_a: Any, vector_b: Any) -> dict[str, Any]:
    """Cosine of two vectors.

    Aligned vectors return 1. Opposite vectors return -1. A zero vector is
    ``undefined_zero_vector`` with ``cosine = None``. It is not reported as 0.
    """
    left = _flat(vector_a, "vector_a")
    right = _flat(vector_b, "vector_b")
    if left.shape != right.shape:
        raise ValueError(f"vector shapes differ: {left.shape} vs {right.shape}.")
    norm_a = float(np.linalg.norm(left))
    norm_b = float(np.linalg.norm(right))
    if norm_a == 0.0 or norm_b == 0.0:
        return {
            "cosine": None,
            "status": "undefined_zero_vector",
            "norm_a": norm_a,
            "norm_b": norm_b,
        }
    cosine = float(np.dot(left, right) / (norm_a * norm_b))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return {
        "cosine": cosine,
        "status": "ok",
        "norm_a": norm_a,
        "norm_b": norm_b,
    }


def counterfactual_alignment(
    z_hat_factual: Any,
    z_hat_counterfactual: Any,
    z_target_factual: Any,
    z_target_counterfactual: Any,
) -> dict[str, Any]:
    """Predictor and target contrasts, plus cosine of their difference vectors.

    The cosine compares ``(z_hat_cf - z_hat_f)`` with ``(z_tgt_cf - z_tgt_f)``.
    It is undefined when either difference is the zero vector.
    """
    hat_f = _flat(z_hat_factual, "z_hat_factual")
    hat_c = _flat(z_hat_counterfactual, "z_hat_counterfactual")
    tgt_f = _flat(z_target_factual, "z_target_factual")
    tgt_c = _flat(z_target_counterfactual, "z_target_counterfactual")
    if hat_f.shape != hat_c.shape or tgt_f.shape != tgt_c.shape:
        raise ValueError("factual and counterfactual latents must share a shape.")
    if hat_f.shape != tgt_f.shape:
        raise ValueError("predictor and target latents must share a shape.")
    return {
        "predictor": latent_contrast(hat_f, hat_c),
        "target": latent_contrast(tgt_f, tgt_c),
        "alignment": cosine_alignment(hat_c - hat_f, tgt_c - tgt_f),
    }


def variance_normalized_mse(mse: float, reference_variance: float) -> dict[str, Any]:
    """Divide a latent MSE by a declared reference variance.

    ``reference_variance <= 0`` yields status ``undefined_nonpositive_variance``
    and ``value = None``.
    """
    mse_f = float(mse)
    ref = float(reference_variance)
    if not np.isfinite(mse_f) or not np.isfinite(ref):
        return {
            "value": None,
            "mse": None if not np.isfinite(mse_f) else mse_f,
            "reference_variance": None if not np.isfinite(ref) else ref,
            "status": "undefined_nonfinite",
        }
    if ref <= 0.0:
        return {
            "value": None,
            "mse": mse_f,
            "reference_variance": ref,
            "status": "undefined_nonpositive_variance",
        }
    return {
        "value": mse_f / ref,
        "mse": mse_f,
        "reference_variance": ref,
        "status": "ok",
    }


def summarize_defined(
    values: Sequence[float | None],
    statuses: Sequence[str],
) -> dict[str, Any]:
    """Mean and median over defined measurements, with an undefined count.

    Statuses ``measured`` and ``ok`` are defined. Every other status is
    counted in ``n_undefined`` and excluded from the mean. Nothing is dropped
    without that count.
    """
    if len(values) != len(statuses):
        raise ValueError("values and statuses must have the same length.")
    defined: list[float] = []
    n_undefined = 0
    for value, status in zip(values, statuses):
        if status in _MEASURED and value is not None and np.isfinite(float(value)):
            defined.append(float(value))
        else:
            n_undefined += 1
    array = np.asarray(defined, dtype=np.float64)
    return {
        "n_total": int(len(values)),
        "n_defined": int(array.size),
        "n_undefined": int(n_undefined),
        "mean": None if array.size == 0 else float(array.mean()),
        "median": None if array.size == 0 else float(np.median(array)),
    }


def _require_layer(name: str, layer: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(layer, Mapping) or "status" not in layer:
        raise ValueError(f"layer {name} must be a mapping with a status.")
    return dict(layer)


def assemble_cascade(
    *,
    physical: Mapping[str, Any],
    observation: Mapping[str, Any],
    representation: Mapping[str, Any],
    predictor: Mapping[str, Any],
    decision: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    """Pack one intervention. There is no collapsed score field."""
    layers = {
        "physical": _require_layer("physical", physical),
        "observation": _require_layer("observation", observation),
        "representation": _require_layer("representation", representation),
        "predictor": _require_layer("predictor", predictor),
        "decision": _require_layer("decision", decision),
    }
    payload = {
        "schema": "cascade_v1",
        "action": _require_layer("action", action),
        "layers": layers,
    }
    if "score" in payload or "overall" in payload:
        raise AssertionError("cascade payload must not collapse layers.")
    return payload
