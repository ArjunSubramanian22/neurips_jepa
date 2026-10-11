"""Public entry for directional counterfactual metrics.

The implementation lives in ``src.metrics.counterfactual_direction``.
"""

from src.metrics.counterfactual_direction import (
    CONTRAST_INVISIBLE,
    CONTRAST_NEAR_ZERO,
    CONTRAST_NONFINITE,
    CONTRAST_VALID,
    EPSILON,
    HIGH_VISIBILITY_AFFECTED,
    NORM_TAU,
    assign_quintiles,
    bootstrap_valid_cosine,
    directional_contrast,
    summarize_metric,
)

__all__ = [
    "CONTRAST_INVISIBLE",
    "CONTRAST_NEAR_ZERO",
    "CONTRAST_NONFINITE",
    "CONTRAST_VALID",
    "EPSILON",
    "HIGH_VISIBILITY_AFFECTED",
    "NORM_TAU",
    "assign_quintiles",
    "bootstrap_valid_cosine",
    "directional_contrast",
    "summarize_metric",
]
