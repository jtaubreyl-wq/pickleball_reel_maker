# FILE: src/rally/scoring/scoring_utils.py

# ============================================================
# SECTION: Imports
# ============================================================

import math
from typing import List


# ============================================================
# SECTION: Clamp Utility
# ============================================================

def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    Clamp a numeric value into a fixed range.

    Purpose:
        Prevents highlight scores and intermediate signals from
        exploding or collapsing to invalid values.

    Notes:
        - Handles NaN and infinities safely.
        - Used throughout HighlightScoringEngine.
    """
    if value is None or math.isnan(value) or math.isinf(value):
        return min_value
    return max(min_value, min(value, max_value))


# ============================================================
# SECTION: Normalization Utilities
# ============================================================

def normalize(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a value into [0, 1].

    If min == max, returns 0 to avoid division-by-zero.

    Example:
        normalize(75, 0, 100) → 0.75
    """
    if max_value == min_value:
        return 0.0
    return clamp((value - min_value) / (max_value - min_value))


def normalize_safe(value: float, min_value: float, max_value: float, default: float = 0.0) -> float:
    """
    Safe normalization that:
        - Handles None
        - Handles NaN
        - Handles reversed ranges
        - Applies a default fallback

    Used for speed, duration, and hit-count normalization.
    """
    if value is None or math.isnan(value):
        return default
    if max_value <= min_value:
        return default
    return clamp((value - min_value) / (max_value - min_value))


# ============================================================
# SECTION: Variance Utilities
# ============================================================

def safe_variance(values: List[float]) -> float:
    """
    Compute variance safely for small lists.

    Returns:
        0 if fewer than 2 samples.

    Why this matters:
        RallyMetadataEngine and HighlightScoringEngine rely on variance
        to measure pace and intensity. A broken variance calculation
        can collapse highlight_score to zero.
    """
    if not values or len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def normalize_variance_for_speed(
    variance: float,
    max_expected_variance: float = 25.0
) -> float:
    """
    Convert raw ball velocity variance into a normalized [0,1] score.

    Purpose:
        Higher variance → faster pace → more exciting rally.

    Notes:
        - Uses clamp() to ensure safe bounds.
        - max_expected_variance is tunable per dataset.
    """
    if variance is None or math.isnan(variance):
        return 0.0
    return clamp(variance / max_expected_variance)


# ============================================================
# SECTION: Composite Helpers
# ============================================================

def weighted_sum(pairs: List[tuple[float, float]]) -> float:
    """
    Compute a weighted sum of (value, weight) pairs.

    Example:
        weighted_sum([(0.8, 2.0), (0.5, 1.0)]) → 2.1

    Used by:
        - HighlightScoringEngine
        - Quality/Difficulty scoring
    """
    total = 0.0
    for value, weight in pairs:
        if value is None or weight is None:
            continue
        total += float(value) * float(weight)
    return total


def safe_mean(values: List[float]) -> float:
    """
    Compute a safe mean:
        - Returns 0 for empty lists
        - Ignores NaN values
    """
    if not values:
        return 0.0
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)
