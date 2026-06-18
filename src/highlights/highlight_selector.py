# FILE: src/highlights/highlight_selector.py

from typing import List
from .models import Rally


# ============================================================
# SECTION: Overlap Check
# ============================================================

def has_overlap(a: Rally, b: Rally) -> bool:
    """Check if two rallies overlap in time."""
    return not (a.end_time <= b.start_time or b.end_time <= a.start_time)


# ============================================================
# SECTION: Variety Helpers
# ============================================================

def too_close_in_time(a: Rally, b: Rally, min_gap_s: float = 3.0) -> bool:
    """
    Prevent selecting rallies that occur too close together.
    Encourages variety in the final highlight reel.
    """
    return abs(a.start_time - b.start_time) < min_gap_s


def too_similar(a: Rally, b: Rally) -> bool:
    """
    Prevent selecting rallies that are too similar in metadata.
    Encourages variety in rally types.
    """
    # Same ending event → too similar
    if getattr(a, "ending_event", None) == getattr(b, "ending_event", None):
        return True

    # Similar hit count → too similar
    if abs(a.hit_count - b.hit_count) <= 2:
        return True

    # Similar duration → too similar
    if abs(a.duration_seconds - b.duration_seconds) < 1.0:
        return True

    return False


# ============================================================
# SECTION: Highlight Selection (Forgiving + Variety)
# ============================================================

def select_top_highlights(
    rallies: List[Rally],
    max_highlights: int = 10,
    min_duration_s: float = 0.2,
) -> List[Rally]:
    """
    Forgiving Highlight Selector with Variety Enforcement.

    NEW BEHAVIOR:
        • Always selects top N rallies
        • Allows short rallies (scoring engine already soft-floors)
        • Enforces variety (time spacing + metadata diversity)
        • Avoids overlapping clips
        • Sorted by highlight_score (descending)
    """

    # --------------------------------------------------------
    # STEP 1 — Filter invalid rallies
    # --------------------------------------------------------

    valid = []
    for r in rallies:
        if r.start_time is None or r.end_time is None:
            continue

        duration = r.end_time - r.start_time
        if duration < min_duration_s:
            continue

        if r.highlight_score is None:
            continue

        valid.append(r)

    if not valid:
        return []

    # --------------------------------------------------------
    # STEP 2 — Sort by highlight_score (descending)
    # --------------------------------------------------------

    sorted_rallies = sorted(valid, key=lambda r: r.highlight_score, reverse=True)

    # --------------------------------------------------------
    # STEP 3 — Select top N with variety enforcement
    # --------------------------------------------------------

    selected: List[Rally] = []

    for candidate in sorted_rallies:
        if len(selected) >= max_highlights:
            break

        # Skip if overlapping
        if any(has_overlap(candidate, s) for s in selected):
            continue

        # Skip if too close in time
        if any(too_close_in_time(candidate, s) for s in selected):
            continue

        # Skip if too similar in metadata
        if any(too_similar(candidate, s) for s in selected):
            continue

        selected.append(candidate)

    # --------------------------------------------------------
    # STEP 4 — If we still have fewer than N, fill remaining slots
    # --------------------------------------------------------

    if len(selected) < max_highlights:
        for candidate in sorted_rallies:
            if len(selected) >= max_highlights:
                break

            if candidate not in selected:
                # Only enforce overlap here
                if any(has_overlap(candidate, s) for s in selected):
                    continue
                selected.append(candidate)

    return selected


# ============================================================
# SECTION: Export Utility
# ============================================================

def export_rallies_to_dict(rallies: List[Rally]) -> List[dict]:
    """Export selected rallies to JSON/CSV-friendly format."""
    return [
        {
            "rally_id": r.rally_id,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "highlight_score": r.highlight_score,
            "duration": r.duration_seconds,
            "metadata": r.metadata or {},
        }
        for r in rallies
    ]
