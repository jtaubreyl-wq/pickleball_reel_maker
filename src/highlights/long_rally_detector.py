# FILE: src/highlights/long_rally_detector.py

# ============================================================
# SECTION: Imports & Data Models
# ============================================================

from dataclasses import dataclass
from typing import Iterable, List

from rally.metadata.rally_metadata import RallyMetadata


# ============================================================
# SECTION: Configuration Model (Story 6.1)
# ============================================================

@dataclass
class LongRallyConfig:
    """
    EPIC 6 — Story 6.1
    Configuration for long rally detection.

    duration_threshold_seconds:
        Minimum rally duration (in seconds) to qualify as a long rally.

    hit_threshold:
        Minimum number of ball hits to qualify as a long rally.
    """
    duration_threshold_seconds: float = 8.0
    hit_threshold: int = 6


# ============================================================
# SECTION: Long Rally Detector (Story 6.1)
# ============================================================

class LongRallyDetector:
    """
    EPIC 6 — Story 6.1

    Purpose:
        Determine whether a rally qualifies as a "long rally" based on:
            • Rally duration
            • Number of hits

    Output:
        Writes the following fields into RallyMetadata:
            • is_long_rally: bool
            • long_rally_duration: float
            • long_rally_hit_count: int
    """

    # --------------------------------------------------------
    # SECTION: Initialization
    # --------------------------------------------------------

    def __init__(self, config: LongRallyConfig = LongRallyConfig()):
        self.config = config

    # --------------------------------------------------------
    # SECTION: Duration Computation
    # --------------------------------------------------------

    @staticmethod
    def compute_duration_seconds(start_frame: int, end_frame: int, fps: float) -> float:
        """
        Compute rally duration in seconds.

        Ensures non-negative duration.
        """
        if fps <= 0:
            return 0.0
        duration = max(0, end_frame - start_frame) / fps
        return duration

    # --------------------------------------------------------
    # SECTION: Process Single Rally
    # --------------------------------------------------------

    def process_rally(self, rally: RallyMetadata, fps: float) -> RallyMetadata:
        """
        Update a single RallyMetadata object with long-rally fields.

        rally.start_frame, rally.end_frame, rally.hit_count must be valid.
        """

        # --- Compute duration safely ---
        duration = self.compute_duration_seconds(
            rally.start_frame,
            rally.end_frame,
            fps
        )

        # --- Hit count fallback ---
        hit_count = rally.hit_count if rally.hit_count is not None else 0

        # --- Determine if long rally ---
        is_long = (
            duration >= self.config.duration_threshold_seconds
            or hit_count >= self.config.hit_threshold
        )

        # --- Write EPIC 6 fields directly into metadata ---
        rally.is_long_rally = bool(is_long)
        rally.long_rally_duration = float(duration)
        rally.long_rally_hit_count = int(hit_count)

        return rally

    # --------------------------------------------------------
    # SECTION: Batch Processing
    # --------------------------------------------------------

    def process_all(self, rallies: Iterable[RallyMetadata], fps: float) -> List[RallyMetadata]:
        """
        Batch-process a list of rallies.

        Ensures all rallies receive valid long-rally metadata.
        """
        processed = []
        for r in rallies:
            processed.append(self.process_rally(r, fps))
        return processed
