# FILE: src/highlights/models.py

# ============================================================
# SECTION: Imports & Data Models
# ============================================================

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# ============================================================
# SECTION: Rally Model (Core Highlight Object)
# ============================================================

@dataclass
class Rally:
    """
    Core rally object used throughout EPIC 5 & EPIC 6.

    This model is intentionally rich because:
        • Highlight scoring depends on metadata
        • Highlight selection depends on duration & score
        • Reel assembler needs timestamps
        • Debug overlays need rally_id, frames, and metadata

    Fields:
        rally_id: Unique identifier for the rally
        start_time: Start time in seconds
        end_time: End time in seconds
        highlight_score: Final score from scoring engine
        metadata: Arbitrary metadata dict (winner, forced error, long rally, etc.)

    Additional fields added for stability:
        start_frame / end_frame: Frame indices
        fps: Frames per second
        winner: "Team A" / "Team B" / None
        ending_event: "winner_shot", "forced_error", "unforced_error", "unknown"
        is_long_rally: bool
        long_rally_duration: float
        long_rally_hit_count: int
        quality_score: float
        difficulty_score: float
        highlight_reasons: List[str]
        selected_for_highlight: bool
    """

    # --------------------------------------------------------
    # SECTION: Core Required Fields
    # --------------------------------------------------------

    rally_id: str
    start_time: float
    end_time: float
    highlight_score: float
    metadata: Optional[dict] = field(default_factory=dict)

    # --------------------------------------------------------
    # SECTION: Frame-Level Information
    # --------------------------------------------------------

    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    fps: Optional[float] = None

    # --------------------------------------------------------
    # SECTION: Winner / Forced Error (Story 6.2)
    # --------------------------------------------------------

    winner: Optional[str] = None
    ending_event: Optional[str] = None  # "winner_shot", "forced_error", "unforced_error", "unknown"

    # --------------------------------------------------------
    # SECTION: Long Rally (Story 6.1)
    # --------------------------------------------------------

    is_long_rally: bool = False
    long_rally_duration: float = 0.0
    long_rally_hit_count: int = 0

    # --------------------------------------------------------
    # SECTION: Scoring (Story 6.3)
    # --------------------------------------------------------

    quality_score: float = 0.0
    difficulty_score: float = 0.0
    highlight_reasons: List[str] = field(default_factory=list)

    # --------------------------------------------------------
    # SECTION: Highlight Selection (Story 6.4)
    # --------------------------------------------------------

    selected_for_highlight: bool = False

    # --------------------------------------------------------
    # SECTION: Computed Properties
    # --------------------------------------------------------

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return max(0.0, self.end_time - self.start_time)

    @property
    def duration_frames(self) -> Optional[int]:
        """Duration in frames (if fps and frame indices are available)."""
        if self.start_frame is None or self.end_frame is None or not self.fps:
            return None
        return max(0, self.end_frame - self.start_frame)

    @property
    def is_valid(self) -> bool:
        """
        A rally is valid if:
            • start_time < end_time
            • highlight_score > 0
            • duration > 0.5s
        """
        return (
            self.start_time is not None
            and self.end_time is not None
            and self.duration >= 0.5
            and self.highlight_score > 0
        )
