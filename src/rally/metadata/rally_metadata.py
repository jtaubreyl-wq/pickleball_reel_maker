# ============================================================
# FILE: src/rally/metadata/rally_metadata.py
# ============================================================

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

BoundingBox = Tuple[int, int, int, int]
Point2D = Tuple[float, float]


# ============================================================
# SECTION: RallyMetadata (EPIC 5 + EPIC 6)
# ============================================================

@dataclass
class RallyMetadata:
    """
    Master metadata object for a single rally.

    This MUST stay in sync with:
        • RallyMetadataEngine
        • LongRallyDetector
        • WinnerForcedErrorDetector
        • HighlightScoringEngine
        • HighlightSelector
        • ReelAssembler

    If any field is missing, the highlight pipeline will fail.
    """

    # --------------------------------------------------------
    # Story 5.1 — Rally Start Detector
    # --------------------------------------------------------
    start_frame: int

    # --------------------------------------------------------
    # Story 5.2 — Rally End Detector
    # --------------------------------------------------------
    end_frame: int

    # --------------------------------------------------------
    # Story 5.3 — Rally Segment Builder
    # --------------------------------------------------------
    rally_id: int
    segment_path: Optional[str] = None

    # --------------------------------------------------------
    # Story 5.4 — Rally Metadata Engine
    # --------------------------------------------------------
    duration_seconds: float = 0.0
    hit_count: int = 0

    # Ball trajectory
    trajectory: List[Point2D] = field(default_factory=list)

    # Player bounding boxes per frame
    bounding_boxes: Dict[str, List[BoundingBox]] = field(default_factory=dict)

    # Ball speed metrics
    ball_speeds: List[float] = field(default_factory=list)
    avg_ball_speed: float = 0.0
    max_ball_speed: float = 0.0
    ball_velocity_variance: float = 0.0

    # Player involvement
    players_involved: List[str] = field(default_factory=list)
    last_hitter: Optional[str] = None
    player_positions: Dict[str, Point2D] = field(default_factory=dict)

    # 🔥 REQUIRED FIX
    metadata: Dict = field(default_factory=dict)

    # --------------------------------------------------------
    # Advanced Motion Metrics (NEW)
    # --------------------------------------------------------
    curvature_score: float = 0.0
    direction_change_score: float = 0.0
    pace_score: float = 0.0
    volley_score: float = 0.0
    rally_complexity_score: float = 0.0

    # --------------------------------------------------------
    # Quality & difficulty scores
    # --------------------------------------------------------
    quality_score: float = 0.0
    difficulty_score: float = 0.0

    """
    ============================================================
    EPIC 6 — HIGHLIGHT LOGIC
    ============================================================
    """

    # --------------------------------------------------------
    # Story 6.1 — Long Rally Detection
    # --------------------------------------------------------
    is_long_rally: bool = False
    long_rally_duration: float = 0.0
    long_rally_hit_count: int = 0

    # --------------------------------------------------------
    # Story 6.2 — Winner / Forced Error Detection
    # --------------------------------------------------------
    winner: Optional[str] = None
    ending_event: Optional[str] = None   # "winner_shot", "forced_error", "unforced_error", "unknown"

    # --------------------------------------------------------
    # Story 6.3 — Highlight Scoring Engine
    # --------------------------------------------------------
    highlight_score: float = 0.0
    highlight_reasons: List[str] = field(default_factory=list)

    # --------------------------------------------------------
    # Story 6.4 — Highlight Selection & Ranking
    # --------------------------------------------------------
    selected_for_highlight: bool = False

    # ============================================================
    # SECTION: Convenience Properties
    # ============================================================

    @property
    def duration(self) -> float:
        return max(0.0, self.duration_seconds)

    @property
    def has_valid_duration(self) -> bool:
        return self.duration_seconds >= 0.5

    @property
    def has_valid_score(self) -> bool:
        return self.highlight_score > 0.0

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "rally_id": self.rally_id,
            "duration": self.duration_seconds,
            "hit_count": self.hit_count,
            "quality_score": self.quality_score,
            "difficulty_score": self.difficulty_score,
            "highlight_score": self.highlight_score,
            "winner": self.winner,
            "ending_event": self.ending_event,
            "is_long_rally": self.is_long_rally,
            "reasons": self.highlight_reasons,
        }
