from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

BoundingBox = Tuple[int, int, int, int]
Point2D = Tuple[float, float]


@dataclass
class RallyMetadata:
    """
    ============================================================
    EPIC 5 — RALLY SEGMENTATION + METADATA ENGINE
    ============================================================
    """

    # Story 5.1 — Rally Start Detector
    start_frame: int

    # Story 5.2 — Rally End Detector
    end_frame: int

    # Story 5.3 — Rally Segment Builder
    rally_id: int
    segment_path: Optional[str] = None

    # Story 5.4 — Rally Metadata Engine
    duration_seconds: float = 0.0
    hit_count: int = 0

    # Ball trajectory (list of (x, y))
    trajectory: List[Point2D] = field(default_factory=list)

    # Player bounding boxes per frame
    # NOTE: pipeline expects dict[str, list[BoundingBox]]
    bounding_boxes: Dict[str, List[BoundingBox]] = field(default_factory=dict)

    # Ball speed metrics
    ball_speeds: List[float] = field(default_factory=list)
    avg_ball_speed: float = 0.0
    max_ball_speed: float = 0.0

    # Additional ball metrics expected by pipeline
    ball_velocity_variance: float = 0.0   # <-- Added for pipeline compatibility

    # Player involvement
    players_involved: List[str] = field(default_factory=list)
    last_hitter: Optional[str] = None
    player_positions: dict = field(default_factory=dict)

    # Quality & difficulty scores
    quality_score: float = 0.0
    difficulty_score: float = 0.0

    """
    ============================================================
    EPIC 6 — HIGHLIGHT LOGIC
    ============================================================
    """

    # Story 6.1 — Long Rally Detection
    is_long_rally: bool = False
    long_rally_duration: float = 0.0
    long_rally_hit_count: int = 0

    # Alias for pipeline compatibility
    long_rally: bool = False   # <-- Added to match pipeline expectations

    # Story 6.2 — Winner / Forced Error Detection
    winner: Optional[str] = None
    ending_event: Optional[str] = None   # "winner_shot", "forced_error", "unforced_error"

    # Story 6.3 — Highlight Scoring Engine
    highlight_score: float = 0.0
    highlight_reasons: List[str] = field(default_factory=list)

    # Story 6.4 — Highlight Selection & Ranking
    selected_for_highlight: bool = False
