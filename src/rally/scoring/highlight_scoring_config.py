# FILE: src/rally/scoring/highlight_scoring_config.py

from dataclasses import dataclass

@dataclass
class HighlightScoringConfig:
    """
    Updated configuration for the forgiving HighlightScoringEngine.

    This config is tuned for:
        • Noisy real-world tracking
        • Forgiving rally detection
        • Curvature-based action scoring
        • Speed burst scoring
        • Direction-change scoring
        • Soft scoring floor (never zero)
        • Always selecting top N highlights
    """

    # --------------------------------------------------------
    # SECTION: Duration Scoring
    # --------------------------------------------------------

    # A rally is considered "long" at this duration
    long_rally_seconds: float = 6.0

    # Weight for long rallies
    long_rally_weight: float = 1.0


    # --------------------------------------------------------
    # SECTION: Ending Event Weights
    # --------------------------------------------------------

    winner_weight: float = 1.5
    forced_error_weight: float = 1.0
    unforced_error_weight: float = 0.5


    # --------------------------------------------------------
    # SECTION: Speed / Pace Scoring
    # --------------------------------------------------------

    # Normalization constant for speed bursts
    speed_variance_norm: float = 35.0

    # Weight applied to speed burst scoring
    speed_variance_weight: float = 1.0


    # --------------------------------------------------------
    # SECTION: Quality & Difficulty Weights
    # --------------------------------------------------------

    quality_weight: float = 0.8
    difficulty_weight: float = 0.8


    # --------------------------------------------------------
    # SECTION: Hit Count & Duration Bonuses
    # --------------------------------------------------------

    high_hit_count_threshold: int = 8
    high_hit_count_weight: float = 1.0

    high_duration_threshold: float = 10.0
    high_duration_weight: float = 1.0


    # --------------------------------------------------------
    # SECTION: Curvature & Direction Scoring
    # --------------------------------------------------------

    curvature_action_weight: float = 1.0
    direction_change_weight: float = 0.8


    # --------------------------------------------------------
    # SECTION: Soft Score Floor
    # --------------------------------------------------------

    # Minimum score ANY rally can have
    soft_floor: float = 0.2


    # --------------------------------------------------------
    # SECTION: Final Score Clamping
    # --------------------------------------------------------

    max_score: float = 10.0
    min_score: float = 0.0
