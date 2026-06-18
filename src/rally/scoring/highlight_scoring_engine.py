# FILE: src/rally/scoring/highlight_scoring_engine.py

# ============================================================
# SECTION: Imports
# ============================================================

from typing import Dict, List
import numpy as np

from .highlight_scoring_config import HighlightScoringConfig
from .scoring_utils import clamp


# ============================================================
# SECTION: Forgiving Highlight Scoring Engine
# ============================================================

class HighlightScoringEngine:
    """
    Forgiving Highlight Scoring Engine

    NEW FEATURES:
        • Scores ALL rallies (never returns 0)
        • Soft scoring floor (ensures inclusion)
        • Curvature & direction-change scoring
        • Speed burst scoring
        • Rally momentum scoring
        • Baseline reset bonus (optional)
        • More forgiving quality/difficulty weighting
        • Designed for noisy real-world tracking

    Notes:
        The baseline reset bonus is optional and only applied when
        metadata.metadata["players_reset_to_baseline"] is present and True.
        This integrates cleanly with the updated RallyMetadataEngine.
    """

    def __init__(self, config: HighlightScoringConfig | None = None):
        """
        Initialize the scoring engine.

        Args:
            config: Optional HighlightScoringConfig instance.
        """
        self.config = config or HighlightScoringConfig()

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def score_rally(self, metadata) -> Dict:
        """
        Compute highlight_score and reasons for a single rally.

        Expected metadata fields:
            - duration_seconds
            - hit_count
            - ball_speeds
            - curvature_values (optional)
            - direction_changes (optional)
            - quality_score
            - difficulty_score
            - ending_event
            - metadata.metadata["players_reset_to_baseline"] (optional)

        Returns:
            dict with:
                "highlight_score": float
                "reasons": List[str]
        """

        score = 0.0
        reasons: List[str] = []

        # -----------------------------------------------------
        # 1. Duration Score (soft)
        # -----------------------------------------------------
        dur = metadata.duration_seconds
        if dur > 0:
            dur_score = min(dur / self.config.long_rally_seconds, 1.0) * 1.0
            score += dur_score
            if dur_score > 0.5:
                reasons.append("duration")

        # -----------------------------------------------------
        # 2. Hit Count Score
        # -----------------------------------------------------
        hits = metadata.hit_count
        if hits > 0:
            hit_score = min(hits / self.config.high_hit_count_threshold, 1.0) * 1.0
            score += hit_score
            if hit_score > 0.5:
                reasons.append("many_hits")

        # -----------------------------------------------------
        # 3. Speed Burst Score
        # -----------------------------------------------------
        speeds = metadata.ball_speeds or []
        if speeds:
            max_speed = max(speeds)
            speed_score = min(max_speed / self.config.speed_variance_norm, 1.0) * 1.2
            score += speed_score
            if speed_score > 0.4:
                reasons.append("speed_burst")

        # -----------------------------------------------------
        # 4. Curvature Score (volleys, impacts)
        # -----------------------------------------------------
        curv = getattr(metadata, "curvature_values", [])
        if curv:
            max_curv = max(curv)
            curv_score = min(max_curv * 5.0, 1.0) * 1.0
            score += curv_score
            if curv_score > 0.3:
                reasons.append("curvature_action")

        # -----------------------------------------------------
        # 5. Direction Change Score (dynamic play)
        # -----------------------------------------------------
        dirs = getattr(metadata, "direction_changes", [])
        if dirs:
            avg_dir = np.mean(np.abs(dirs))
            dir_score = min(avg_dir / 20.0, 1.0) * 0.8
            score += dir_score
            if dir_score > 0.3:
                reasons.append("direction_changes")

        # -----------------------------------------------------
        # 6. Ending Event Bonuses
        # -----------------------------------------------------
        end_event = getattr(metadata, "ending_event", None)

        if end_event == "winner_shot":
            score += 1.5
            reasons.append("winner_shot")

        elif end_event == "forced_error":
            score += 1.0
            reasons.append("forced_error")

        elif end_event == "unforced_error":
            score += 0.5
            reasons.append("unforced_error")

        # -----------------------------------------------------
        # 7. Quality & Difficulty (softened)
        # -----------------------------------------------------
        if metadata.quality_score > 0:
            q = metadata.quality_score * 0.8
            score += q
            if q > 0.4:
                reasons.append("quality")

        if metadata.difficulty_score > 0:
            d = metadata.difficulty_score * 0.8
            score += d
            if d > 0.4:
                reasons.append("difficulty")

        # -----------------------------------------------------
        # 8. Baseline Reset Bonus (NEW)
        # -----------------------------------------------------
        # If players clearly reset to opposite baselines at rally end,
        # we reward the rally with a small bonus.
        #
        # This bonus is intentionally small so it does not distort
        # highlight selection but helps differentiate rallies with
        # clean transitions and good flow.
        #
        # metadata.metadata is a dict inside RallyMetadata.
        baseline_reset = False
        if hasattr(metadata, "metadata") and isinstance(metadata.metadata, dict):
            baseline_reset = metadata.metadata.get("players_reset_to_baseline", False)

        if baseline_reset:
            score += 0.4
            reasons.append("baseline_reset")

        # -----------------------------------------------------
        # 9. Rally Momentum (fallback)
        # -----------------------------------------------------
        if score < 1.0 and dur > 0.4:
            score += 0.5
            reasons.append("momentum")

        # -----------------------------------------------------
        # 10. Soft Floor (NEVER return 0)
        # -----------------------------------------------------
        score = max(score, 0.2)

        # -----------------------------------------------------
        # 11. Clamp final score
        # -----------------------------------------------------
        score = clamp(score, self.config.min_score, self.config.max_score)

        return {
            "highlight_score": round(score, 4),
            "reasons": reasons,
        }
