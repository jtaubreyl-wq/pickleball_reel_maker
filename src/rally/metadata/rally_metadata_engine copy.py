# FILE: src/rally/metadata/rally_metadata_engine.py

from __future__ import annotations

# ============================================================
# SECTION: Imports & Type Definitions
# ============================================================

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import math

from .rally_metadata import RallyMetadata, BoundingBox, Point2D


# ============================================================
# SECTION: Rally Metadata Engine
# ============================================================

class RallyMetadataEngine:
    """
    Story 5.4 — Rally Metadata Engine

    Purpose:
        Consume a RallySegment (from RallySegmentBuilder) and produce a
        fully-populated RallyMetadata object that downstream components
        (long-rally detector, winner/error detector, highlight selector,
        reel assembler) can rely on.

    Why this matters for the “1-second highlight” bug:
        If metadata is weak or incomplete:
            - duration_seconds may be ~0
            - hit_count may be 0
            - quality_score / difficulty_score may be 0
            - highlight_score stays 0
        → Highlight selector rejects all rallies
        → Reel assembler falls back to a 1-second clip.

        This upgraded engine:
            - Enforces sane minimum duration
            - Produces non-trivial hit counts when possible
            - Computes quality & difficulty scores
            - Computes a highlight_score + reasons
            - Ensures consistent, safe metadata for selection.
    """

    def __init__(self, fps_default: float = 30.0):
        """
        fps_default:
            Fallback FPS used when a segment does not carry its own fps.
        """
        self.fps_default = fps_default
        self._rally_counter = 0

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------

    def generate_metadata(self, segment) -> RallyMetadata:
        """
        Generate a RallyMetadata object from a RallySegment.

        Expected fields on `segment`:
            - start_frame: int
            - end_frame: int
            - trajectory: List[dict] with keys "x", "y"
            - speeds: List[float]
            - player_boxes: Dict[str, List[BoundingBox]] (may be empty)
            - fps: Optional[float]
            - last_hitter: Optional[str]
            - player_positions: Dict[str, Any]

        Returns:
            RallyMetadata instance with:
                - duration, hit_count
                - speed stats
                - quality_score, difficulty_score
                - highlight_score, highlight_reasons
                - long-rally flags
        """

        # --- Assign rally ID if missing ---
        if not hasattr(segment, "rally_id"):
            self._rally_counter += 1
            segment.rally_id = self._rally_counter

        # --- FPS fallback ---
        fps = getattr(segment, "fps", self.fps_default) or self.fps_default

        # --- Duration (with safety minimum) ---
        duration_seconds = self._compute_duration(
            segment.start_frame, segment.end_frame, fps
        )

        # --- Hit count ---
        hit_count = self._estimate_hit_count(segment.speeds)

        # --- Player boxes (normalize to dict) ---
        player_boxes: Dict[str, List[BoundingBox]] = segment.player_boxes or {}

        # --- Players involved ---
        players_involved = self._compute_players_involved(player_boxes)

        # --- Speed stats ---
        avg_speed, max_speed, var_speed = self._compute_speed_stats(segment.speeds)

        # --- Quality & difficulty ---
        quality_score = self._compute_quality_score(
            duration_seconds=duration_seconds,
            hit_count=hit_count,
            speeds=segment.speeds,
            player_boxes=player_boxes,
        )

        difficulty_score = self._compute_difficulty_score(
            speeds=segment.speeds,
            player_boxes=player_boxes,
        )

        # --- Trajectory → list of (x, y) ---
        trajectory_xy: List[Point2D] = [
            (p["x"], p["y"])
            for p in segment.trajectory
            if isinstance(p, dict) and "x" in p and "y" in p
        ]

        # --- Long rally flags (simple heuristic) ---
        is_long_rally, long_rally_duration, long_rally_hit_count = (
            self._compute_long_rally_flags(duration_seconds, hit_count)
        )

        # --- Highlight score & reasons (core for selection) ---
        highlight_score, highlight_reasons = self._compute_highlight_score_and_reasons(
            duration_seconds=duration_seconds,
            hit_count=hit_count,
            quality_score=quality_score,
            difficulty_score=difficulty_score,
            is_long_rally=is_long_rally,
        )

        metadata = RallyMetadata(
            # Core identifiers
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            rally_id=segment.rally_id,
            segment_path=getattr(segment, "segment_path", None),

            # Basic stats
            duration_seconds=duration_seconds,
            hit_count=hit_count,

            # Geometry / trajectory
            trajectory=trajectory_xy,
            bounding_boxes=player_boxes,

            # Ball speed stats
            ball_speeds=list(segment.speeds),
            avg_ball_speed=avg_speed,
            max_ball_speed=max_speed,
            ball_velocity_variance=var_speed,

            # Player context
            players_involved=players_involved,
            last_hitter=getattr(segment, "last_hitter", None),
            player_positions=getattr(segment, "player_positions", {}),

            # Quality & difficulty
            quality_score=quality_score,
            difficulty_score=difficulty_score,

            # Long rally flags
            is_long_rally=is_long_rally,
            long_rally_duration=long_rally_duration,
            long_rally_hit_count=long_rally_hit_count,

            # Winner / ending event (filled later by winner/error detector)
            winner=None,
            ending_event=None,

            # Highlight selection fields
            highlight_score=highlight_score,
            highlight_reasons=highlight_reasons,
            selected_for_highlight=False,
        )

        return metadata

    # ------------------------------------------------------------
    # CORE COMPUTATIONS
    # ------------------------------------------------------------

    def _compute_duration(self, start_frame: int, end_frame: int, fps: float) -> float:
        """
        Compute rally duration in seconds, with a small safety floor
        to avoid zero-duration rallies that break highlight selection.
        """
        raw = max(0.0, (end_frame - start_frame) / fps)
        # Safety: treat ultra-short segments as at least one frame long
        return max(raw, 1.0 / fps if end_frame > start_frame else 0.0)

    def _estimate_hit_count(self, speeds: List[float]) -> int:
        """
        Estimate number of hits from speed changes.

        Heuristic:
            - Use sign changes in speed derivative
            - Filter by a dynamic threshold
            - Ensure at least 1 hit if there is any motion
        """
        if not speeds:
            return 0

        speeds_arr = np.array(speeds, dtype=float)
        if len(speeds_arr) < 3:
            # Minimal heuristic: at least 1 hit if we have motion
            return max(1, len(speeds_arr) // 2)

        diffs = np.diff(speeds_arr)
        sign_changes = np.sign(diffs[:-1]) * np.sign(diffs[1:]) < 0
        candidate_indices = np.where(sign_changes)[0] + 1

        threshold = max(0.3 * speeds_arr.max(), 1.0)
        hits = [i for i in candidate_indices if speeds_arr[i] > threshold]

        return max(1, len(hits))

    def _compute_players_involved(
        self,
        player_boxes: Dict[str, List[BoundingBox]],
    ) -> List[str]:
        """
        Determine which players were meaningfully present in the rally.
        """
        if not player_boxes:
            return []

        involved: List[str] = []
        total_frames = max((len(v) for v in player_boxes.values()), default=0)
        if total_frames == 0:
            return []

        for player_id, boxes in player_boxes.items():
            presence_ratio = len(boxes) / total_frames
            if presence_ratio >= 0.1:
                involved.append(player_id)

        return involved

    def _compute_speed_stats(self, speeds: List[float]) -> Tuple[float, float, float]:
        """
        Compute average, max, and variance of ball speed.
        """
        if not speeds:
            return 0.0, 0.0, 0.0
        arr = np.array(speeds, dtype=float)
        avg = float(arr.mean())
        max_ = float(arr.max())
        var = float(arr.var())
        return avg, max_, var

    def _compute_quality_score(
        self,
        duration_seconds: float,
        hit_count: int,
        speeds: List[float],
        player_boxes: Dict[str, List[BoundingBox]],
    ) -> float:
        """
        Compute a quality score in [0, 1] based on:
            - duration
            - hit count
            - speed variance
            - player movement intensity
        """
        duration_score = min(duration_seconds / 10.0, 1.0)
        hit_score = min(hit_count / 20.0, 1.0)

        if speeds:
            var_score = min(float(np.var(speeds)) / 50.0, 1.0)
        else:
            var_score = 0.0

        movement_score = self._estimate_movement_intensity(player_boxes)

        return float(
            0.3 * duration_score
            + 0.3 * hit_score
            + 0.2 * var_score
            + 0.2 * movement_score
        )

    def _compute_difficulty_score(
        self,
        speeds: List[float],
        player_boxes: Dict[str, List[BoundingBox]],
    ) -> float:
        """
        Compute a difficulty score in [0, 1] based on:
            - average ball speed
            - player movement intensity
        """
        if speeds:
            speed_score = min(float(np.mean(speeds)) / 20.0, 1.0)
        else:
            speed_score = 0.0

        movement_score = self._estimate_movement_intensity(player_boxes)

        return float(0.5 * speed_score + 0.5 * movement_score)

    def _estimate_movement_intensity(
        self,
        player_boxes: Dict[str, List[BoundingBox]],
    ) -> float:
        """
        Estimate how much players moved during the rally.

        Heuristic:
            - Compute center of each bounding box
            - Sum distances between consecutive centers
            - Normalize by a rough scale factor
        """
        if not player_boxes:
            return 0.0

        total_distance = 0.0
        count = 0

        for boxes in player_boxes.values():
            if len(boxes) < 2:
                continue
            centers = [((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1, x2, y2) in boxes]
            for (x0, y0), (x1, y1) in zip(centers[:-1], centers[1:]):
                total_distance += math.hypot(x1 - x0, y1 - y0)
                count += 1

        if count == 0:
            return 0.0

        avg_step = total_distance / count
        # Normalize by an approximate pixel scale
        return float(min(avg_step / 100.0, 1.0))

    def _compute_long_rally_flags(
        self,
        duration_seconds: float,
        hit_count: int,
        duration_thresh: float = 10.0,
        hit_thresh: int = 20,
    ) -> Tuple[bool, float, int]:
        """
        Simple long-rally heuristic used as a fallback
        (LongRallyDetector can refine this later).
        """
        is_long = duration_seconds >= duration_thresh or hit_count >= hit_thresh
        long_duration = duration_seconds if is_long else 0.0
        long_hits = hit_count if is_long else 0
        return is_long, long_duration, long_hits

    # ------------------------------------------------------------
    # HIGHLIGHT SCORE & REASONS
    # ------------------------------------------------------------

    def _compute_highlight_score_and_reasons(
        self,
        duration_seconds: float,
        hit_count: int,
        quality_score: float,
        difficulty_score: float,
        is_long_rally: bool,
    ) -> Tuple[float, List[str]]:
        """
        Compute a scalar highlight_score and a list of human-readable
        reasons that explain why this rally is interesting.

        This is a key piece for avoiding the “all scores = 0” situation
        that leads to no highlights being selected.
        """
        reasons: List[str] = []

        # Base score from quality & difficulty
        base = 0.6 * quality_score + 0.4 * difficulty_score

        # Duration bonus
        if duration_seconds >= 8.0:
            base += 0.1
            reasons.append("long_rally_duration")

        # Hit count bonus
        if hit_count >= 10:
            base += 0.1
            reasons.append("many_shots")

        # Long rally flag bonus
        if is_long_rally:
            base += 0.1
            reasons.append("epic_long_rally")

        # Clamp to [0, 1]
        score = float(max(0.0, min(1.0, base)))

        # If still extremely low but rally is non-trivial, give a small floor
        if score < 0.05 and duration_seconds > 1.0 and hit_count > 1:
            score = 0.1
            reasons.append("non_trivial_rally_floor")

        return score, reasons
