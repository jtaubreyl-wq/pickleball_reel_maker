# FILE: src/rally/metadata/rally_metadata_engine.py

from __future__ import annotations

# ============================================================
# SECTION: Imports & Type Definitions
# ============================================================

from typing import List, Dict, Optional, Any, Tuple
from types import SimpleNamespace
import numpy as np
import math

from .rally_metadata import RallyMetadata, BoundingBox, Point2D
from src.rally.scoring.highlight_scoring_engine import HighlightScoringEngine


# ============================================================
# SECTION: Rally Metadata Engine
# ============================================================

class RallyMetadataEngine:
    """
    Fully upgraded Rally Metadata Engine.

    Produces:
        • duration, hit_count
        • speed statistics
        • curvature / direction-change metrics
        • pace / volley / complexity metrics
        • quality_score + difficulty_score
        • highlight_score via HighlightScoringEngine
        • inferred final player positions
        • baseline reset detection (players ending on opposite sides)

    NOTE:
        We do NOT modify RallyMetadata constructor fields.
        Instead, new optional fields are stored inside metadata.metadata
        to avoid breaking your existing pipeline.
    """

    def __init__(self, fps_default: float = 30.0):
        self.fps_default = fps_default
        self._rally_counter = 0
        self.scoring_engine = HighlightScoringEngine()

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------

    def generate_metadata(self, segment) -> RallyMetadata:
        """
        Generate a fully-populated RallyMetadata object from a RallySegment.

        This method:
            • Computes all rally metrics
            • Infers final player positions if missing
            • Computes baseline reset flag
            • Passes all metrics to HighlightScoringEngine
        """

        # --- Assign rally ID if missing ---
        if not hasattr(segment, "rally_id"):
            self._rally_counter += 1
            segment.rally_id = self._rally_counter

        # --- FPS fallback ---
        fps = getattr(segment, "fps", self.fps_default) or self.fps_default

        # --- Duration ---
        duration_seconds = self._compute_duration(
            segment.start_frame, segment.end_frame, fps
        )

        # --- Hit count ---
        hit_count = self._estimate_hit_count(segment.speeds)

        # --- Player boxes ---
        player_boxes: Dict[str, List[BoundingBox]] = (
            getattr(segment, "player_boxes", {}) or {}
        )

        # --- Players involved ---
        players_involved = self._compute_players_involved(player_boxes)

        # --- Speed stats ---
        avg_speed, max_speed, var_speed = self._compute_speed_stats(segment.speeds)

        # --- Trajectory list of (x, y) ---
        trajectory_xy: List[Point2D] = [
            (float(p["x"]), float(p["y"]))
            for p in getattr(segment, "trajectory", [])
            if isinstance(p, dict) and "x" in p and "y" in p
        ]

        # --- Curvature / direction / pace / volley / complexity ---
        (
            curvature_score,
            direction_change_score,
            pace_score,
            volley_score,
            rally_complexity_score,
            direction_change_count,
            direction_change_rate,
        ) = self._compute_motion_metrics(
            trajectory_xy,
            segment.speeds,
            fps,
        )

        # --- Quality & difficulty ---
        quality_score = self._compute_quality_score(
            duration_seconds=duration_seconds,
            hit_count=hit_count,
            speeds=segment.speeds,
            curvature_score=curvature_score,
            pace_score=pace_score,
        )

        difficulty_score = self._compute_difficulty_score(
            speeds=segment.speeds,
            direction_change_score=direction_change_score,
            rally_complexity_score=rally_complexity_score,
        )

        # --- Long rally flags ---
        is_long_rally, long_rally_duration, long_rally_hit_count = (
            self._compute_long_rally_flags(duration_seconds, hit_count)
        )

        # --- Ending event ---
        ending_event = getattr(segment, "ending_event", None)

        # ------------------------------------------------------------
        # PLAYER POSITION / BASELINE LOGIC
        # ------------------------------------------------------------

        # 1. Get existing player positions if provided
        player_positions: Dict[str, Any] = getattr(segment, "player_positions", {}) or {}

        # 2. If missing, infer from bounding boxes
        if not player_positions:
            inferred = self._infer_final_player_positions(player_boxes)
            if inferred:
                player_positions = inferred

        # 3. Compute baseline reset flag
        players_reset_to_baseline = self._compute_players_reset_to_baseline(player_positions)

        # ------------------------------------------------------------
        # HIGHLIGHT SCORING
        # ------------------------------------------------------------

        scoring_input = SimpleNamespace(
            duration_seconds=duration_seconds,
            hit_count=hit_count,
            ball_speeds=list(segment.speeds),
            quality_score=quality_score,
            difficulty_score=difficulty_score,
            ending_event=ending_event,
            is_long_rally=is_long_rally,
            curvature_score=curvature_score,
            direction_change_score=direction_change_score,
            direction_change_count=direction_change_count,
            direction_change_rate=direction_change_rate,
            pace_score=pace_score,
            volley_score=volley_score,
            rally_complexity_score=rally_complexity_score,
        )

        score_result = self.scoring_engine.score_rally(scoring_input)
        highlight_score = score_result["highlight_score"]
        highlight_reasons = score_result["reasons"]

        # ------------------------------------------------------------
        # BUILD METADATA OBJECT
        # ------------------------------------------------------------

        metadata = RallyMetadata(
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            rally_id=segment.rally_id,
            segment_path=getattr(segment, "segment_path", None),

            duration_seconds=duration_seconds,
            hit_count=hit_count,

            trajectory=trajectory_xy,
            bounding_boxes=player_boxes,

            ball_speeds=list(segment.speeds),
            avg_ball_speed=avg_speed,
            max_ball_speed=max_speed,
            ball_velocity_variance=var_speed,

            curvature_score=curvature_score,
            direction_change_score=direction_change_score,
            pace_score=pace_score,
            volley_score=volley_score,
            rally_complexity_score=rally_complexity_score,

            players_involved=players_involved,
            last_hitter=getattr(segment, "last_hitter", None),
            player_positions=player_positions,

            quality_score=quality_score,
            difficulty_score=difficulty_score,

            is_long_rally=is_long_rally,
            long_rally_duration=long_rally_duration,
            long_rally_hit_count=long_rally_hit_count,

            winner=getattr(segment, "winner", None),
            ending_event=ending_event,

            highlight_score=highlight_score,
            highlight_reasons=highlight_reasons,
            selected_for_highlight=False,
        )

        # Store new optional fields safely inside metadata.metadata
        metadata.metadata["players_reset_to_baseline"] = players_reset_to_baseline

        return metadata

    # ============================================================
    # CORE COMPUTATIONS
    # ============================================================

    def _compute_duration(self, start_frame: int, end_frame: int, fps: float) -> float:
        raw = max(0.0, (end_frame - start_frame) / fps)
        return max(raw, 1.0 / fps if end_frame > start_frame else 0.0)

    def _estimate_hit_count(self, speeds: List[float]) -> int:
        if not speeds or len(speeds) < 3:
            return max(1, len(speeds) // 5)

        arr = np.asarray(speeds, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std()) or 1.0
        threshold = mean + 0.5 * std

        hits = 0
        for i in range(1, len(arr) - 1):
            if arr[i] > threshold and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
                hits += 1

        return max(hits, 1)

    def _compute_players_involved(
        self,
        player_boxes: Dict[str, List[BoundingBox]],
    ) -> List[str]:
        return [pid for pid, boxes in player_boxes.items() if boxes]

    def _compute_speed_stats(self, speeds: List[float]) -> Tuple[float, float, float]:
        if not speeds:
            return 0.0, 0.0, 0.0
        arr = np.asarray(speeds, dtype=float)
        return float(arr.mean()), float(arr.max()), float(arr.var())

    # ------------------------------------------------------------
    # MOTION METRICS
    # ------------------------------------------------------------

    def _compute_motion_metrics(
        self,
        trajectory: List[Point2D],
        speeds: List[float],
        fps: float,
    ) -> Tuple[float, float, float, float, float, int, float]:

        if len(trajectory) < 3:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0

        # --- Direction angles ---
        directions = []
        for i in range(1, len(trajectory)):
            x0, y0 = trajectory[i - 1]
            x1, y1 = trajectory[i]
            dx, dy = x1 - x0, y1 - y0
            if dx == 0 and dy == 0:
                directions.append(directions[-1] if directions else 0.0)
            else:
                directions.append(math.atan2(dy, dx))

        # --- Angle deltas ---
        deltas = []
        for i in range(1, len(directions)):
            d = directions[i] - directions[i - 1]
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            deltas.append(abs(d))

        if not deltas:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0

        # --- Curvature ---
        mean_abs_delta = float(np.mean(deltas))
        curvature_score = min(mean_abs_delta / (math.pi / 2), 1.0)

        # --- Direction changes ---
        change_threshold = 0.15
        direction_change_count = int(sum(1 for d in deltas if d > change_threshold))
        direction_change_score = min(
            direction_change_count / max(5, len(trajectory) / 10), 1.0
        )

        # --- Pace & volley ---
        if speeds:
            arr = np.asarray(speeds, dtype=float)
            avg_speed = float(arr.mean())
            pace_score = min(avg_speed / 20.0, 1.0)

            high_speed_thresh = max(5.0, 0.5 * float(arr.max()))
            volley_score = min(
                sum(1 for s in speeds if s >= high_speed_thresh) / len(speeds),
                1.0,
            )
        else:
            pace_score = 0.0
            volley_score = 0.0

        # --- Complexity ---
        rally_complexity_score = min(
            0.3 * curvature_score
            + 0.25 * direction_change_score
            + 0.25 * pace_score
            + 0.2 * volley_score,
            1.0,
        )

        # --- Direction change rate ---
        duration_s = max(len(trajectory) / fps, 1e-3)
        direction_change_rate = direction_change_count / duration_s

        return (
            curvature_score,
            direction_change_score,
            pace_score,
            volley_score,
            rally_complexity_score,
            direction_change_count,
            direction_change_rate,
        )

    # ------------------------------------------------------------
    # QUALITY / DIFFICULTY
    # ------------------------------------------------------------

    def _compute_quality_score(
        self,
        duration_seconds: float,
        hit_count: int,
        speeds: List[float],
        curvature_score: float,
        pace_score: float,
    ) -> float:
        dur_norm = min(duration_seconds / 12.0, 1.0)
        hit_norm = min(hit_count / 12.0, 1.0)
        speed_norm = min(np.mean(speeds) / 20.0, 1.0) if speeds else 0.0

        return float(
            0.4 * dur_norm
            + 0.3 * hit_norm
            + 0.2 * speed_norm
            + 0.1 * curvature_score
        )

    def _compute_difficulty_score(
        self,
        speeds: List[float],
        direction_change_score: float,
        rally_complexity_score: float,
    ) -> float:
        speed_score = min(np.mean(speeds) / 25.0, 1.0) if speeds else 0.0

        return float(
            0.5 * speed_score
            + 0.25 * direction_change_score
            + 0.25 * rally_complexity_score
        )

    # ------------------------------------------------------------
    # LONG RALLY FLAGS
    # ------------------------------------------------------------

    def _compute_long_rally_flags(
        self,
        duration_seconds: float,
        hit_count: int,
    ) -> Tuple[bool, float, int]:
        is_long = duration_seconds >= 6.0 or hit_count >= 8
        return (
            is_long,
            duration_seconds if is_long else 0.0,
            hit_count if is_long else 0,
        )

    # ============================================================
    # PLAYER POSITION INFERENCE / BASELINE LOGIC
    # ============================================================

    def _infer_final_player_positions(
        self,
        player_boxes: Dict[str, List[BoundingBox]],
    ) -> Dict[str, Any]:
        """
        Infer final player positions from the last bounding box of each player.

        Returns:
            Dict[player_id -> {"cx": float, "cy": float}]
        """
        final_positions: Dict[str, Any] = {}

        for player_id, boxes in player_boxes.items():
            if not boxes:
                continue

            last_box = boxes[-1]

            # Handle both object and dict-style bounding boxes
            if hasattr(last_box, "x"):
                cx = float(last_box.x + last_box.w / 2.0)
                cy = float(last_box.y + last_box.h / 2.0)
            else:
                x = float(last_box.get("x", 0.0))
                y = float(last_box.get("y", 0.0))
                w = float(last_box.get("w", 0.0))
                h = float(last_box.get("h", 0.0))
                cx = x + w / 2.0
                cy = y + h / 2.0

            final_positions[player_id] = {"cx": cx, "cy": cy}

        return final_positions

    def _compute_players_reset_to_baseline(
        self,
        player_positions: Dict[str, Any],
    ) -> bool:
        """
        Determine whether players ended the rally on opposite sides of the court,
        approximating a 'baseline reset'.

        This is intentionally simple and non-destructive:
            • If we have at least 2 players with 'cy' positions,
              we check whether they are on opposite halves of the court.
            • We do NOT require court bounds here; we infer relative positions.

        Returns:
            True if players appear to have reset to opposite sides.
            False otherwise.
        """
        if not player_positions or len(player_positions) < 2:
            return False

        # Extract all cy values
        cy_values = [pos.get("cy") for pos in player_positions.values() if "cy" in pos]
        if len(cy_values) < 2:
            return False

        # Simple heuristic:
        # If the vertical spread is large, players are likely on opposite sides.
        spread = max(cy_values) - min(cy_values)

        # Threshold is heuristic; can be tuned later
        return spread > 100.0
