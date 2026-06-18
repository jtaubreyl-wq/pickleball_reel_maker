# ============================================================
# FILE: src/tracking/ball/ball_tracking_pipeline.py
# ============================================================
"""
BallTrackingPipeline

Implements the full ball‑tracking → smoothing → trajectory analytics →
multi‑rally start/end detection → rally segmentation → metadata →
winner/forced‑error detection → highlight scoring pipeline.

Stories implemented:
    4.1 – YOLO Ball Detection
    4.2 – Basic Smoothing (EMA + SG + Outlier Removal)
    4.4 – Advanced Smoothing (SG + Kalman)
    4.3 – Trajectory Analytics
    5.1 – Rally Start Detection
    5.2 – Rally End Detection
    5.3 – Rally Segment Builder
    5.4 – Rally Metadata Engine
    6.2 – Winner / Forced Error Detection
    6.3 – Highlight Scoring Engine
"""
import cv2
from typing import Dict, List
import time
from collections import deque

# -----------------------------
# Story 4.1 – YOLO Ball Detection
# -----------------------------
from src.tracking.ball.ball_tracker import BallTracker

# -----------------------------
# Story 4.2 – Basic Smoothing
# -----------------------------
from src.tracking.ball.ball_trajectory_smoothing import (
    TrajectorySmoother as Story4_2Smoother,
)

# -----------------------------
# Story 4.4 – Advanced Smoothing
# -----------------------------
from src.tracking.ball.ball_trajectory_smoothing_advanced import (
    TrajectorySmoother as Story4_4Smoother,
)

# -----------------------------
# Story 4.3 – Trajectory Analytics
# -----------------------------
from src.tracking.ball.ball_trajectory_extractor import BallTrajectoryExtractor

# -----------------------------
# Story 5.1 – Rally Start Detection
# -----------------------------
from src.rally.detection.rally_start_detector import (
    RallyStartDetector,
    CourtBounds,
    BallState,
    PlayerState,
)

# -----------------------------
# Story 5.2 – Rally End Detection
# -----------------------------
from src.rally.detection.rally_end_detector import RallyEndDetector

# -----------------------------
# Story 5.3 – Rally Segment Builder
# -----------------------------
from src.rally.segmentation.rally_segment_builder import RallySegmentBuilder

# -----------------------------
# Story 5.4 – Rally Metadata Engine
# -----------------------------
from src.rally.metadata.rally_metadata_engine import RallyMetadataEngine

# -----------------------------
# Story 6.2 – Winner / Forced Error Detection
# -----------------------------
from src.rally.detection.winner_forced_error_detector import WinnerForcedErrorDetector

# -----------------------------
# Story 6.3 – Highlight Scoring Engine
# -----------------------------
from src.rally.scoring.highlight_scoring_engine import HighlightScoringEngine
from src.rally.scoring.highlight_scoring_config import HighlightScoringConfig

# -----------------------------
# Story  – PlayerDetector
# -----------------------------
from src.tracking.player.player_detector import PlayerDetector
from src.tracking.player.player_tracker_sort import PlayerTrackerSORT
from src.tracking.player.player_state_extractor import PlayerStateExtractor


# ---------------------------------------------------------
# Debug helper
# ---------------------------------------------------------
def debug(msg: str):
    print(f"[DEBUG][Pipeline] {msg}", flush=True)


# ============================================================
# CLASS: BallTrackingPipeline
# ============================================================
class BallTrackingPipeline:
    """
    Full ball tracking pipeline with multi‑rally detection.
    """

    # ---------------------------------------------------------
    # INIT
    # ---------------------------------------------------------
    def __init__(self, detector, tracker: BallTracker, fps: float, court):

        debug("Initializing BallTrackingPipeline...")

        self.detector = detector
        self.tracker = tracker
        self.fps = fps
        self.court = court

        # Build CourtBounds
        self.court_bounds = CourtBounds(
            left=self.court.left,
            right=self.court.right,
            top=self.court.top,
            bottom=self.court.bottom,
        )
        debug(f"Court bounds loaded: {self.court_bounds}")

        # Smoothing + trajectory
        self.story4_2_smoother = Story4_2Smoother()
        self.story4_4_smoother = Story4_4Smoother()
        self.extractor = BallTrajectoryExtractor()

        # Rally detectors
        self.rally_start_detector = RallyStartDetector(
            court_bounds=self.court_bounds,
            fps=self.fps,
            velocity_threshold=3.0,
            min_sustain_frames=3,
        )

        self.rally_end_detector = RallyEndDetector(
            court_bounds=self.court_bounds,
            fps=self.fps,
            stop_speed_threshold=1.0,
            min_stop_frames=4,
        )

        # Rally segmentation + metadata
        self.rally_segment_builder = RallySegmentBuilder()
        self.rally_metadata_engine = RallyMetadataEngine(fps_default=self.fps)

        # Winner detection + scoring
        self.winner_detector = WinnerForcedErrorDetector(
            court_bounds=self.court_bounds,
            forced_error_distance_thresh=1.0,
        )
        self.scoring_engine = HighlightScoringEngine(
            config=HighlightScoringConfig()
        )

        # Rolling context for dynamic detection
        window_frames = int(self.fps * 3) if self.fps > 0 else 90
        self.trajectory_buffer = deque(maxlen=window_frames)
        self.speed_buffer = deque(maxlen=window_frames)

        self.start_events: List[Dict] = []
        self.end_events: List[Dict] = []

        # PlayerDetector + Tracker + StateExtractor
        self.player_detector = PlayerDetector(model_path="models/yolov8s.pt")
        self.player_tracker = PlayerTrackerSORT()
        self.player_state_extractor = PlayerStateExtractor(court_bounds=self.court_bounds)

        debug(
            f"Rolling buffers initialized: window_frames={window_frames}, "
            f"trajectory_buffer.maxlen={self.trajectory_buffer.maxlen}, "
            f"speed_buffer.maxlen={self.speed_buffer.maxlen}"
        )

        debug("BallTrackingPipeline initialized.")

    # =====================================================================
    # Loads a frame by index from the video
    # =====================================================================
    def _load_frame(self, frame_idx: int):
        import cv2 as cv2_local
        if not hasattr(self, "_frame_reader"):
            self._frame_reader = cv2_local.VideoCapture(self.video_path)

        self._frame_reader.set(cv2_local.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = self._frame_reader.read()
        if not ret:
            return None
        return frame

    # =====================================================================
    # PROCESS VIDEO
    # =====================================================================
    def process_video(self, video_path: str) -> Dict:
        """
        Main pipeline entry point.
        """

        debug("===== BEGIN PIPELINE: process_video() =====")
        debug(f"Video path: {video_path}")

        # Reset detectors
        self.rally_start_detector.reset_rally_state()
        self.rally_end_detector.reset_rally_state()
        self.start_events = []
        self.end_events = []

        if hasattr(self.rally_segment_builder, "reset"):
            self.rally_segment_builder.reset()

        self.trajectory_buffer.clear()
        self.speed_buffer.clear()

        self.video_path = video_path


        # ---------------------------------------------------------
        # Story 4.1 – YOLO Ball Detection
        # ---------------------------------------------------------
        debug("STEP 4.1: Running YOLO inference...")
        try:
            start = time.time()
            detections = self.detector.run_inference(video_path)
            debug(f"YOLO inference returned in {time.time() - start:.2f}s")
            debug(f"YOLO detections complete. Frames detected: {len(detections)}")
        except Exception as e:
            debug("ERROR in YOLO detection!")
            raise e

        # ---------------------------------------------------------
        # Story 4.2 – Tracking
        # ---------------------------------------------------------
        debug("STEP 4.2: Running ball tracking...")
        try:
            raw_track = self.tracker.track_sequence(detections)
            debug(f"Tracking complete. Raw track points: {len(raw_track)}")
        except Exception as e:
            debug("ERROR in tracking!")
            raise e

        # ---------------------------------------------------------
        # Story 4.2 – Basic Smoothing
        # ---------------------------------------------------------
        debug("STEP 4.2b: Basic smoothing...")
        try:
            clean_track = self.story4_2_smoother.smooth(raw_track)
            debug(f"Basic smoothing complete. Clean track points: {len(clean_track)}")
        except Exception as e:
            debug("ERROR in basic smoothing!")
            raise e

        # ---------------------------------------------------------
        # Story 4.4 – Advanced Smoothing
        # ---------------------------------------------------------
        debug("STEP 4.4: Advanced smoothing...")
        try:
            smooth_track = self.story4_4_smoother.smooth(clean_track)
            debug(
                f"Advanced smoothing complete. Smooth track points: {len(smooth_track)}"
            )
        except Exception as e:
            debug("ERROR in advanced smoothing!")
            raise e

        # ---------------------------------------------------------
        # Story 4.3 – Trajectory Analytics
        # ---------------------------------------------------------
        debug("STEP 4.3: Extracting trajectory...")
        try:
            trajectory = self.extractor.extract(smooth_track)
            debug(f"Trajectory extraction complete. Points: {len(trajectory)}")

            trajectory = [p for p in trajectory if isinstance(p, dict)]
            debug(f"Trajectory filtered to dict points: {len(trajectory)}")

        except Exception as e:
            debug("ERROR in trajectory extraction!")
            raise e

        # ---------------------------------------------------------
        # Story 5.1 + 5.2 – Multi‑rally Start/End Detection
        # ---------------------------------------------------------
        debug("STEP 5.1 + 5.2: Detecting rally start/end events (multi‑rally, dynamic)...")

        try:
            self.start_events = []
            self.end_events = []
            rally_active = False

            for point in trajectory:

                # Build BallState
                ball_state = BallState(
                    frame_idx=point["frame"],
                    x=point["x"],
                    y=point["y"],
                    vx=point["vx"],
                    vy=point["vy"],
                )

                # NEW: Player detection + tracking + state extraction
                frame = self._load_frame(point["frame"])
                player_dets = self.player_detector.detect_players(frame)
                tracked_players = self.player_tracker.update(player_dets)

                players = self.player_state_extractor.build_player_state(
                    tracked_players=tracked_players,
                    ball_state=ball_state,
                )

                # Rolling context
                self.trajectory_buffer.append(point)
                speed = (point["vx"] ** 2 + point["vy"] ** 2) ** 0.5
                self.speed_buffer.append(speed)

                # Rally start detection
                if not rally_active:
                    try:
                        event = self.rally_start_detector.detect(
                            ball_state=ball_state,
                            player_state=players,
                            trajectory=list(self.trajectory_buffer),
                            speeds=list(self.speed_buffer),
                        )
                    except TypeError:
                        event = self.rally_start_detector.detect(ball_state, players)

                    if event:
                        debug(f"Rally start detected at frame {event['frame']}")
                        self.start_events.append(event)
                        rally_active = True
                        self.rally_end_detector.start_rally(event["frame"])

                # Rally end detection
                else:
                    try:
                        event = self.rally_end_detector.detect(
                            ball_state=ball_state,
                            player_state=players,
                            trajectory=list(self.trajectory_buffer),
                            speeds=list(self.speed_buffer),
                        )
                    except TypeError:
                        event = self.rally_end_detector.detect(ball_state, players)

                    if event:
                        debug(f"Rally end detected at frame {event['frame']}")
                        self.end_events.append(event)
                        rally_active = False
                        self.rally_start_detector.reset_rally_state()

            debug(f"Rally start events detected: {len(self.start_events)}")
            debug(f"Rally end events detected: {len(self.end_events)}")

        except Exception as e:
            debug("ERROR in multi‑rally start/end detection!")
            raise e

        # ---------------------------------------------------------
        # Story 5.4 → 6.3 – Metadata, Winner Detection, Scoring
        # ---------------------------------------------------------
        debug("STEP 5.4 → 6.3: Generating metadata, winner detection, scoring...")
        rally_metadata = []

        try:
            for segment in rally_segments:
                metadata = self.rally_metadata_engine.generate_metadata(segment)

                # Winner / Forced Error
                result = self.winner_detector.analyze(
                    trajectory=metadata.trajectory,
                    last_hitter=metadata.last_hitter,
                    player_positions=metadata.player_positions,
                )
                metadata.winner = result.winner
                metadata.ending_event = result.ending_event

                # Highlight scoring
                score_result = self.scoring_engine.score_rally(metadata)
                metadata.highlight_score = score_result["highlight_score"]
                metadata.highlight_reasons = score_result["reasons"]

                rally_metadata.append(metadata)

            debug(f"Metadata entries generated: {len(rally_metadata)}")

        except Exception as e:
            debug("ERROR in metadata/winner/scoring pipeline!")
            raise e

        debug("===== PIPELINE COMPLETE =====")

        return {
            "raw_track": raw_track,
            "clean_track": clean_track,
            "smooth_track": smooth_track,
            "trajectory": trajectory,
            "start_events": self.start_events,
            "end_events": self.end_events,
            "rally_segments": rally_segments,
            "rally_metadata": rally_metadata,
        }
    
    
