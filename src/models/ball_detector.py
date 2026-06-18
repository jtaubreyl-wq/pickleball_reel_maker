# ============================================================
# FILE: src/models/ball_detector.py
# ============================================================

from ultralytics import YOLO
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================
# SECTION: Ball Detector (Story 4.1) — Now with infer_video()
# ============================================================

class BallDetector:
    """
    Legacy Story 4.1 Ball Detector — now upgraded to support:
        • predict(frame)
        • run_inference(video_path)  (legacy slow path)
        • infer_video(video_path)    (new fast path)
        • batch inference
        • frame skipping
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        resize_factor: float = 1.0,
        frame_skip: int = 1,
        batch_size: int = 16,
    ):
        # Auto-select device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf_threshold = conf_threshold
        self.resize_factor = resize_factor
        self.frame_skip = frame_skip
        self.batch_size = batch_size

        # Your model has exactly 1 class: {0: "Ball"}
        self.ball_class_id = 0
        self.ball_class_name = "Ball"

        logger.info(f"[BallDetector] Loaded model on {self.device}")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_factor != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=self.resize_factor,
                fy=self.resize_factor,
                interpolation=cv2.INTER_LINEAR,
            )
        return frame

    # --------------------------------------------------------
    # Predict on a single frame
    # --------------------------------------------------------

    def predict(self, frame: np.ndarray) -> List[Dict]:
        if frame is None or frame.size == 0:
            return []

        frame_proc = self._preprocess(frame)

        results = self.model.predict(
            frame_proc,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            if cls != self.ball_class_id:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Scale back if resized
            if self.resize_factor != 1.0:
                scale = 1.0 / self.resize_factor
                x1 *= scale
                y1 *= scale
                x2 *= scale
                y2 *= scale

            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": conf,
                "class_id": cls,
                "class_name": self.ball_class_name,
            })

        return detections

    # --------------------------------------------------------
    # Legacy full-video inference (slow)
    # --------------------------------------------------------

    def run_inference(self, video_path: str) -> Dict[int, List[Dict]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_idx = 0
        all_detections: Dict[int, List[Dict]] = {}

        logger.info(f"[BallDetector] Starting legacy inference on {video_path}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            all_detections[frame_idx] = self.predict(frame)
            frame_idx += 1

        cap.release()
        logger.info(f"[BallDetector] Completed legacy inference: {frame_idx} frames")
        return all_detections

    # --------------------------------------------------------
    # NEW: Fast batch + skip inference (YoloDetector-compatible)
    # --------------------------------------------------------

    def infer_video(self, video_path: str) -> List[List[Dict]]:
        """
        Fast path:
            • Frame skipping
            • Batch inference
            • Same output format as YoloDetector.infer_video()
        """

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_idx = 0
        batch_frames = []
        batch_indices = []
        all_results: List[List[Dict]] = []

        logger.info(f"[BallDetector] Starting fast infer_video() on {video_path}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Skip frames
            if frame_idx % self.frame_skip != 0:
                all_results.append([])  # maintain alignment
                frame_idx += 1
                continue

            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            # Run batch
            if len(batch_frames) == self.batch_size:
                all_results.extend(self._infer_batch(batch_frames, batch_indices))
                batch_frames, batch_indices = [], []

            frame_idx += 1

        # Final partial batch
        if batch_frames:
            all_results.extend(self._infer_batch(batch_frames, batch_indices))

        cap.release()
        logger.info(f"[BallDetector] Completed fast infer_video(): {len(all_results)} frames")
        return all_results

    # --------------------------------------------------------
    # Batch inference helper
    # --------------------------------------------------------

    def _infer_batch(self, frames: List[np.ndarray], indices: List[int]) -> List[List[Dict]]:
        """
        Runs YOLO on a batch of frames and returns list[list[detections]]
        """

        # Preprocess batch
        proc_frames = [self._preprocess(f) for f in frames]

        results = self.model(
            proc_frames,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )

        output = []
        for frame_idx, frame, result in zip(indices, frames, results):
            detections = []

            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)

                if cls != self.ball_class_id:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Scale back if resized
                if self.resize_factor != 1.0:
                    scale = 1.0 / self.resize_factor
                    x1 *= scale
                    y1 *= scale
                    x2 *= scale
                    y2 *= scale

                detections.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "conf": conf,
                    "class_id": cls,
                    "class_name": self.ball_class_name,
                })

            output.append(detections)

        return output
