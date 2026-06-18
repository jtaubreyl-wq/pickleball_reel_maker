# FILE: src/models/yolo_detector.py

from typing import List, Dict, Literal, Optional
import time
import logging
import cv2
import torch
from ultralytics import YOLO
import numpy as np

logger = logging.getLogger(__name__)

BBox = Dict[str, float]


# ============================================================
# SECTION: YOLO Detector Wrapper (Optimized + Rally‑Safe)
# ============================================================

class YoloDetector:
    """
    Unified YOLOv8/YOLOv10 detector wrapper with:
        • GPU acceleration
        • Half precision
        • Batch inference
        • Frame skipping
        • Standardized bbox output
        • Debug logging
        • Corrupted‑frame protection
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[Literal["cpu", "cuda"]] = None,
        half: bool = True,
        imgsz: int = 1280,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        frame_skip: int = 2,
        batch_size: int = 16,
    ) -> None:

        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.frame_skip = frame_skip
        self.batch_size = batch_size

        # Auto‑select device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load YOLO model
        t0 = time.time()
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Optional half precision
        if self.device == "cuda" and half:
            try:
                self.model.model.half()
                logger.info("[YoloDetector] Using half precision on GPU")
            except Exception:
                logger.warning("[YoloDetector] Half precision not supported, using full precision")

        self.load_time = time.time() - t0
        logger.info(f"[YoloDetector] Loaded {model_path} on {self.device} in {self.load_time:.3f}s")

    # --------------------------------------------------------
    # SECTION: YOLO → Standardized BBox Format
    # --------------------------------------------------------

    def _to_bboxes(self, result, frame_width: int, frame_height: int) -> List[BBox]:

        bboxes: List[BBox] = []

        xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else np.empty((0, 4))
        conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.empty((0,))
        cls = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.empty((0,))

        names = self.model.names

        for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
            class_id = int(cl)
            class_name = names.get(class_id, str(class_id))

            # Clamp coordinates
            x1 = max(0, min(frame_width, x1))
            y1 = max(0, min(frame_height, y1))
            x2 = max(0, min(frame_width, x2))
            y2 = max(0, min(frame_height, y2))

            bboxes.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(c),
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

        return bboxes

    # --------------------------------------------------------
    # SECTION: Single‑Frame Inference (unchanged)
    # --------------------------------------------------------

    @torch.no_grad()
    def infer(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> List[BBox]:

        if frame is None or frame.size == 0:
            logger.warning("[YoloDetector] Empty or corrupted frame")
            return []

        h, w = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        if frame_idx is not None and frame_idx % 30 == 0:
            for r in results:
                logger.debug(f"[YoloDetector] Frame {frame_idx} raw boxes: {r.boxes.xyxy.cpu().numpy()}")

        return self._to_bboxes(results[0], frame_width=w, frame_height=h)

    # --------------------------------------------------------
    # SECTION: High‑Performance Video Inference
    # --------------------------------------------------------

    @torch.no_grad()
    def infer_video(self, video_path: str) -> List[List[BBox]]:
        """
        High‑performance YOLO inference:
            • Frame skipping
            • Batch inference
            • Streaming decode
        """

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_idx = 0
        batch_frames = []
        batch_indices = []
        all_results: List[List[BBox]] = []

        while True:
            ret, frame = cap.read()
            if not ret:
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
        return all_results

    # --------------------------------------------------------
    # SECTION: Batch Inference Helper
    # --------------------------------------------------------

    def _infer_batch(self, frames: List[np.ndarray], indices: List[int]) -> List[List[BBox]]:

        results = self.model(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        output = []
        for frame_idx, frame, result in zip(indices, frames, results):
            h, w = frame.shape[:2]
            output.append(self._to_bboxes(result, frame_width=w, frame_height=h))

        return output


# ============================================================
# SECTION: Backwards‑Compatible Wrapper
# ============================================================

def run_yolo_tracking(frames, model_path="yolov8n.pt") -> List[List[BBox]]:
    """
    Legacy API — still supported.
    Uses single‑frame inference (slow).
    """

    detector = YoloDetector(model_path=model_path)

    all_detections = []
    for idx, frame in enumerate(frames):
        all_detections.append(detector.infer(frame, frame_idx=idx))

    return all_detections
