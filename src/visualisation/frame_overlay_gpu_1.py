# FILE: src/visualisation/frame_overlay_gpu.py

import cv2
import numpy as np
import os

class FrameOverlayGPU:
    """
    Applies a transparent PNG frame overlay to a video.

    Improvements:
    - Validates input clip has frames
    - Uses H.264/AVC1 instead of mp4v (fixes empty output on Windows)
    - Logs frame counts + debug info
    - Handles corrupted/empty clips gracefully
    - Faster alpha blending
    """

    def __init__(self, frame_path: str):
        self.frame_path = frame_path

        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame PNG not found: {frame_path}")

        self.frame_png = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if self.frame_png is None:
            raise ValueError(f"Failed to load PNG: {frame_path}")

        if self.frame_png.shape[2] != 4:
            raise ValueError("Frame PNG must contain an alpha channel (RGBA).")

        print(f"[FrameOverlay] Loaded frame PNG: {frame_path}")

    def apply_to_video(self, clip_path: str, output_path: str):
        print(f"[FrameOverlay] Applying overlay to: {clip_path}")

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {clip_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print("[FrameOverlay] WARNING: Input clip has 0 frames. Skipping overlay.")
            cap.release()
            return

        print(f"[FrameOverlay] Clip info: {width}x{height} @ {fps} FPS ({total_frames} frames)")

        # Resize overlay to match video resolution
        overlay = cv2.resize(self.frame_png, (width, height))

        # Split overlay into RGB + alpha
        overlay_rgb = overlay[:, :, :3].astype(np.float32)
        overlay_alpha = (overlay[:, :, 3] / 255.0).astype(np.float32)
        overlay_alpha = overlay_alpha[..., np.newaxis]  # shape (H, W, 1)

        # Use H.264 for reliable output
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for: {output_path}")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame.astype(np.float32)

            # Alpha blend: frame*(1-a) + overlay*a
            blended = frame * (1 - overlay_alpha) + overlay_rgb * overlay_alpha
            blended = blended.astype(np.uint8)

            out.write(blended)
            frame_count += 1

        cap.release()
        out.release()

        print(f"[FrameOverlay] Overlay complete. Frames processed: {frame_count}")
        print(f"[FrameOverlay] Output saved → {output_path}")
