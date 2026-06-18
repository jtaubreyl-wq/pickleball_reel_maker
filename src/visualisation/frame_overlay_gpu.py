import cv2
import numpy as np
import os
import shutil
import subprocess


class FrameOverlayGPU:
    """
    Production‑grade GPU overlay engine.

    Pipeline:
    1. Load PNG (must be RGBA)
    2. Auto‑crop transparent padding
    3. Pad to perfect 16:9
    4. Scale + center‑crop to match video
    5. Alpha‑blend onto each frame
    6. Write temporary MP4 (OpenCV)
    7. FFmpeg re‑encode → H.264 + yuv420p (Messenger‑compatible)
    """

    # ------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------

    def __init__(self, frame_path: str):
        self.frame_path = frame_path

        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame PNG not found: {frame_path}")

        self.frame_png = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if self.frame_png is None:
            raise ValueError(f"Failed to load PNG: {frame_path}")

        if self.frame_png.shape[2] != 4:
            raise ValueError("Frame PNG must contain an alpha channel (RGBA).")

        print(f"[FrameOverlay] Loaded PNG: {frame_path}")
        print(f"[FrameOverlay] Original PNG resolution: {self.frame_png.shape[1]}x{self.frame_png.shape[0]}")

        # Step 1 — Auto-crop transparent padding
        self.frame_png = self._auto_crop(self.frame_png)
        print(f"[FrameOverlay] Cropped PNG resolution: {self.frame_png.shape[1]}x{self.frame_png.shape[0]}")

        # Step 2 — Pad to perfect 16:9
        self.frame_png = self._pad_to_aspect_ratio(self.frame_png)
        print(f"[FrameOverlay] Padded PNG resolution (16:9): {self.frame_png.shape[1]}x{self.frame_png.shape[0]}")

    # ------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------

    def _auto_crop(self, img):
        alpha = img[:, :, 3]
        mask = (alpha > 50).astype(np.uint8) * 255

        coords = cv2.findNonZero(mask)
        if coords is None:
            raise ValueError("PNG is fully transparent — nothing to overlay.")

        x, y, w, h = cv2.boundingRect(coords)
        print(f"[FrameOverlay] Auto-crop bounding box: x={x}, y={y}, w={w}, h={h}")
        return img[y:y+h, x:x+w]

    def _pad_to_aspect_ratio(self, img, target_ar=16/9):
        h, w = img.shape[:2]
        current_ar = w / h

        if abs(current_ar - target_ar) < 0.001:
            print("[FrameOverlay] PNG already 16:9 — no padding needed.")
            return img

        if current_ar > target_ar:
            # Too wide → pad vertically
            new_h = int(w / target_ar)
            pad = new_h - h
            top = pad // 4
            #bottom = pad - top
            bottom = 20
            padded = cv2.copyMakeBorder(img, top, bottom, 0, 0,
                                        cv2.BORDER_CONSTANT, value=[0,0,0,0])
            print(f"[FrameOverlay] Added vertical padding: top={top}, bottom={bottom}")
            return padded

        else:
            # Too tall → pad horizontally
            new_w = int(h * target_ar)
            pad = new_w - w
            left = pad // 2
            right = pad - left
            padded = cv2.copyMakeBorder(img, 0, 0, left, right,
                                        cv2.BORDER_CONSTANT, value=[0,0,0,0])
            print(f"[FrameOverlay] Added horizontal padding: left={left}, right={right}")
            return padded

    def _scale_and_center_crop(self, img, target_w, target_h, scale=1.07):
        new_w = int(target_w * scale)
        new_h = int(target_h * scale)

        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        x_start = (new_w - target_w) // 2
        y_start = (new_h - target_h) // 2

        cropped = scaled[y_start:y_start+target_h, x_start:x_start+target_w]

        print(f"[FrameOverlay] Scaled overlay to {new_w}x{new_h} → cropped to {target_w}x{target_h}")
        return cropped

    # ------------------------------------------------------------
    # FFmpeg RE-ENCODE
    # ------------------------------------------------------------

    def _ffmpeg_reencode(self, temp_path, final_path):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg not found. Install FFmpeg to enable final MP4 encoding.")

        cmd = [
            "ffmpeg", "-y",
            "-i", temp_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            final_path
        ]

        print(f"[FrameOverlay] Re-encoding with FFmpeg → {final_path}")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if not os.path.exists(final_path):
            raise RuntimeError("FFmpeg failed to produce final MP4.")

        print("[FrameOverlay] FFmpeg re-encode complete.")

    # ------------------------------------------------------------
    # MAIN PUBLIC METHOD
    # ------------------------------------------------------------

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

        print(f"[FrameOverlay] Video resolution: {width}x{height} @ {fps} FPS")

        # Step 3 — Scale overlay
        overlay = self._scale_and_center_crop(self.frame_png, width, height)

        overlay_rgb = overlay[:, :, :3].astype(np.float32)
        overlay_alpha = (overlay[:, :, 3] / 255.0).astype(np.float32)[..., None]

        print(f"[FrameOverlay] Final overlay resolution: {overlay_rgb.shape[1]}x{overlay_rgb.shape[0]}")

        # TEMPORARY OUTPUT (OpenCV)
        temp_output = output_path.replace(".mp4", "_temp.mp4")

        # Use mp4v for temp file (OpenCV safe)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for: {temp_output}")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame.astype(np.float32)
            blended = frame * (1 - overlay_alpha) + overlay_rgb * overlay_alpha
            blended = blended.astype(np.uint8)

            out.write(blended)
            frame_count += 1

        cap.release()
        out.release()

        print(f"[FrameOverlay] Overlay complete. Frames processed: {frame_count}")
        print(f"[FrameOverlay] Temporary output saved → {temp_output}")

        # FINAL RE-ENCODE
        self._ffmpeg_reencode(temp_output, output_path)

        # Cleanup
        if os.path.exists(temp_output):
            os.remove(temp_output)

        print(f"[FrameOverlay] Final MP4 saved → {output_path}")
