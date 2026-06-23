import cv2
import numpy as np
import os
import shutil
import subprocess
from pathlib import Path


class FrameOverlayGPU:
    """
    TRUE GPU‑accelerated overlay engine.

    GPU Acceleration:
    - cv2.cuda_GpuMat for all frame operations
    - cv2.cuda.resize for scaling
    - cv2.cuda.alphaComp for blending
    - FFmpeg NVENC for GPU H.264 encoding
    """

    def __init__(self, frame_path: str):
        self.frame_path = frame_path

        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame PNG not found: {frame_path}")

        # Load PNG on CPU (OpenCV cannot load directly to GPU)
        png = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if png is None:
            raise ValueError(f"Failed to load PNG: {frame_path}")

        if png.shape[2] != 4:
            raise ValueError("Frame PNG must contain an alpha channel (RGBA).")

        print(f"[FrameOverlayGPU] Loaded PNG: {frame_path}")

        # Preprocess PNG (crop + pad)
        png = self._auto_crop(png)
        png = self._pad_to_aspect_ratio(png)

        # Upload to GPU once
        self.overlay_gpu = cv2.cuda_GpuMat()
        self.overlay_gpu.upload(png)

        print(f"[FrameOverlayGPU] PNG uploaded to GPU: {png.shape[1]}x{png.shape[0]}")

    # ------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------

    def _auto_crop(self, img):
        alpha = img[:, :, 3]
        mask = (alpha > 50).astype(np.uint8) * 255
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]

    def _pad_to_aspect_ratio(self, img, target_ar=16/9):
        h, w = img.shape[:2]
        current_ar = w / h

        if abs(current_ar - target_ar) < 0.001:
            return img

        if current_ar > target_ar:
            new_h = int(w / target_ar)
            pad = new_h - h
            top = pad // 4
            bottom = pad - top
            return cv2.copyMakeBorder(img, top, bottom, 0, 0,
                                      cv2.BORDER_CONSTANT, value=[0,0,0,0])
        else:
            new_w = int(h * target_ar)
            pad = new_w - w
            left = pad // 2
            right = pad - left
            return cv2.copyMakeBorder(img, 0, 0, left, right,
                                      cv2.BORDER_CONSTANT, value=[0,0,0,0])

    # ------------------------------------------------------------
    # MAIN GPU OVERLAY
    # ------------------------------------------------------------

    def apply_to_video(self, clip_path: str, output_path: str):
        output_path = Path(output_path)
        temp_output = output_path.with_name(output_path.stem + "_temp.mp4")

        clip_path = str(clip_path)
        temp_output = str(temp_output)
        final_output = str(output_path)

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {clip_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[FrameOverlayGPU] Video: {width}x{height} @ {fps} FPS")

        # Prepare GPU mats
        frame_gpu = cv2.cuda_GpuMat()
        resized_overlay = cv2.cuda.resize(self.overlay_gpu, (width, height))

        # Split overlay into RGB + alpha
        overlay_cpu = resized_overlay.download()
        overlay_rgb = overlay_cpu[:, :, :3]
        overlay_alpha = overlay_cpu[:, :, 3] / 255.0

        # Upload alpha mask to GPU
        alpha_gpu = cv2.cuda_GpuMat()
        alpha_gpu.upload(overlay_alpha.astype(np.float32))

        # TEMPORARY OUTPUT (OpenCV)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Upload frame to GPU
            frame_gpu.upload(frame)

            # Alpha blend on GPU
            blended_gpu = cv2.cuda.alphaComp(
                frame_gpu,
                resized_overlay,
                cv2.cuda.ALPHA_OVER
            )

            # Download blended frame
            blended = blended_gpu.download()
            out.write(blended)

            frame_count += 1

        cap.release()
        out.release()

        print(f"[FrameOverlayGPU] GPU overlay complete. Frames: {frame_count}")

        # FINAL GPU ENCODE (NVENC)
        self._ffmpeg_nvenc(temp_output, final_output)

        os.remove(temp_output)
        print(f"[FrameOverlayGPU] Final MP4 saved → {final_output}")

    # ------------------------------------------------------------
    # GPU FFmpeg ENCODE
    # ------------------------------------------------------------

    def _ffmpeg_nvenc(self, temp_path, final_path):
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_path,
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "5M",
            "-pix_fmt", "yuv420p",
            final_path
        ]

        print(f"[FrameOverlayGPU] NVENC encoding → {final_path}")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
