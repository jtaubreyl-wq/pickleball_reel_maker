# ============================================================
# FILE: src/ingestion/video_ingestion_async.py
# ============================================================

# SECTION: Imports & Setup
# ============================================================

import asyncio
import os
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import cv2
import av
import logging
import numpy as np
from pathlib import Path

from src.extractor.async_frame_extractor import extract_frames_async

logger = logging.getLogger(__name__)


# ============================================================
# SECTION: Exceptions
# ============================================================

class VideoValidationError(Exception):
    """Raised when a video fails validation or cannot be processed."""
    pass


# ============================================================
# SECTION: Metadata Model
# ============================================================

@dataclass
class VideoMetadata:
    """
    Container for extracted video metadata.
    """
    path: str
    fps: float
    width: int
    height: int
    codec: str
    duration_sec: float
    size_bytes: int
    is_variable_framerate: bool


# ============================================================
# SECTION: ASYNC METADATA EXTRACTION
# ============================================================

async def extract_metadata(path: str) -> VideoMetadata:
    """
    Extract metadata from a video file using PyAV.
    """

    def _sync_extract() -> VideoMetadata:
        container = av.open(path)
        video_stream = next(s for s in container.streams if s.type == "video")

        fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0
        width = video_stream.codec_context.width
        height = video_stream.codec_context.height
        codec = video_stream.codec_context.name
        duration_sec = float(container.duration / 1_000_000) if container.duration else 0.0
        size_bytes = os.path.getsize(path)

        # VFR detection
        timestamps = []
        for packet in container.demux(video_stream):
            if packet.pts is not None:
                timestamps.append(packet.pts * packet.time_base)
            if len(timestamps) > 50:
                break

        is_vfr = False
        if len(timestamps) > 2:
            diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
            is_vfr = (max(diffs) - min(diffs)) > 0.002

        container.close()

        return VideoMetadata(
            path=path,
            fps=fps,
            width=width,
            height=height,
            codec=codec,
            duration_sec=duration_sec,
            size_bytes=size_bytes,
            is_variable_framerate=is_vfr,
        )

    return await asyncio.to_thread(_sync_extract)


# ============================================================
# SECTION: ASYNC VALIDATION
# ============================================================

async def validate_video(path: str) -> VideoMetadata:
    """
    Validate video extension and extract metadata.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".mp4", ".mov", ".mkv", ".avi"):
        raise VideoValidationError(f"Unsupported format: {ext}")

    metadata = await extract_metadata(path)

    if metadata.fps <= 0:
        raise VideoValidationError("Invalid FPS detected")

    if metadata.width <= 0 or metadata.height <= 0:
        raise VideoValidationError("Invalid resolution detected")

    return metadata


# ============================================================
# SECTION: ASYNC CFR NORMALIZATION
# ============================================================

async def normalize_to_cfr(input_path: str, output_path: str, target_fps: float):
    """
    Convert a VFR (Variable Frame Rate) video to CFR (Constant Frame Rate).
    """

    def _sync_normalize():
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise VideoValidationError("Cannot open video for CFR normalization")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

        frame_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue

            writer.write(frame)
            frame_count += 1

        cap.release()
        writer.release()

        logger.info(f"[CFR] Normalized {frame_count} frames → {output_path}")

    await asyncio.to_thread(_sync_normalize)


# ============================================================
# SECTION: FFmpeg DIRECT PIPE MODE
# ============================================================

async def extract_frames_pipe(
    video_path: str,
    target_fps: float,
) -> Dict[str, Any]:
    """
    Fast extraction: FFmpeg → raw RGB24 → Python (no disk I/O).
    """

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={target_fps}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    meta = await extract_metadata(video_path)
    frame_size = meta.width * meta.height * 3

    frames: List[np.ndarray] = []
    frame_count = 0

    while True:
        raw = await proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((meta.height, meta.width, 3))
        frames.append(frame)
        frame_count += 1

    await proc.wait()

    logger.info(f"[PipeExtractor] Extracted {frame_count} frames via FFmpeg pipe")

    return {
        "frame_count": frame_count,
        "fps": target_fps,
        "width": meta.width,
        "height": meta.height,
        "frames": frames,
    }


# ============================================================
# SECTION: FFmpeg NVDEC GPU DECODE MODE
# ============================================================

async def extract_frames_nvdec(
    video_path: str,
    target_fps: float,
) -> Dict[str, Any]:
    """
    Ultra-fast GPU-accelerated frame extraction using NVDEC.

    Requirements:
        • NVIDIA GPU
        • FFmpeg compiled with CUDA/NVDEC support
    """

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", video_path,
        "-vf", f"fps={target_fps},hwdownload,format=rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    meta = await extract_metadata(video_path)
    frame_size = meta.width * meta.height * 3

    frames: List[np.ndarray] = []
    frame_count = 0

    while True:
        raw = await proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((meta.height, meta.width, 3))
        frames.append(frame)
        frame_count += 1

    await proc.wait()

    logger.info(f"[NVDEC] Extracted {frame_count} frames via NVDEC GPU decode")

    return {
        "frame_count": frame_count,
        "fps": target_fps,
        "width": meta.width,
        "height": meta.height,
        "frames": frames,
    }


# ============================================================
# SECTION: ASYNC INGESTION PIPELINE
# ============================================================

async def ingest_video(
    path: str,
    target_fps: Optional[float] = None,
    extract_frames: bool = False,
    output_dir: Optional[str] = None,
    mode: str = "mp",  # "mp" | "pipe" | "nvdec"
) -> Tuple[str, VideoMetadata, Optional[Dict[str, Any]]]:
    """
    Full ingestion pipeline:
        1. Validate video
        2. Extract metadata
        3. If VFR → normalize to CFR
        4. Optional frame extraction (mp / pipe / nvdec)
        5. Return final path + metadata + frame extraction result
    """

    metadata = await validate_video(path)
    final_path = path

    # Normalize VFR to CFR if requested
    if metadata.is_variable_framerate and target_fps:
        base, ext = os.path.splitext(path)
        output_path = f"{base}_cfr{ext}"

        logger.info(f"[Ingestion] VFR detected → normalizing to CFR at {target_fps} FPS")
        await normalize_to_cfr(path, output_path, target_fps)

        final_path = output_path
        metadata = await extract_metadata(final_path)

    frame_result: Optional[Dict[str, Any]] = None

    if extract_frames:
        eff_fps = target_fps or metadata.fps

        if mode == "pipe":
            logger.info("[Ingestion] Extracting frames via FFmpeg direct pipe...")
            frame_result = await extract_frames_pipe(
                video_path=final_path,
                target_fps=eff_fps,
            )

        elif mode == "nvdec":
            logger.info("[Ingestion] Extracting frames via NVDEC GPU decode...")
            frame_result = await extract_frames_nvdec(
                video_path=final_path,
                target_fps=eff_fps,
            )

        else:
            logger.info("[Ingestion] Extracting frames via multiprocessing FFmpeg...")
            frame_result = await extract_frames_async(
                video_path=Path(final_path),
                output_dir=Path(output_dir) if output_dir else None,
                jpeg_quality=92,
            )

    logger.info(f"[Ingestion] Final video ready: {final_path}")

    return final_path, metadata, frame_result
