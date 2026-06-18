# ============================================================
# FILE: src/extractor/async_frame_extractor.py
# ============================================================

import cv2
import asyncio
import logging
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================
# SECTION: Worker function (runs in separate process)
# ============================================================

def _extract_chunk(args):
    """
    Worker: extract frames for a time slice using FFmpeg.
    Runs in a separate process → true parallelism.
    """
    video_path, out_dir, start_time, duration, fps, jpeg_quality = args

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(out_dir / f"frame_%06d.jpg")

    vf = f"fps={fps},format=yuvj420p"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(video_path),
        "-qscale:v", str(int((100 - jpeg_quality) / 5)),  # maps quality to qscale
        "-vf", vf,
        pattern,
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        return {"error": str(e), "start": start_time, "duration": duration}


# ============================================================
# SECTION: Async Frame Extractor (Story 3.x)
# ============================================================

async def extract_frames_async(
    video_path: Path,
    output_dir: Optional[Path] = None,
    jpeg_quality: int = 92,
) -> Dict[str, Any]:
    """
    High‑performance frame extractor using multiprocessing.

    Purpose:
        • True parallelism (FFmpeg per worker)
        • No RAM blow‑up (workers stream directly)
        • Same metadata contract as Story 3.x
        • Async wrapper for compatibility with ingestion pipeline

    Returns:
        {
            "frame_count": int,
            "fps": float,
            "width": int,
            "height": int,
            "paths": List[Path] | None
        }
    """

    # --------------------------------------------------------
    # STEP 1 — Validate video path
    # --------------------------------------------------------

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    # --------------------------------------------------------
    # STEP 2 — Extract metadata
    # --------------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    logger.info(f"[ExtractorMP] Video opened: {video_path}")
    logger.info(f"[ExtractorMP] FPS={fps}, Size={width}x{height}, Duration={duration:.2f}s")

    # --------------------------------------------------------
    # STEP 3 — Prepare output directory
    # --------------------------------------------------------

    saved_paths: List[Path] = []

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # STEP 4 — Build multiprocessing tasks
    # --------------------------------------------------------

    workers = mp.cpu_count()
    chunk_duration = duration / workers if workers > 0 else duration

    tasks = []
    for i in range(workers):
        start_time = i * chunk_duration
        tasks.append(
            (
                str(video_path),
                str(output_dir),
                start_time,
                chunk_duration,
                fps,
                jpeg_quality,
            )
        )

    # --------------------------------------------------------
    # STEP 5 — Run workers in parallel
    # --------------------------------------------------------

    logger.info(f"[ExtractorMP] Launching {workers} FFmpeg workers...")

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: mp.Pool(workers).map(_extract_chunk, tasks)
    )

    # Check for worker errors
    for r in results:
        if isinstance(r, dict) and "error" in r:
            logger.error(f"[ExtractorMP] Worker error: {r}")

    # --------------------------------------------------------
    # STEP 6 — Collect frames in sorted order
    # --------------------------------------------------------

    if output_dir:
        saved_paths = sorted(Path(output_dir).glob("*.jpg"))
        frame_count = len(saved_paths)
    else:
        frame_count = total_frames

    logger.info(f"[ExtractorMP] Extraction complete: {frame_count} frames")

    # --------------------------------------------------------
    # STEP 7 — Build return metadata
    # --------------------------------------------------------

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "paths": saved_paths if output_dir else None,
    }
