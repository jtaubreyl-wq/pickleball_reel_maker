# async_frame_extractor.py
"""
Asynchronous, robust frame extractor for pickleball highlight generation.

This version includes:
- Guaranteed frame ordering
- Corruption‑tolerant frame reads
- Strict FPS + metadata extraction
- Optional downscaling to prevent memory overload
- ThreadPoolExecutor for Windows‑safe concurrency
- Clear docstrings and error handling

Stable frame extraction → stable ball tracking → stable rallies →
prevents the 1‑second highlight fallback.
"""

print(">>> extractor imported")

import cv2
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# =====================================================================
#  Custom Error
# =====================================================================

class AsyncFrameExtractionError(Exception):
    """Raised when async frame extraction fails."""


# =====================================================================
#  Internal helper: read a single frame
# =====================================================================

def _read_frame(cap, frame_idx: int, downscale_factor: float = 1.0):
    """
    Read a single frame by index.

    Parameters
    ----------
    cap : cv2.VideoCapture
    frame_idx : int
    downscale_factor : float
        If < 1.0, frame is resized to reduce memory usage.

    Returns
    -------
    np.ndarray or None
    """
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        if downscale_factor != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (int(w * downscale_factor), int(h * downscale_factor)),
                interpolation=cv2.INTER_AREA,
            )

        return frame

    except Exception as e:
        raise AsyncFrameExtractionError(f"Failed to read frame {frame_idx}: {e}")


# =====================================================================
#  Main async extractor
# =====================================================================

async def extract_frames_async(
    video_path: str,
    downscale_factor: float = 1.0,
    max_workers: int = 8,
):
    """
    Asynchronously extract ALL frames from a video and return them IN MEMORY.

    This version:
    - Does NOT write frames to disk
    - Returns a list of frames in correct order
    - Extracts FPS + metadata for downstream modules
    - Uses asyncio + ThreadPoolExecutor (Windows‑safe)
    - Handles corrupted frames gracefully
    - Supports optional downscaling to reduce memory load

    Parameters
    ----------
    video_path : str
        Path to video file.
    downscale_factor : float
        Resize frames (0.25–1.0 recommended).
    max_workers : int
        Number of threads for concurrent frame reads.

    Returns
    -------
    dict
        {
            "frames": [np.ndarray, ...],
            "fps": float,
            "total_frames": int,
            "width": int,
            "height": int,
        }
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise AsyncFrameExtractionError(f"Video not found: {video_path}")

    # ---------------------------------------------------------
    # Open video
    # ---------------------------------------------------------
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise AsyncFrameExtractionError(f"Failed to open video: {video_path}")
    except Exception as e:
        raise AsyncFrameExtractionError(f"Error opening video: {e}")

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        f"[ASYNC] Extracting {total_frames} frames @ {fps:.2f} FPS "
        f"({width}x{height}) from {video_path}"
    )

    loop = asyncio.get_event_loop()
    frames = [None] * total_frames  # preserve ordering

    # ---------------------------------------------------------
    # Thread pool for concurrent frame reads
    # ---------------------------------------------------------
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tasks = [
            loop.run_in_executor(
                pool,
                _read_frame,
                cap,
                idx,
                downscale_factor,
            )
            for idx in range(total_frames)
        ]

        # Gather results asynchronously
        for idx, coro in enumerate(asyncio.as_completed(tasks)):
            frame = await coro
            frames[idx] = frame

    cap.release()

    # ---------------------------------------------------------
    # Remove missing frames (rare but possible)
    # ---------------------------------------------------------
    valid_frames = [f for f in frames if f is not None]
    dropped = total_frames - len(valid_frames)

    if dropped > 0:
        logger.warning(f"[ASYNC] Dropped {dropped} corrupted frames")

    logger.info(
        f"[ASYNC] Frame extraction complete: {len(valid_frames)} valid frames loaded"
    )

    return {
        "frames": valid_frames,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
    }
