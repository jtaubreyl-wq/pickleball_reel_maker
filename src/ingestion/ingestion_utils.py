# FILE: src/ingestion/ingestion_utils.py

# ============================================================
# SECTION: Imports & Setup
# ============================================================

"""
Utility helpers for the ingestion pipeline.

Why this matters:
    If ingestion paths or directories are incorrect, the ingestion pipeline
    may fail silently. That leads to:
        • Missing normalized video
        • Missing extracted frames
        • Incorrect frame counts
        • Rally detectors failing
        • Highlight selector rejecting all rallies
        → Final output becomes a 1‑second fallback clip.

This module centralizes safe path handling and directory creation.
"""

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================
# SECTION: Directory Helpers
# ============================================================

def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists.

    Parameters:
        path: str or Path — directory to create

    Returns:
        Path object for the directory.

    Behavior:
        • Creates directory and parents if needed
        • Logs creation
        • Returns a usable Path object
    """
    p = Path(path)
    if not p.exists():
        logger.info(f"[IngestionUtils] Creating directory: {p}")
        p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
# SECTION: Temporary Output Path Helpers
# ============================================================

def get_temp_output_path(input_path: str | Path, suffix: str) -> str:
    """
    Generate a temporary output path based on the input file.

    Example:
        input:  match.mp4
        suffix: _cfr
        output: match_cfr.mp4

    Why this matters:
        CFR normalization, transcoding, and ingestion stages rely on
        predictable, collision‑free filenames.

    Parameters:
        input_path: original video path
        suffix: string to append before extension

    Returns:
        str — new file path with suffix applied
    """
    input_path = str(input_path)
    base, ext = os.path.splitext(input_path)
    return f"{base}{suffix}{ext}"


# ============================================================
# SECTION: Safe Path Builders
# ============================================================

def build_normalized_video_path(input_path: str | Path, temp_dir: str | Path, suffix: str = "_cfr") -> Path:
    """
    Build a safe path for the normalized (CFR) video output.

    Ensures:
        • temp_dir exists
        • filename is derived from input
        • suffix is applied correctly

    Returns:
        Path to normalized video.
    """
    temp_dir = ensure_directory(temp_dir)
    out_name = get_temp_output_path(Path(input_path).name, suffix)
    out_path = temp_dir / out_name
    logger.info(f"[IngestionUtils] Normalized video path: {out_path}")
    return out_path


def build_frame_output_dir(base_dir: str | Path, video_name: str) -> Path:
    """
    Build a directory for extracted frames for a specific video.

    Example:
        base_dir = "data/frames"
        video_name = "match.mp4"
        → data/frames/match/

    Ensures:
        • Directory exists
        • Name is sanitized
    """
    safe_name = Path(video_name).stem
    out_dir = ensure_directory(Path(base_dir) / safe_name)
    logger.info(f"[IngestionUtils] Frame output directory: {out_dir}")
    return out_dir


# ============================================================
# SECTION: Validation Helpers
# ============================================================

def validate_video_extension(path: str | Path, allowed_exts: tuple[str, ...]) -> bool:
    """
    Validate that a video file has an allowed extension.

    Returns:
        True if valid, False otherwise.
    """
    ext = Path(path).suffix.lower()
    valid = ext in allowed_exts
    if not valid:
        logger.warning(f"[IngestionUtils] Invalid video extension: {ext}")
    return valid


def validate_path_exists(path: str | Path) -> bool:
    """
    Check if a file exists and log if missing.
    """
    p = Path(path)
    if not p.exists():
        logger.error(f"[IngestionUtils] File not found: {p}")
        return False
    return True
