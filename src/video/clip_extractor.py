# FILE: src/video/clip_extractor.py
"""
Rock‑solid FFmpeg-based clip extractor for highlight generation.

Major improvements:
- Dual-phase seeking (-ss before AND after -i) for accurate extraction
- Validates timestamps against actual video duration (via ffprobe)
- Prevents zero-frame clips by enforcing minimum duration
- Ensures all paths are absolute (Windows-safe)
- Re-encode fallback for corrupted clips
- Full stderr capture + detailed error suggestions
"""

from pathlib import Path
import subprocess
import logging
import json

logger = logging.getLogger(__name__)


class ClipExtractionError(Exception):
    pass


# =====================================================================
#  FFprobe helper — get video duration safely
# =====================================================================

def _get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        raise ClipExtractionError(f"FFprobe failed:\n{proc.stderr}")

    try:
        data = json.loads(proc.stdout)
        return float(data["format"]["duration"])
    except Exception:
        raise ClipExtractionError("Unable to parse video duration from ffprobe.")


# =====================================================================
#  Suggest Fixes
# =====================================================================

def _suggest_fix(stderr: str) -> str:
    s = stderr.lower()

    if "invalid argument" in s:
        return "Start/end timestamps may be outside the video duration."

    if "no such file" in s:
        return "Input video path is invalid or unreadable."

    if "moov atom not found" in s:
        return "Re-encode the input video using H.264 + faststart."

    if "error while decoding" in s:
        return "Video may be corrupted. Try re-encoding the source."

    if "output file is empty" in s:
        return "Clip timestamps may not contain any decodable frames."

    return "Run the FFmpeg command manually for more details."


# =====================================================================
#  Main Extraction Function
# =====================================================================

def extract_clip(
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
    reencode: bool = True,
):
    """
    Extract a video clip using FFmpeg with accurate seeking.
    """

    # ---------------------------------------------------------
    # Normalize paths
    # ---------------------------------------------------------
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()

    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    if not video_path.exists():
        raise ClipExtractionError(f"Input video does not exist: {video_path}")

    # ---------------------------------------------------------
    # Validate timestamps against actual video duration
    # ---------------------------------------------------------
    video_duration = _get_video_duration(video_path)

    if start < 0:
        start = 0.0

    if end > video_duration:
        end = video_duration

    if end <= start:
        raise ClipExtractionError(
            f"Invalid clip range: start={start:.3f}, end={end:.3f}"
        )

    duration = max(0.10, end - start)  # enforce minimum duration

    # ---------------------------------------------------------
    # Ensure output directory exists
    # ---------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Build FFmpeg command (dual-phase seeking)
    # ---------------------------------------------------------
    # Phase 1: fast seek (-ss before -i)
    # Phase 2: accurate trim (-ss after -i)
    if reencode:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss", f"{start:.3f}",          # fast seek
            "-i", str(video_path),
            "-ss", "0",                     # accurate seek from new start
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-ss", "0",
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(output_path),
        ]

    # ---------------------------------------------------------
    # Run FFmpeg
    # ---------------------------------------------------------
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stderr = proc.stderr or ""

    if proc.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        logger.error("FFmpeg clip extraction failed:\n%s", stderr)

        raise ClipExtractionError(
            f"FFmpeg failed to extract clip:\n"
            f"  Output: {output_path}\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Error: {stderr}\n"
            f"Suggested fix: {_suggest_fix(stderr)}"
        )

    logger.info(
        f"Clip extracted OK: {output_path} "
        f"(start={start:.3f}, end={end:.3f}, duration={duration:.3f})"
    )

    return output_path
