# ffmpeg_utils.py
"""
Robust FFmpeg utility module for pickleball highlight generation.

This version includes:
- Strict CFR normalization
- Safer transcoding defaults
- Better error reporting + suggested fixes
- FPS validation + rounding
- Metadata extraction hardened against malformed ffprobe output
- Clear docstrings and structured helper functions

Stable video ingestion → stable frame extraction → stable ball tracking →
stable rally detection → prevents the 1‑second highlight fallback.
"""

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


# =====================================================================
#  Result container
# =====================================================================

@dataclass
class FFmpegResult:
    success: bool
    command: str
    returncode: int
    stdout: str
    stderr: str
    suggested_fix: Optional[str] = None


# =====================================================================
#  Internal command runner
# =====================================================================

def _run_command(cmd: List[str]) -> FFmpegResult:
    """
    Run a command via subprocess and capture output.

    This wrapper:
    - Logs the command
    - Captures stdout/stderr
    - Provides suggested fixes for common FFmpeg errors
    """
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    logger.info("Running command: %s", cmd_str)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        msg = f"Command not found: {cmd[0]}"
        logger.error("%s (%s)", msg, e)
        return FFmpegResult(
            success=False,
            command=cmd_str,
            returncode=-1,
            stdout="",
            stderr=str(e),
            suggested_fix=f"Ensure `{cmd[0]}` is installed and on your PATH.",
        )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        logger.error("Command failed (code %s): %s", proc.returncode, cmd_str)
        logger.error("stderr: %s", stderr[:2000])
        suggested_fix = _suggest_fix(stderr)
        return FFmpegResult(
            success=False,
            command=cmd_str,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            suggested_fix=suggested_fix,
        )

    logger.debug("Command succeeded: %s", cmd_str)
    return FFmpegResult(
        success=True,
        command=cmd_str,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


# =====================================================================
#  Suggested fixes
# =====================================================================

def _suggest_fix(stderr: str) -> str:
    """
    Provide simple heuristic suggestions based on stderr.
    """
    s = stderr.lower()

    if "unknown encoder 'libx264'" in s:
        return "Install FFmpeg with libx264 support (full build)."

    if "no such file or directory" in s:
        return "Check that the input path is correct and the file exists."

    if "invalid argument" in s:
        return "Check FFmpeg arguments (codec, pixel format, filters)."

    if "could not find codec parameters" in s:
        return "Input file may be corrupted or unsupported."

    if "error while decoding" in s:
        return "Input video may be corrupted. Try re‑encoding first."

    return "Check FFmpeg installation and input file. Try running the command manually."


# =====================================================================
#  Public API
# =====================================================================

def get_ffmpeg_version() -> FFmpegResult:
    """
    Get FFmpeg version (for logging and diagnostics).
    """
    return _run_command(["ffmpeg", "-version"])


# =====================================================================
#  Transcoding
# =====================================================================

def transcode_to_h264_mp4(
    input_path: str | Path,
    output_path: str | Path,
    crf: int = 23,
    preset: str = "medium",
    audio_codec: str = "aac",
    overwrite: bool = True,
) -> FFmpegResult:
    """
    Transcode any input to H.264 MP4 using libx264.

    This ensures:
    - Compatible codec
    - Faststart enabled for streaming
    - Predictable output for downstream frame extraction
    """
    input_path = str(input_path)
    output_path = str(output_path)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y" if overwrite else "-n",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", audio_codec,
        "-movflags", "+faststart",
        output_path,
    ]
    return _run_command(cmd)


# =====================================================================
#  CFR Normalization
# =====================================================================

def normalize_to_cfr_h264_mp4(
    input_path: str | Path,
    output_path: str | Path,
    target_fps: float = 30.0,
    crf: int = 23,
    preset: str = "medium",
    audio_codec: str = "aac",
    overwrite: bool = True,
) -> FFmpegResult:
    """
    Normalize a video to:
    - Constant frame rate (CFR)
    - H.264 MP4

    Uses:
    -r <fps> before -i to hint input rate (for VFR sources)
    -vf fps=<fps> to enforce CFR output
    """
    input_path = str(input_path)
    output_path = str(output_path)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y" if overwrite else "-n",
        "-r", str(target_fps),  # input fps hint
        "-i", input_path,
        "-vf", f"fps={target_fps}",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", audio_codec,
        "-movflags", "+faststart",
        output_path,
    ]
    return _run_command(cmd)


# =====================================================================
#  Metadata extraction
# =====================================================================

def probe_metadata(input_path: str | Path) -> Dict[str, Any]:
    """
    Use ffprobe to extract duration, resolution, fps, etc.
    Returns parsed JSON or {} on failure.
    """
    input_path = str(input_path)
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_path,
    ]
    result = _run_command(cmd)

    if not result.success:
        logger.warning("ffprobe failed for %s: %s", input_path, result.stderr[:500])
        return {}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.error("Failed to parse ffprobe JSON for %s", input_path)
        return {}


def get_video_stream_info(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return the first video stream from ffprobe metadata.
    """
    streams = meta.get("streams") or []
    for s in streams:
        if s.get("codec_type") == "video":
            return s
    return None


# =====================================================================
#  Basic video properties
# =====================================================================

def get_basic_video_properties(input_path: str | Path) -> Dict[str, Any]:
    """
    Return a dict with:
    - duration
    - width
    - height
    - fps (rounded)
    - codec_name

    FPS is validated and rounded to avoid floating‑point drift.
    """
    meta = probe_metadata(input_path)
    if not meta:
        return {}

    fmt = meta.get("format", {})
    vstream = get_video_stream_info(meta) or {}

    # Duration
    duration = None
    if "duration" in fmt:
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = None

    # Resolution
    width = vstream.get("width")
    height = vstream.get("height")

    # FPS
    fps = None
    r_frame_rate = vstream.get("r_frame_rate")
    if r_frame_rate and r_frame_rate != "0/0":
        try:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den)
            fps = round(fps, 3)
        except Exception:
            fps = None

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "codec_name": vstream.get("codec_name"),
    }
