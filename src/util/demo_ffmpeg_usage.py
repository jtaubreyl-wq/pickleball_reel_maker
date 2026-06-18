# demo_ffmpeg_usage.py
"""
Demonstration script for FFmpeg utilities used in the pickleball
highlight generation pipeline.

This version includes:
- Clear logging and error reporting
- Validation of CFR normalization
- FPS and metadata inspection
- Early exit on failure (prevents downstream 1‑second highlight bug)
- Descriptions and structured output

This script is safe to run standalone and helps verify that your
FFmpeg installation and transcoding pipeline are functioning correctly.
"""

import logging
from ffmpeg_utils import (
    get_ffmpeg_version,
    transcode_to_h264_mp4,
    normalize_to_cfr_h264_mp4,
    get_basic_video_properties,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ffmpeg_demo")


# =====================================================================
#  Helper: Pretty print metadata
# =====================================================================

def print_video_properties(label: str, props: dict):
    """
    Print video metadata in a readable format.
    """
    logger.info(f"\n--- {label} ---")
    for k, v in props.items():
        logger.info(f"{k}: {v}")


# =====================================================================
#  Main Demo
# =====================================================================

if __name__ == "__main__":
    logger.info("=== FFmpeg Demo: Validating Video Pipeline ===")

    # ---------------------------------------------------------
    # 1) Check FFmpeg version
    # ---------------------------------------------------------
    version = get_ffmpeg_version()
    logger.info(f"FFmpeg version success: {version.success}")

    if version.success:
        logger.info(f"FFmpeg version: {version.stdout.splitlines()[0]}")
    else:
        logger.error("FFmpeg version check failed.")
        logger.error(version.stderr)
        raise SystemExit(1)

    # ---------------------------------------------------------
    # 2) Transcode to H.264 MP4
    # ---------------------------------------------------------
    input_file = "input_any.ext"
    output_h264 = "output_h264.mp4"

    logger.info(f"\nTranscoding → H.264: {input_file} → {output_h264}")
    result = transcode_to_h264_mp4(input_file, output_h264)

    if not result.success:
        logger.error("Transcode failed.")
        logger.error(result.stderr)
        logger.error(f"Suggested fix: {result.suggested_fix}")
        raise SystemExit(1)

    logger.info("Transcode successful.")

    # ---------------------------------------------------------
    # 3) Normalize to CFR + H.264
    # ---------------------------------------------------------
    output_cfr = "output_cfr_h264.mp4"
    logger.info(f"\nNormalizing → CFR + H.264: {input_file} → {output_cfr}")

    norm = normalize_to_cfr_h264_mp4(input_file, output_cfr, target_fps=30)

    if not norm.success:
        logger.error("CFR normalization failed.")
        logger.error(norm.stderr)
        logger.error(f"Suggested fix: {norm.suggested_fix}")
        raise SystemExit(1)

    logger.info("CFR normalization successful.")

    # ---------------------------------------------------------
    # 4) Probe metadata
    # ---------------------------------------------------------
    props = get_basic_video_properties(output_cfr)

    if not props:
        logger.error("Failed to read video metadata.")
        raise SystemExit(1)

    print_video_properties("Normalized Video Properties", props)

    # ---------------------------------------------------------
    # 5) Validate CFR + FPS
    # ---------------------------------------------------------
    fps = props.get("fps")
    if fps is None or abs(fps - 30) > 0.5:
        logger.warning(
            f"WARNING: Expected CFR 30 FPS, but detected {fps}. "
            "This may cause timing issues → broken rallies → 1‑second highlight."
        )
    else:
        logger.info("FPS validated: CFR 30 FPS confirmed.")

    logger.info("\n=== FFmpeg Demo Complete ===")
