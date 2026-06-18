# FILE: src/video/reel_assembler.py

"""
Unified highlight reel assembler with GPU frame overlay.

Pipeline:
1. Validate + auto-repair clips
2. FFmpeg concat → stitched.mp4
3. Apply Pickleverse frame using FrameOverlayGPU
4. Output final highlight reel

Features:
- Absolute path enforcement (Windows‑safe)
- Ensures .mp4 extension
- Validates clip existence + non-zero size
- Auto-repair corrupted clips (re-encode)
- Deterministic ordering
- POSIX paths for FFmpeg
- UTF‑8 (no BOM), LF-only concat list
- Full FFmpeg stderr capture
- GPU overlay support
"""

from pathlib import Path
import subprocess
import logging

from src.visualisation.frame_overlay_gpu import FrameOverlayGPU

FRAME_PNG = "assets/Pickleverse_frame_1.png"

logger = logging.getLogger(__name__)


class ReelAssemblyError(Exception):
    pass


# =====================================================================
#  Helper: Suggest fixes based on FFmpeg stderr
# =====================================================================

def _suggest_fix(stderr: str) -> str:
    s = stderr.lower()

    if "unsafe file name" in s:
        return "FFmpeg rejected a path. Ensure all clip paths are absolute."

    if "no such file" in s:
        return "One or more clip files do not exist. Verify clip extraction."

    if "invalid data" in s or "moov atom" in s:
        return "A clip may be corrupted. Re-extract or re-encode the clip."

    if "concat" in s and "codec" in s:
        return "Ensure all clips use the same codec (H.264 + AAC)."

    return "Run the FFmpeg command manually for more details."


# =====================================================================
#  Helper: Re-encode a corrupted clip
# =====================================================================

def _repair_clip(path: Path) -> Path:
    repaired = path.with_name(path.stem + "_repaired.mp4")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i", str(path),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(repaired),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        logger.error("Clip repair failed:\n%s", proc.stderr)
        raise ReelAssemblyError(f"Failed to repair corrupted clip: {path}")

    logger.info(f"Repaired corrupted clip → {repaired}")
    return repaired


# =====================================================================
#  Unified ReelAssembler class
# =====================================================================

class ReelAssembler:
    def __init__(self):
        self.overlay = FrameOverlayGPU(FRAME_PNG)

    # ---------------------------------------------------------
    # Validate + repair clips
    # ---------------------------------------------------------
    def _validate_clips(self, clips):
        validated = []

        for clip in clips:
            clip = Path(clip).resolve()

            # enforce .mp4
            if clip.suffix.lower() != ".mp4":
                clip = clip.with_suffix(".mp4")

            if not clip.exists():
                raise ReelAssemblyError(f"Clip not found: {clip}")

            if clip.stat().st_size == 0:
                logger.warning(f"Clip is empty, attempting repair: {clip}")
                clip = _repair_clip(clip)

            if not clip.exists() or clip.stat().st_size == 0:
                raise ReelAssemblyError(f"Clip invalid even after repair: {clip}")

            validated.append(clip)

        return sorted(validated)

    # ---------------------------------------------------------
    # FFmpeg concat stitching
    # ---------------------------------------------------------
    def _stitch_clips(self, clips, stitched_path):
        stitched_path = Path(stitched_path).resolve()
        stitched_path.parent.mkdir(parents=True, exist_ok=True)

        list_file = stitched_path.parent / "clips.txt"

        # Write concat list
        with list_file.open("w", encoding="utf-8", newline="\n") as f:
            for clip in clips:
                f.write(f"file '{clip.as_posix()}'\n")

        # Remove trailing newline
        text = list_file.read_text(encoding="utf-8").rstrip("\n")
        list_file.write_text(text, encoding="utf-8", newline="\n")

        logger.debug(f"FFmpeg concat list:\n{text}")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(stitched_path),
        ]

        logger.info(f"Stitching clips → {stitched_path}")

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            stderr = proc.stderr or ""
            logger.error("FFmpeg stitching failed:\n%s", stderr)
            raise ReelAssemblyError(
                f"Failed to stitch clips: {stitched_path}\n"
                f"Suggested fix: {_suggest_fix(stderr)}"
            )

        return stitched_path

    # ---------------------------------------------------------
    # Public API: assemble final highlight reel
    # ---------------------------------------------------------
    def assemble_reel(self, clips, output_path):
        if not clips:
            raise ReelAssemblyError("No clips provided to assemble_reel().")

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Validate + repair
        validated = self._validate_clips(clips)

        # 2. Stitch into temp file
        stitched_path = output_path.parent / "stitched.mp4"
        stitched_path = self._stitch_clips(validated, stitched_path)

        # 3. Apply Pickleverse frame overlay
        logger.info("Applying Pickleverse frame overlay...")
        self.overlay.apply_to_video(stitched_path, output_path)

        logger.info(f"Highlight reel created successfully → {output_path}")
        return output_path
