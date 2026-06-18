# FILE: src/ingestion/ingestion_config.py

# ============================================================
# SECTION: Ingestion Configuration (EPIC 1)
# ============================================================
"""
Configuration values for the ingestion pipeline.

Why this matters:
    Incorrect ingestion settings → incorrect FPS → incorrect rally durations →
    → invalid rally segments → highlight selector rejects everything →
    → final output becomes a 1‑second fallback clip.

This config centralizes all ingestion‑related constants so the pipeline
remains stable and predictable.
"""

# ============================================================
# SECTION: Allowed Video Formats
# ============================================================

# Video file extensions supported by the ingestion pipeline.
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")


# ============================================================
# SECTION: FPS Normalization
# ============================================================

# Target FPS for frame normalization.
# If the source video FPS differs, ingestion will resample.
DEFAULT_TARGET_FPS = 60

# Minimum acceptable FPS before treating video as corrupted.
MIN_ACCEPTABLE_FPS = 10

# Maximum acceptable FPS to prevent absurd metadata.
MAX_ACCEPTABLE_FPS = 240


# ============================================================
# SECTION: Frame Extraction Settings
# ============================================================

# Number of frames to process per async chunk.
# Smaller chunks = lower memory usage, safer for long videos.
ASYNC_CHUNK_SIZE = 256

# JPEG quality for extracted frames (0–100).
FRAME_JPEG_QUALITY = 92


# ============================================================
# SECTION: Temporary Directories
# ============================================================

# Directory for storing normalized or intermediate ingestion outputs.
TEMP_DIR = "data/tmp_ingestion"

# Directory for storing extracted frames (if enabled).
FRAME_OUTPUT_DIR = "data/frames"


# ============================================================
# SECTION: Video Validation Settings
# ============================================================

# Minimum resolution allowed for ingestion.
MIN_WIDTH = 320
MIN_HEIGHT = 240

# Maximum resolution allowed (to prevent memory blow‑ups).
MAX_WIDTH = 3840
MAX_HEIGHT = 2160


# ============================================================
# SECTION: Logging & Debugging
# ============================================================

# Enable verbose ingestion logging.
ENABLE_INGESTION_LOGS = True

# Enable frame‑level debug logs.
ENABLE_FRAME_DEBUG = False


# ============================================================
# SECTION: Helper Functions (Optional)
# ============================================================

def is_valid_extension(path: str) -> bool:
    """
    Check if a file path has a valid video extension.
    """
    return any(path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


def is_valid_resolution(width: int, height: int) -> bool:
    """
    Validate that the video resolution is within acceptable bounds.
    """
    return (
        MIN_WIDTH <= width <= MAX_WIDTH
        and MIN_HEIGHT <= height <= MAX_HEIGHT
    )


def is_valid_fps(fps: float) -> bool:
    """
    Validate that the FPS is within acceptable bounds.
    """
    return MIN_ACCEPTABLE_FPS <= fps <= MAX_ACCEPTABLE_FPS
