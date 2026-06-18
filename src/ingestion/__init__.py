"""
Ingestion package for the pickleball video‑processing pipeline.

This package contains modules responsible for:
- Loading raw video files
- Validating integrity and metadata
- Normalizing frame rates (VFR → CFR)
- Streaming frames for large (>4GB) videos
- Providing async ingestion entrypoints for the full pipeline
"""

from .video_ingestion_async import (
    ingest_video,
    extract_metadata,
    validate_video,
)
