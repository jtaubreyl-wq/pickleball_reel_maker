# src/rally/segmentation/rally_segment_builder.py

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RallyEvent:
    """Represents a start or end event detected by the rally detectors."""
    frame: int
    timestamp: float


@dataclass
class RallySegment:
    """Represents a fully constructed rally segment."""
    rally_id: int
    start_frame: int
    end_frame: int
    duration_s: float


class RallySegmentBuilder:
    """
    Story 5.3 — Rally Segment Builder
    ---------------------------------
    Consumes start/end events and produces validated rally segments.
    """

    MIN_DURATION_S = 1.0  # Filter out noise

    def __init__(self):
        self.start_events: List[RallyEvent] = []
        self.end_events: List[RallyEvent] = []

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def add_start_events(self, events: List[Dict]):
        """Add start events from RallyStartDetector."""
        for e in events:
            self.start_events.append(RallyEvent(e["frame"], e["timestamp"]))
        logger.info(f"Loaded {len(events)} rally start events.")

    def add_end_events(self, events: List[Dict]):
        """Add end events from RallyEndDetector."""
        for e in events:
            self.end_events.append(RallyEvent(e["frame"], e["timestamp"]))
        logger.info(f"Loaded {len(events)} rally end events.")

    def build_segments(self) -> List[RallySegment]:
        """Main entry point — pairs start/end events and returns segments."""
        if not self.start_events:
            logger.warning("No start events found — cannot build segments.")
            return []

        if not self.end_events:
            logger.warning("No end events found — cannot build segments.")
            return []

        # Sort chronologically
        self.start_events.sort(key=lambda e: e.frame)
        self.end_events.sort(key=lambda e: e.frame)

        segments = []
        used_end_indices = set()

        for i, start in enumerate(self.start_events):
            end = self._find_matching_end(start, used_end_indices)

            if end is None:
                logger.warning(f"No valid end found for start at frame {start.frame}. Skipping.")
                continue

            duration = end.timestamp - start.timestamp
            if duration < self.MIN_DURATION_S:
                logger.warning(
                    f"Discarding rally: duration too short ({duration:.2f}s) "
                    f"start={start.frame}, end={end.frame}"
                )
                continue

            segment = RallySegment(
                rally_id=len(segments) + 1,
                start_frame=start.frame,
                end_frame=end.frame,
                duration_s=duration,
            )
            segments.append(segment)

        logger.info(f"Built {len(segments)} rally segments.")
        return segments

    def save_segments(self, segments: List[RallySegment], output_path: str):
        """Save segments to JSON."""
        data = [
            {
                "rally_id": s.rally_id,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "duration_s": s.duration_s,
            }
            for s in segments
        ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(segments)} rally segments → {output_path}")

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _find_matching_end(
        self,
        start: RallyEvent,
        used_end_indices: set
    ) -> Optional[RallyEvent]:
        """
        Find the nearest end event that occurs AFTER the start event.
        Ensures each end event is used only once.
        """
        for idx, end in enumerate(self.end_events):
            if idx in used_end_indices:
                continue
            if end.frame <= start.frame:
                continue

            used_end_indices.add(idx)
            return end

        return None
