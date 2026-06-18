# FILE: src/court/court.py

# ============================================================
# SECTION: Imports & Data Model
# ============================================================

from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================
# SECTION: Court Model (Core Geometry Object)
# ============================================================

@dataclass
class Court:
    """
    Core court geometry model used across the entire pipeline.

    Fields:
        left, right, top, bottom:
            Pixel coordinates defining the full court bounding box.

    Why this matters:
        • RallyStartDetector uses a shrunk "active zone"
        • RallyEndDetector uses out-of-bounds logic
        • WinnerForcedErrorDetector uses court boundaries
        • Trajectory smoothing uses court limits
        • Highlight scoring uses court-relative metrics

    This upgraded version includes:
        • Normalized bounds
        • Width/height helpers
        • Court center
        • Active zone shrinker
        • Conversion to dict for detectors
    """

    left: float
    right: float
    top: float
    bottom: float

    # --------------------------------------------------------
    # SECTION: Basic Geometry Helpers
    # --------------------------------------------------------

    @property
    def width(self) -> float:
        """Court width in pixels."""
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        """Court height in pixels."""
        return max(0.0, self.bottom - self.top)

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the court."""
        return (
            self.left + self.width / 2,
            self.top + self.height / 2,
        )

    # --------------------------------------------------------
    # SECTION: Normalized Bounds
    # --------------------------------------------------------

    def as_bounds_dict(self) -> Dict[str, float]:
        """
        Convert to a normalized dict format used by:
            • WinnerForcedErrorDetector
            • RallyEndDetector
            • Metadata engines
        """
        return {
            "xmin": self.left,
            "xmax": self.right,
            "ymin": self.top,
            "ymax": self.bottom,
        }

    # --------------------------------------------------------
    # SECTION: Active Zone Shrinker
    # --------------------------------------------------------

    def shrink(self, margin: float = 0.12) -> "Court":
        """
        Shrink the court bounds by a percentage margin on all sides.

        Used by RallyStartDetector to define the "active play zone".

        margin:
            Fraction of width/height to remove from each side.
        """
        dx = self.width * margin
        dy = self.height * margin

        return Court(
            left=self.left + dx,
            right=self.right - dx,
            top=self.top + dy,
            bottom=self.bottom - dy,
        )

    # --------------------------------------------------------
    # SECTION: Containment Checks
    # --------------------------------------------------------

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is inside the court bounds."""
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def contains_strict(self, x: float, y: float) -> bool:
        """Strict containment (no touching edges)."""
        return self.left < x < self.right and self.top < y < self.bottom

    # --------------------------------------------------------
    # SECTION: Debug Representation
    # --------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Court(left={self.left}, right={self.right}, "
            f"top={self.top}, bottom={self.bottom}, "
            f"width={self.width:.1f}, height={self.height:.1f})"
        )
