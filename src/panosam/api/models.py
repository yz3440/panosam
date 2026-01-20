"""Public, pipeline-first API models.

These types are designed for ergonomic library usage and stable serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Sequence, runtime_checkable

from PIL import Image

from ..sam.models import FlatMaskResult, SphereMaskResult


class PerspectivePreset(str, Enum):
    """Pre-defined perspective configurations for common object scales."""

    DEFAULT = "default"
    ZOOMED_IN = "zoomed_in"
    ZOOMED_OUT = "zoomed_out"
    WIDEANGLE = "wideangle"


@dataclass(frozen=True)
class SegmentationOptions:
    """Inference options passed to the underlying SAM engine."""

    threshold: float = 0.5
    mask_threshold: float = 0.5
    simplify_tolerance: float = 0.005


@dataclass(frozen=True)
class DedupOptions:
    """Deduplication options applied after multi-view segmentation."""

    min_iou: float = 0.3
    use_union: bool = True


@runtime_checkable
class SegmentationEngine(Protocol):
    """Protocol for segmentation engines (structural typing).

    Any class with a matching `segment()` method can be used.
    No inheritance required.
    """

    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        simplify_tolerance: float = 0.005,
    ) -> list[FlatMaskResult]:
        """Segment objects in an image using a text prompt."""


@dataclass(frozen=True)
class SegmentationResult:
    """Segmentation output plus metadata, with preview-tool-friendly JSON export."""

    prompt: str
    masks: Sequence[SphereMaskResult]
    image_path: Optional[str] = None
    perspective_preset: Optional[str] = None
    perspective_presets: Optional[Sequence[str]] = None

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "image_path": self.image_path,
            "perspective_preset": self.perspective_preset,
            "perspective_presets": list(self.perspective_presets)
            if self.perspective_presets is not None
            else None,
            "masks": [m.to_dict() for m in self.masks],
        }

    def save_json(self, path: str) -> None:
        """Save segmentation results in a JSON file suitable for the preview tool."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

