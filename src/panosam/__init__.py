"""PanoSAM: SAM3 segmentation for panoramic images."""

__version__ = "0.1.0"

from .api import (
    DedupOptions,
    PanoSAM,
    PerspectivePreset,
    SegmentationEngine,
    SegmentationOptions,
    SegmentationResult,
)

from .image.models import PanoramaImage, PerspectiveImage, PerspectiveMetadata
from .image.perspectives import (
    # Perspective generation API
    generate_perspectives,
    combine_perspectives,
    # Pre-defined perspective sets
    DEFAULT_IMAGE_PERSPECTIVES,
    ZOOMED_IN_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES,
    WIDEANGLE_IMAGE_PERSPECTIVES,
)

from .sam.models import FlatMaskResult, SphereMaskResult
from .sam.utils import extract_mask_contours, visualize_masks, visualize_sphere_masks

from .dedup.detection import SphereMaskDeduplicationEngine

__all__ = [
    # Version
    "__version__",
    # Pipeline-first public API
    "PanoSAM",
    "PerspectivePreset",
    "SegmentationEngine",
    "SegmentationOptions",
    "DedupOptions",
    "SegmentationResult",
    # Image module
    "PanoramaImage",
    "PerspectiveImage",
    "PerspectiveMetadata",
    # Perspective generation API
    "generate_perspectives",
    "combine_perspectives",
    # Pre-defined perspective sets
    "DEFAULT_IMAGE_PERSPECTIVES",
    "ZOOMED_IN_IMAGE_PERSPECTIVES",
    "ZOOMED_OUT_IMAGE_PERSPECTIVES",
    "WIDEANGLE_IMAGE_PERSPECTIVES",
    # Mask models (needed for custom engines)
    "FlatMaskResult",
    "SphereMaskResult",
    # Deduplication module
    "SphereMaskDeduplicationEngine",
    # Visualization utilities (advanced)
    "extract_mask_contours",
    "visualize_masks",
    "visualize_sphere_masks",
]
