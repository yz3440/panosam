"""PanoSAM: SAM3 segmentation for panoramic images."""

__version__ = "0.1.0"

from .image.models import PanoramaImage, PerspectiveImage, PerspectiveMetadata
from .image.constants import (
    DEFAULT_IMAGE_PERSPECTIVES,
    ZOOMED_IN_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES,
)

from .sam.models import FlatMaskResult, SphereMaskResult
from .sam.engine import SAM3Engine
from .sam.utils import extract_mask_contours, visualize_masks, visualize_sphere_masks

from .dedup.detection import SphereMaskDeduplicationEngine

__all__ = [
    # Version
    "__version__",
    # Image module
    "PanoramaImage",
    "PerspectiveImage",
    "PerspectiveMetadata",
    "DEFAULT_IMAGE_PERSPECTIVES",
    "ZOOMED_IN_IMAGE_PERSPECTIVES",
    "ZOOMED_OUT_IMAGE_PERSPECTIVES",
    # SAM module
    "FlatMaskResult",
    "SphereMaskResult",
    "SAM3Engine",
    "extract_mask_contours",
    "visualize_masks",
    "visualize_sphere_masks",
    # Deduplication module
    "SphereMaskDeduplicationEngine",
]
