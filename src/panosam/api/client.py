"""Pipeline-first client API (Facade).

The `PanoSAM` class holds reusable configuration and dependencies (segmentation engine,
deduper, perspective views) to make batch usage ergonomic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..dedup.detection import SphereMaskDeduplicationEngine
from ..image.models import PanoramaImage, PerspectiveMetadata
from ..image.perspectives import (
    DEFAULT_IMAGE_PERSPECTIVES,
    WIDEANGLE_IMAGE_PERSPECTIVES,
    ZOOMED_IN_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES,
    combine_perspectives,
)
from ..sam.models import SphereMaskResult
from .models import (
    DedupOptions,
    PerspectivePreset,
    SegmentationEngine,
    SegmentationOptions,
    SegmentationResult,
)


PanoramaInput = Union[str, Image.Image, np.ndarray, PanoramaImage]
ViewsInput = Union[
    PerspectivePreset,
    Sequence[PerspectivePreset],
    Sequence[PerspectiveMetadata],
]


_PRESET_MAP: dict[PerspectivePreset, List[PerspectiveMetadata]] = {
    PerspectivePreset.DEFAULT: DEFAULT_IMAGE_PERSPECTIVES,
    PerspectivePreset.ZOOMED_IN: ZOOMED_IN_IMAGE_PERSPECTIVES,
    PerspectivePreset.ZOOMED_OUT: ZOOMED_OUT_IMAGE_PERSPECTIVES,
    PerspectivePreset.WIDEANGLE: WIDEANGLE_IMAGE_PERSPECTIVES,
}


def _ensure_panorama(panorama: PanoramaInput) -> tuple[PanoramaImage, Optional[str]]:
    if isinstance(panorama, PanoramaImage):
        return panorama, None

    if isinstance(panorama, str):
        panorama_id = os.path.splitext(os.path.basename(panorama))[0] or "panorama"
        return PanoramaImage(panorama_id=panorama_id, image=panorama), panorama

    return PanoramaImage(panorama_id="panorama", image=panorama), None


def _resolve_views(views: ViewsInput) -> tuple[List[PerspectiveMetadata], Optional[str], Optional[List[str]]]:
    """Resolve user views input into concrete PerspectiveMetadata.

    Returns:
        (perspectives, perspective_preset, perspective_presets)
    """
    # Single preset
    if isinstance(views, PerspectivePreset):
        return list(_PRESET_MAP[views]), views.value, None

    # Sequence input: determine element type by inspection
    views_list = list(views)
    if len(views_list) == 0:
        raise ValueError("views must not be empty")

    first = views_list[0]
    if isinstance(first, PerspectivePreset):
        presets = [v for v in views_list if isinstance(v, PerspectivePreset)]
        if len(presets) != len(views_list):
            raise TypeError("views must be all PerspectivePreset or all PerspectiveMetadata")
        perspective_sets = [list(_PRESET_MAP[p]) for p in presets]
        combined = combine_perspectives(*perspective_sets)
        return list(combined), None, [p.value for p in presets]

    if isinstance(first, PerspectiveMetadata):
        metas = [v for v in views_list if isinstance(v, PerspectiveMetadata)]
        if len(metas) != len(views_list):
            raise TypeError("views must be all PerspectivePreset or all PerspectiveMetadata")
        return metas, None, None

    raise TypeError("Unsupported views type")


@dataclass
class PanoSAM:
    """High-level segmentation client for panoramas.

    Args:
        engine: Segmentation engine (required). Use SAM3Engine or a custom engine
            conforming to the SegmentationEngine protocol.
        views: Perspective preset(s) or custom PerspectiveMetadata list.
        deduper: Optional custom deduplication engine.
        default_options: Default segmentation options (threshold, mask_threshold, etc.).
        default_dedup: Default deduplication options (min_iou, use_union).

    Example:
        >>> from panosam import PanoSAM, PerspectivePreset
        >>> from panosam.engines.sam3 import SAM3Engine
        >>> engine = SAM3Engine()
        >>> ps = PanoSAM(engine=engine, views=PerspectivePreset.DEFAULT)
        >>> result = ps.segment("panorama.jpg", "car")
    """

    engine: SegmentationEngine
    views: ViewsInput = PerspectivePreset.DEFAULT
    deduper: Optional[SphereMaskDeduplicationEngine] = None
    default_options: SegmentationOptions = field(default_factory=SegmentationOptions)
    default_dedup: DedupOptions = field(default_factory=DedupOptions)

    def __post_init__(self) -> None:
        self._perspectives, self._preset, self._presets = _resolve_views(self.views)
        self._deduper = self.deduper or SphereMaskDeduplicationEngine(
            min_iou=self.default_dedup.min_iou
        )

    def segment(
        self,
        panorama: PanoramaInput,
        prompt: str,
        *,
        options: Optional[SegmentationOptions] = None,
        dedup: Optional[DedupOptions] = None,
    ) -> SegmentationResult:
        opts = options or self.default_options
        dopt = dedup or self.default_dedup

        pano, image_path = _ensure_panorama(panorama)

        # Deduper may need different thresholds per call; create a temporary one if so.
        deduper = (
            self._deduper
            if getattr(self._deduper, "min_iou", None) == dopt.min_iou
            else SphereMaskDeduplicationEngine(min_iou=dopt.min_iou)
        )

        per_frame: List[List[SphereMaskResult]] = []

        for i, perspective in enumerate(
            tqdm(self._perspectives, desc="Segmenting perspectives", unit="view")
        ):
            perspective_image = pano.generate_perspective_image(perspective)
            pil_image = perspective_image.get_perspective_image()

            flat_masks = self.engine.segment(
                image=pil_image,
                text_prompt=prompt,
                threshold=opts.threshold,
                mask_threshold=opts.mask_threshold,
                simplify_tolerance=opts.simplify_tolerance,
            )

            sphere_masks: List[SphereMaskResult] = []
            for flat_mask in flat_masks:
                sphere_mask = flat_mask.to_sphere(
                    horizontal_fov=perspective.horizontal_fov,
                    vertical_fov=perspective.vertical_fov,
                    yaw_offset=perspective.yaw_offset,
                    pitch_offset=perspective.pitch_offset,
                )
                # Ensure uniqueness across frames
                if sphere_mask.mask_id:
                    sphere_mask.mask_id = f"p{i:02d}_{sphere_mask.mask_id}"
                else:
                    sphere_mask.mask_id = f"p{i:02d}_{prompt}_{len(sphere_masks)}"
                sphere_masks.append(sphere_mask)

            per_frame.append(sphere_masks)

        masks = deduper.deduplicate_frames(per_frame, use_union=dopt.use_union)

        return SegmentationResult(
            prompt=prompt,
            image_path=image_path,
            perspective_preset=self._preset,
            perspective_presets=self._presets,
            masks=masks,
        )

