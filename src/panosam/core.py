"""High-level Python API for PanoSAM.

This module provides simple entrypoints intended for typical library usage:

- `segment()` runs SAM3 over a panorama (via perspective projections) and returns
  deduplicated masks in spherical coordinates.
- `segment_multi()` runs multi-scale segmentation by combining multiple presets.
"""

from __future__ import annotations

import json
import os
from typing import List, Literal, Optional, Sequence, Union

from PIL import Image
import numpy as np

from .dedup.detection import SphereMaskDeduplicationEngine
from .image.models import PanoramaImage, PerspectiveMetadata
from .image.perspectives import (
    DEFAULT_IMAGE_PERSPECTIVES,
    WIDEANGLE_IMAGE_PERSPECTIVES,
    ZOOMED_IN_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES,
    combine_perspectives,
)
from .sam.engine import SAM3Engine
from .sam.models import SphereMaskResult


PresetName = Literal["default", "zoomed_in", "zoomed_out", "wideangle"]


_PRESET_MAP: dict[PresetName, List[PerspectiveMetadata]] = {
    "default": DEFAULT_IMAGE_PERSPECTIVES,
    "zoomed_in": ZOOMED_IN_IMAGE_PERSPECTIVES,
    "zoomed_out": ZOOMED_OUT_IMAGE_PERSPECTIVES,
    "wideangle": WIDEANGLE_IMAGE_PERSPECTIVES,
}


PanoramaInput = Union[str, Image.Image, np.ndarray, PanoramaImage]


def _ensure_panorama(panorama: PanoramaInput) -> tuple[PanoramaImage, Optional[str]]:
    """Normalize user input to a `PanoramaImage` and best-effort source path."""
    if isinstance(panorama, PanoramaImage):
        return panorama, None

    if isinstance(panorama, str):
        panorama_id = os.path.splitext(os.path.basename(panorama))[0] or "panorama"
        return PanoramaImage(panorama_id=panorama_id, image=panorama), panorama

    # in-memory input
    return PanoramaImage(panorama_id="panorama", image=panorama), None


def _resolve_perspectives(
    *,
    preset: PresetName,
    perspectives: Optional[Sequence[PerspectiveMetadata]],
) -> List[PerspectiveMetadata]:
    if perspectives is not None:
        return list(perspectives)
    return list(_PRESET_MAP[preset])


def save_results_json(
    path: str,
    *,
    prompt: str,
    image_path: Optional[str],
    preset: Optional[str],
    presets: Optional[Sequence[str]],
    masks: Sequence[SphereMaskResult],
) -> None:
    """Save segmentation results in a JSON file suitable for the preview tool."""
    payload = {
        "prompt": prompt,
        "image_path": image_path,
        "perspective_preset": preset,
        "perspective_presets": list(presets) if presets is not None else None,
        "masks": [m.to_dict() for m in masks],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def segment(
    panorama: PanoramaInput,
    prompt: str,
    *,
    preset: PresetName = "default",
    perspectives: Optional[Sequence[PerspectiveMetadata]] = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_iou: float = 0.3,
    use_union: bool = True,
    save_json: Optional[str] = None,
    engine: Optional[SAM3Engine] = None,
) -> List[SphereMaskResult]:
    """Run PanoSAM segmentation on a panorama image and return deduplicated masks.

    Args:
        panorama: Panorama image input. Can be a file path, PIL Image, numpy array,
            or an existing `PanoramaImage`.
        prompt: Text prompt describing objects to segment (e.g., "car", "person").
        preset: One of: "default", "zoomed_in", "zoomed_out", "wideangle".
            Ignored if `perspectives` is provided.
        perspectives: Custom list of `PerspectiveMetadata`. If provided, overrides
            `preset`.
        threshold: Confidence threshold for SAM3 detections.
        mask_threshold: Threshold for binary mask generation.
        min_iou: Minimum IoU for deduplication.
        use_union: If True, merge overlapping masks using polygon union.
        save_json: Optional path to write results JSON (preview-tool friendly).
        engine: Optional shared `SAM3Engine` instance to reuse across calls.

    Returns:
        List of deduplicated `SphereMaskResult` objects in spherical coordinates.
    """
    pano, image_path = _ensure_panorama(panorama)
    perspective_list = _resolve_perspectives(preset=preset, perspectives=perspectives)

    sam_engine = engine or SAM3Engine()
    dedup_engine = SphereMaskDeduplicationEngine(min_iou=min_iou)

    per_frame: List[List[SphereMaskResult]] = []

    for i, perspective in enumerate(perspective_list):
        perspective_image = pano.generate_perspective_image(perspective)
        pil_image = perspective_image.get_perspective_image()

        flat_masks = sam_engine.segment(
            image=pil_image,
            text_prompt=prompt,
            threshold=threshold,
            mask_threshold=mask_threshold,
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

    deduplicated = dedup_engine.deduplicate_frames(per_frame, use_union=use_union)

    if save_json is not None:
        save_results_json(
            save_json,
            prompt=prompt,
            image_path=image_path,
            preset=preset if perspectives is None else None,
            presets=None,
            masks=deduplicated,
        )

    return deduplicated


def segment_multi(
    panorama: PanoramaInput,
    prompt: str,
    *,
    presets: Sequence[PresetName] = ("default",),
    perspectives: Optional[Sequence[PerspectiveMetadata]] = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_iou: float = 0.3,
    use_union: bool = True,
    save_json: Optional[str] = None,
    engine: Optional[SAM3Engine] = None,
) -> List[SphereMaskResult]:
    """Run multi-scale segmentation by combining multiple perspective presets.

    If `perspectives` is provided, it is used directly and `presets` is ignored.
    """
    used_presets = perspectives is None
    if perspectives is None:
        perspective_sets: List[List[PerspectiveMetadata]] = [
            list(_PRESET_MAP[p]) for p in presets
        ]
        combined = combine_perspectives(*perspective_sets)
        perspectives = combined

    result = segment(
        panorama,
        prompt,
        preset="default",
        perspectives=perspectives,
        threshold=threshold,
        mask_threshold=mask_threshold,
        min_iou=min_iou,
        use_union=use_union,
        save_json=None,  # handled below to include presets metadata
        engine=engine,
    )

    if save_json is not None:
        _, image_path = _ensure_panorama(panorama)
        save_results_json(
            save_json,
            prompt=prompt,
            image_path=image_path,
            preset=None,
            presets=list(presets) if used_presets else None,
            masks=result,
        )

    return result

