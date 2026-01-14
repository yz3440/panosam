"""Command-line interface for PanoSAM."""

import argparse
import json
import os
import sys
from typing import List

from . import (
    PanoramaImage,
    PerspectiveMetadata,
    SAM3Engine,
    SphereMaskDeduplicationEngine,
    SphereMaskResult,
    FlatMaskResult,
    DEFAULT_IMAGE_PERSPECTIVES,
    ZOOMED_IN_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES,
    WIDEANGLE_IMAGE_PERSPECTIVES,
)


PRESET_MAP = {
    "default": DEFAULT_IMAGE_PERSPECTIVES,
    "zoomed_in": ZOOMED_IN_IMAGE_PERSPECTIVES,
    "zoomed_out": ZOOMED_OUT_IMAGE_PERSPECTIVES,
    "wideangle": WIDEANGLE_IMAGE_PERSPECTIVES,
}


# =============================================================================
# Cache utilities
# =============================================================================


def load_cached_masks(
    intermediates_dir: str,
    perspective_index: int,
    perspective: PerspectiveMetadata,
) -> List[SphereMaskResult] | None:
    """Load sphere masks from cached intermediate JSON if it exists.

    Args:
        intermediates_dir: Directory containing intermediate files.
        perspective_index: Index of the perspective (0-based).
        perspective: The perspective metadata.

    Returns:
        List of SphereMaskResult if cache exists, None otherwise.
    """
    if intermediates_dir is None:
        return None

    json_filename = f"visualization_{perspective_index:03d}_yaw{perspective.yaw_offset:.0f}_fov{perspective.horizontal_fov:.0f}.json"
    json_path = os.path.join(intermediates_dir, json_filename)

    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        masks = []
        for mask_dict in data.get("masks", []):
            # Handle both old format (polygon) and new format (polygons)
            if "polygons" in mask_dict:
                polygons = [[tuple(p) for p in poly] for poly in mask_dict["polygons"]]
            elif "polygon" in mask_dict:
                # Legacy format: single polygon
                polygons = (
                    [[tuple(p) for p in mask_dict["polygon"]]]
                    if mask_dict["polygon"]
                    else []
                )
            else:
                polygons = []

            # Add perspective prefix to mask_id for uniqueness
            original_mask_id = mask_dict["mask_id"]
            unique_mask_id = f"p{perspective_index:02d}_{original_mask_id}"
            mask = SphereMaskResult(
                polygons=polygons,
                score=mask_dict["score"],
                label=mask_dict["label"],
                mask_id=unique_mask_id,
                center_yaw=mask_dict["center_yaw"],
                center_pitch=mask_dict["center_pitch"],
            )
            masks.append(mask)
        return masks
    except Exception:
        return None


# =============================================================================


def get_perspectives(preset: str) -> List[PerspectiveMetadata]:
    """Get perspective configuration by preset name."""
    return PRESET_MAP.get(preset, DEFAULT_IMAGE_PERSPECTIVES)


def get_perspectives_multi(presets: List[str]) -> List[PerspectiveMetadata]:
    """Get combined perspectives from multiple presets."""
    all_perspectives = []
    for preset in presets:
        all_perspectives.extend(PRESET_MAP.get(preset, []))
    return all_perspectives


def run_panosam(
    image_path: str,
    text_prompt: str,
    output_path: str | None = None,
    perspective_preset: str = "default",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_iou: float = 0.3,
    verbose: bool = True,
    save_perspectives: bool = False,
    save_visualizations: bool = False,
    intermediates_dir: str | None = None,
    use_cache: bool = False,
) -> List[SphereMaskResult]:
    """Run PanoSAM segmentation on a panorama image.

    Args:
        image_path: Path to the equirectangular panorama image.
        text_prompt: Text describing objects to segment (e.g., "car", "person").
        output_path: Path to save JSON results. If None, auto-generates from image path.
        perspective_preset: Perspective configuration preset ("default", "zoomed_in", "zoomed_out").
        threshold: Confidence threshold for detections.
        mask_threshold: Threshold for binary mask generation.
        min_iou: Minimum IoU for deduplication.
        verbose: Whether to print progress messages.
        save_perspectives: Whether to save perspective images.
        save_visualizations: Whether to save perspective images with mask overlays.
        intermediates_dir: Directory to save intermediate files. Defaults to <image>_panosam/.
        use_cache: If True, load masks from cached intermediate JSONs instead of running SAM3.

    Returns:
        List of deduplicated SphereMaskResult objects.
    """
    # Initialize engines (SAM3 only needed if not using cache)
    sam_engine = None
    if not use_cache:
        if verbose:
            print("Initializing SAM3 engine...")
        sam_engine = SAM3Engine()
    dedup_engine = SphereMaskDeduplicationEngine(min_iou=min_iou)

    # Load panorama
    if verbose:
        print(f"Loading panorama: {image_path}")
    panorama_id = os.path.splitext(os.path.basename(image_path))[0]
    panorama = PanoramaImage(panorama_id, image_path)

    # Setup intermediates directory if needed
    if save_perspectives or save_visualizations or use_cache:
        if intermediates_dir is None:
            intermediates_dir = f"{os.path.splitext(image_path)[0]}_panosam"
        if save_perspectives or save_visualizations:
            os.makedirs(intermediates_dir, exist_ok=True)
        if verbose:
            if use_cache:
                print(f"Using cache from: {intermediates_dir}")
            else:
                print(f"Saving intermediates to: {intermediates_dir}")

    # Get perspectives
    perspectives = get_perspectives(perspective_preset)
    perspective_count = len(perspectives)

    if verbose:
        print(
            f"Processing {perspective_count} perspectives with prompt: '{text_prompt}'"
        )

    # Process each perspective
    all_sphere_masks_per_perspective: List[List[SphereMaskResult]] = []

    for i, perspective in enumerate(perspectives):
        if verbose:
            print(
                f"  [{i+1}/{perspective_count}] Processing perspective (yaw={perspective.yaw_offset}°)"
            )

        # Check for cached results first (for testing)
        cached_masks = (
            load_cached_masks(intermediates_dir, i, perspective) if use_cache else None
        )

        if cached_masks is not None:
            # Use cached masks - skip SAM3
            sphere_masks = cached_masks
            if verbose:
                print(f"    Loaded {len(sphere_masks)} masks from cache")
        else:
            # Initialize SAM3 lazily if needed
            if sam_engine is None:
                if verbose:
                    print("    Initializing SAM3 engine...")
                sam_engine = SAM3Engine()

            # Generate perspective view
            perspective_image = panorama.generate_perspective_image(perspective)
            pil_image = perspective_image.get_perspective_image()

            # Save perspective image if requested
            if save_perspectives:
                persp_filename = f"perspective_{i:03d}_yaw{perspective.yaw_offset:.0f}_fov{perspective.horizontal_fov:.0f}.jpg"
                pil_image.save(os.path.join(intermediates_dir, persp_filename))

            # Run SAM3 segmentation
            flat_masks, raw_masks = sam_engine.segment(
                image=pil_image,
                text_prompt=text_prompt,
                threshold=threshold,
                mask_threshold=mask_threshold,
                return_raw_masks=True,
            )

            if verbose and len(flat_masks) > 0:
                print(f"    Found {len(flat_masks)} masks")

            # Convert to sphere coordinates and add perspective prefix to mask_id
            sphere_masks = []
            for flat_mask in flat_masks:
                sphere_mask = flat_mask.to_sphere(
                    horizontal_fov=perspective.horizontal_fov,
                    vertical_fov=perspective.vertical_fov,
                    yaw_offset=perspective.yaw_offset,
                    pitch_offset=perspective.pitch_offset,
                )
                # Make mask_id unique by adding perspective prefix
                sphere_mask.mask_id = f"p{i:02d}_{sphere_mask.mask_id}"
                sphere_masks.append(sphere_mask)

            # Save visualization if requested
            if save_visualizations and len(raw_masks) > 0:
                from .sam.utils import visualize_masks

                vis_image = visualize_masks(pil_image, raw_masks)
                vis_filename = f"visualization_{i:03d}_yaw{perspective.yaw_offset:.0f}_fov{perspective.horizontal_fov:.0f}.jpg"
                vis_image.convert("RGB").save(
                    os.path.join(intermediates_dir, vis_filename)
                )

                # Save intermediate JSON with sphere coordinates for this perspective
                json_filename = f"visualization_{i:03d}_yaw{perspective.yaw_offset:.0f}_fov{perspective.horizontal_fov:.0f}.json"
                with open(os.path.join(intermediates_dir, json_filename), "w") as f:
                    json.dump(
                        {
                            "prompt": text_prompt,
                            "perspective": {
                                "yaw_offset": perspective.yaw_offset,
                                "pitch_offset": perspective.pitch_offset,
                                "horizontal_fov": perspective.horizontal_fov,
                                "vertical_fov": perspective.vertical_fov,
                            },
                            "masks": [mask.to_dict() for mask in sphere_masks],
                        },
                        f,
                        indent=2,
                    )

        all_sphere_masks_per_perspective.append(sphere_masks)

    # Count total masks before dedup
    total_before = sum(len(masks) for masks in all_sphere_masks_per_perspective)
    if verbose:
        print(f"Total masks before deduplication: {total_before}")
        print("Running incremental deduplication across all frames...")

    # Use incremental frame-based deduplication
    all_sphere_masks = dedup_engine.deduplicate_frames(all_sphere_masks_per_perspective)

    if verbose:
        print(f"Total masks after deduplication: {len(all_sphere_masks)}")

    # Save results
    if output_path is None:
        output_path = f"{os.path.splitext(image_path)[0]}.panosam.json"

    results_dicts = [mask.to_dict() for mask in all_sphere_masks]

    with open(output_path, "w") as f:
        json.dump(
            {
                "prompt": text_prompt,
                "image_path": image_path,
                "perspective_preset": perspective_preset,
                "masks": results_dicts,
            },
            f,
            indent=2,
        )

    if verbose:
        print(f"Results saved to: {output_path}")

    return all_sphere_masks


def run_panosam_multi(
    image_path: str,
    text_prompt: str,
    output_path: str | None = None,
    perspective_presets: List[str] = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_iou: float = 0.3,
    verbose: bool = True,
    save_perspectives: bool = False,
    save_visualizations: bool = False,
    intermediates_dir: str | None = None,
    use_cache: bool = False,
) -> List[SphereMaskResult]:
    """Run PanoSAM with multiple perspective presets.

    This mode combines multiple zoom levels to detect objects of different sizes,
    using incremental deduplication to merge overlapping masks across all frames.

    Args:
        image_path: Path to the equirectangular panorama image.
        text_prompt: Text describing objects to segment (e.g., "car", "window").
        output_path: Path to save JSON results. If None, auto-generates from image path.
        perspective_presets: List of preset names to combine (e.g., ["zoomed_out", "wideangle"]).
        threshold: Confidence threshold for detections.
        mask_threshold: Threshold for binary mask generation.
        min_iou: Minimum IoU for deduplication.
        verbose: Whether to print progress messages.
        save_perspectives: Whether to save perspective images.
        save_visualizations: Whether to save perspective images with mask overlays.
        intermediates_dir: Directory to save intermediate files. Defaults to <image>_panosam/.
        use_cache: If True, load masks from cached intermediate JSONs instead of running SAM3.

    Returns:
        List of deduplicated SphereMaskResult objects.
    """
    if perspective_presets is None:
        perspective_presets = ["default"]

    # Initialize engines (SAM3 only needed if not using cache)
    sam_engine = None
    if not use_cache:
        if verbose:
            print("Initializing SAM3 engine...")
        sam_engine = SAM3Engine()
    dedup_engine = SphereMaskDeduplicationEngine(min_iou=min_iou)

    # Load panorama
    if verbose:
        print(f"Loading panorama: {image_path}")
    panorama_id = os.path.splitext(os.path.basename(image_path))[0]
    panorama = PanoramaImage(panorama_id, image_path)

    # Setup intermediates directory if needed
    if save_perspectives or save_visualizations or use_cache:
        if intermediates_dir is None:
            intermediates_dir = f"{os.path.splitext(image_path)[0]}_panosam"
        if save_perspectives or save_visualizations:
            os.makedirs(intermediates_dir, exist_ok=True)
        if verbose:
            if use_cache:
                print(f"Using cache from: {intermediates_dir}")
            else:
                print(f"Saving intermediates to: {intermediates_dir}")

    # Get all perspectives from all presets
    all_perspectives = get_perspectives_multi(perspective_presets)
    total_perspectives = len(all_perspectives)

    if verbose:
        print(f"Using presets: {', '.join(perspective_presets)}")
        print(
            f"Processing {total_perspectives} total perspectives with prompt: '{text_prompt}'"
        )

    # Process each perspective and collect masks per frame
    all_sphere_masks_per_frame: List[List[SphereMaskResult]] = []

    for i, perspective in enumerate(all_perspectives):
        if verbose:
            print(
                f"  [{i+1}/{total_perspectives}] Processing perspective "
                f"(yaw={perspective.yaw_offset}°, fov={perspective.horizontal_fov}°)"
            )

        # Check for cached results first (for testing)
        cached_masks = (
            load_cached_masks(intermediates_dir, i, perspective) if use_cache else None
        )

        if cached_masks is not None:
            # Use cached masks - skip SAM3
            sphere_masks = cached_masks
            if verbose:
                print(f"    Loaded {len(sphere_masks)} masks from cache")
        else:
            # Initialize SAM3 lazily if needed
            if sam_engine is None:
                if verbose:
                    print("    Initializing SAM3 engine...")
                sam_engine = SAM3Engine()

            # Generate perspective view
            perspective_image = panorama.generate_perspective_image(perspective)
            pil_image = perspective_image.get_perspective_image()

            # Save perspective image if requested
            if save_perspectives:
                persp_filename = f"perspective_{i:03d}_yaw{perspective.yaw_offset:.0f}_fov{perspective.horizontal_fov:.0f}.jpg"
                pil_image.save(os.path.join(intermediates_dir, persp_filename))

            # Run SAM3 segmentation
            flat_masks, raw_masks = sam_engine.segment(
                image=pil_image,
                text_prompt=text_prompt,
                threshold=threshold,
                mask_threshold=mask_threshold,
                return_raw_masks=True,
            )

            if verbose and len(flat_masks) > 0:
                print(f"    Found {len(flat_masks)} masks")

            # Convert to sphere coordinates and add perspective prefix to mask_id
            sphere_masks = []
            for flat_mask in flat_masks:
                sphere_mask = flat_mask.to_sphere(
                    horizontal_fov=perspective.horizontal_fov,
                    vertical_fov=perspective.vertical_fov,
                    yaw_offset=perspective.yaw_offset,
                    pitch_offset=perspective.pitch_offset,
                )
                # Make mask_id unique by adding perspective prefix
                sphere_mask.mask_id = f"p{i:02d}_{sphere_mask.mask_id}"
                sphere_masks.append(sphere_mask)

            # Save visualization if requested
            if save_visualizations and len(raw_masks) > 0:
                from .sam.utils import visualize_masks

                vis_image = visualize_masks(pil_image, raw_masks)
                vis_filename = f"visualization_{i:03d}_yaw{perspective.yaw_offset:.0f}_fov{perspective.horizontal_fov:.0f}.jpg"
                vis_image.convert("RGB").save(
                    os.path.join(intermediates_dir, vis_filename)
                )

                # Save intermediate JSON with sphere coordinates for this perspective
                json_filename = f"visualization_{i:03d}_yaw{perspective.yaw_offset:.0f}_fov{perspective.horizontal_fov:.0f}.json"
                with open(os.path.join(intermediates_dir, json_filename), "w") as f:
                    json.dump(
                        {
                            "prompt": text_prompt,
                            "perspective": {
                                "yaw_offset": perspective.yaw_offset,
                                "pitch_offset": perspective.pitch_offset,
                                "horizontal_fov": perspective.horizontal_fov,
                                "vertical_fov": perspective.vertical_fov,
                            },
                            "masks": [mask.to_dict() for mask in sphere_masks],
                        },
                        f,
                        indent=2,
                    )

        all_sphere_masks_per_frame.append(sphere_masks)

    # Count total masks before dedup
    total_before = sum(len(masks) for masks in all_sphere_masks_per_frame)
    if verbose:
        print(f"Total masks before deduplication: {total_before}")
        print("Running incremental deduplication across all frames...")

    # Use incremental frame-based deduplication
    # This properly handles objects spanning 3+ frames
    deduplicated_masks = dedup_engine.deduplicate_frames(all_sphere_masks_per_frame)

    if verbose:
        print(f"Total masks after deduplication: {len(deduplicated_masks)}")

    # Save results
    if output_path is None:
        output_path = f"{os.path.splitext(image_path)[0]}.panosam.json"

    results_dicts = [mask.to_dict() for mask in deduplicated_masks]

    with open(output_path, "w") as f:
        json.dump(
            {
                "prompt": text_prompt,
                "image_path": image_path,
                "perspective_presets": perspective_presets,
                "total_perspectives": total_perspectives,
                "dedup_mode": "incremental",
                "masks": results_dicts,
            },
            f,
            indent=2,
        )

    if verbose:
        print(f"Results saved to: {output_path}")

    return deduplicated_masks


def flat_to_equirectangular(x: float, y: float) -> tuple[float, float]:
    """Convert normalized flat coordinates to equirectangular spherical coordinates.

    Args:
        x: Horizontal coordinate (0-1, left to right)
        y: Vertical coordinate (0-1, top to bottom)

    Returns:
        Tuple of (yaw, pitch) in degrees.
        - yaw: -180 to 180 (left to right)
        - pitch: 90 to -90 (top to bottom)
    """
    yaw = (x - 0.5) * 360  # 0 -> -180, 0.5 -> 0, 1 -> 180
    pitch = (0.5 - y) * 180  # 0 -> 90, 0.5 -> 0, 1 -> -90
    return yaw, pitch


def run_panosam_direct(
    image_path: str,
    text_prompt: str,
    output_path: str | None = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    verbose: bool = True,
) -> List[SphereMaskResult]:
    """Run SAM3 directly on the original equirectangular image without perspective projection.

    This mode is useful for benchmarking - it runs SAM3 directly on the
    equirectangular image without any perspective transformation or deduplication.
    Coordinates are still converted to spherical (yaw/pitch) for consistency.

    Args:
        image_path: Path to the equirectangular panorama image.
        text_prompt: Text describing objects to segment (e.g., "car", "person").
        output_path: Path to save JSON results. If None, auto-generates from image path.
        threshold: Confidence threshold for detections.
        mask_threshold: Threshold for binary mask generation.
        verbose: Whether to print progress messages.

    Returns:
        List of SphereMaskResult objects (in spherical coordinates).
    """
    from PIL import Image

    # Initialize SAM3 engine
    if verbose:
        print("Initializing SAM3 engine...")
    sam_engine = SAM3Engine()

    # Load image directly
    if verbose:
        print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    if verbose:
        print(f"Image size: {image.size[0]}x{image.size[1]}")
        print(f"Running SAM3 with prompt: '{text_prompt}'")

    # Run SAM3 directly on the image
    flat_masks = sam_engine.segment(
        image=image,
        text_prompt=text_prompt,
        threshold=threshold,
        mask_threshold=mask_threshold,
    )

    if verbose:
        print(f"Found {len(flat_masks)} masks")

    # Convert flat coordinates to spherical (equirectangular mapping)
    from .sam.models import _calculate_spherical_centroid

    sphere_masks = []
    for flat_mask in flat_masks:
        # Convert each polygon vertex from (x, y) to (yaw, pitch)
        sphere_polygons = []
        for polygon in flat_mask.polygons:
            sphere_polygon = [flat_to_equirectangular(x, y) for x, y in polygon]
            if sphere_polygon:
                sphere_polygons.append(sphere_polygon)

        # Calculate centroid in spherical coordinates
        if sphere_polygons:
            center_yaw, center_pitch = _calculate_spherical_centroid(sphere_polygons)
        else:
            center_yaw = 0.0
            center_pitch = 0.0

        sphere_mask = SphereMaskResult(
            polygons=sphere_polygons,
            score=flat_mask.score,
            label=flat_mask.label,
            mask_id=flat_mask.mask_id,
            center_yaw=center_yaw,
            center_pitch=center_pitch,
        )
        sphere_masks.append(sphere_mask)

    # Save results
    if output_path is None:
        output_path = f"{os.path.splitext(image_path)[0]}.panosam.direct.json"

    results_dicts = [mask.to_dict() for mask in sphere_masks]

    with open(output_path, "w") as f:
        json.dump(
            {
                "prompt": text_prompt,
                "image_path": image_path,
                "mode": "direct",
                "image_width": image.size[0],
                "image_height": image.size[1],
                "masks": results_dicts,
            },
            f,
            indent=2,
        )

    if verbose:
        print(f"Results saved to: {output_path}")

    return sphere_masks


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PanoSAM: Run SAM3 segmentation on panoramic images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  panosam --image panorama.jpg --prompt "car"
  panosam --image panorama.jpg --prompt "person" --preset zoomed_in
  panosam --image panorama.jpg --prompt "sign" --output results.json
  
  # Multiple presets (for multi-scale detection):
  panosam --image panorama.jpg --prompt "window" --preset zoomed_out wideangle
  
  # Save intermediate perspective images and visualizations:
  panosam --image panorama.jpg --prompt "car" --save-perspectives --save-visualizations
  
  # Use cached intermediate results (skip SAM3, for testing dedup):
  panosam --image panorama.jpg --prompt "window" --preset wideangle --use-cache
  
  # Direct mode (no perspective projection, for benchmarking):
  panosam --image panorama.jpg --prompt "car" --direct
        """,
    )

    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to the equirectangular panorama image",
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text prompt describing objects to segment (e.g., 'car', 'person')",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for JSON results (default: <image>.panosam.json)",
    )

    parser.add_argument(
        "--preset",
        type=str,
        nargs="+",
        choices=["default", "zoomed_in", "zoomed_out", "wideangle"],
        default=["default"],
        help="Perspective preset(s). Use multiple for multi-scale detection (default: default)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )

    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary mask generation (default: 0.5)",
    )

    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.3,
        help="Minimum IoU for deduplication (default: 0.3)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--direct",
        "-d",
        action="store_true",
        help="Direct mode: run SAM3 on original image without perspective projection (for benchmarking)",
    )

    parser.add_argument(
        "--save-perspectives",
        action="store_true",
        help="Save perspective images to intermediates directory",
    )

    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save perspective images with mask overlays to intermediates directory",
    )

    parser.add_argument(
        "--intermediates-dir",
        type=str,
        default=None,
        help="Directory for intermediate files (default: <image>_panosam/)",
    )

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load masks from cached intermediate JSONs instead of running SAM3 (for testing)",
    )

    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.direct:
            # Direct mode - no perspective projection or deduplication
            run_panosam_direct(
                image_path=args.image,
                text_prompt=args.prompt,
                output_path=args.output,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                verbose=not args.quiet,
            )
        elif len(args.preset) == 1:
            # Single preset
            run_panosam(
                image_path=args.image,
                text_prompt=args.prompt,
                output_path=args.output,
                perspective_preset=args.preset[0],
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                min_iou=args.min_iou,
                verbose=not args.quiet,
                save_perspectives=args.save_perspectives,
                save_visualizations=args.save_visualizations,
                intermediates_dir=args.intermediates_dir,
                use_cache=args.use_cache,
            )
        else:
            # Multiple presets
            run_panosam_multi(
                image_path=args.image,
                text_prompt=args.prompt,
                output_path=args.output,
                perspective_presets=args.preset,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                min_iou=args.min_iou,
                verbose=not args.quiet,
                save_perspectives=args.save_perspectives,
                save_visualizations=args.save_visualizations,
                intermediates_dir=args.intermediates_dir,
                use_cache=args.use_cache,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
