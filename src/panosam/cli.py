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
    DEFAULT_IMAGE_PERSPECTIVES,
    ZOOMED_IN_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES,
)


def get_perspectives(preset: str) -> List[PerspectiveMetadata]:
    """Get perspective configuration by preset name."""
    presets = {
        "default": DEFAULT_IMAGE_PERSPECTIVES,
        "zoomed_in": ZOOMED_IN_IMAGE_PERSPECTIVES,
        "zoomed_out": ZOOMED_OUT_IMAGE_PERSPECTIVES,
    }
    return presets.get(preset, DEFAULT_IMAGE_PERSPECTIVES)


def run_panosam(
    image_path: str,
    text_prompt: str,
    output_path: str | None = None,
    perspective_preset: str = "default",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_iou: float = 0.3,
    verbose: bool = True,
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

    Returns:
        List of deduplicated SphereMaskResult objects.
    """
    # Initialize engines
    if verbose:
        print("Initializing SAM3 engine...")
    sam_engine = SAM3Engine()
    dedup_engine = SphereMaskDeduplicationEngine(min_iou=min_iou)

    # Load panorama
    if verbose:
        print(f"Loading panorama: {image_path}")
    panorama_id = os.path.splitext(os.path.basename(image_path))[0]
    panorama = PanoramaImage(panorama_id, image_path)

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
                f"  [{i+1}/{perspective_count}] Generating perspective (yaw={perspective.yaw_offset}°)"
            )

        # Generate perspective view
        perspective_image = panorama.generate_perspective_image(perspective)
        pil_image = perspective_image.get_perspective_image()

        # Run SAM3 segmentation
        flat_masks = sam_engine.segment(
            image=pil_image,
            text_prompt=text_prompt,
            threshold=threshold,
            mask_threshold=mask_threshold,
        )

        if verbose and len(flat_masks) > 0:
            print(f"    Found {len(flat_masks)} masks")

        # Convert to sphere coordinates
        sphere_masks = [
            flat_mask.to_sphere(
                horizontal_fov=perspective.horizontal_fov,
                vertical_fov=perspective.vertical_fov,
                yaw_offset=perspective.yaw_offset,
                pitch_offset=perspective.pitch_offset,
            )
            for flat_mask in flat_masks
        ]

        all_sphere_masks_per_perspective.append(sphere_masks)

    # Deduplication across adjacent perspectives
    if verbose:
        print("Removing duplicates across perspectives...")

    for i in range(perspective_count):
        first_idx = i
        second_idx = (i + 1) % perspective_count

        (
            all_sphere_masks_per_perspective[first_idx],
            all_sphere_masks_per_perspective[second_idx],
        ) = dedup_engine.remove_duplication_for_two_lists(
            all_sphere_masks_per_perspective[first_idx],
            all_sphere_masks_per_perspective[second_idx],
        )

    # Flatten results
    all_sphere_masks = [
        mask
        for perspective_masks in all_sphere_masks_per_perspective
        for mask in perspective_masks
    ]

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
        choices=["default", "zoomed_in", "zoomed_out"],
        default="default",
        help="Perspective configuration preset (default: default)",
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

    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        run_panosam(
            image_path=args.image,
            text_prompt=args.prompt,
            output_path=args.output,
            perspective_preset=args.preset,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            min_iou=args.min_iou,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
