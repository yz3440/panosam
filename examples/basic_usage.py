#!/usr/bin/env python3
"""Basic PanoSAM usage example.

This script demonstrates the simplest way to run SAM3 segmentation
on a panoramic image using the high-level `panosam.segment()` API.

Usage:
    python examples/basic_usage.py path/to/panorama.jpg "car"
"""

import sys
import panosam as ps


def main():
    if len(sys.argv) < 3:
        print("Usage: python basic_usage.py <image_path> <prompt>")
        print("Example: python basic_usage.py assets/test-pano.jpg car")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]

    print(f"Segmenting '{prompt}' in {image_path}...")

    # Run segmentation - this is the simplest API
    results = ps.segment(
        image_path,
        prompt,
        save_json=image_path.replace(".jpg", ".panosam.json"),
    )

    print(f"Found {len(results)} objects")
    for i, mask in enumerate(results):
        print(f"  [{i}] score={mask.score:.2f}, center=({mask.center_yaw:.1f}, {mask.center_pitch:.1f})")


if __name__ == "__main__":
    main()
