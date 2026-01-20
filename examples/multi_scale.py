#!/usr/bin/env python3
"""Multi-scale detection example.

This script demonstrates how to combine multiple perspective presets
for detecting objects of varying sizes in a panorama.

Usage:
    python examples/multi_scale.py path/to/panorama.jpg "window"
"""

import sys
import panosam as ps
from panosam.engines.sam3 import SAM3Engine


def main():
    if len(sys.argv) < 3:
        print("Usage: python multi_scale.py <image_path> <prompt>")
        print("Example: python multi_scale.py assets/test-pano.jpg window")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]

    print(f"Running multi-scale segmentation for '{prompt}'...")

    # Initialize the SAM3 engine (requires panosam[sam3] dependencies)
    engine = SAM3Engine()

    # Combine zoomed_out (large objects) and wideangle (very large objects)
    client = ps.PanoSAM(
        engine=engine,
        views=[ps.PerspectivePreset.ZOOMED_OUT, ps.PerspectivePreset.WIDEANGLE],
    )
    result = client.segment(image_path, prompt=prompt)
    result.save_json(image_path.replace(".jpg", ".multiscale.json"))

    print(f"Found {len(result.masks)} objects across multiple scales")
    for i, mask in enumerate(result.masks):
        print(
            f"  [{i}] score={mask.score:.2f}, center=({mask.center_yaw:.1f}, {mask.center_pitch:.1f})"
        )


if __name__ == "__main__":
    main()
