#!/usr/bin/env python3
"""Basic PanoSAM usage example.

This script demonstrates the simplest way to run SAM3 segmentation
on a panoramic image using the pipeline-first `PanoSAM` API.

Usage:
    python examples/basic_usage.py path/to/panorama.jpg "car"
"""

import sys
import panosam as ps
from panosam.engines.sam3 import SAM3Engine


def main():
    if len(sys.argv) < 3:
        print("Usage: python basic_usage.py <image_path> <prompt>")
        print("Example: python basic_usage.py assets/test-pano.jpg car")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]

    print(f"Segmenting '{prompt}' in {image_path}...")

    # Initialize the SAM3 engine (requires panosam[sam3] dependencies)
    engine = SAM3Engine()

    # Create the client with the engine
    client = ps.PanoSAM(engine=engine, views=ps.PerspectivePreset.DEFAULT)
    result = client.segment(image_path, prompt=prompt)
    result.save_json(image_path.replace(".jpg", ".panosam.json"))

    print(f"Found {len(result.masks)} objects")
    for i, mask in enumerate(result.masks):
        print(
            f"  [{i}] score={mask.score:.2f}, center=({mask.center_yaw:.1f}, {mask.center_pitch:.1f})"
        )


if __name__ == "__main__":
    main()
