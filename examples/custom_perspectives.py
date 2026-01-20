#!/usr/bin/env python3
"""Custom perspective configuration example.

This script demonstrates how to create custom perspective configurations
for specialized use cases (e.g., covering ceiling/floor, dense overlap).

Usage:
    python examples/custom_perspectives.py path/to/panorama.jpg "light"
"""

import sys
import panosam as ps
from panosam.engines.sam3 import SAM3Engine


def main():
    if len(sys.argv) < 3:
        print("Usage: python custom_perspectives.py <image_path> <prompt>")
        print("Example: python custom_perspectives.py assets/test-pano.jpg light")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]

    # Create custom perspectives that cover ceiling and floor
    # (useful for detecting lights, ceiling fixtures, floor patterns)
    perspectives = ps.generate_perspectives(
        fov=60,  # 60 degree field of view
        resolution=2048,  # 2048x2048 output images
        overlap=0.5,  # 50% overlap between views
        pitch_angles=[-45, 0, 45],  # Look up, straight, and down
    )

    print(f"Using {len(perspectives)} custom perspectives (including ceiling/floor)")
    print(f"Segmenting '{prompt}'...")

    # Initialize the SAM3 engine (requires panosam[sam] dependencies)
    engine = SAM3Engine()

    client = ps.PanoSAM(engine=engine, views=perspectives)
    result = client.segment(image_path, prompt=prompt)
    result.save_json(image_path.replace(".jpg", ".custom.json"))

    print(f"Found {len(result.masks)} objects")
    for i, mask in enumerate(result.masks):
        print(
            f"  [{i}] score={mask.score:.2f}, center=({mask.center_yaw:.1f}, {mask.center_pitch:.1f})"
        )


if __name__ == "__main__":
    main()
