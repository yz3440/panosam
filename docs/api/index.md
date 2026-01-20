# API Reference

PanoSAM provides a modular API for panorama segmentation. The main components are:

## Core Components

- **[SAM3 Engine](sam.md)**: The segmentation engine using HuggingFace Transformers and the SAM3 model.
- **[Deduplication](dedup.md)**: Sphere mask deduplication using GeoPandas polygon overlap detection.
- **[Image Models](image.md)**: Classes for representing panoramas and perspective projections.

## Quick Start

```python
import panosam as ps

# Load a panorama image
panorama = ps.PanoramaImage("my_pano", "path/to/panorama.jpg")

# Initialize the SAM3 engine
engine = ps.SAM3Engine()

# Generate perspectives and run segmentation
for perspective in ps.DEFAULT_IMAGE_PERSPECTIVES:
    persp_img = panorama.generate_perspective_image(perspective)
    results = engine.segment(persp_img.get_perspective_image(), "car")
    # Process results...
```

## Module Structure

```
panosam/
├── sam/           # SAM3 segmentation engine
│   ├── engine.py  # SAM3Engine class
│   ├── models.py  # Mask result models
│   └── utils.py   # Coordinate conversion utilities
├── dedup/         # Deduplication engine
│   └── detection.py  # SphereMaskDeduplicationEngine
└── image/         # Image handling
    ├── models.py  # PanoramaImage, PerspectiveImage
    └── constants.py  # Perspective presets
```
