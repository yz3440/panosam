# API Reference

PanoSAM provides a simple high-level API for panorama segmentation.

## High-Level API

The main entry points are `segment()` and `segment_multi()`:

```python
import panosam as ps

# Simple usage
results = ps.segment("panorama.jpg", "car")

# Multi-scale detection
results = ps.segment_multi("panorama.jpg", "window", presets=["zoomed_out", "wideangle"])
```

::: panosam.core.segment
    options:
      show_root_heading: true

::: panosam.core.segment_multi
    options:
      show_root_heading: true

## Low-Level Components

For advanced use cases, PanoSAM exposes the underlying components:

- **[SAM3 Engine](sam.md)**: The segmentation engine using HuggingFace Transformers.
- **[Deduplication](dedup.md)**: Sphere mask deduplication using GeoPandas.
- **[Image Models](image.md)**: Classes for panoramas and perspective projections.

## Module Structure

```
panosam/
├── core.py        # High-level segment() API
├── sam/           # SAM3 segmentation engine
│   ├── engine.py  # SAM3Engine class
│   └── models.py  # FlatMaskResult, SphereMaskResult
├── dedup/         # Deduplication engine
│   └── detection.py
└── image/         # Image handling
    ├── models.py     # PanoramaImage, PerspectiveMetadata
    └── constants.py  # Presets, generate_perspectives()
```
