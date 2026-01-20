# API Reference

PanoSAM provides a simple high-level API for panorama segmentation.

## High-Level API

The main entry point is the `PanoSAM` client:

```python
import panosam as ps

# Simple usage
client = ps.PanoSAM(views=ps.PerspectivePreset.DEFAULT)
result = client.segment("panorama.jpg", "car")

# Save preview-tool JSON
result.save_json("results.panosam.json")
```

::: panosam.api.client.PanoSAM
    options:
      show_root_heading: true

::: panosam.api.models.SegmentationResult
    options:
      show_root_heading: true

::: panosam.api.models.PerspectivePreset
    options:
      show_root_heading: true

## Low-Level Components

For advanced use cases, PanoSAM exposes the underlying components:

- **[SAM3 Engine](sam.md)**: The segmentation engine using HuggingFace Transformers.
- **[Deduplication](dedup.md)**: Sphere mask deduplication using GeoPandas.
- **[Image Models](image.md)**: Classes for panoramas and perspective projections.

## Module Structure

```text
panosam/
├── api/           # Pipeline-first public API
│   ├── client.py  # PanoSAM client
│   └── models.py  # Result + option types
├── core.py        # Legacy functional API (advanced/internal)
├── sam/           # SAM3 segmentation engine
│   ├── engine.py  # SAM3Engine class
│   └── models.py  # FlatMaskResult, SphereMaskResult
├── dedup/         # Deduplication engine
│   └── detection.py
└── image/         # Image handling
    ├── models.py     # PanoramaImage, PerspectiveMetadata
    └── perspectives.py  # Presets, generate_perspectives()
```
