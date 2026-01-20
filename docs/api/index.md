# API Reference

## Client API

The `PanoSAM` class is the main entry point.

::: panosam.api.client.PanoSAM
    options:
      show_root_heading: true

::: panosam.api.models.SegmentationResult
    options:
      show_root_heading: true

::: panosam.api.models.PerspectivePreset
    options:
      show_root_heading: true

::: panosam.api.models.SegmentationOptions
    options:
      show_root_heading: true

::: panosam.api.models.DedupOptions
    options:
      show_root_heading: true

## Module Structure

```text
panosam/
├── api/              # Client API
│   ├── client.py     # PanoSAM
│   └── models.py     # SegmentationResult, options, SegmentationEngine protocol
├── engines/          # Segmentation engines (lazily imported)
│   └── sam3.py       # SAM3Engine (requires [sam])
├── sam/              # Mask models
│   ├── models.py     # FlatMaskResult, SphereMaskResult
│   └── utils.py      # Visualization (requires [viz])
├── dedup/            # Deduplication
│   └── detection.py  # SphereMaskDeduplicationEngine
├── image/            # Panorama handling
│   ├── models.py     # PanoramaImage, PerspectiveMetadata
│   └── perspectives.py  # Presets, generate_perspectives()
└── geometry.py       # Coordinate conversion utilities
```

## Submodules

- [Engines](sam.md) - `SegmentationEngine` protocol and `SAM3Engine`
- [Image](image.md) - Panorama and perspective classes
- [Deduplication](dedup.md) - Mask deduplication
- [Geometry](geometry.md) - Coordinate conversion
- [Visualization](visualization.md) - Mask visualization (requires `[viz]`)
