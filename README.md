# PanoSAM

PanoSAM is a Python library for running [SAM3](https://huggingface.co/facebook/sam3) (Segment Anything Model 3) segmentation on equirectangular panorama images. It automatically handles perspective projection, coordinate conversion from flat to spherical coordinates, and intelligent deduplication of overlapping masks.

## Demo

This is a demo using the built-in [preview tool](#interactive-preview-tool) with test results in `/assets` folder.

`https://github.com/user-attachments/assets/edef8666-a7dd-4bf9-b86a-2144f28e17e1`

The [test image](./assets/test-pano.jpg) is taken by the author himself and is copyright-free. Feel free to use it as you wish.

## Features

- Support for SAM3 (Segment Anything Model 3) via HuggingFace Transformers
- Text-prompted segmentation (e.g., "car", "person", "sign")
- Automatic perspective generation from equirectangular panoramas
- Spherical coordinate conversion for polygon masks
- IoU-based duplication detection and removal across perspectives
- MPS/CUDA GPU acceleration support
- Interactive preview tool for visualization

## Installation

### Quick Install (from GitHub)

Install directly from GitHub with pip:

```bash
# Full install with SAM3 segmentation support
pip install "panosam[sam] @ git+https://github.com/yz3440/panosam.git"

# Or lightweight install (only image/dedup utilities, no ML dependencies)
pip install "panosam @ git+https://github.com/yz3440/panosam.git"
```

### Install a Specific Version

```bash
pip install "panosam[sam] @ git+https://github.com/yz3440/panosam.git@v0.1.0"
```

### Development Install

1. Clone the repository:

   ```bash
   git clone https://github.com/yz3440/panosam.git
   cd panosam
   ```

2. Install with uv (recommended):

   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Sync dependencies
   uv sync
   ```

   Or with pip:

   ```bash
   pip install -e ".[sam]"
   ```

### HuggingFace Authentication (Required for SAM3)

SAM3 requires HuggingFace authentication:

```bash
# Accept the SAM3 license at https://huggingface.co/facebook/sam3
# Then login:
huggingface-cli login
```

## Usage

### Basic Usage

```python
import panosam as ps

# Create a reusable segmentation client (defaults shown)
client = ps.PanoSAM(views=ps.PerspectivePreset.DEFAULT)

# Segment all cars in a panorama
result = client.segment("panorama.jpg", prompt="car")

# Results are SphereMaskResult objects with spherical coordinates
for mask in result.masks:
    print(f"Found {mask.label} at yaw={mask.center_yaw:.1f}, pitch={mask.center_pitch:.1f}")
    print(f"  Score: {mask.score:.2f}")
    print(f"  Polygons: {len(mask.polygons)}")
```

### Save Results

```python
# Save results as JSON (compatible with the preview tool)
client = ps.PanoSAM(views=ps.PerspectivePreset.DEFAULT)
result = client.segment("panorama.jpg", "car")
result.save_json("results.panosam.json")
```

### Perspective Presets

Use different presets depending on the size of objects you're detecting:

```python
# For small objects (e.g., signs, small fixtures)
client = ps.PanoSAM(views=ps.PerspectivePreset.ZOOMED_IN)
result = client.segment("panorama.jpg", "sign")

# For large objects (e.g., buildings, vehicles)
client = ps.PanoSAM(views=ps.PerspectivePreset.WIDEANGLE)
result = client.segment("panorama.jpg", "car")
```

| Preset       | FOV   | Resolution | Perspectives | Best For           |
| ------------ | ----- | ---------- | ------------ | ------------------ |
| `default`    | 45°   | 2048x2048  | 16           | General use        |
| `zoomed_in`  | 22.5° | 1024x1024  | 32           | Small objects      |
| `zoomed_out` | 60°   | 2500x2500  | 12           | Large objects      |
| `wideangle`  | 90°   | 2500x2500  | 8            | Very large objects |

### Multi-Scale Detection

Combine multiple presets for detecting objects of varying sizes:

```python
client = ps.PanoSAM(views=[ps.PerspectivePreset.ZOOMED_OUT, ps.PerspectivePreset.WIDEANGLE])
result = client.segment("panorama.jpg", "window")
```

### Custom Perspectives

For advanced control, create custom perspective configurations:

```python
# Cover ceiling and floor (useful for lights, floor patterns)
perspectives = ps.generate_perspectives(
    fov=60,
    resolution=2048,
    overlap=0.5,
    pitch_angles=[-45, 0, 45],  # Look up, straight, and down
)

client = ps.PanoSAM(views=perspectives)
result = client.segment("panorama.jpg", "light")
```

### Reuse Engine Across Calls

For batch processing, reuse the SAM3 engine to avoid reloading the model:

```python
engine = ps.SAM3Engine()

for image_path in image_paths:
    client = ps.PanoSAM(engine=engine, views=ps.PerspectivePreset.DEFAULT)
    result = client.segment(image_path, "car")
    # Process results...
```

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "prompt": "car",
  "image_path": "panorama.jpg",
  "perspective_preset": "default",
  "masks": [
    {
      "polygons": [[[yaw1, pitch1], [yaw2, pitch2], ...]],
      "score": 0.95,
      "label": "car",
      "mask_id": "car_0",
      "center_yaw": 45.2,
      "center_pitch": -5.3
    }
  ]
}
```

- `polygons`: List of polygon coordinates in (yaw, pitch) degrees
- `score`: Confidence score (0-1)
- `center_yaw/pitch`: Centroid of the polygon

## Deduplication Algorithm

When the same object appears in multiple perspective views, PanoSAM uses an incremental merging algorithm to deduplicate masks:

### Overview

```text
Frame 1 → Add all masks to master list
Frame 2 → For each mask:
            If overlaps with master list → Union & merge
            Else → Add to master list
Frame 3 → Same process...
...
Final → Validate all polygons
```

### Overlap Detection

Two masks are considered duplicates if either:

- **IoU >= 0.3** (Intersection over Union)
- **Intersection Ratio >= 0.5** (intersection area / smaller mask area)

This catches both similar-sized overlapping masks and cases where a small mask is contained within a larger one.

## Interactive Preview Tool

PanoSAM includes a web-based interactive preview tool for visualizing segmentation results. To use it:

```bash
cd preview && python -m http.server
```

Then open `http://localhost:8000` in your browser.

Drag and drop the JSON result file and your panorama image to the interface. You can orbit around the panorama and hover over masks to highlight them.

## Examples

See the [examples/](examples/) folder for complete working scripts:

- `basic_usage.py` - Simplest usage with `PanoSAM`
- `multi_scale.py` - Combining presets for multi-scale detection
- `custom_perspectives.py` - Creating custom perspective configurations

## Related Projects

- [PanoOCR](https://github.com/yz3440/panoocr) - OCR for panoramic images (similar architecture)
- [SAM3](https://huggingface.co/facebook/sam3) - Meta's Segment Anything Model 3
- [py360convert](https://github.com/sunset1995/py360convert) - Equirectangular projection library

## License

MIT License - see [LICENSE](LICENSE) for details.
