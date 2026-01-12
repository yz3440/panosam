# PanoSAM

PanoSAM is a Python library for running [SAM3](https://huggingface.co/facebook/sam3) (Segment Anything Model 3) segmentation on equirectangular panorama images. It automatically handles perspective projection, coordinate conversion from flat to spherical coordinates, and intelligent deduplication of overlapping masks.

## Demo

This is a demo using the built-in [preview tool](#interactive-preview-tool) with test results.

<!-- TODO: Add demo video -->
<!-- https://github.com/user-attachments/assets/VIDEO_ID -->

## Features

- Support for SAM3 (Segment Anything Model 3) via HuggingFace Transformers
- Text-prompted segmentation (e.g., "car", "person", "sign")
- Automatic perspective generation from equirectangular panoramas
- Spherical coordinate conversion for polygon masks
- IoU-based duplication detection and removal across perspectives
- MPS/CUDA GPU acceleration support
- Interactive preview tool for visualization

## Installation

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
   pip install -e .
   ```

3. Login to HuggingFace (required for SAM3 model access):

   ```bash
   # First, accept the license at https://huggingface.co/facebook/sam3
   huggingface-cli login
   ```

## Usage

### Basic Usage

The basic usage is showcased in [`run_panosam.py`](run_panosam.py)

To run the script, simply execute:

```bash
uv run panosam --image [path-to-panorama-image] --prompt [text-prompt]
```

For example:

```bash
uv run panosam --image assets/test-pano.jpg --prompt "car"
```

The result will be saved in the same folder as the original image, with the same filename but with a different extension: `.panosam.json`

### Command Line Options

```bash
panosam --help

Options:
  --image, -i       Path to the equirectangular panorama image (required)
  --prompt, -p      Text prompt describing objects to segment (required)
  --output, -o      Output path for JSON results (default: <image>.panosam.json)
  --preset          Perspective preset: default, zoomed_in, zoomed_out, wideangle
  --threshold       Confidence threshold for detections (default: 0.5)
  --mask-threshold  Threshold for binary mask generation (default: 0.5)
  --min-iou         Minimum IoU for deduplication (default: 0.3)
  --direct, -d      Direct mode: run SAM3 on original image (for benchmarking)
  --quiet, -q       Suppress progress output
```

### Direct Mode (Benchmarking)

For benchmarking purposes, you can run SAM3 directly on the original equirectangular image without perspective projection or deduplication:

```bash
uv run panosam --image panorama.jpg --prompt "car" --direct
```

This mode still outputs spherical coordinates (yaw/pitch) by mapping the flat pixel coordinates to equirectangular projection. The output file will be named `<image>.panosam.direct.json`.

| Aspect           | Normal Mode                | Direct Mode              |
| ---------------- | -------------------------- | ------------------------ |
| Perspective      | Multiple projections       | None (original image)    |
| Deduplication    | Yes (IoU-based)            | No                       |
| Coordinates      | Spherical (yaw/pitch)      | Spherical (yaw/pitch)    |
| Output file      | `.panosam.json`            | `.panosam.direct.json`   |

## Core Components

The core building blocks of `panosam` are:

- `SAM3Engine`: Represents the SAM3 segmentation engine using HuggingFace Transformers.
- `SphereMaskDeduplicationEngine`: Represents a duplication detection engine using GeoPandas IoU.
- `PanoramaImage`: Represents an equirectangular panorama image.
- `PerspectiveMetadata`: Represents a perspective view of the panorama.

By default, `run_panosam.py` uses `ps.DEFAULT_IMAGE_PERSPECTIVES` which is 16 perspectives with the following settings:

- pixel_width: 2048
- pixel_height: 2048
- 45° horizontal field of view
- 0° pitch offset
- 22.5° yaw interval

You can also specify different perspective settings by directly constructing `PerspectiveMetadata` objects:

```python
import panosam as ps

perspective = ps.PerspectiveMetadata(
    pixel_width=1024,
    pixel_height=1024,
    horizontal_fov=45,
    vertical_fov=45,
    yaw_offset=0,
    pitch_offset=0,
)
```

### Perspective Presets

| Preset       | FOV   | Resolution | Perspectives | Best For      |
| ------------ | ----- | ---------- | ------------ | ------------- |
| `default`    | 45°   | 2048×2048  | 16           | General use   |
| `zoomed_in`  | 22.5° | 1024×1024  | 32           | Small objects |
| `zoomed_out` | 60°   | 2500×2500  | 12           | Large objects |

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "prompt": "car",
  "image_path": "panorama.jpg",
  "perspective_preset": "default",
  "masks": [
    {
      "polygon": [[yaw1, pitch1], [yaw2, pitch2], ...],
      "score": 0.95,
      "label": "car",
      "mask_id": "car_0",
      "center_yaw": 45.2,
      "center_pitch": -5.3
    }
  ]
}
```

- `polygon`: List of (yaw, pitch) coordinates in degrees
- `score`: Confidence score (0-1)
- `center_yaw/pitch`: Centroid of the polygon

## Interactive Preview Tool

PanoSAM includes a web-based interactive preview tool that allows you to visualize segmentation results on the panorama image. It's located in `preview/index.html`. To run it:

```bash
cd preview && python -m http.server
```

Then, open your browser and navigate to `http://localhost:8000`.

Simply drag and drop the JSON result file and your panorama image to the interface, and you should see the segmentation masks overlaid on the panorama image. You can orbit around the panorama and hover over masks to highlight them.

## Related Projects

- [PanoOCR](https://github.com/yz3440/panoocr) - OCR for panoramic images (similar architecture)
- [SAM3](https://huggingface.co/facebook/sam3) - Meta's Segment Anything Model 3
- [py360convert](https://github.com/sunset1995/py360convert) - Equirectangular projection library

## License

MIT License - see [LICENSE](LICENSE) for details.
