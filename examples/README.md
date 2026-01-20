# PanoSAM Examples

Example scripts demonstrating PanoSAM usage.

## Prerequisites

1. Install PanoSAM with SAM3 support:
   ```bash
   pip install "panosam[sam] @ git+https://github.com/yz3440/panosam.git"
   ```

2. Login to HuggingFace (required for SAM3 model access):
   ```bash
   huggingface-cli login
   ```
   You must also accept the SAM3 license at https://huggingface.co/facebook/sam3

## Examples

### basic_usage.py

The simplest way to segment objects in a panorama:

```bash
python examples/basic_usage.py assets/test-pano.jpg "car"
```

### multi_scale.py

Combine multiple presets for detecting objects of different sizes:

```bash
python examples/multi_scale.py assets/test-pano.jpg "window"
```

### custom_perspectives.py

Create custom perspective configurations (e.g., include ceiling/floor):

```bash
python examples/custom_perspectives.py assets/test-pano.jpg "light"
```

## Output

All examples save results as JSON files compatible with the preview tool.
To visualize results:

```bash
cd preview && python -m http.server
```

Then open http://localhost:8000 and drag-drop your panorama + JSON file.
