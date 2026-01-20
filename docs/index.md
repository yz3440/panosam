# PanoSAM

SAM3 segmentation for equirectangular panorama images. Handles perspective projection, coordinate conversion, and mask deduplication.

## Installation

```bash
# Lightweight (bring your own engine)
pip install "panosam @ git+https://github.com/yz3440/panosam.git"

# With SAM3
pip install "panosam[sam3]"

# With visualization
pip install "panosam[full] @ git+https://github.com/yz3440/panosam.git"
```

SAM3 requires HuggingFace authentication. Accept the license at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then:

```bash
huggingface-cli login
```

## Quick Start

```python
import panosam as ps
from panosam.engines.sam3 import SAM3Engine

engine = SAM3Engine()
client = ps.PanoSAM(engine=engine)
result = client.segment("panorama.jpg", prompt="car")
result.save_json("results.panosam.json")
```

## Demo

<video controls width="100%">
  <source src="https://github.com/user-attachments/assets/edef8666-a7dd-4bf9-b86a-2144f28e17e1" type="video/mp4">
</video>

The preview tool visualizes segmentation results on an interactive 3D sphere.

```bash
cd preview && python -m http.server
```

Open `http://localhost:8000` and drag in the JSON result file and panorama image.

## Next

- [Examples](examples.md) - Working scripts
- [API Reference](api/index.md) - Full documentation
