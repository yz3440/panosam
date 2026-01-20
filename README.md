# PanoSAM

SAM3 segmentation for equirectangular panorama images.

Handles perspective projection, coordinate conversion, and mask deduplication.

https://github.com/user-attachments/assets/61a546ac-3fce-4c26-b87c-3e09b0e4c331


## Installation

```bash
pip install "panosam @ git+https://github.com/yz3440/panosam.git"
```

| Extra    | Dependencies                                    | Use Case              |
| -------- | ----------------------------------------------- | --------------------- |
| (none)   | pillow, numpy, geopandas, shapely, py360convert | Bring your own engine |
| `[sam]`  | + torch, transformers, opencv                   | SAM3 segmentation     |
| `[viz]`  | + opencv, matplotlib                            | Visualization         |
| `[full]` | sam + viz                                       | All features          |

SAM3 requires HuggingFace authentication:

```bash
huggingface-cli login
```

## Usage

```python
import panosam as ps
from panosam.engines.sam3 import SAM3Engine

engine = SAM3Engine()
client = ps.PanoSAM(engine=engine, views=ps.PerspectivePreset.DEFAULT)
result = client.segment("panorama.jpg", prompt="car")

for mask in result.masks:
    print(f"{mask.label}: yaw={mask.center_yaw:.1f}, pitch={mask.center_pitch:.1f}")

result.save_json("results.panosam.json")
```

## Custom Engine

Any class with a matching `segment()` method works:

```python
class MyEngine:
    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        simplify_tolerance: float = 0.005,
    ) -> list[ps.FlatMaskResult]:
        ...

client = ps.PanoSAM(engine=MyEngine())
```

## Perspective Presets

| Preset       | FOV   | Resolution | Perspectives |
| ------------ | ----- | ---------- | ------------ |
| `DEFAULT`    | 45°   | 2048x2048  | 16           |
| `ZOOMED_IN`  | 22.5° | 1024x1024  | 32           |
| `ZOOMED_OUT` | 60°   | 2500x2500  | 12           |
| `WIDEANGLE`  | 90°   | 2500x2500  | 8            |

## Documentation

See [yz3440.github.io/panosam](https://yz3440.github.io/panosam/) for full API reference and examples.

## License

MIT
