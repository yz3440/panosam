# Engines

PanoSAM uses dependency injection for segmentation engines. Provide any object with a matching `segment()` method.

## SegmentationEngine Protocol

::: panosam.api.models.SegmentationEngine
    options:
      show_root_heading: true

## SAM3Engine

The built-in engine using Meta's SAM3 model. Requires the `[sam]` extra.

```bash
pip install "panosam[sam] @ git+https://github.com/yz3440/panosam.git"
```

::: panosam.engines.sam3.SAM3Engine
    options:
      show_root_heading: true
      members:
        - __init__
        - segment
        - segment_with_boxes
        - get_raw_masks

## Custom Engines

Any class with a compatible `segment()` method works:

```python
import panosam as ps
from PIL import Image

class MyEngine:
    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        simplify_tolerance: float = 0.005,
    ) -> list[ps.FlatMaskResult]:
        # Return list of FlatMaskResult
        ...

client = ps.PanoSAM(engine=MyEngine())
```

## Mask Results

::: panosam.sam.models.FlatMaskResult
    options:
      show_root_heading: true

::: panosam.sam.models.SphereMaskResult
    options:
      show_root_heading: true
