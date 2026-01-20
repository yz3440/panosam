# Image

Classes for equirectangular panoramas and perspective projections.

## PanoramaImage

::: panosam.image.models.PanoramaImage
    options:
      show_root_heading: true
      members:
        - __init__
        - generate_perspective_image
        - get_image
        - get_image_array

## PerspectiveImage

::: panosam.image.models.PerspectiveImage
    options:
      show_root_heading: true
      members:
        - __init__
        - get_perspective_metadata
        - get_perspective_image_array
        - get_perspective_image

## PerspectiveMetadata

::: panosam.image.models.PerspectiveMetadata
    options:
      show_root_heading: true

## Perspective Generation

::: panosam.image.perspectives.generate_perspectives
    options:
      show_root_heading: true

::: panosam.image.perspectives.combine_perspectives
    options:
      show_root_heading: true

## Presets

Pre-configured perspective sets:

```python
import panosam as ps

ps.DEFAULT_IMAGE_PERSPECTIVES      # 16 perspectives, 45° FOV, 2048x2048
ps.ZOOMED_IN_IMAGE_PERSPECTIVES    # 32 perspectives, 22.5° FOV, 1024x1024
ps.ZOOMED_OUT_IMAGE_PERSPECTIVES   # 12 perspectives, 60° FOV, 2500x2500
ps.WIDEANGLE_IMAGE_PERSPECTIVES    # 8 perspectives, 90° FOV, 2500x2500
```
