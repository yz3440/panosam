# Deduplication

Removes duplicate masks when the same object appears in multiple perspective views.

## Algorithm

Masks are deduplicated using IoU (Intersection over Union) on spherical polygons:

1. Process perspectives sequentially
2. For each mask, check overlap with existing masks
3. If IoU >= threshold or intersection ratio >= 0.5, merge via union
4. Otherwise, add as new mask

## SphereMaskDeduplicationEngine

::: panosam.dedup.detection.SphereMaskDeduplicationEngine
    options:
      show_root_heading: true
      members:
        - __init__
        - check_duplication
        - deduplicate_list
        - deduplicate_frames

## Data Classes

::: panosam.dedup.detection.PolygonIntersection
    options:
      show_root_heading: true
