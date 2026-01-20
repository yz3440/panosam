"""Utility functions for SAM3 mask processing and visualization.

Note: These functions require optional dependencies:
- extract_mask_contours: requires opencv-python (part of [sam] extra)
- visualize_masks: requires opencv-python + matplotlib (part of [viz] extra)
- visualize_sphere_masks: requires opencv-python + matplotlib (part of [viz] extra)
"""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image


def _check_cv2():
    """Check if opencv is available."""
    try:
        import cv2

        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for this function.\n"
            "Install with: pip install opencv-python\n"
            "Or install panosam with SAM3 support: pip install 'panosam[sam]'"
        )


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib

        return matplotlib
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization functions.\n"
            "Install with: pip install matplotlib\n"
            "Or install panosam with visualization support: pip install 'panosam[viz]'"
        )


def extract_mask_contours(
    mask: np.ndarray,
    simplify_tolerance: float = 0.001,
) -> List[List[Tuple[float, float]]]:
    """Extract contours from a binary mask.

    Args:
        mask: Binary mask as numpy array (H, W).
        simplify_tolerance: Tolerance for polygon simplification (0-1).

    Returns:
        List of contours, each contour is a list of (x, y) tuples in normalized coords.

    Note:
        Requires opencv-python. Install with `pip install opencv-python`.
    """
    cv2 = _check_cv2()

    # Ensure mask is uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape[:2]
    result = []

    for contour in contours:
        # Simplify the contour
        epsilon = simplify_tolerance * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        # Convert to normalized coordinates (0-1)
        polygon = [(float(pt[0][0]) / w, float(pt[0][1]) / h) for pt in simplified]

        if len(polygon) >= 3:  # Only include valid polygons
            result.append(polygon)

    return result


def visualize_masks(
    image: Image.Image,
    masks: List[np.ndarray],
    scores: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    alpha: float = 0.5,
) -> Image.Image:
    """Visualize segmentation masks on an image.

    Args:
        image: Base image to draw masks on.
        masks: List of binary masks as numpy arrays.
        scores: Optional confidence scores for each mask.
        labels: Optional labels for each mask.
        alpha: Transparency for mask overlay (0-1).

    Returns:
        Image with masks overlaid.

    Note:
        Requires opencv-python and matplotlib.
        Install with `pip install 'panosam[viz]'`.
    """
    cv2 = _check_cv2()
    matplotlib = _check_matplotlib()

    image = image.convert("RGBA")
    result = image.copy()

    n_masks = len(masks)
    if n_masks == 0:
        return result

    # Generate colors
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for idx, mask in enumerate(masks):
        color = colors[idx]

        # Ensure mask is the right shape
        if mask.dtype != np.uint8:
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        else:
            mask_uint8 = mask

        # Resize mask if needed
        if mask_uint8.shape[:2] != (image.height, image.width):
            mask_uint8 = cv2.resize(
                mask_uint8, (image.width, image.height), interpolation=cv2.INTER_NEAREST
            )

        # Create colored overlay
        mask_pil = Image.fromarray(mask_uint8)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha_mask = mask_pil.point(lambda v: int(v * alpha))
        overlay.putalpha(alpha_mask)
        result = Image.alpha_composite(result, overlay)

    return result


def visualize_sphere_masks(
    panorama: Image.Image,
    sphere_masks: List["SphereMaskResult"],  # Forward reference
    alpha: float = 0.5,
) -> Image.Image:
    """Visualize sphere mask results on an equirectangular panorama.

    Args:
        panorama: Equirectangular panorama image.
        sphere_masks: List of SphereMaskResult objects.
        alpha: Transparency for mask overlay (0-1).

    Returns:
        Panorama with masks overlaid.

    Note:
        Requires opencv-python and matplotlib.
        Install with `pip install 'panosam[viz]'`.
    """
    cv2 = _check_cv2()
    matplotlib = _check_matplotlib()

    panorama = panorama.convert("RGBA")
    result = panorama.copy()

    n_masks = len(sphere_masks)
    if n_masks == 0:
        return result

    # Generate colors
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    w, h = panorama.size

    for idx, sphere_mask in enumerate(sphere_masks):
        color = colors[idx]

        # Convert spherical polygon to equirectangular pixel coordinates
        if len(sphere_mask.polygon) < 3:
            continue

        # Convert yaw/pitch to pixel coordinates
        # yaw: -180 to 180 -> 0 to w
        # pitch: -90 to 90 -> h to 0
        pixel_polygon = []
        for yaw, pitch in sphere_mask.polygon:
            x = int((yaw + 180) / 360 * w) % w
            y = int((90 - pitch) / 180 * h)
            y = max(0, min(h - 1, y))
            pixel_polygon.append((x, y))

        # Create mask from polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(pixel_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # Create overlay
        mask_pil = Image.fromarray(mask)
        overlay = Image.new("RGBA", panorama.size, color + (0,))
        alpha_mask = mask_pil.point(lambda v: int(v * alpha))
        overlay.putalpha(alpha_mask)
        result = Image.alpha_composite(result, overlay)

    return result
