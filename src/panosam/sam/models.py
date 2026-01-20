from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..geometry import calculate_spherical_centroid, perspective_to_sphere


@dataclass
class FlatMaskResult:
    """A segmentation mask result in flat/perspective image coordinates.

    Attributes:
        polygons: List of polygons, each polygon is a list of (x, y) tuples
                  in normalized coordinates (0-1 range, where 0,0 is top-left).
        score: Confidence score for this mask (0-1).
        label: Optional text label for the segmented object.
        mask_id: Optional unique identifier for this mask.
    """

    polygons: List[List[Tuple[float, float]]]
    score: float
    label: Optional[str] = None
    mask_id: Optional[str] = None

    def to_sphere(
        self,
        horizontal_fov: float,
        vertical_fov: float,
        yaw_offset: float,
        pitch_offset: float,
    ) -> "SphereMaskResult":
        """Convert flat mask result to spherical coordinates.

        Uses proper 3D rotation to accurately map perspective image coordinates
        to equirectangular spherical coordinates.

        Args:
            horizontal_fov: Horizontal field of view in degrees.
            vertical_fov: Vertical field of view in degrees.
            yaw_offset: Horizontal offset of the perspective in degrees.
            pitch_offset: Vertical offset of the perspective in degrees.

        Returns:
            SphereMaskResult with polygons in spherical coordinates.
        """
        if (
            horizontal_fov is None
            or vertical_fov is None
            or yaw_offset is None
            or pitch_offset is None
        ):
            raise ValueError("Missing parameters")
        if horizontal_fov < 0 or vertical_fov < 0:
            raise ValueError("FOV must be positive")

        # Convert each polygon to spherical coordinates
        sphere_polygons = []
        for polygon in self.polygons:
            sphere_polygon = []
            for u, v in polygon:
                yaw, pitch = perspective_to_sphere(
                    u, v, horizontal_fov, vertical_fov, yaw_offset, pitch_offset
                )
                sphere_polygon.append((yaw, pitch))
            if sphere_polygon:
                sphere_polygons.append(sphere_polygon)

        # Calculate centroid using proper spherical averaging
        # This handles wrap-around at ±180° correctly
        if sphere_polygons:
            center_yaw, center_pitch = calculate_spherical_centroid(sphere_polygons)
        else:
            center_yaw = yaw_offset
            center_pitch = pitch_offset

        return SphereMaskResult(
            polygons=sphere_polygons,
            score=self.score,
            label=self.label,
            mask_id=self.mask_id,
            center_yaw=center_yaw,
            center_pitch=center_pitch,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "polygons": self.polygons,
            "score": self.score,
            "label": self.label,
            "mask_id": self.mask_id,
        }

    @classmethod
    def from_binary_mask(
        cls,
        mask: np.ndarray,
        score: float,
        label: Optional[str] = None,
        mask_id: Optional[str] = None,
        simplify_tolerance: float = 0.001,
        min_contour_area_ratio: float = 0.01,
    ) -> "FlatMaskResult":
        """Create a FlatMaskResult from a binary mask.

        Extracts ALL significant contours from the mask, not just the largest.

        Args:
            mask: Binary mask as numpy array (H, W) with values 0 or 1/255.
            score: Confidence score for this mask.
            label: Optional text label.
            mask_id: Optional unique identifier.
            simplify_tolerance: Tolerance for polygon simplification (0-1).
            min_contour_area_ratio: Minimum contour area as ratio of largest contour.
                                    Contours smaller than this are discarded.

        Returns:
            FlatMaskResult with normalized polygon coordinates.
        """
        import cv2

        # Ensure mask is uint8
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return cls(polygons=[], score=score, label=label, mask_id=mask_id)

        # Get the largest contour area for filtering
        contour_areas = [cv2.contourArea(c) for c in contours]
        max_area = max(contour_areas)
        min_area = max_area * min_contour_area_ratio

        # Process all significant contours
        h, w = mask.shape[:2]
        polygons = []

        for contour, area in zip(contours, contour_areas):
            if area < min_area:
                continue  # Skip tiny contours

            # Simplify the contour
            epsilon = simplify_tolerance * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)

            # Need at least 3 points for a polygon
            if len(simplified) < 3:
                continue

            # Convert to normalized coordinates (0-1)
            polygon = [(float(pt[0][0]) / w, float(pt[0][1]) / h) for pt in simplified]
            polygons.append(polygon)

        return cls(polygons=polygons, score=score, label=label, mask_id=mask_id)


@dataclass
class SphereMaskResult:
    """A segmentation mask result in spherical/panoramic coordinates.

    Attributes:
        polygons: List of polygons, each polygon is a list of (yaw, pitch) tuples in degrees.
        score: Confidence score for this mask (0-1).
        label: Optional text label for the segmented object.
        mask_id: Optional unique identifier for this mask.
        center_yaw: Yaw of the polygon centroid in degrees.
        center_pitch: Pitch of the polygon centroid in degrees.
    """

    polygons: List[List[Tuple[float, float]]]
    score: float
    label: Optional[str] = None
    mask_id: Optional[str] = None
    center_yaw: float = 0.0
    center_pitch: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "polygons": self.polygons,
            "score": self.score,
            "label": self.label,
            "mask_id": self.mask_id,
            "center_yaw": self.center_yaw,
            "center_pitch": self.center_pitch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SphereMaskResult":
        """Create from dictionary representation."""
        # Handle both old format (polygon) and new format (polygons)
        if "polygons" in data:
            polygons = [[tuple(p) for p in poly] for poly in data["polygons"]]
        elif "polygon" in data:
            # Legacy format: single polygon
            polygons = [[tuple(p) for p in data["polygon"]]] if data["polygon"] else []
        else:
            polygons = []

        return cls(
            polygons=polygons,
            score=data["score"],
            label=data.get("label"),
            mask_id=data.get("mask_id"),
            center_yaw=data.get("center_yaw", 0.0),
            center_pitch=data.get("center_pitch", 0.0),
        )

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the bounding box of all polygons.

        Returns:
            Tuple of (min_yaw, min_pitch, max_yaw, max_pitch) in degrees.
        """
        all_points = [pt for polygon in self.polygons for pt in polygon]
        if not all_points:
            return (0, 0, 0, 0)

        yaws = [p[0] for p in all_points]
        pitches = [p[1] for p in all_points]

        return (min(yaws), min(pitches), max(yaws), max(pitches))

    def get_area_estimate(self) -> float:
        """Estimate the total area of all polygons using the shoelace formula.

        Returns:
            Estimated area in square degrees.
        """
        total_area = 0.0
        for polygon in self.polygons:
            if len(polygon) < 3:
                continue

            # Shoelace formula
            n = len(polygon)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += polygon[i][0] * polygon[j][1]
                area -= polygon[j][0] * polygon[i][1]

            total_area += abs(area) / 2.0

        return total_area
