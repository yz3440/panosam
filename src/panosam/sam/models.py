from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import math
import numpy as np


def _calculate_spherical_centroid(
    polygons: List[List[Tuple[float, float]]],
) -> Tuple[float, float]:
    """Calculate the centroid of spherical polygon(s) using 3D averaging.

    This handles wrap-around at ±180° correctly by converting to 3D Cartesian
    coordinates, averaging in 3D space, and converting back.

    Args:
        polygons: List of polygons, each polygon is a list of (yaw, pitch) tuples in degrees.

    Returns:
        (center_yaw, center_pitch) in degrees.
    """
    # Flatten all polygon points
    all_points = [pt for polygon in polygons for pt in polygon]

    if not all_points:
        return 0.0, 0.0

    # Convert to 3D Cartesian and accumulate
    sum_x, sum_y, sum_z = 0.0, 0.0, 0.0

    for yaw_deg, pitch_deg in all_points:
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)

        # Spherical to Cartesian (pitch = latitude, yaw = longitude)
        # x = cos(pitch) * sin(yaw)  [East direction]
        # y = sin(pitch)              [Up direction]
        # z = cos(pitch) * cos(yaw)  [North direction]
        x = math.cos(pitch_rad) * math.sin(yaw_rad)
        y = math.sin(pitch_rad)
        z = math.cos(pitch_rad) * math.cos(yaw_rad)

        sum_x += x
        sum_y += y
        sum_z += z

    n = len(all_points)
    avg_x = sum_x / n
    avg_y = sum_y / n
    avg_z = sum_z / n

    # Convert back to spherical
    # Handle the degenerate case where the centroid is at origin
    magnitude = math.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
    if magnitude < 1e-10:
        # Points are symmetrically distributed, fall back to simple average
        # This is rare but can happen
        center_yaw = sum(p[0] for p in all_points) / n
        center_pitch = sum(p[1] for p in all_points) / n
        return center_yaw, center_pitch

    # Normalize
    avg_x /= magnitude
    avg_y /= magnitude
    avg_z /= magnitude

    # Convert to spherical coordinates
    center_yaw = math.degrees(math.atan2(avg_x, avg_z))
    # Clamp to avoid domain errors from floating point issues
    center_pitch = math.degrees(math.asin(max(-1.0, min(1.0, avg_y))))

    return center_yaw, center_pitch


def _perspective_to_sphere(
    u: float,
    v: float,
    horizontal_fov: float,
    vertical_fov: float,
    yaw_offset: float,
    pitch_offset: float,
) -> Tuple[float, float]:
    """Convert perspective image coordinates to spherical coordinates.

    Uses proper 3D rotation to handle camera orientation correctly.
    This is the inverse of py360convert's e2p transformation.

    Args:
        u: Horizontal coordinate (0-1, left to right).
        v: Vertical coordinate (0-1, top to bottom).
        horizontal_fov: Horizontal field of view in degrees.
        vertical_fov: Vertical field of view in degrees.
        yaw_offset: Camera yaw (horizontal rotation) in degrees.
        pitch_offset: Camera pitch (vertical rotation) in degrees.

    Returns:
        Tuple of (yaw, pitch) in degrees, where:
        - yaw: -180 to 180 (left to right in equirectangular)
        - pitch: -90 to 90 (bottom to top)
    """
    # Convert to centered coordinates (-0.5 to 0.5)
    # x: positive = right, y: positive = up
    x = u - 0.5
    y = 0.5 - v

    # Calculate 3D direction vector in camera's local frame
    # Camera looks along +Z axis, X is right, Y is up
    # The image plane is at Z=1, with extent based on FOV
    half_h_fov = math.radians(horizontal_fov) / 2
    half_v_fov = math.radians(vertical_fov) / 2

    # Direction in camera local coordinates
    X_local = x * 2 * math.tan(half_h_fov)
    Y_local = y * 2 * math.tan(half_v_fov)
    Z_local = 1.0

    # Normalize to unit vector
    r = math.sqrt(X_local**2 + Y_local**2 + Z_local**2)
    X_local /= r
    Y_local /= r
    Z_local /= r

    # Rotate by camera orientation to get world coordinates
    # Order: first pitch (rotation around X), then yaw (rotation around Y)
    pitch_rad = math.radians(pitch_offset)
    yaw_rad = math.radians(yaw_offset)

    cos_pitch = math.cos(pitch_rad)
    sin_pitch = math.sin(pitch_rad)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # Rotation by pitch (around X axis)
    # This tilts the view up/down
    # For positive pitch (looking up), forward direction rotates toward +Y
    X_pitched = X_local
    Y_pitched = Y_local * cos_pitch + Z_local * sin_pitch
    Z_pitched = -Y_local * sin_pitch + Z_local * cos_pitch

    # Rotation by yaw (around Y axis)
    # This pans the view left/right
    X_world = X_pitched * cos_yaw + Z_pitched * sin_yaw
    Y_world = Y_pitched
    Z_world = -X_pitched * sin_yaw + Z_pitched * cos_yaw

    # Convert world direction to spherical coordinates
    # yaw = atan2(X, Z), pitch = asin(Y)
    world_yaw = math.degrees(math.atan2(X_world, Z_world))
    world_pitch = math.degrees(math.asin(np.clip(Y_world, -1.0, 1.0)))

    return world_yaw, world_pitch


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
                yaw, pitch = _perspective_to_sphere(
                    u, v, horizontal_fov, vertical_fov, yaw_offset, pitch_offset
                )
                sphere_polygon.append((yaw, pitch))
            if sphere_polygon:
                sphere_polygons.append(sphere_polygon)

        # Calculate centroid using proper spherical averaging
        # This handles wrap-around at ±180° correctly
        if sphere_polygons:
            center_yaw, center_pitch = _calculate_spherical_centroid(sphere_polygons)
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
