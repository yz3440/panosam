from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class FlatMaskResult:
    """A segmentation mask result in flat/perspective image coordinates.
    
    Attributes:
        polygon: List of (x, y) tuples representing the mask contour in normalized
                 coordinates (0-1 range, where 0,0 is top-left).
        score: Confidence score for this mask (0-1).
        label: Optional text label for the segmented object.
        mask_id: Optional unique identifier for this mask.
    """
    polygon: List[Tuple[float, float]]
    score: float
    label: Optional[str] = None
    mask_id: Optional[str] = None

    def __uv_to_yaw_pitch(
        self, horizontal_fov: float, vertical_fov: float, u: float, v: float
    ) -> Tuple[float, float]:
        """Convert UV coordinate to yaw and pitch.
        
        Args:
            horizontal_fov: Horizontal field of view in degrees.
            vertical_fov: Vertical field of view in degrees.
            u: Horizontal coordinate (0-1, left to right).
            v: Vertical coordinate (0-1, top to bottom).
            
        Returns:
            Tuple of (yaw, pitch) in degrees.
        """
        if horizontal_fov is None or vertical_fov is None or u is None or v is None:
            raise ValueError("Missing parameters")

        if horizontal_fov < 0 or vertical_fov < 0:
            raise ValueError("FOV must be positive")

        # Translate the origin to the center of the image
        u = u - 0.5
        v = 0.5 - v

        yaw = math.atan2(2 * u * math.tan(math.radians(horizontal_fov) / 2), 1)
        pitch = math.atan2(2 * v * math.tan(math.radians(vertical_fov) / 2), 1)

        return math.degrees(yaw), math.degrees(pitch)

    def to_sphere(
        self,
        horizontal_fov: float,
        vertical_fov: float,
        yaw_offset: float,
        pitch_offset: float,
    ) -> "SphereMaskResult":
        """Convert flat mask result to spherical coordinates.
        
        Args:
            horizontal_fov: Horizontal field of view in degrees.
            vertical_fov: Vertical field of view in degrees.
            yaw_offset: Horizontal offset of the perspective in degrees.
            pitch_offset: Vertical offset of the perspective in degrees.
            
        Returns:
            SphereMaskResult with polygon in spherical coordinates.
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

        # Convert each polygon vertex to spherical coordinates
        sphere_polygon = []
        for u, v in self.polygon:
            yaw, pitch = self.__uv_to_yaw_pitch(horizontal_fov, vertical_fov, u, v)
            sphere_polygon.append((yaw + yaw_offset, pitch + pitch_offset))

        # Calculate centroid for the result
        if len(sphere_polygon) > 0:
            center_yaw = sum(p[0] for p in sphere_polygon) / len(sphere_polygon)
            center_pitch = sum(p[1] for p in sphere_polygon) / len(sphere_polygon)
        else:
            center_yaw = yaw_offset
            center_pitch = pitch_offset

        return SphereMaskResult(
            polygon=sphere_polygon,
            score=self.score,
            label=self.label,
            mask_id=self.mask_id,
            center_yaw=center_yaw,
            center_pitch=center_pitch,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "polygon": self.polygon,
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
    ) -> "FlatMaskResult":
        """Create a FlatMaskResult from a binary mask.
        
        Args:
            mask: Binary mask as numpy array (H, W) with values 0 or 1/255.
            score: Confidence score for this mask.
            label: Optional text label.
            mask_id: Optional unique identifier.
            simplify_tolerance: Tolerance for polygon simplification (0-1).
            
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
            return cls(polygon=[], score=score, label=label, mask_id=mask_id)
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify the contour
        epsilon = simplify_tolerance * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to normalized coordinates (0-1)
        h, w = mask.shape[:2]
        polygon = [(float(pt[0][0]) / w, float(pt[0][1]) / h) for pt in simplified]
        
        return cls(polygon=polygon, score=score, label=label, mask_id=mask_id)


@dataclass
class SphereMaskResult:
    """A segmentation mask result in spherical/panoramic coordinates.
    
    Attributes:
        polygon: List of (yaw, pitch) tuples in degrees representing the mask contour.
        score: Confidence score for this mask (0-1).
        label: Optional text label for the segmented object.
        mask_id: Optional unique identifier for this mask.
        center_yaw: Yaw of the polygon centroid in degrees.
        center_pitch: Pitch of the polygon centroid in degrees.
    """
    polygon: List[Tuple[float, float]]
    score: float
    label: Optional[str] = None
    mask_id: Optional[str] = None
    center_yaw: float = 0.0
    center_pitch: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "polygon": self.polygon,
            "score": self.score,
            "label": self.label,
            "mask_id": self.mask_id,
            "center_yaw": self.center_yaw,
            "center_pitch": self.center_pitch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SphereMaskResult":
        """Create from dictionary representation."""
        return cls(
            polygon=[tuple(p) for p in data["polygon"]],
            score=data["score"],
            label=data.get("label"),
            mask_id=data.get("mask_id"),
            center_yaw=data.get("center_yaw", 0.0),
            center_pitch=data.get("center_pitch", 0.0),
        )

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the bounding box of the polygon.
        
        Returns:
            Tuple of (min_yaw, min_pitch, max_yaw, max_pitch) in degrees.
        """
        if len(self.polygon) == 0:
            return (0, 0, 0, 0)
        
        yaws = [p[0] for p in self.polygon]
        pitches = [p[1] for p in self.polygon]
        
        return (min(yaws), min(pitches), max(yaws), max(pitches))

    def get_area_estimate(self) -> float:
        """Estimate the area of the polygon using the shoelace formula.
        
        Returns:
            Estimated area in square degrees.
        """
        if len(self.polygon) < 3:
            return 0.0
        
        # Shoelace formula
        n = len(self.polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.polygon[i][0] * self.polygon[j][1]
            area -= self.polygon[j][0] * self.polygon[i][1]
        
        return abs(area) / 2.0
