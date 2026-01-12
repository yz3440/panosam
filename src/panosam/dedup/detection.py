"""Sphere mask deduplication using GeoPandas polygon overlap."""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import geopandas as gpd
from shapely.geometry import Polygon

from ..sam.models import SphereMaskResult


@dataclass
class PolygonIntersection:
    """Result of polygon intersection analysis.
    
    Attributes:
        polygon_1_area: Area of the first polygon.
        polygon_2_area: Area of the second polygon.
        intersection_area: Area of intersection.
        union_area: Area of union.
        iou: Intersection over Union ratio.
        intersection_ratio: Intersection area / min(area1, area2).
    """
    polygon_1_area: float
    polygon_2_area: float
    intersection_area: float
    union_area: float
    iou: float
    intersection_ratio: float


# Default thresholds
DEFAULT_MIN_IOU = 0.3
DEFAULT_MIN_INTERSECTION_RATIO = 0.5


class SphereMaskDeduplicationEngine:
    """Engine for detecting and removing duplicate sphere masks.
    
    Uses GeoPandas for polygon intersection calculations with IoU-based
    deduplication strategy.
    
    Attributes:
        min_iou: Minimum IoU threshold to consider masks as duplicates.
        min_intersection_ratio: Minimum intersection ratio threshold.
    """
    
    def __init__(
        self,
        min_iou: float = DEFAULT_MIN_IOU,
        min_intersection_ratio: float = DEFAULT_MIN_INTERSECTION_RATIO,
    ):
        """Initialize the deduplication engine.
        
        Args:
            min_iou: Minimum IoU to consider masks as duplicates.
            min_intersection_ratio: Minimum intersection/min(area) ratio.
        """
        self.min_iou = min_iou
        self.min_intersection_ratio = min_intersection_ratio
    
    def __sphere_mask_to_polygon(self, sphere_mask: SphereMaskResult) -> Optional[Polygon]:
        """Convert a sphere mask to a Shapely Polygon.
        
        Args:
            sphere_mask: The sphere mask result.
            
        Returns:
            Shapely Polygon or None if invalid.
        """
        if len(sphere_mask.polygon) < 3:
            return None
        
        # Convert (yaw, pitch) tuples to Polygon
        # Note: We use yaw as x and pitch as y
        points = [(yaw, pitch) for yaw, pitch in sphere_mask.polygon]
        
        try:
            polygon = Polygon(points)
            if not polygon.is_valid:
                # Try to fix invalid polygon
                polygon = polygon.buffer(0)
            return polygon
        except Exception:
            return None
    
    def __polygon_to_gdf(self, polygon: Polygon) -> gpd.GeoDataFrame:
        """Convert a Shapely Polygon to a GeoDataFrame.
        
        Uses EPSG:4326 (WGS84) for geographic coordinates and converts
        to EPSG:3857 for area calculations.
        
        Args:
            polygon: Shapely Polygon.
            
        Returns:
            GeoDataFrame with the polygon.
        """
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[polygon])
        gdf = gdf.to_crs(3857)
        return gdf
    
    def __sphere_mask_to_gdf(self, sphere_mask: SphereMaskResult) -> Optional[gpd.GeoDataFrame]:
        """Convert a sphere mask to a GeoDataFrame.
        
        Args:
            sphere_mask: The sphere mask result.
            
        Returns:
            GeoDataFrame or None if conversion failed.
        """
        polygon = self.__sphere_mask_to_polygon(sphere_mask)
        if polygon is None or polygon.is_empty:
            return None
        return self.__polygon_to_gdf(polygon)
    
    def __get_intersection(
        self, gdf_1: gpd.GeoDataFrame, gdf_2: gpd.GeoDataFrame
    ) -> Optional[PolygonIntersection]:
        """Calculate intersection metrics between two polygons.
        
        Args:
            gdf_1: First polygon as GeoDataFrame.
            gdf_2: Second polygon as GeoDataFrame.
            
        Returns:
            PolygonIntersection or None if no intersection.
        """
        try:
            gdf_intersection = gpd.overlay(gdf_1, gdf_2, how="intersection")
            
            if gdf_intersection.empty:
                return None
            
            polygon_1_area = gdf_1.area.values[0]
            polygon_2_area = gdf_2.area.values[0]
            intersection_area = gdf_intersection.area.values[0]
            
            # Calculate union area
            union_area = polygon_1_area + polygon_2_area - intersection_area
            
            # Calculate IoU
            iou = intersection_area / union_area if union_area > 0 else 0
            
            # Calculate intersection ratio
            min_area = min(polygon_1_area, polygon_2_area)
            intersection_ratio = intersection_area / min_area if min_area > 0 else 0
            
            return PolygonIntersection(
                polygon_1_area=polygon_1_area,
                polygon_2_area=polygon_2_area,
                intersection_area=intersection_area,
                union_area=union_area,
                iou=iou,
                intersection_ratio=intersection_ratio,
            )
        except Exception:
            return None
    
    def __intersect_masks(
        self,
        mask_1: SphereMaskResult,
        mask_2: SphereMaskResult,
    ) -> Optional[PolygonIntersection]:
        """Calculate intersection between two sphere masks.
        
        Args:
            mask_1: First sphere mask.
            mask_2: Second sphere mask.
            
        Returns:
            PolygonIntersection or None if no intersection.
        """
        gdf_1 = self.__sphere_mask_to_gdf(mask_1)
        gdf_2 = self.__sphere_mask_to_gdf(mask_2)
        
        if gdf_1 is None or gdf_2 is None:
            return None
        
        return self.__get_intersection(gdf_1, gdf_2)
    
    def check_duplication(
        self, mask_1: SphereMaskResult, mask_2: SphereMaskResult
    ) -> bool:
        """Check if two masks are duplicates based on spatial overlap.
        
        Args:
            mask_1: First sphere mask.
            mask_2: Second sphere mask.
            
        Returns:
            True if masks are considered duplicates.
        """
        intersection = self.__intersect_masks(mask_1, mask_2)
        
        if intersection is None:
            return False
        
        # Check IoU threshold
        if intersection.iou >= self.min_iou:
            return True
        
        # Check intersection ratio threshold
        if intersection.intersection_ratio >= self.min_intersection_ratio:
            return True
        
        return False
    
    def __select_better_mask(
        self, mask_1: SphereMaskResult, mask_2: SphereMaskResult
    ) -> Tuple[SphereMaskResult, SphereMaskResult]:
        """Select which mask to keep and which to remove.
        
        Prefers masks with higher confidence scores.
        
        Args:
            mask_1: First sphere mask.
            mask_2: Second sphere mask.
            
        Returns:
            Tuple of (keep, remove) masks.
        """
        if mask_1.score >= mask_2.score:
            return mask_1, mask_2
        return mask_2, mask_1
    
    def remove_duplication_for_two_lists(
        self,
        masks_0: List[SphereMaskResult],
        masks_1: List[SphereMaskResult],
    ) -> Tuple[List[SphereMaskResult], List[SphereMaskResult]]:
        """Remove duplicate masks between two lists.
        
        Compares each mask in masks_0 with each mask in masks_1 and removes
        duplicates, keeping the mask with higher confidence.
        
        Args:
            masks_0: First list of sphere masks.
            masks_1: Second list of sphere masks.
            
        Returns:
            Tuple of (filtered_masks_0, filtered_masks_1).
        """
        duplications = []
        for i, mask_0 in enumerate(masks_0):
            for j, mask_1 in enumerate(masks_1):
                if self.check_duplication(mask_0, mask_1):
                    duplications.append((i, j))
        
        indices_to_remove_from_masks_0 = set()
        indices_to_remove_from_masks_1 = set()
        
        for idx_0, idx_1 in duplications:
            keep, remove = self.__select_better_mask(masks_0[idx_0], masks_1[idx_1])
            
            if keep is masks_0[idx_0]:
                indices_to_remove_from_masks_1.add(idx_1)
            else:
                indices_to_remove_from_masks_0.add(idx_0)
        
        # Filter out duplicates
        filtered_masks_0 = [
            mask for i, mask in enumerate(masks_0) 
            if i not in indices_to_remove_from_masks_0
        ]
        filtered_masks_1 = [
            mask for i, mask in enumerate(masks_1) 
            if i not in indices_to_remove_from_masks_1
        ]
        
        return filtered_masks_0, filtered_masks_1
    
    def deduplicate_list(
        self, masks: List[SphereMaskResult]
    ) -> List[SphereMaskResult]:
        """Remove duplicates within a single list of masks.
        
        Compares all pairs of masks and removes duplicates, keeping
        the mask with higher confidence.
        
        Args:
            masks: List of sphere masks.
            
        Returns:
            Deduplicated list of sphere masks.
        """
        if len(masks) <= 1:
            return masks
        
        indices_to_remove = set()
        
        for i in range(len(masks)):
            if i in indices_to_remove:
                continue
            for j in range(i + 1, len(masks)):
                if j in indices_to_remove:
                    continue
                
                if self.check_duplication(masks[i], masks[j]):
                    keep, remove = self.__select_better_mask(masks[i], masks[j])
                    if remove is masks[i]:
                        indices_to_remove.add(i)
                        break
                    else:
                        indices_to_remove.add(j)
        
        return [mask for i, mask in enumerate(masks) if i not in indices_to_remove]
