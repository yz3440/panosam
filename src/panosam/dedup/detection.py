"""Sphere mask deduplication using GeoPandas polygon overlap."""

from dataclasses import dataclass
from typing import List, Optional

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

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

    def __sphere_mask_to_polygon(
        self, sphere_mask: SphereMaskResult
    ) -> Optional[Polygon]:
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

    def __sphere_mask_to_gdf(
        self, sphere_mask: SphereMaskResult
    ) -> Optional[gpd.GeoDataFrame]:
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

    def __fix_polygon(self, poly) -> Optional[Polygon]:
        """Fix a potentially invalid polygon and return a valid simple polygon.

        Args:
            poly: Input polygon or geometry (may be invalid/self-intersecting).

        Returns:
            A valid simple Polygon, or None if unfixable.
        """
        if poly is None or poly.is_empty:
            return None

        # Step 1: Make valid if needed
        if not poly.is_valid:
            poly = make_valid(poly)

        # Step 2: Handle GeometryCollection - take largest polygon
        if poly.geom_type == "GeometryCollection":
            poly_geoms = [
                g for g in poly.geoms if g.geom_type == "Polygon" and not g.is_empty
            ]
            if not poly_geoms:
                return None
            poly = max(poly_geoms, key=lambda p: p.area)

        # Step 3: Handle MultiPolygon - take largest polygon
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)

        if poly.geom_type != "Polygon" or poly.is_empty:
            return None

        # Step 4: Buffer by 0 to clean up any remaining issues
        poly = poly.buffer(0)

        # Step 5: Handle if buffer created MultiPolygon
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)

        if poly.geom_type != "Polygon" or poly.is_empty:
            return None

        # Step 6: Simplify slightly to reduce complexity
        if len(poly.exterior.coords) > 100:
            tolerance = (
                max(
                    poly.bounds[2] - poly.bounds[0],
                    poly.bounds[3] - poly.bounds[1],
                )
                * 0.001
            )
            poly = poly.simplify(tolerance, preserve_topology=True)

        # Final check
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)

        return poly if poly.is_valid and poly.geom_type == "Polygon" else None

    def __validate_and_fix_mask(self, mask: SphereMaskResult) -> SphereMaskResult:
        """Validate and fix a single mask's polygon if needed.

        Args:
            mask: The sphere mask to validate/fix.

        Returns:
            The mask with a valid polygon, or original if unfixable.
        """
        poly = self.__sphere_mask_to_polygon(mask)
        fixed_poly = self.__fix_polygon(poly)

        if fixed_poly is None:
            return mask  # Return original if we can't fix it

        # Check if polygon changed
        if fixed_poly.equals(poly):
            return mask

        # Create new mask with fixed polygon
        coords = list(fixed_poly.exterior.coords)[:-1]
        fixed_polygon = [(float(x), float(y)) for x, y in coords]

        centroid = fixed_poly.centroid
        return SphereMaskResult(
            polygon=fixed_polygon,
            score=mask.score,
            label=mask.label,
            mask_id=mask.mask_id,
            center_yaw=float(centroid.x),
            center_pitch=float(centroid.y),
        )

    def __merge_masks(self, masks: List[SphereMaskResult]) -> SphereMaskResult:
        """Merge multiple overlapping masks into one by taking their polygon union.

        Args:
            masks: List of overlapping sphere masks to merge.

        Returns:
            A single SphereMaskResult with the union polygon and best score.
        """
        if len(masks) == 1:
            return self.__validate_and_fix_mask(masks[0])

        # Convert all masks to Shapely polygons and fix them first
        polygons = []
        for mask in masks:
            poly = self.__sphere_mask_to_polygon(mask)
            fixed_poly = self.__fix_polygon(poly)
            if fixed_poly is not None:
                polygons.append(fixed_poly)

        if not polygons:
            # Fallback: return the mask with highest score
            return max(masks, key=lambda m: m.score)

        # Union all polygons
        try:
            union_poly = unary_union(polygons)

            # Fix the union result
            union_poly = self.__fix_polygon(union_poly)

            if union_poly is None:
                raise ValueError("Could not create valid union polygon")

            # Get exterior coordinates
            coords = list(union_poly.exterior.coords)[:-1]  # Remove closing point
            union_polygon = [(float(x), float(y)) for x, y in coords]

            # Calculate centroid
            centroid = union_poly.centroid
            center_yaw = float(centroid.x)
            center_pitch = float(centroid.y)

            # Use the best score among merged masks
            best_score = max(m.score for m in masks)

            # Use label from highest-scoring mask
            best_mask = max(masks, key=lambda m: m.score)

            return SphereMaskResult(
                polygon=union_polygon,
                score=best_score,
                label=best_mask.label,
                mask_id=f"{best_mask.mask_id}_merged",
                center_yaw=center_yaw,
                center_pitch=center_pitch,
            )
        except Exception:
            pass

        # Fallback: return the mask with highest score
        return max(masks, key=lambda m: m.score)

    def deduplicate_list(
        self, masks: List[SphereMaskResult], use_union: bool = True
    ) -> List[SphereMaskResult]:
        """Remove duplicates within a single list of masks using incremental merging.

        This handles objects that span multiple frames correctly by maintaining
        a master list and comparing each mask against it. When masks overlap,
        they are merged using polygon union to capture the full extent of the object.

        Args:
            masks: List of sphere masks.
            use_union: If True, merge overlapping masks using polygon union.
                      If False, keep only the highest-scoring mask.

        Returns:
            Deduplicated list of sphere masks.
        """
        if len(masks) <= 1:
            return list(masks)

        # Use incremental merging approach
        master_list: List[SphereMaskResult] = []

        for mask in masks:
            # Check if this mask overlaps with any mask in the master list
            overlapping_indices = []
            for i, master_mask in enumerate(master_list):
                if self.check_duplication(mask, master_mask):
                    overlapping_indices.append(i)

            if not overlapping_indices:
                # No overlap - add to master list
                master_list.append(mask)
            else:
                # Overlaps with one or more masks in master list
                candidates = [mask] + [master_list[i] for i in overlapping_indices]

                if use_union:
                    # Merge all overlapping masks using polygon union
                    merged_mask = self.__merge_masks(candidates)
                else:
                    # Just keep the best score
                    merged_mask = max(candidates, key=lambda m: m.score)

                # Remove all overlapping masks from master list (in reverse order)
                for i in sorted(overlapping_indices, reverse=True):
                    master_list.pop(i)

                # Add the merged mask
                master_list.append(merged_mask)

        # Final pass: validate and fix all masks
        return [self.__validate_and_fix_mask(m) for m in master_list]

    def deduplicate_frames(
        self, frames: List[List[SphereMaskResult]], use_union: bool = True
    ) -> List[SphereMaskResult]:
        """Deduplicate masks across multiple frames using incremental merging with union.

        Processes frames one by one, maintaining a master list. Each mask from
        a new frame is compared against the entire master list. When masks overlap,
        they are merged using polygon union to capture the full extent of objects
        that span multiple frames (e.g., a car visible in frames A, B, C where
        A and C see small parts and B sees the middle).

        Args:
            frames: List of frames, each containing a list of sphere masks.
            use_union: If True, merge overlapping masks using polygon union.
                      If False, keep only the highest-scoring mask.

        Returns:
            Deduplicated list of sphere masks.
        """
        if not frames:
            return []

        # Start with the first frame as the master list
        master_list: List[SphereMaskResult] = list(frames[0])

        # Process each subsequent frame
        for frame_idx, frame_masks in enumerate(frames[1:], start=1):
            for mask in frame_masks:
                # Check if this mask overlaps with any mask in the master list
                overlapping_indices = []
                for i, master_mask in enumerate(master_list):
                    if self.check_duplication(mask, master_mask):
                        overlapping_indices.append(i)

                if not overlapping_indices:
                    # No overlap - add to master list
                    master_list.append(mask)
                else:
                    # Overlaps with one or more masks
                    candidates = [mask] + [master_list[i] for i in overlapping_indices]

                    if use_union:
                        # Merge all overlapping masks using polygon union
                        merged_mask = self.__merge_masks(candidates)
                    else:
                        # Just keep the best score
                        merged_mask = max(candidates, key=lambda m: m.score)

                    # Remove all overlapping masks (in reverse order to preserve indices)
                    for i in sorted(overlapping_indices, reverse=True):
                        master_list.pop(i)

                    # Add the merged mask
                    master_list.append(merged_mask)

        # Final pass: validate and fix all masks
        return [self.__validate_and_fix_mask(m) for m in master_list]
