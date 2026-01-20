"""Sphere mask deduplication using GeoPandas polygon overlap."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm

from ..geometry import calculate_spherical_centroid
from ..sam.models import SphereMaskResult


# Maximum allowed yaw span for a single object (degrees).
# Any merged polygon spanning more than this is invalid (wrap-around error).
MAX_VALID_YAW_SPAN = 120.0


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
DEFAULT_MIN_IOU = 0.9
DEFAULT_MIN_INTERSECTION_RATIO = 0.9


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
        # Cache for pre-loaded GeoDataFrames keyed by mask identifier
        self._gdf_cache: dict[str, gpd.GeoDataFrame] = {}

    def _get_mask_key(self, mask: SphereMaskResult) -> str:
        """Generate a unique key for a mask based on its polygon content.

        Args:
            mask: The sphere mask.

        Returns:
            A unique string key for cache lookup.
        """
        # Use mask_id combined with polygons hash for uniqueness
        poly_hash = hash(tuple(tuple(p) for poly in mask.polygons for p in poly))
        return f"{mask.mask_id}_{poly_hash}"

    def _preload_gdfs(self, masks: List[SphereMaskResult]) -> None:
        """Pre-load all masks as GeoDataFrames into the cache.

        Args:
            masks: List of sphere masks to pre-load.
        """
        for mask in masks:
            key = self._get_mask_key(mask)
            if key not in self._gdf_cache:
                gdf = self._sphere_mask_to_gdf(mask)
                if gdf is not None:
                    self._gdf_cache[key] = gdf

    def _get_cached_gdf(self, mask: SphereMaskResult) -> Optional[gpd.GeoDataFrame]:
        """Get the GeoDataFrame for a mask from cache, or compute it.

        Args:
            mask: The sphere mask.

        Returns:
            GeoDataFrame or None if conversion failed.
        """
        key = self._get_mask_key(mask)
        if key in self._gdf_cache:
            return self._gdf_cache[key]
        # Fallback: compute and cache if not found
        gdf = self._sphere_mask_to_gdf(mask)
        if gdf is not None:
            self._gdf_cache[key] = gdf
        return gdf

    def _update_cache_for_merged(self, merged_mask: SphereMaskResult) -> None:
        """Add a newly merged mask to the cache.

        Args:
            merged_mask: The newly merged mask.
        """
        key = self._get_mask_key(merged_mask)
        gdf = self._sphere_mask_to_gdf(merged_mask)
        if gdf is not None:
            self._gdf_cache[key] = gdf

    def _clear_cache(self) -> None:
        """Clear the GeoDataFrame cache."""
        self._gdf_cache.clear()

    def _normalize_yaw(self, yaw: float) -> float:
        """Normalize yaw to [-180, 180) range."""
        while yaw >= 180:
            yaw -= 360
        while yaw < -180:
            yaw += 360
        return yaw

    def _shift_polygon_yaw(self, poly: Polygon, shift: float) -> Polygon:
        """Shift all yaw coordinates of a polygon by a given amount.

        Args:
            poly: Input polygon with (yaw, pitch) coordinates.
            shift: Amount to shift yaw by (degrees).

        Returns:
            New polygon with shifted yaw coordinates.
        """
        coords = list(poly.exterior.coords)
        shifted_coords = [(self._normalize_yaw(x + shift), y) for x, y in coords]
        return Polygon(shifted_coords)

    def _get_yaw_bounds(self, poly: Polygon) -> Tuple[float, float]:
        """Get min and max yaw of a polygon."""
        coords = list(poly.exterior.coords)
        yaws = [x for x, y in coords]
        return min(yaws), max(yaws)

    def _get_yaw_span(self, poly: Polygon) -> float:
        """Get the yaw span of a polygon."""
        min_yaw, max_yaw = self._get_yaw_bounds(poly)
        return max_yaw - min_yaw

    def _polygons_need_wrap_handling(self, polygons: List[Polygon]) -> bool:
        """Check if polygons need wrap-around handling.

        Returns True if polygons span across the ±180° boundary, which would
        cause incorrect results from standard polygon operations.
        """
        has_positive = False
        has_negative = False

        for poly in polygons:
            coords = list(poly.exterior.coords)
            for x, y in coords:
                if x > 90:
                    has_positive = True
                if x < -90:
                    has_negative = True

        # If we have polygons on both sides of the wrap boundary, we need handling
        return has_positive and has_negative

    def _sphere_mask_to_shapely(
        self, sphere_mask: SphereMaskResult
    ) -> Optional[MultiPolygon]:
        """Convert a sphere mask to Shapely MultiPolygon.

        Args:
            sphere_mask: The sphere mask result.

        Returns:
            Shapely MultiPolygon or None if invalid.
        """
        shapely_polygons = []

        for polygon_coords in sphere_mask.polygons:
            if len(polygon_coords) < 3:
                continue

            # Convert (yaw, pitch) tuples to Polygon
            points = [(yaw, pitch) for yaw, pitch in polygon_coords]

            try:
                polygon = Polygon(points)
                if not polygon.is_valid:
                    # Try to fix invalid polygon
                    polygon = polygon.buffer(0)
                if polygon.is_valid and not polygon.is_empty:
                    if polygon.geom_type == "Polygon":
                        shapely_polygons.append(polygon)
                    elif polygon.geom_type == "MultiPolygon":
                        shapely_polygons.extend(polygon.geoms)
            except Exception:
                continue

        if not shapely_polygons:
            return None

        return MultiPolygon(shapely_polygons)

    def _multipolygon_to_gdf(self, multi_polygon: MultiPolygon) -> gpd.GeoDataFrame:
        """Convert a Shapely MultiPolygon to a GeoDataFrame.

        Uses EPSG:4326 (WGS84) for geographic coordinates and converts
        to a cylindrical equal-area projection for accurate area calculations.

        Note: We use ESRI:54034 (World Cylindrical Equal Area) instead of
        EPSG:3857 (Web Mercator) because Web Mercator has severe area
        distortion at high latitudes. At pitch=45°, areas are ~2x inflated;
        near the poles it becomes extreme. This caused masks at high pitch
        values (upper/lower parts of panoramas) to be incorrectly merged
        because their projected areas were massively inflated.

        Args:
            multi_polygon: Shapely MultiPolygon.

        Returns:
            GeoDataFrame with the geometry.
        """
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[multi_polygon])
        # Use Cylindrical Equal Area projection for accurate area calculations
        # This preserves area regardless of latitude/pitch
        gdf = gdf.to_crs("ESRI:54034")
        return gdf

    def _sphere_mask_to_gdf(
        self, sphere_mask: SphereMaskResult
    ) -> Optional[gpd.GeoDataFrame]:
        """Convert a sphere mask to a GeoDataFrame.

        Args:
            sphere_mask: The sphere mask result.

        Returns:
            GeoDataFrame or None if conversion failed.
        """
        multi_polygon = self._sphere_mask_to_shapely(sphere_mask)
        if multi_polygon is None or multi_polygon.is_empty:
            return None
        return self._multipolygon_to_gdf(multi_polygon)

    def _get_intersection(
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

    def _intersect_masks(
        self,
        mask_1: SphereMaskResult,
        mask_2: SphereMaskResult,
    ) -> Optional[PolygonIntersection]:
        """Calculate intersection between two sphere masks.

        Uses cached GeoDataFrames if available for better performance.

        Args:
            mask_1: First sphere mask.
            mask_2: Second sphere mask.

        Returns:
            PolygonIntersection or None if no intersection.
        """
        gdf_1 = self._get_cached_gdf(mask_1)
        gdf_2 = self._get_cached_gdf(mask_2)

        if gdf_1 is None or gdf_2 is None:
            return None

        return self._get_intersection(gdf_1, gdf_2)

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
        intersection = self._intersect_masks(mask_1, mask_2)

        if intersection is None:
            return False

        # Check IoU threshold
        if intersection.iou >= self.min_iou:
            return True

        # Check intersection ratio threshold
        if intersection.intersection_ratio >= self.min_intersection_ratio:
            return True

        return False

    def _fix_polygon(self, poly) -> Optional[Polygon]:
        """Fix a potentially invalid polygon and return a valid simple polygon.

        Args:
            poly: Input polygon or geometry (may be invalid/self-intersecting).

        Returns:
            A valid simple Polygon, or None if unfixable.
        """
        if poly is None or poly.is_empty:
            return None

        # Only accept simple Polygons
        if poly.geom_type != "Polygon":
            return None

        # Make valid if needed (fixes self-intersections)
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Reject if still not a valid simple Polygon
        if poly.geom_type != "Polygon" or poly.is_empty or not poly.is_valid:
            return None

        return poly

    def _validate_and_fix_mask(self, mask: SphereMaskResult) -> SphereMaskResult:
        """Validate and fix a mask's polygons if needed.

        Args:
            mask: The sphere mask to validate/fix.

        Returns:
            The mask with valid polygons, or original if unfixable.
        """
        multi_poly = self._sphere_mask_to_shapely(mask)
        if multi_poly is None:
            return mask  # Return original if we can't fix it

        fixed_polygons = []
        for geom in multi_poly.geoms:
            fixed_poly = self._fix_polygon(geom)
            if fixed_poly is not None:
                coords = list(fixed_poly.exterior.coords)[:-1]
                fixed_polygons.append([(float(x), float(y)) for x, y in coords])

        if not fixed_polygons:
            return mask  # Return original if nothing fixed

        # Use proper spherical centroid (handles wrap-around at ±180°)
        center_yaw, center_pitch = calculate_spherical_centroid(fixed_polygons)
        return SphereMaskResult(
            polygons=fixed_polygons,
            score=mask.score,
            label=mask.label,
            mask_id=mask.mask_id,
            center_yaw=center_yaw,
            center_pitch=center_pitch,
        )

    def _merge_masks(
        self, masks: List[SphereMaskResult]
    ) -> Optional[SphereMaskResult]:
        """Merge multiple overlapping masks into one using polygon union.

        Handles spherical wrap-around at ±180° by shifting coordinates when needed.
        Returns None if the merge would create an invalid polygon (e.g., spanning
        more than MAX_VALID_YAW_SPAN degrees in yaw).

        The result may contain multiple polygons if the merged area is non-contiguous.

        Args:
            masks: List of overlapping sphere masks to merge.

        Returns:
            A single SphereMaskResult with the union polygon(s) and best score,
            or None if the merge is invalid.
        """
        if len(masks) == 1:
            return self._validate_and_fix_mask(masks[0])

        # Collect all Shapely polygons from all masks
        all_polygons = []
        for mask in masks:
            multi_poly = self._sphere_mask_to_shapely(mask)
            if multi_poly is not None:
                all_polygons.extend(multi_poly.geoms)

        if not all_polygons:
            # Fallback: return the mask with highest score
            return max(masks, key=lambda m: m.score)

        # Check if we need to handle wrap-around at ±180°
        needs_wrap_handling = self._polygons_need_wrap_handling(all_polygons)
        yaw_shift = 180.0 if needs_wrap_handling else 0.0

        # Shift polygons if needed to avoid wrap-around issues
        if yaw_shift != 0:
            all_polygons = [
                self._shift_polygon_yaw(p, yaw_shift) for p in all_polygons
            ]

        # Union all polygons
        try:
            union_result = unary_union(all_polygons)

            # Make valid if needed
            if not union_result.is_valid:
                union_result = union_result.buffer(0)

            if union_result.is_empty:
                raise ValueError("Empty union result")

            # Shift back if we shifted earlier
            result_polygons = []

            # Handle both Polygon and MultiPolygon results
            if union_result.geom_type == "Polygon":
                geoms = [union_result]
            elif union_result.geom_type == "MultiPolygon":
                geoms = list(union_result.geoms)
            else:
                # GeometryCollection or other - extract polygons
                geoms = [g for g in union_result.geoms if g.geom_type == "Polygon"]

            for geom in geoms:
                if yaw_shift != 0:
                    geom = self._shift_polygon_yaw(geom, -yaw_shift)

                # Validate the polygon doesn't span too much in yaw
                yaw_span = self._get_yaw_span(geom)
                if yaw_span > MAX_VALID_YAW_SPAN:
                    continue  # Skip invalid polygons but continue with others

                # Get exterior coordinates
                coords = list(geom.exterior.coords)[:-1]  # Remove closing point
                result_polygons.append([(float(x), float(y)) for x, y in coords])

            if not result_polygons:
                # All polygons were invalid - reject merge
                return None

            # Calculate centroid using proper spherical averaging
            center_yaw, center_pitch = calculate_spherical_centroid(result_polygons)

            # Use the best score among merged masks
            best_score = max(m.score for m in masks)

            # Use label from highest-scoring mask
            best_mask = max(masks, key=lambda m: m.score)

            # Concatenate mask_ids
            merged_ids = "&".join(m.mask_id for m in masks if m.mask_id)

            return SphereMaskResult(
                polygons=result_polygons,
                score=best_score,
                label=best_mask.label,
                mask_id=merged_ids,
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
        they can be merged using polygon union to capture the full extent of the object.

        Args:
            masks: List of sphere masks.
            use_union: If True, merge overlapping masks using polygon union.
                      If False, keep only the highest-scoring mask.

        Returns:
            Deduplicated list of sphere masks.
        """
        if len(masks) <= 1:
            return list(masks)

        # Pre-load all GeoDataFrames into cache for faster processing
        self._preload_gdfs(masks)

        try:
            # Use incremental merging approach
            master_list: List[SphereMaskResult] = []

            for mask in masks:
                # Check overlaps ONE BY ONE (not all at once)
                indices_to_remove = []
                masks_to_merge = [mask]
                new_mask_survives = True

                for i, master_mask in enumerate(master_list):
                    if not self.check_duplication(mask, master_mask):
                        continue

                    # Found an overlap - compare scores
                    if mask.score >= master_mask.score:
                        # New mask wins this comparison
                        indices_to_remove.append(i)
                        if use_union:
                            masks_to_merge.append(master_mask)
                    else:
                        # Existing mask wins
                        if use_union:
                            # Merge into existing mask
                            indices_to_remove.append(i)
                            masks_to_merge = [master_mask, mask]
                        else:
                            # Discard new mask
                            new_mask_survives = False
                        break

                if new_mask_survives:
                    # Remove all masks that are being merged
                    for i in sorted(indices_to_remove, reverse=True):
                        master_list.pop(i)

                    if use_union and len(masks_to_merge) > 1:
                        # Merge overlapping masks
                        merged = self._merge_masks(masks_to_merge)
                        if merged is not None:
                            master_list.append(merged)
                            self._update_cache_for_merged(merged)
                        else:
                            # Merge failed - keep the best scoring one
                            best = max(masks_to_merge, key=lambda m: m.score)
                            master_list.append(best)
                            self._update_cache_for_merged(best)
                    else:
                        # No union - just add the mask
                        master_list.append(mask)
                        self._update_cache_for_merged(mask)

            # Final pass: validate and fix all masks
            return [self._validate_and_fix_mask(m) for m in master_list]
        finally:
            # Clear cache to free memory
            self._clear_cache()

    def deduplicate_frames(
        self,
        frames: List[List[SphereMaskResult]],
        use_union: bool = True,
    ) -> List[SphereMaskResult]:
        """Deduplicate masks across multiple frames using incremental merging with union.

        Processes frames one by one, maintaining a master list. Each mask from
        a new frame is compared against the entire master list. When masks overlap,
        they can be merged using polygon union to capture the full extent of objects
        that span multiple frames.

        Args:
            frames: List of frames, each containing a list of sphere masks.
            use_union: If True, merge overlapping masks using polygon union.
                      If False, keep only the highest-scoring mask.

        Returns:
            Deduplicated list of sphere masks.
        """
        if not frames:
            return []

        # Pre-load all GeoDataFrames from all frames into cache
        all_masks = [mask for frame in frames for mask in frame]
        self._preload_gdfs(all_masks)

        try:
            # Start with the first frame as the master list
            master_list: List[SphereMaskResult] = list(frames[0])

            # Process each subsequent frame
            for frame_masks in tqdm(
                frames[1:],
                desc="Deduplicating",
                unit="frame",
                initial=1,
                total=len(frames),
            ):
                for mask in frame_masks:
                    # Check overlaps one by one
                    indices_to_remove = []
                    masks_to_merge = [mask]
                    new_mask_survives = True

                    for i, master_mask in enumerate(master_list):
                        if not self.check_duplication(mask, master_mask):
                            continue

                        # Found an overlap - compare scores
                        if mask.score >= master_mask.score:
                            # New mask wins this comparison
                            indices_to_remove.append(i)
                            if use_union:
                                masks_to_merge.append(master_mask)
                        else:
                            # Existing mask wins
                            if use_union:
                                # Merge into existing mask
                                indices_to_remove.append(i)
                                masks_to_merge = [master_mask, mask]
                            else:
                                new_mask_survives = False
                            break

                    if new_mask_survives:
                        # Remove all masks that are being merged
                        for i in sorted(indices_to_remove, reverse=True):
                            master_list.pop(i)

                        if use_union and len(masks_to_merge) > 1:
                            # Merge overlapping masks
                            merged = self._merge_masks(masks_to_merge)
                            if merged is not None:
                                master_list.append(merged)
                                self._update_cache_for_merged(merged)
                            else:
                                # Merge failed - keep the best scoring one
                                best = max(masks_to_merge, key=lambda m: m.score)
                                master_list.append(best)
                                self._update_cache_for_merged(best)
                        else:
                            # No union or single mask - just add the mask
                            master_list.append(mask)
                            self._update_cache_for_merged(mask)

            # Final pass: validate and fix all masks
            return [self._validate_and_fix_mask(m) for m in master_list]
        finally:
            # Clear cache to free memory
            self._clear_cache()
