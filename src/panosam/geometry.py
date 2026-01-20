"""Shared geometry and coordinate-conversion utilities.

This module contains math that is used by multiple subpackages (e.g. `sam` and
`dedup`) to avoid cross-module coupling.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def calculate_spherical_centroid(
    polygons: List[List[Tuple[float, float]]],
) -> Tuple[float, float]:
    """Calculate the centroid of spherical polygon(s) using 3D averaging.

    This handles wrap-around at ±180° correctly by converting to 3D Cartesian
    coordinates, averaging in 3D space, and converting back.

    Args:
        polygons: List of polygons, each polygon is a list of (yaw, pitch) tuples
            in degrees.

    Returns:
        (center_yaw, center_pitch) in degrees.
    """
    all_points = [pt for polygon in polygons for pt in polygon]
    if not all_points:
        return 0.0, 0.0

    sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
    for yaw_deg, pitch_deg in all_points:
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)

        # Spherical to Cartesian (pitch = latitude, yaw = longitude)
        # x = cos(pitch) * sin(yaw)  [East direction]
        # y = sin(pitch)             [Up direction]
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

    magnitude = math.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
    if magnitude < 1e-10:
        # Degenerate case: points are symmetrically distributed.
        center_yaw = sum(p[0] for p in all_points) / n
        center_pitch = sum(p[1] for p in all_points) / n
        return center_yaw, center_pitch

    avg_x /= magnitude
    avg_y /= magnitude
    avg_z /= magnitude

    center_yaw = math.degrees(math.atan2(avg_x, avg_z))
    center_pitch = math.degrees(math.asin(max(-1.0, min(1.0, avg_y))))

    return center_yaw, center_pitch


def perspective_to_sphere(
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
        Tuple of (yaw, pitch) in degrees.
    """
    # Convert to centered coordinates (-0.5 to 0.5)
    # x: positive = right, y: positive = up
    x = u - 0.5
    y = 0.5 - v

    half_h_fov = math.radians(horizontal_fov) / 2
    half_v_fov = math.radians(vertical_fov) / 2

    # Direction in camera local coordinates (camera looks along +Z)
    X_local = x * 2 * math.tan(half_h_fov)
    Y_local = y * 2 * math.tan(half_v_fov)
    Z_local = 1.0

    # Normalize to unit vector
    r = math.sqrt(X_local**2 + Y_local**2 + Z_local**2)
    X_local /= r
    Y_local /= r
    Z_local /= r

    # Rotate by camera orientation to get world coordinates
    pitch_rad = math.radians(pitch_offset)
    yaw_rad = math.radians(yaw_offset)

    cos_pitch = math.cos(pitch_rad)
    sin_pitch = math.sin(pitch_rad)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # Rotation by pitch (around X axis)
    X_pitched = X_local
    Y_pitched = Y_local * cos_pitch + Z_local * sin_pitch
    Z_pitched = -Y_local * sin_pitch + Z_local * cos_pitch

    # Rotation by yaw (around Y axis)
    X_world = X_pitched * cos_yaw + Z_pitched * sin_yaw
    Y_world = Y_pitched
    Z_world = -X_pitched * sin_yaw + Z_pitched * cos_yaw

    world_yaw = math.degrees(math.atan2(X_world, Z_world))
    world_pitch = math.degrees(math.asin(np.clip(Y_world, -1.0, 1.0)))

    return world_yaw, world_pitch

