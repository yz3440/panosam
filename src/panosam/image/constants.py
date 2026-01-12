from .models import PerspectiveMetadata
from typing import List


def initialize_default_perspectives() -> List[PerspectiveMetadata]:
    """Initialize default perspective configurations.

    Creates 16 perspectives with:
    - 2048x2048 pixel resolution
    - 45° FOV
    - 22.5° yaw intervals
    - 0° pitch
    """
    PIXEL_SIZE = 2048

    pixel_width = PIXEL_SIZE
    pixel_height = PIXEL_SIZE

    perspectives = []
    horizontal_fov = 45
    vertical_fov = 45

    yaw_offset_count = round(360 / horizontal_fov * 2)
    yaw_angle_offsets = []
    interval = 360 / yaw_offset_count
    for k in range(yaw_offset_count):
        yaw_angle_offsets.append(k * interval - 180)
    pitch_angle_offsets = [0]

    for yaw_angle_offset in yaw_angle_offsets:
        for pitch_angle_offset in pitch_angle_offsets:
            perspective = PerspectiveMetadata(
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                horizontal_fov=horizontal_fov,
                vertical_fov=vertical_fov,
                yaw_offset=yaw_angle_offset,
                pitch_offset=pitch_angle_offset,
            )
            perspectives.append(perspective)
    return perspectives


DEFAULT_IMAGE_PERSPECTIVES = initialize_default_perspectives()
"""List[PerspectiveMetadata]: Default perspectives with 45° FOV and 22.5° yaw intervals."""


def initialize_zoomed_in_perspectives() -> List[PerspectiveMetadata]:
    """Initialize zoomed-in perspective configurations.

    Creates 32 perspectives with:
    - 1024x1024 pixel resolution
    - 22.5° FOV
    - 11.25° yaw intervals
    - 0° pitch
    """
    PIXEL_SIZE = 1024

    pixel_width = PIXEL_SIZE
    pixel_height = PIXEL_SIZE

    perspectives = []
    horizontal_fov = 22.5
    vertical_fov = 22.5

    yaw_offset_count = round(360 / horizontal_fov * 2)
    yaw_angle_offsets = []
    interval = 360 / yaw_offset_count
    for k in range(yaw_offset_count):
        yaw_angle_offsets.append(k * interval - 180)
    pitch_angle_offsets = [0]

    for yaw_angle_offset in yaw_angle_offsets:
        for pitch_angle_offset in pitch_angle_offsets:
            perspective = PerspectiveMetadata(
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                horizontal_fov=horizontal_fov,
                vertical_fov=vertical_fov,
                yaw_offset=yaw_angle_offset,
                pitch_offset=pitch_angle_offset,
            )
            perspectives.append(perspective)
    return perspectives


ZOOMED_IN_IMAGE_PERSPECTIVES = initialize_zoomed_in_perspectives()
"""List[PerspectiveMetadata]: Zoomed-in perspectives with 22.5° FOV."""


def initialize_zoomed_out_perspectives() -> List[PerspectiveMetadata]:
    """Initialize zoomed-out perspective configurations.

    Creates 12 perspectives with:
    - 2500x2500 pixel resolution
    - 60° FOV
    - 30° yaw intervals
    - 0° pitch
    """
    PIXEL_SIZE = 2500

    pixel_width = PIXEL_SIZE
    pixel_height = PIXEL_SIZE

    perspectives = []
    horizontal_fov = 60
    vertical_fov = 60

    yaw_offset_count = round(360 / horizontal_fov * 2)
    yaw_angle_offsets = []
    interval = 360 / yaw_offset_count
    for k in range(yaw_offset_count):
        yaw_angle_offsets.append(k * interval - 180)
    pitch_angle_offsets = [0]

    for yaw_angle_offset in yaw_angle_offsets:
        for pitch_angle_offset in pitch_angle_offsets:
            perspective = PerspectiveMetadata(
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                horizontal_fov=horizontal_fov,
                vertical_fov=vertical_fov,
                yaw_offset=yaw_angle_offset,
                pitch_offset=pitch_angle_offset,
            )
            perspectives.append(perspective)
    return perspectives


ZOOMED_OUT_IMAGE_PERSPECTIVES = initialize_zoomed_out_perspectives()
"""List[PerspectiveMetadata]: Zoomed-out perspectives with 60° FOV."""
