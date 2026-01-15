# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Object placement and pose randomization utilities.
Handles randomizing object positions, rotations, and scales.
"""

import logging
import random

from pxr import Gf, Usd

from .transforms import TransformUtils

logger = logging.getLogger(__name__)


class ObjectPlacer:
    """Manages object placement and pose randomization."""

    @staticmethod
    def offset_range(
        range_coords: tuple[float, float, float, float, float, float],
        offset: tuple[float, float, float]
    ) -> tuple[float, float, float, float, float, float]:
        """
        Shift AABB by translation vector.

        Applies same offset to min and max corners, moving box without changing dimensions.

        Args:
            range_coords: AABB as (min_x, min_y, min_z, max_x, max_y, max_z) in meters
            offset: Translation vector as (offset_x, offset_y, offset_z) in meters

        Returns:
            Translated AABB as (new_min_x, new_min_y, new_min_z, new_max_x, new_max_y, new_max_z)

        Example:
            # Room bounds in room-local space
            room_bounds = (-2.0, -1.5, 0.0, 2.0, 1.5, 3.0)

            # Room is offset from world origin
            room_offset = (5.0, 3.0, 0.5)

            # Convert to world space
            world_bounds = ObjectPlacer.offset_range(room_bounds, room_offset)
            # Returns: (3.0, 1.5, 0.5, 7.0, 4.5, 3.5)
        """
        return (
            range_coords[0] + offset[0],  # min_x
            range_coords[1] + offset[1],  # min_y
            range_coords[2] + offset[2],  # min_z
            range_coords[3] + offset[0],  # max_x
            range_coords[4] + offset[1],  # max_y
            range_coords[5] + offset[2],  # max_z
        )

    @staticmethod
    def randomize_poses(
        prims: list[Usd.Prim],
        location_range: tuple[float, float, float, float, float, float],
        rotation_range: tuple[float, float],
        scale_range: tuple[float, float]
    ) -> None:
        """
        Apply random position/rotation/scale to prims within AABB.

        Samples uniform 6-DOF pose + scale for each prim independently. Simpler than randomize_poses_with_surfaces.

        Args:
            prims: List of USD prims to transform (typically distractor objects)
            location_range: Placement AABB as (min_x, min_y, min_z, max_x, max_y, max_z) in meters
            rotation_range: Euler angle range as (min_deg, max_deg) applied to all three axes
            scale_range: Uniform scale factor range as (min_scale, max_scale)
                        1.0 = original size, 0.5 = half size, 2.0 = double size

        Example:
            # Place 10 shape distractors randomly in working area
            ObjectPlacer.randomize_poses(
                prims=distractor_shapes,
                location_range=(-2, -1.5, 0.5, 2, 1.5, 2.0),  # 4x3x1.5m volume
                rotation_range=(0, 360),  # Full random orientation
                scale_range=(0.05, 0.15)  # Small objects (5-15cm)
            )
        """
        for prim in prims:
            rand_loc = Gf.Vec3d(
                random.uniform(location_range[0], location_range[3]),
                random.uniform(location_range[1], location_range[4]),
                random.uniform(location_range[2], location_range[5]),
            )
            rand_rot = Gf.Vec3f(
                random.uniform(rotation_range[0], rotation_range[1]),
                random.uniform(rotation_range[0], rotation_range[1]),
                random.uniform(rotation_range[0], rotation_range[1]),
            )
            rand_scale = random.uniform(scale_range[0], scale_range[1])
            TransformUtils.set_transform_attributes(
                prim,
                location=rand_loc,
                rotation=rand_rot,
                scale=Gf.Vec3f(rand_scale, rand_scale, rand_scale)
            )

    @staticmethod
    def randomize_poses_with_surfaces(
        prims: list[Usd.Prim],
        floor_bounds: tuple[float, float, float, float, float, float],
        furniture_surfaces: list[dict],
        surface_placement_ratio: float = 0.3,
        rotation_range: tuple[float, float] = (0, 360),
        scale_range: tuple[float, float] = (0.95, 1.15),
        surface_height_offset: float = 0.02,
        description: str = "objects"
    ) -> None:
        """
        Place objects on floor and furniture surfaces with physics-aware positioning.

        Splits objects by ratio, places on furniture (Z-rotation only) or floor (full rotation) with bbox-based spawn heights.
        Accounts for object dimensions to prevent clipping.

        Args:
            prims: List of USD prims to place (typically target assets like YCB objects)
            floor_bounds: Room floor AABB as (min_x, min_y, min_z, max_x, max_y, max_z) in meters
            furniture_surfaces: List of surface dicts from EnvironmentAnalyzer.find_furniture_surfaces()
                               Each dict contains: 'bounds', 'surface_height', 'area'
            surface_placement_ratio: Fraction (0.0-1.0) of objects to place on furniture surfaces
                                    0.0 = all on floor, 1.0 = all on furniture, 0.3 = 30% on furniture
            rotation_range: Euler angle range in degrees as (min, max)
                           Surface objects: only Z-axis uses this
                           Floor objects: all axes use this
            scale_range: Uniform scale factor range as (min, max)
                        Example: (0.95, 1.15) = 95-115% of original size (±5% variation)
            surface_height_offset: Vertical offset above surface in meters (e.g., 0.02 = 2cm clearance)
            description: Object type name for logging (e.g., "target objects", "labeled assets")
        """
        if not prims:
            return

        num_surface_objects = int(len(prims) * surface_placement_ratio)
        num_floor_objects = len(prims) - num_surface_objects

        shuffled_prims = list(prims)
        random.shuffle(shuffled_prims)

        surface_objects = shuffled_prims[:num_surface_objects]
        floor_objects = shuffled_prims[num_surface_objects:]

        if surface_objects and furniture_surfaces:
            logger.info(f"Placing {len(surface_objects)} {description} on furniture surfaces")
            for prim in surface_objects:
                surface = random.choice(furniture_surfaces)
                min_x, min_y, max_x, max_y = surface['bounds']
                rand_x = random.uniform(min_x, max_x)
                rand_y = random.uniform(min_y, max_y)
                rand_z = surface['surface_height'] + surface_height_offset

                rand_loc = Gf.Vec3d(rand_x, rand_y, rand_z)
                rand_rot = Gf.Vec3f(0, 0, random.uniform(rotation_range[0], rotation_range[1]))
                rand_scale = random.uniform(scale_range[0], scale_range[1])

                TransformUtils.set_transform_attributes(
                    prim,
                    location=rand_loc,
                    rotation=rand_rot,
                    scale=Gf.Vec3f(rand_scale, rand_scale, rand_scale)
                )
        elif surface_objects:
            logger.info(f"Warning: {len(surface_objects)} {description} intended for surfaces, but no surfaces found. Placing on floor instead.")
            floor_objects.extend(surface_objects)

        if floor_objects:
            logger.info(f"Placing {len(floor_objects)} {description} on floor")

            floor_spawn_offset = 0.05
            horizontal_buffer = 0.1

            for prim in floor_objects:
                bbox = TransformUtils.get_prim_bounding_box(prim)
                if bbox:
                    bbox_min, bbox_max = bbox
                    object_half_height = (bbox_max[2] - bbox_min[2]) / 2
                    object_half_width = max((bbox_max[0] - bbox_min[0]) / 2, (bbox_max[1] - bbox_min[1]) / 2)
                else:
                    object_half_height = 0.05
                    object_half_width = 0.05
                spawn_min_x = floor_bounds[0] + horizontal_buffer + object_half_width
                spawn_max_x = floor_bounds[3] - horizontal_buffer - object_half_width
                spawn_min_y = floor_bounds[1] + horizontal_buffer + object_half_width
                spawn_max_y = floor_bounds[4] - horizontal_buffer - object_half_width

                if spawn_min_x >= spawn_max_x or spawn_min_y >= spawn_max_y:
                    logger.warning(f"Object too large for floor bounds, placing at center")
                    spawn_x = (floor_bounds[0] + floor_bounds[3]) / 2
                    spawn_y = (floor_bounds[1] + floor_bounds[4]) / 2
                else:
                    spawn_x = random.uniform(spawn_min_x, spawn_max_x)
                    spawn_y = random.uniform(spawn_min_y, spawn_max_y)

                rand_loc = Gf.Vec3d(
                    spawn_x,
                    spawn_y,
                    floor_bounds[2] + object_half_height + floor_spawn_offset
                )

                rand_rot = Gf.Vec3f(
                    random.uniform(rotation_range[0], rotation_range[1]),
                    random.uniform(rotation_range[0], rotation_range[1]),
                    random.uniform(rotation_range[0], rotation_range[1])
                )

                rand_scale = random.uniform(scale_range[0], scale_range[1])

                TransformUtils.set_transform_attributes(
                    prim,
                    location=rand_loc,
                    rotation=rand_rot,
                    scale=Gf.Vec3f(rand_scale, rand_scale, rand_scale)
                )