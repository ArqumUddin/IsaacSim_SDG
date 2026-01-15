# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Environment analysis utilities.
Handles room bounds calculation, wall detection, and furniture surface finding.
"""

import logging

import omni.usd
from pxr import Gf

from .environment_setup import EnvironmentSetup
from .transforms import TransformUtils

logger = logging.getLogger(__name__)


class EnvironmentAnalyzer:
    """Analyzes environment geometry for navigation and placement."""

    @staticmethod
    def get_matching_prim_location(match_string: str, root_path: str | None = None) -> tuple[float, float, float]:
        """
        Get world-space position of first prim matching name substring.

        Searches stage for prim containing match_string, extracts translation from xformOp:translate or xformOp:transform.

        Args:
            match_string: Substring to match in prim path (case-sensitive)
                         Example: "spawn" matches "/Environment/spawn_point"
            root_path: Optional root path to narrow search scope (e.g., "/Environment")
                      If None, searches entire stage

        Returns:
            Tuple of (x, y, z) world coordinates in meters
            Returns (0, 0, 0) if no matching prim found or transform unavailable
        """
        prim = EnvironmentSetup.find_matching_prims(
            match_strings=[match_string], root_path=root_path, prim_type="Xform", first_match_only=True
        )
        if prim is None:
            logger.info(f"Could not find matching prim, returning (0, 0, 0)")
            return (0, 0, 0)
        if prim.HasAttribute("xformOp:translate"):
            return prim.GetAttribute("xformOp:translate").Get()
        elif prim.HasAttribute("xformOp:transform"):
            return prim.GetAttribute("xformOp:transform").Get().ExtractTranslation()
        else:
            logger.info(f"Could not find location attribute for '{prim.GetPath()}', returning (0, 0, 0)")
            return (0, 0, 0)

    @staticmethod
    def get_surface_height(
        match_string: str,
        offset: float,
        add_offset: bool = True,
        root_path: str | None = None,
        default_value: float | None = None
    ) -> float | None:
        """
        Calculate camera height constraint from surface prim Z-coordinate.

        Finds surface prim, applies offset: add_offset=True for floor (min height), False for ceiling (max height).
        Prevents camera clipping through surfaces.

        Args:
            match_string: Substring to match in prim path (case-sensitive)
                         Examples: "floor", "Floor", "ceiling", "Ceiling"
            offset: Clearance distance from surface in meters (e.g., 0.2 for 20cm buffer)
            add_offset: If True, adds offset (for floor/min). If False, subtracts (for ceiling/max)
            root_path: Optional root path to narrow search (e.g., "/Environment")
            default_value: Fallback constraint if surface prim not found (None = no constraint)

        Returns:
            Height constraint in meters (world Z-coordinate), or default_value if surface not found
            None if surface not found and no default_value provided
        """
        surface_prim = EnvironmentSetup.find_matching_prims(
            match_strings=[match_string], root_path=root_path, prim_type="Xform", first_match_only=True
        )

        if surface_prim is None:
            if default_value is not None:
                logger.info(f"Warning: Could not find {match_string} prim. Using default value of {default_value}")
            else:
                logger.info(f"Warning: Could not find {match_string} prim. No constraint will be applied.")
            return default_value

        if surface_prim.HasAttribute("xformOp:translate"):
            surface_loc = surface_prim.GetAttribute("xformOp:translate").Get()
        elif surface_prim.HasAttribute("xformOp:transform"):
            surface_loc = surface_prim.GetAttribute("xformOp:transform").Get().ExtractTranslation()
        else:
            if default_value is not None:
                logger.info(f"Warning: {match_string.capitalize()} prim found but no transform attribute. Using default value of {default_value}")
            else:
                logger.info(f"Warning: {match_string.capitalize()} prim found but no transform attribute. No constraint will be applied.")
            return default_value

        constraint_height = surface_loc[2] + offset if add_offset else surface_loc[2] - offset
        constraint_type = "min" if add_offset else "max"
        logger.info(f"{match_string.capitalize()} detected at Z={surface_loc[2]:.3f}, setting {constraint_type} camera height to {constraint_height:.3f}")
        return constraint_height

    @staticmethod
    def get_wall_data(root_path: str = "/Environment", min_wall_size: float = 0.5) -> list[dict]:
        """
        Extract wall geometry for collision detection and camera bouncing.

        Scans for wall prims, computes bbox/center/inward-facing normal for each, filters exterior walls and decorative elements.
        Normals validated to point toward room interior for correct camera reflection.

        Args:
            root_path: Root path to search for wall prims (typically "/Environment")
            min_wall_size: Minimum dimension in meters to qualify as a wall (filters decorative elements)
                          Applied to both width (horizontal extent) and height (vertical extent)

        Returns:
            List of wall data dictionaries, one per qualifying wall:
                - 'prim': Usd.Prim reference to the wall prim
                - 'center': Gf.Vec3d wall center point in world coordinates
                - 'normal': Gf.Vec3d inward-facing normal vector (normalized, points toward room center)
                - 'bbox_min': Gf.Vec3d bounding box minimum corner (world coordinates)
                - 'bbox_max': Gf.Vec3d bounding box maximum corner (world coordinates)
                - 'width': float horizontal extent (max of X or Y dimension)
                - 'height': float vertical extent (Z dimension)
        """
        stage = omni.usd.get_context().get_stage()

        env_prim = stage.GetPrimAtPath(root_path)
        if env_prim:
            logger.debug(f"Top-level children under {root_path}:")
            for child in env_prim.GetChildren():
                name = child.GetName()
                prim_type = child.GetTypeName()
                logger.debug(f"  {name} ({prim_type})")
                if child.GetChildren():
                    for grandchild in list(child.GetChildren())[:5]:
                        gc_name = grandchild.GetName()
                        gc_type = grandchild.GetTypeName()
                        logger.debug(f"    └─ {gc_name} ({gc_type})")
                    if len(list(child.GetChildren())) > 5:
                        logger.debug(f"    └─ ... and {len(list(child.GetChildren())) - 5} more")

        wall_prims = EnvironmentSetup.find_matching_prims(
            match_strings=["wall", "Wall", "WALL"], root_path=root_path, prim_type="Xform", first_match_only=False
        )

        logger.debug(f"find_matching_prims returned {len(wall_prims) if wall_prims else 0} wall prim(s)")
        if wall_prims:
            for wp in wall_prims[:5]:
                logger.debug(f"  Found wall prim: {wp.GetPath()}")

        if not wall_prims:
            logger.info(f"Warning: No wall prims found at '{root_path}'")
            return []

        wall_data = []

        stage_center = Gf.Vec3d(0, 0, 0)

        for wall_prim in wall_prims:
            wall_path = str(wall_prim.GetPath())
            if "_exterior" in wall_path.lower():
                logger.debug(f"Skipping exterior wall: {wall_path}")
                continue

            bbox = TransformUtils.get_prim_bounding_box(wall_prim)
            if bbox is None:
                logger.debug(f"Skipping wall (no bbox): {wall_path}")
                continue

            bbox_min, bbox_max = bbox

            width = max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1])
            height = bbox_max[2] - bbox_min[2]

            if width < min_wall_size or height < min_wall_size:
                logger.debug(f"Skipping wall (too small: w={width:.2f}, h={height:.2f}): {wall_path}")
                continue

            center = Gf.Vec3d(
                (bbox_min[0] + bbox_max[0]) / 2,
                (bbox_min[1] + bbox_max[1]) / 2,
                (bbox_min[2] + bbox_max[2]) / 2
            )

            transform = TransformUtils.get_prim_world_transform(wall_prim)
            if transform is None:
                to_center = stage_center - center
                normal = Gf.Vec3d(to_center[0], to_center[1], 0).GetNormalized()
            else:
                normal = TransformUtils.extract_normal_from_transform(transform, axis=0)
                to_center = stage_center - center
                if normal.GetDot(to_center) < 0:
                    normal = -normal

            wall_data.append({
                'prim': wall_prim,
                'center': center,
                'normal': normal,
                'bbox_min': bbox_min,
                'bbox_max': bbox_max,
                'width': width,
                'height': height
            })
            logger.debug(f"Successfully added wall: {wall_path} (w={width:.2f}, h={height:.2f})")

        logger.info(f"Found {len(wall_data)} wall(s) in environment")
        return wall_data

    @staticmethod
    def calculate_room_bounds(
        wall_data: list[dict],
        floor_height: float,
        ceiling_height: float | None,
        wall_clearance: float = 0.3,
        root_path: str = "/Environment"
    ) -> tuple[float, float, float, float, float, float]:
        """
        Compute navigable AABB for camera/object placement.

        Calculates bounds from wall positions with clearance buffer, optionally constrained by floor bbox.
        Z bounds from provided floor/ceiling heights.

        Args:
            wall_data: Wall geometry from get_wall_data() - list of wall dicts with 'center' keys
            floor_height: Minimum Z-coordinate (typically from get_surface_height("floor"))
            ceiling_height: Maximum Z-coordinate (typically from get_surface_height("ceiling"))
                           If None, defaults to floor_height + 3.0 meters
            wall_clearance: Inward buffer from wall centers in meters (e.g., 0.3 = 30cm clearance)
            root_path: Root path to search for floor prim (used as secondary constraint)

        Returns:
            Tuple of (min_x, min_y, min_z, max_x, max_y, max_z) defining the navigable AABB
            All coordinates in world space (meters)

        """
        if not wall_data:
            logger.info(f"Warning: No wall data provided, using default 10m x 10m area")
            wall_min_x, wall_min_y = -5.0, -5.0
            wall_max_x, wall_max_y = 5.0, 5.0
        else:
            wall_min_x = min(wall['center'][0] for wall in wall_data) + wall_clearance
            wall_max_x = max(wall['center'][0] for wall in wall_data) - wall_clearance
            wall_min_y = min(wall['center'][1] for wall in wall_data) + wall_clearance
            wall_max_y = max(wall['center'][1] for wall in wall_data) - wall_clearance

        floor_min_x, floor_min_y = -100.0, -100.0
        floor_max_x, floor_max_y = 100.0, 100.0

        floor_prim = EnvironmentSetup.find_matching_prims(
            match_strings=["floor", "Floor", "FLOOR"], root_path=root_path, prim_type="Xform", first_match_only=True
        )

        if floor_prim:
            floor_bbox = TransformUtils.get_prim_bounding_box(floor_prim)
            if floor_bbox:
                f_min, f_max = floor_bbox
                floor_buffer = 0.1
                floor_min_x, floor_min_y = f_min[0] + floor_buffer, f_min[1] + floor_buffer
                floor_max_x, floor_max_y = f_max[0] - floor_buffer, f_max[1] - floor_buffer
                logger.info(f"Floor bounds found: X=[{floor_min_x:.2f}, {floor_max_x:.2f}], Y=[{floor_min_y:.2f}, {floor_max_y:.2f}]")

        if wall_data:
            min_x, max_x = wall_min_x, wall_max_x
            min_y, max_y = wall_min_y, wall_max_y
            logger.info(f"Using wall-based bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")

            if floor_prim and floor_min_x != -100.0:
                potential_min_x = max(wall_min_x, floor_min_x)
                potential_max_x = min(wall_max_x, floor_max_x)
                potential_min_y = max(wall_min_y, floor_min_y)
                potential_max_y = min(wall_max_y, floor_max_y)

                if potential_min_x < potential_max_x and potential_min_y < potential_max_y:
                    min_x, max_x = potential_min_x, potential_max_x
                    min_y, max_y = potential_min_y, potential_max_y
                    logger.info(f"Tightened with floor: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")
        elif floor_prim:
            min_x, max_x = floor_min_x, floor_max_x
            min_y, max_y = floor_min_y, floor_max_y
            logger.warning(f"No walls found, using floor bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")
        else:
            logger.error("No walls or floor found! Using default bounds.")
            min_x, max_x = -5.0, 5.0
            min_y, max_y = -5.0, 5.0

        min_z = floor_height
        max_z = ceiling_height if ceiling_height is not None else floor_height + 3.0

        room_width = max_x - min_x
        room_depth = max_y - min_y
        room_height = max_z - min_z
        logger.info(f"Final Room bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}], Z=[{min_z:.2f}, {max_z:.2f}]")
        logger.info(f"Room dimensions: {room_width:.2f}m (W) x {room_depth:.2f}m (D) x {room_height:.2f}m (H)")

        return (min_x, min_y, min_z, max_x, max_y, max_z)

    @staticmethod
    def find_furniture_surfaces(
        root_path: str = "/Environment",
        furniture_keywords: list[str] | None = None,
        min_surface_area: float = 0.1
    ) -> list[dict]:
        """
        Find horizontal furniture surfaces for object placement.

        Scans for furniture prims, extracts top surface bbox (bbox_max Z), filters by min area.
        Shrinks bounds by 5cm margin for stable placement.

        Args:
            root_path: Root path to search for furniture prims (typically "/Environment")
            furniture_keywords: List of name substrings to match (case-sensitive)
                               Defaults to: ["Table", "Counter", "Shelf", "Desk", "Cabinet"]
                               Matches any prim whose path contains one of these strings
            min_surface_area: Minimum horizontal area in m² to qualify as placement surface
                             Filters out small items like coasters, picture frames, etc.
                             Example: 0.1 = surfaces must be at least 10cm × 10cm (0.1m²)

        Returns:
            List of surface data dictionaries, one per qualifying furniture piece:
                - 'prim': Usd.Prim reference to the furniture prim
                - 'surface_height': float Z-coordinate of top surface in world space (meters)
                - 'bounds': tuple (min_x, min_y, max_x, max_y) defining placement area
                           Shrunk by 5cm margin for stability
                - 'center': Gf.Vec3d center point of top surface (x, y, surface_height)
                - 'area': float horizontal surface area in square meters
        """
        if furniture_keywords is None:
            furniture_keywords = ["Table", "Counter", "Shelf", "Desk", "Cabinet"]

        logger.debug(f"Searching for furniture with keywords: {furniture_keywords}")

        furniture_surfaces = []

        for keyword in furniture_keywords:
            matching_prims = EnvironmentSetup.find_matching_prims(
                match_strings=[keyword], root_path=root_path, prim_type="Xform", first_match_only=False
            )

            if not matching_prims:
                logger.debug(f"No prims found matching '{keyword}'")
                continue
            else:
                logger.debug(f"Found {len(matching_prims)} prim(s) matching '{keyword}'")

            for furniture_prim in matching_prims:
                furniture_path = str(furniture_prim.GetPath())
                bbox = TransformUtils.get_prim_bounding_box(furniture_prim)
                if bbox is None:
                    logger.debug(f"Skipping furniture (no bbox): {furniture_path}")
                    continue

                bbox_min, bbox_max = bbox
                width = bbox_max[0] - bbox_min[0]
                depth = bbox_max[1] - bbox_min[1]
                surface_area = width * depth
                if surface_area < min_surface_area:
                    logger.debug(f"Skipping furniture (area too small: {surface_area:.3f}m²): {furniture_path}")
                    continue

                surface_height = bbox_max[2]
                margin = 0.05
                bounds = (
                    bbox_min[0] + margin,
                    bbox_min[1] + margin,
                    bbox_max[0] - margin,
                    bbox_max[1] - margin
                )
                center = Gf.Vec3d(
                    (bbox_min[0] + bbox_max[0]) / 2,
                    (bbox_min[1] + bbox_max[1]) / 2,
                    surface_height
                )

                furniture_surfaces.append({
                    'prim': furniture_prim,
                    'surface_height': surface_height,
                    'bounds': bounds,
                    'center': center,
                    'area': surface_area
                })
                logger.debug(f"Successfully added furniture: {furniture_path} (area={surface_area:.3f}m²)")

        logger.info(f"Found {len(furniture_surfaces)} furniture surface(s) for object placement")
        return furniture_surfaces
