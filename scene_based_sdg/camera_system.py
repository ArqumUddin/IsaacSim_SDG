# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Camera positioning system for synthetic data generation.
Supports both Brownian motion traversal and object-centric positioning.
"""

import logging
import math
import random

from pxr import Gf, Usd, UsdGeom

from scene_based_sdg.transforms import TransformUtils

logger = logging.getLogger(__name__)


class CameraSystem:
    """
    Manages camera positioning for synthetic data generation.

    Supports two positioning strategies:
    - Brownian motion: Random walk through environment for scene exploration
    - Object-centric: Spherical orbit around target objects for focused captures
    """

    @staticmethod
    def raycast_to_walls_2d(
        start_pos: tuple[float, float, float],
        direction_vector: Gf.Vec3d,
        wall_data: list[dict],
        wall_buffer: float,
        max_distance: float = 100.0
    ) -> tuple[bool, float, Gf.Vec3d | None, dict | None]:
        """
        Perform 2D ray-box intersection test for wall collision detection.

        Projects 3D ray onto XY plane and tests intersection with wall AABBs using slab method.
        Wall bounding boxes are expanded by wall_buffer to create early collision detection.

        Args:
            start_pos: Camera position (x, y, z) in world coordinates
            direction_vector: Movement direction as 3D vector
            wall_data: List of wall dictionaries from EnvironmentAnalyzer.get_wall_data()
                       Each dict contains: bbox_min, bbox_max, normal
            wall_buffer: Safety distance from walls (meters) - expands bounding boxes
            max_distance: Maximum raycast distance (meters)

        Returns:
            Tuple of (hit_detected, hit_distance, wall_normal, hit_wall_data):
            - hit_detected: True if ray intersects any wall within max_distance
            - hit_distance: Distance to closest intersection (or max_distance if no hit)
            - wall_normal: Outward-facing normal vector of hit wall (None if no hit)
            - hit_wall_data: Dictionary of the hit wall (None if no hit)
        """
        ray_dir_2d = Gf.Vec2d(direction_vector[0], direction_vector[1])
        if ray_dir_2d.GetLength() < 1e-8:
            return (False, 0.0, None, None)
        ray_dir_2d = ray_dir_2d.GetNormalized()
        ray_origin_2d = Gf.Vec2d(start_pos[0], start_pos[1])

        closest_hit_distance = max_distance
        closest_wall = None
        closest_normal = None

        for wall in wall_data:
            bbox_min_2d = Gf.Vec2d(
                wall['bbox_min'][0] - wall_buffer,
                wall['bbox_min'][1] - wall_buffer
            )
            bbox_max_2d = Gf.Vec2d(
                wall['bbox_max'][0] + wall_buffer,
                wall['bbox_max'][1] + wall_buffer
            )

            t_min = 0.0
            t_max = max_distance

            valid_intersection = True
            for i in range(2):
                if abs(ray_dir_2d[i]) < 1e-8:
                    if ray_origin_2d[i] < bbox_min_2d[i] or ray_origin_2d[i] > bbox_max_2d[i]:
                        valid_intersection = False
                        break
                else:
                    t1 = (bbox_min_2d[i] - ray_origin_2d[i]) / ray_dir_2d[i]
                    t2 = (bbox_max_2d[i] - ray_origin_2d[i]) / ray_dir_2d[i]

                    if t1 > t2:
                        t1, t2 = t2, t1

                    t_min = max(t_min, t1)
                    t_max = min(t_max, t2)

                    if t_min > t_max:
                        valid_intersection = False
                        break

            if valid_intersection and t_min <= t_max and t_min < closest_hit_distance and t_min > 0:
                closest_hit_distance = t_min
                closest_wall = wall
                closest_normal = Gf.Vec3d(wall['normal'][0], wall['normal'][1], 0).GetNormalized()

        if closest_wall is not None:
            return (True, closest_hit_distance, closest_normal, closest_wall)
        else:
            return (False, closest_hit_distance, None, None)

    @staticmethod
    def reflect_vector(incident: Gf.Vec3d, normal: Gf.Vec3d) -> Gf.Vec3d:
        """
        Reflect incident vector off surface using formula: r = i - 2(i·n)n.

        Wall normals from EnvironmentAnalyzer are inward-facing, so this negates them before applying reflection.

        Args:
            incident: Incoming direction vector (typically normalized from camera movement)
            normal: Surface normal vector from wall_data (inward-facing convention)

        Returns:
            Reflected direction vector (normalized) representing new movement direction
        """
        outward_normal = -normal
        dot_product = incident.GetDot(outward_normal)
        reflected = incident - 2.0 * dot_product * outward_normal
        return reflected.GetNormalized()

    @staticmethod
    def generate_brownian_motion_path(
        room_bounds: tuple[float, float, float, float, float, float],
        num_steps: int,
        step_distance: float,
        height: float,
        direction_change_range: tuple[float, float],
        wall_buffer: float,
        bounce_angle_range: tuple[float, float],
        initial_direction: str | float,
        floor_height: float,
        wall_data: list[dict] | None = None
    ) -> list[dict]:
        """
        Generate random walk camera path with wall bouncing.

        Simulates 2D random walk (constant height) with raycasting for wall collision or AABB fallback.
        Each step applies random direction change and moves forward, bouncing off walls using vector reflection.

        Args:
            room_bounds: Room AABB (min_x, min_y, min_z, max_x, max_y, max_z) in world coordinates
            num_steps: Number of waypoints to generate (typically equals num_captures_per_env)
            step_distance: Distance traveled per step in meters (e.g., 0.1 = 10cm steps)
            height: Camera height above floor_height in meters (e.g., 0.6 = 60cm for robot POV)
            direction_change_range: Random direction change per step in degrees (e.g., (-10, 10))
            wall_buffer: Safety distance from walls in meters (e.g., 0.3 = 30cm clearance)
            bounce_angle_range: Random perturbation added to reflection angle in degrees (e.g., (30, 60))
            initial_direction: Starting direction - "random" for 0-360°, or specific angle in degrees
            floor_height: Actual floor Z-coordinate from environment (not relative offset)
            wall_data: Optional wall geometry from EnvironmentAnalyzer.get_wall_data() for raycasting

        Returns:
            List of waypoint dictionaries in temporal order:
                [{"position": (x, y, z), "direction": angle_deg}, ...]
            Each waypoint represents one camera position along the path.
        """
        min_x, min_y, min_z, max_x, max_y, max_z = room_bounds

        start_x = random.uniform(min_x + wall_buffer, max_x - wall_buffer)
        start_y = random.uniform(min_y + wall_buffer, max_y - wall_buffer)
        start_z = floor_height + height

        current_pos = (start_x, start_y, start_z)

        if initial_direction == "random":
            current_direction = random.uniform(0, 360)
        else:
            current_direction = float(initial_direction) + random.uniform(-15, 15)

        path = []

        logger.info(f"Brownian path starting at ({start_x:.2f}, {start_y:.2f}, {start_z:.2f}), direction={current_direction:.1f}°")

        for step in range(num_steps):
            path.append({
                "position": current_pos,
                "direction": current_direction
            })

            direction_change = random.uniform(direction_change_range[0], direction_change_range[1])
            next_direction = (current_direction + direction_change) % 360

            direction_vector = Gf.Vec3d(
                math.cos(math.radians(next_direction)),
                math.sin(math.radians(next_direction)),
                0
            )

            next_x = current_pos[0] + step_distance * direction_vector[0]
            next_y = current_pos[1] + step_distance * direction_vector[1]
            next_z = current_pos[2]

            collision_detected = False

            if wall_data:
                hit, hit_distance, wall_normal, hit_wall = CameraSystem.raycast_to_walls_2d(
                    start_pos=current_pos,
                    direction_vector=direction_vector,
                    wall_data=wall_data,
                    wall_buffer=wall_buffer,
                    max_distance=step_distance * 1.5
                )

                if hit and hit_distance < step_distance:
                    collision_detected = True

                    reflected_vector = CameraSystem.reflect_vector(direction_vector, wall_normal)

                    reflected_angle = math.degrees(math.atan2(reflected_vector[1], reflected_vector[0]))

                    bounce_perturbation = random.uniform(bounce_angle_range[0], bounce_angle_range[1])
                    if random.random() < 0.5:
                        bounce_perturbation = -bounce_perturbation

                    next_direction = (reflected_angle + bounce_perturbation) % 360

                    next_x = current_pos[0] + step_distance * math.cos(math.radians(next_direction))
                    next_y = current_pos[1] + step_distance * math.sin(math.radians(next_direction))

                    logger.debug(f"Wall bounce at step {step}: reflected {reflected_angle:.1f}° + perturbation {bounce_perturbation:.1f}° = {next_direction:.1f}°")

            if not collision_detected:
                if not (min_x + wall_buffer <= next_x <= max_x - wall_buffer and
                        min_y + wall_buffer <= next_y <= max_y - wall_buffer):
                    collision_detected = True
                    bounce_angle = random.uniform(bounce_angle_range[0], bounce_angle_range[1])
                    next_direction = (current_direction + bounce_angle) % 360
                    next_x = current_pos[0] + step_distance * math.cos(math.radians(next_direction))
                    next_y = current_pos[1] + step_distance * math.sin(math.radians(next_direction))

            next_x = max(min_x + wall_buffer, min(next_x, max_x - wall_buffer))
            next_y = max(min_y + wall_buffer, min(next_y, max_y - wall_buffer))

            current_pos = (next_x, next_y, next_z)
            current_direction = next_direction

        logger.info(f"Generated Brownian path: {num_steps} steps, {step_distance}m per step")
        return path

    @staticmethod
    def place_cameras_on_paths(
        cameras: list[Usd.Prim],
        paths: list[list[dict]],
        step_index: int,
        look_ahead_distance: float,
        pitch_angle: float
    ) -> None:
        """
        Position cameras at specific step along their Brownian motion paths.

        Places each camera at corresponding waypoint, oriented to look ahead along trajectory direction.
        Uses SetLookAt with Z-up to prevent roll and keep horizon level.

        Args:
            cameras: List of camera prims to position (typically self.cameras from main script)
            paths: List of trajectory paths (one per camera) from generate_brownian_motion_path()
                   Each path is a list of waypoint dicts: [{"position": (x,y,z), "direction": angle_deg}, ...]
            step_index: Current timestep index (0 to num_steps-1) - which waypoint to use from each path
            look_ahead_distance: Distance in meters to look ahead along trajectory (e.g., 1.0 = 1 meter ahead)
            pitch_angle: Camera pitch angle in degrees relative to horizon
                        Negative = tilt down (e.g., -5° for slight downward gaze)
                        Positive = tilt up (e.g., +10° for upward perspective)
                        Zero = perfectly horizontal gaze

        Implementation notes:
            - Cameras are positioned independently - each camera can have different paths
            - If step_index exceeds a camera's path length, that camera is skipped with warning
            - Uses Gf.Matrix4d.SetLookAt with Z-up vector to prevent roll (keeps horizon level)
            - Direction angle from waypoint determines horizontal gaze direction
            - Pitch angle tilts camera up/down from the horizontal plane
        """
        for i, cam in enumerate(cameras):
            if i >= len(paths):
                logger.warning(f"No path available for camera {i}")
                continue

            path = paths[i]
            if step_index >= len(path):
                logger.warning(f"Step {step_index} out of range for camera {i} path (length {len(path)})")
                continue

            waypoint = path[step_index]
            position = Gf.Vec3d(*waypoint["position"])
            direction_angle = waypoint["direction"]

            look_at_x = position[0] + look_ahead_distance * math.cos(math.radians(direction_angle))
            look_at_y = position[1] + look_ahead_distance * math.sin(math.radians(direction_angle))
            look_at_z = position[2] + look_ahead_distance * math.tan(math.radians(pitch_angle))
            look_at = Gf.Vec3d(look_at_x, look_at_y, look_at_z)

            view_mat = Gf.Matrix4d()
            view_mat.SetLookAt(position, look_at, Gf.Vec3d(0, 0, 1))  # Z is up
            camera_transform = view_mat.GetInverse()
            orientation = Gf.Quatf(camera_transform.ExtractRotation().GetQuat())

            xform = UsdGeom.Xformable(cam)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(position)
            xform.AddOrientOp().Set(orientation)

            logger.debug(f"Camera {i} at step {step_index}: pos={position}, dir={direction_angle:.1f}°")

    @staticmethod
    def _get_random_pose_on_sphere(
        origin: tuple[float, float, float],
        radius_range: tuple[float, float],
        polar_angle_range: tuple[float, float],
        camera_forward_axis: tuple[float, float, float] = (0, 0, -1),
        keep_level: bool = False,
    ) -> tuple[Gf.Vec3d, Gf.Quatf]:
        """
        Generate random camera pose on sphere looking at origin.

        Uses spherical coordinates to position camera, then calculates orientation quaternion.
        Polar angle: elevation (0°=top, 90°=horizon, 180°=below). Azimuthal: rotation around Z.

        Args:
            origin: Target point to look at (x, y, z)
            radius_range: (min, max) distance from origin in meters
            polar_angle_range: (min, max) elevation angle in degrees
            camera_forward_axis: Camera's forward direction vector (default: (0, 0, -1) for USD cameras)
            keep_level: If True, camera maintains level orientation (no tilt), only rotates around Z-axis

        Returns:
            Tuple of (location, orientation) where:
                - location: 3D position as Gf.Vec3d
                - orientation: Quaternion orientation as Gf.Quatf

        Implementation follows Isaac Sim conventions:
        https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html
        """
        polar_angle_min_rad = math.radians(polar_angle_range[0])
        polar_angle_max_rad = math.radians(polar_angle_range[1])

        radius = random.uniform(radius_range[0], radius_range[1])
        polar_angle = random.uniform(polar_angle_min_rad, polar_angle_max_rad)
        azimuthal_angle = random.uniform(0, 2 * math.pi)

        x = radius * math.sin(polar_angle) * math.cos(azimuthal_angle)
        y = radius * math.sin(polar_angle) * math.sin(azimuthal_angle)
        z = radius * math.cos(polar_angle)

        location = Gf.Vec3d(origin[0] + x, origin[1] + y, origin[2] + z)

        if keep_level:
            # Use SetLookAt with Z-up to prevent camera roll and keep horizon level
            view_mat = Gf.Matrix4d()
            view_mat.SetLookAt(location, Gf.Vec3d(origin), Gf.Vec3d(0, 0, 1))
            camera_transform = view_mat.GetInverse()
            orientation = Gf.Quatf(camera_transform.ExtractRotation().GetQuat())
        else:
            # Allow camera to tilt/roll freely based on viewing direction
            direction = Gf.Vec3d(origin) - location
            direction_normalized = direction.GetNormalized()
            rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), direction_normalized)
            orientation = Gf.Quatf(rotation.GetQuat())

        logger.debug(
            f"Camera pose: origin={origin}, loc={location}, "
            f"radius={radius:.2f}m, polar={math.degrees(polar_angle):.1f}°, "
            f"azimuthal={math.degrees(azimuthal_angle):.1f}°, keep_level={keep_level}"
        )

        return location, orientation

    @staticmethod
    def randomize_object_centric_pose(
        camera: Usd.Prim,
        targets: list[Usd.Prim],
        distance_range: tuple[float, float],
        polar_angle_range: tuple[float, float] = (0, 180),
        look_at_offset: tuple[float, float] = (-0.1, 0.1),
        keep_level: bool = False,
        min_height: float | None = None,
    ) -> None:
        """
        Position camera on sphere around randomly selected target object.

        Selects random target, adds offset to avoid centering bias, positions camera on sphere, orients toward target.
        Ensures objects are visible while providing diverse viewpoints.

        Args:
            camera: Camera prim to position
            targets: List of target object prims to randomly choose from
            distance_range: (min, max) distance from target in meters (e.g., (1.25, 1.5))
            polar_angle_range: (min, max) polar angle in degrees (0° = top-down, 90° = horizon, 180° = below)
            look_at_offset: (min, max) random offset from target center in meters to avoid center bias
            keep_level: If True, camera maintains level orientation (no tilt/roll), only rotates around Z-axis
            min_height: Minimum Z-coordinate for camera (floor constraint). Camera will be clamped above this height.
        """
        if not targets:
            logger.warning("No target objects provided for object-centric camera positioning")
            return

        logger.debug(
            f"Positioning camera around {len(targets)} target objects "
            f"(distance: {distance_range[0]:.2f}-{distance_range[1]:.2f}m, "
            f"polar angle: {polar_angle_range[0]:.0f}-{polar_angle_range[1]:.0f}°, "
            f"keep_level: {keep_level})"
        )

        target_asset = random.choice(targets)

        target_loc = target_asset.GetAttribute("xformOp:translate").Get()
        if target_loc is None:
            logger.warning(f"Target {target_asset.GetPath()} has no translate attribute, skipping camera positioning")
            return

        target_loc_with_offset = (
            target_loc[0] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[1] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[2] + random.uniform(look_at_offset[0], look_at_offset[1]),
        )

        loc, quat = CameraSystem._get_random_pose_on_sphere(
            target_loc_with_offset,
            distance_range,
            polar_angle_range,
            keep_level=keep_level
        )

        # Clamp camera height to stay above floor
        if min_height is not None and loc[2] < min_height:
            loc = Gf.Vec3d(loc[0], loc[1], min_height)
            # Recalculate orientation to look at target from new position
            if keep_level:
                view_mat = Gf.Matrix4d()
                view_mat.SetLookAt(loc, Gf.Vec3d(target_loc_with_offset), Gf.Vec3d(0, 0, 1))
                camera_transform = view_mat.GetInverse()
                quat = Gf.Quatf(camera_transform.ExtractRotation().GetQuat())
            else:
                direction = Gf.Vec3d(target_loc_with_offset) - loc
                direction_normalized = direction.GetNormalized()
                rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction_normalized)
                quat = Gf.Quatf(rotation.GetQuat())
            logger.debug(f"Camera clamped to min_height={min_height:.2f}m")

        TransformUtils.set_transform_attributes(camera, location=loc, orientation=quat)

        logger.debug(f"Camera positioned at {loc}, looking at target {target_asset.GetPath()}")
