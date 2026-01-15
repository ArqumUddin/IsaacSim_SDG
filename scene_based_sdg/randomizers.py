# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Randomization utilities for Replicator.
Handles dome lights, colors, and light properties.
"""

import logging
import random

import omni.replicator.core as rep
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd

from .transforms import TransformUtils

logger = logging.getLogger(__name__)


class Randomizers:
    """Manages Replicator randomization for lighting and materials."""

    @staticmethod
    def register_dome_light_randomizer() -> None:
        """
        Register Replicator randomizer for dome light environment textures.

        Samples from NVIDIA HDR sky textures (cloudy, clear, evening, night) when "randomize_dome_lights" event triggers.
        Provides realistic global illumination with lighting diversity for robust CV model training.
        Dormant until explicitly triggered via rep.trigger.emit_event("randomize_dome_lights").
        """
        assets_root_path = get_assets_root_path()
        dome_textures = [
            assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Clear/mealie_road_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Clear/qwantani_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Evening/evening_road_01_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Night/kloppenheim_02_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Night/moonlit_golf_4k.hdr",
        ]
        with rep.trigger.on_custom_event(event_name="randomize_dome_lights"):
            rep.create.light(light_type="Dome", texture=rep.distribution.choice(dome_textures))

    @staticmethod
    def register_shape_distractors_color_randomizer(shape_distractors: list[Usd.Prim]) -> None:
        """
        Register Replicator randomizer for distractor object colors.

        Randomizes albedo of distractor shapes with uniform RGB sampling when "randomize_shape_distractor_colors" event triggers.
        Prevents color bias and increases visual complexity for robust model training.
        Target objects keep original colors; only specified distractors are affected.

        Args:
            shape_distractors: List of USD prims (typically geometric shapes: spheres, cubes, etc.)
                             These prims will receive random colors when the event is triggered
                             Empty list is allowed but no-op
        """
        with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
            shape_distractors_paths = [prim.GetPath() for prim in shape_distractors]
            shape_distractors_group = rep.create.group(shape_distractors_paths)
            with shape_distractors_group:
                rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

    @staticmethod
    def position_lights_around_cameras(
        lights: list[Usd.Prim],
        cameras: list[Usd.Prim],
        offset_range: tuple[float, float, float, float, float, float] = (-2, -2, 1, 2, 2, 3),
        intensity_range: tuple[float, float] = (1000, 2500)
    ) -> None:
        """
        Position scene lights around cameras with random offsets and intensities.

        Distributes lights evenly among cameras, placing each with random offset in offset_range AABB.
        Creates localized lighting that supplements dome light's global illumination.
        Called during capture loop after camera positioning to ensure consistent illumination.

        Args:
            lights: List of point light prims to position (typically 2-6 lights)
                   Empty list is allowed but results in no scene lighting besides dome
            cameras: List of camera prims that lights should follow
                    Lights are distributed evenly among cameras
            offset_range: 3D AABB defining light placement relative to camera as
                         (min_x, min_y, min_z, max_x, max_y, max_z) in meters
                         Example: (-2, -2, 1, 2, 2, 3) = 4m×4m×2m volume, 1-3m above camera
            intensity_range: Light intensity range in lumens as (min, max)
                            Example: (1000, 2500) varies brightness by 2.5x for diversity
        """
        if not lights or not cameras:
            return

        lights_per_camera = max(1, len(lights) // len(cameras))

        for i, light in enumerate(lights):
            camera_idx = min(i // lights_per_camera, len(cameras) - 1)
            camera = cameras[camera_idx]

            cam_translate = camera.GetAttribute("xformOp:translate")
            if cam_translate:
                cam_pos = cam_translate.Get()
                light_pos = Gf.Vec3d(
                    cam_pos[0] + random.uniform(offset_range[0], offset_range[3]),
                    cam_pos[1] + random.uniform(offset_range[1], offset_range[4]),
                    cam_pos[2] + random.uniform(offset_range[2], offset_range[5])
                )

                TransformUtils.set_transform_attributes(light, location=light_pos)

                if intensity_range:
                    rand_intensity = random.uniform(intensity_range[0], intensity_range[1])
                    light.GetAttribute("inputs:intensity").Set(rand_intensity)
