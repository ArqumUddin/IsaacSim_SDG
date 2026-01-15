# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import os
import random
import yaml
from itertools import cycle

from isaacsim import SimulationApp


logger = logging.getLogger("scene_based_sdg")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[SDG] %(message)s'))
    logger.addHandler(handler)

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to the config file (json or yaml)")
parser.add_argument(
    "--keep-open", action="store_true", help="Keep the app open after completion for scene inspection"
)
args, unknown = parser.parse_known_args()

# Config Loading
config_file = args.config
if not os.path.isfile(config_file):
    logger.error(f" Config file {config_file} does not exist.")
    exit(1)

with open(config_file, "r") as f:
    if config_file.endswith(".json"):
        config = json.load(f)
    elif config_file.endswith(".yaml"):
        config = yaml.safe_load(f)
    else:
        logger.error(f" Config file {config_file} must be .json or .yaml")
        exit(1)

# Launch App
simulation_app = SimulationApp(launch_config={"headless": config.get("launch_config", {}).get("headless", True)})

# Imports after launch
import carb.settings
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import new modular components
from scene_based_sdg.asset_manager import AssetManager
from scene_based_sdg.camera_system import CameraSystem
from scene_based_sdg.common import setup_logging
from scene_based_sdg.environment_analyzer import EnvironmentAnalyzer
from scene_based_sdg.environment_setup import EnvironmentSetup
from scene_based_sdg.object_placer import ObjectPlacer
from scene_based_sdg.physics_engine import PhysicsEngine
from scene_based_sdg.randomizers import Randomizers
from scene_based_sdg.transforms import AdaptiveScaling, TransformUtils

class SceneBasedSDG:
    """
    Main application class for Scene-Based Synthetic Data Generation.

    Features:
    - Loads complex environment scenes (e.g. Infinigen).
    - Spawns and randomizes labeled assets (custom YCB, etc.) and distractors.
    - Manages camera placement strategies.
    - Handles physics settling and motion blur capture steps.
    """
    def __init__(self, app: SimulationApp, config: dict):
        """
        Initialize the Scene-Based SDG environment.

        Args:
            app (SimulationApp): The running Isaac Sim application instance.
            config (dict): Configuration dictionary.
        """
        self.app = app
        self.config = config
        self.debug_mode = config.get("debug_mode", False)

        carb.settings.get_settings().set("/physics/cooking/ujitsoCollisionCooking", False)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        self.cameras = []
        self.render_products = []
        self.writers = []
        self.scene_lights = []

        self.target_assets = []
        self.mesh_distractors = []
        self.shape_distractors = []

        self.unique_labels = set()

        self.assets_root_path = get_assets_root_path()

        self.asset_library = []

    def scan_content_library(self) -> None:
        """
        Scans configured folders for assets and populates self.asset_library.
        Also populates self.unique_labels so writers can be initialized with all possible classes.

        This is called once at the start of the pipeline to get all labels.
        """
        labeled_cfg = self.config.get("labeled_assets", {})
        auto_cfg = labeled_cfg.get("auto_label", {})

        self.asset_library = []

        if auto_cfg:
            scanned = AssetManager.scan_assets(
                folders=auto_cfg.get("folders", []),
                files=auto_cfg.get("files", []),
                regex_replace_pattern=auto_cfg.get("regex_replace_pattern", ""),
                regex_replace_repl=auto_cfg.get("regex_replace_repl", "")
            )
            count = auto_cfg.get("num", 1)
            gravity_chance = auto_cfg.get("gravity_disabled_chance", 0.0)
            if auto_cfg.get("floating", False):
                gravity_chance = 1.0
            scale_min_max = auto_cfg.get("scale_min_max", [1.0, 1.0])

            for asset in scanned:
                asset["count"] = count
                asset["gravity_disabled_chance"] = gravity_chance
                asset["scale_min_max"] = scale_min_max
                self.asset_library.append(asset)
                if "label" in asset:
                    self.unique_labels.add(asset["label"])

        manual_assets = labeled_cfg.get("manual_label", [])
        for item in manual_assets:
            curr_item = item.copy()
            curr_item["count"] = item.get("num", item.get("count", 1))
            self.asset_library.append(curr_item)
            if "label" in curr_item:
                self.unique_labels.add(curr_item["label"])

        logger.info(f"Scanned {len(self.asset_library)} total asset definitions.")
        logger.info(f"Found {len(self.unique_labels)} unique labels.")

    def setup_cameras(self) -> None:
        """
        Creates cameras and render products based on per-camera configuration.
        """
        capture_config = self.config.get("capture", {})
        camera_configs = capture_config.get("cameras", [])

        if not camera_configs:
            logger.error("No cameras defined in config. Please add 'cameras' list to capture config.")
            raise ValueError("cameras list is required in capture config")

        logger.info(f"Setting up {len(camera_configs)} cameras with per-camera configuration")

        collision_enabled = capture_config.get("camera_collision_enabled", False)
        collision_radius = capture_config.get("camera_collision_radius", 0.15)

        stage = omni.usd.get_context().get_stage()

        self.cameras = []
        for cam_cfg in camera_configs:
            cam_idx = cam_cfg["index"]
            cam_prim = stage.DefinePrim(f"/Cameras/cam_{cam_idx}", "Camera")

            properties = cam_cfg.get("properties", {})
            cam_strategy = cam_cfg["strategy"]
            logger.debug(f"Camera {cam_idx}: strategy={cam_strategy}, properties={list(properties.keys())}")

            for key, value in properties.items():
                if cam_prim.HasAttribute(key):
                    cam_prim.GetAttribute(key).Set(value)

            if collision_enabled:
                PhysicsEngine.add_collision_sphere_to_camera(
                    cam_prim=cam_prim,
                    radius=collision_radius,
                    disable_gravity=True,
                    visible=False
                )

            self.cameras.append(cam_prim)

        logger.info(f"Created {len(self.cameras)} cameras" +
                    (f" with collision spheres (r={collision_radius}m)" if collision_enabled else ""))

        self.render_products = []
        resolution = capture_config.get("resolution", (1280, 720))
        disable_rp = capture_config.get("disable_render_products", False)

        for cam in self.cameras:
            rp = rep.create.render_product(cam.GetPath(), resolution, name=f"rp_{cam.GetName()}")
            if disable_rp:
                rp.hydra_texture.set_updates_enabled(False)
            self.render_products.append(rp)
        logger.info(f"Created {len(self.render_products)} render products")

    def setup_writers(self) -> None:
        """
        Initializes and attaches writers to render products.
        
        Iterates through the `writers` configuration, initializes each writer
        using `_setup_writer`, and attaches it to all created RenderProducts.
        """
        writers_config = self.config.get("writers", [])
        self.writers = []
        if self.render_products:
            for writer_cfg in writers_config:
                writer = self._setup_writer(writer_cfg)
                if writer:
                    writer.attach(self.render_products)
                    self.writers.append(writer)
                    logger.info(f"{writer_cfg['type']} output: {writer_cfg.get('kwargs', {}).get('output_dir', '')}")
        logger.info(f"Created {len(self.writers)} writers")

    def _setup_writer(self, config: dict) -> rep.Writer | None:
        """
        Helper to initialize a single Replicator writer from config.
        
        Args:
            config (dict): Writer configuration dictionary (type, kwargs, etc.).
        
        Returns:
            rep.Writer | None: Initialized writer instance or None if failed.
        """
        writer_type = config.get("type", None)
        if writer_type is None:
            logger.info("No writer type specified. No writer will be used.")
            return None

        try:
            writer = rep.writers.get(writer_type)
        except Exception as e:
            logger.info(f"Writer type '{writer_type}' not found. No writer will be used. Error: {e}")
            return None

        writer_kwargs = config.get("kwargs", {})
        if out_dir := writer_kwargs.get("output_dir"):
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.getcwd(), out_dir)
                writer_kwargs["output_dir"] = out_dir

            if writer_type == "CocoWriter":
                if "semantic_types" not in writer_kwargs:
                    writer_kwargs["semantic_types"] = ["class"]
                if "coco_categories" not in writer_kwargs:
                    logger.info("Auto-generating coco_categories...")
                    writer_kwargs["coco_categories"] = self._generate_coco_categories()

        writer.initialize(**writer_kwargs)
        return writer

    def _generate_coco_categories(self):
        """
        Generates the 'coco_categories' dictionary for the CocoWriter.
        """
        categories = {}
        ordered_labels = sorted(list(self.unique_labels))
        logger.info(f"Found {len(ordered_labels)} unique labels: {ordered_labels}")

        for i, label in enumerate(ordered_labels):
            cat_id = i + 1
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            categories[label] = {
                "name": label,
                "id": cat_id,
                "supercategory": "ycb",
                "isthing": 1,
                "color": color
            }
        return categories

    def spawn_assets(self) -> None:
        """
        Spawns a random subset of assets from self.asset_library.
        Clears existing assets first to ensure a fresh set.

        Note: Distractors are loaded once at initialization and reused across environments.

        This is called every time a new environment is loaded.
        """
        stage = omni.usd.get_context().get_stage()

        if stage.GetPrimAtPath("/Assets"):
            omni.kit.commands.execute("DeletePrims", paths=["/Assets"])

        labeled_cfg = self.config.get("labeled_assets", {})
        auto_cfg = labeled_cfg.get("auto_label", {})

        assets_to_spawn = self.asset_library.copy()

        load_all = auto_cfg.get("load_all_assets", False)
        max_types = auto_cfg.get("max_asset_types_to_load")

        if load_all:
            logger.info(f"Loading all {len(assets_to_spawn)} available asset types.")
        elif max_types and len(assets_to_spawn) > max_types:
            logger.info(f"Sub-sampling assets: selecting {max_types} from {len(assets_to_spawn)} available.")
            assets_to_spawn = random.sample(assets_to_spawn, max_types)
        else:
            logger.info(f"Loading {len(assets_to_spawn)} asset types.")

        self.target_assets = []

        logger.info(f"Spawning {len(assets_to_spawn)} unique labeled asset types...")
        for asset_def in assets_to_spawn:
            url = asset_def.get("url")
            label = asset_def.get("label")
            count = asset_def.get("count", 1)
            grav_chance = asset_def.get("gravity_disabled_chance", 0.0)
            scale_range = asset_def.get("scale_min_max", [1.0, 1.0])

            if not os.path.isabs(url) and "://" not in url:
                if not url.startswith("/"):
                     url = self.assets_root_path + url

            for _ in range(count):
                disable_gravity = random.random() < grav_chance
                name_prefix = "floating_" if disable_gravity else "falling_"
                prim_path = omni.usd.get_stage_next_free_path(stage, f"/Assets/{name_prefix}{label}", False)
                try:
                    prim = add_reference_to_stage(usd_path=url, prim_path=prim_path)

                    scale = random.uniform(scale_range[0], scale_range[1])
                    TransformUtils.set_transform_attributes(prim, scale=Gf.Vec3f(scale, scale, scale))

                    PhysicsEngine.add_colliders_and_dynamics(prim, disable_gravity=disable_gravity)
                    remove_labels(prim, include_descendants=True)
                    add_labels(prim, labels=[label], instance_name="class")
                    self.target_assets.append(prim)
                except Exception as e:
                    logger.info(f"Failed to load asset {url}: {e}")

        logger.info(f"Total labeled assets spawned: {len(self.target_assets)}")

        AssetManager.resolve_scale_issues()

    def setup_distractors(self) -> None:
        """
        Loads distractors once at initialization.

        This method is called once during setup. Distractors are then reused
        across all environment iterations by repositioning them rather than
        reloading from USD files.
        """
        distractors_cfg = self.config.get("distractors", {})

        s_float, s_fall = AssetManager.load_shape_distractors(distractors_cfg.get("shape_distractors", {}))
        self.shape_distractors = s_float + s_fall
        logger.info(f"Loaded {len(self.shape_distractors)} shape distractors")

        m_float, m_fall = AssetManager.load_mesh_distractors(distractors_cfg.get("mesh_distractors", {}))
        self.mesh_distractors = m_float + m_fall
        logger.info(f"Loaded {len(self.mesh_distractors)} mesh distractors")

    def setup_scene_lights(self) -> None:
        """
        Creates dynamic scene lights to be randomized.

        Instantiates sphere lights in the stage with proper initialization,
        which are later manipulated by `randomize_lights` during the simulation loop.
        """
        num_lights = self.config.get("capture", {}).get("num_scene_lights", 0)
        self.scene_lights = []
        stage = omni.usd.get_context().get_stage()

        for i in range(num_lights):
            light = stage.DefinePrim(f"/Lights/SphereLight_scene_{i}", "SphereLight")

            if not light.HasAttribute("inputs:intensity"):
                light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float)
            light.GetAttribute("inputs:intensity").Set(1000.0)

            if not light.HasAttribute("inputs:color"):
                light.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f)
            light.GetAttribute("inputs:color").Set(Gf.Vec3f(1.0, 1.0, 1.0))

            if not light.HasAttribute("inputs:radius"):
                light.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float)
            light.GetAttribute("inputs:radius").Set(0.5)

            if not light.HasAttribute("xformOp:translate"):
                UsdGeom.Xformable(light).AddTranslateOp()
            light.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 3))

            self.scene_lights.append(light)

        logger.info(f"Created {len(self.scene_lights)} scene lights with default attributes")

    def setup_randomizers(self) -> None:
        """
        Registers Replicator randomizers.
        
        Sets up Replicator graph nodes for randomizing specific attributes 
        like dome light textures and distractor colors.
        """
        logger.info(f"Registering randomizers")
        Randomizers.register_dome_light_randomizer()
        if self.shape_distractors:
            Randomizers.register_shape_distractors_color_randomizer(self.shape_distractors)

    def run(self) -> None:
        """
        Main execution loop iterating over environments and capturing synthetic data.

        Scans USD files, sets up stage (cameras/lights/writers), loads assets, cycles through environments.
        Each environment: loads USD, analyzes space, randomizes poses, simulates physics, captures frames.
        Supports per-camera strategies (Brownian motion, object-centric) and adaptive object scaling.
        """
        env_config = self.config.get("environments", {})
        env_urls = AssetManager.get_usd_paths(
            files=env_config.get("files", []), 
            folders=env_config.get("folders", []), 
            skip_folder_keywords=[".thumbs"]
        )
        
        capture_cfg = self.config.get("capture", {})
        total_captures = capture_cfg.get("total_captures", 0)
        num_captures_per_env = capture_cfg.get("num_captures_per_env", 5)
        rt_subframes = capture_cfg.get("rt_subframes", 3)
        use_path_tracing = capture_cfg.get("path_tracing", False)
        disable_rp = capture_cfg.get("disable_render_products", False)
        camera_height_offset = capture_cfg.get("camera_height_offset", 0.2)

        log_level = capture_cfg.get("log_level", "INFO")
        setup_logging(log_level=log_level)

        camera_configs = capture_cfg.get("cameras", [])

        if not camera_configs:
            logger.error("No cameras defined in config. Please add 'cameras' list to capture config.")
            raise ValueError("cameras list is required in capture config")

        logger.info(f"Validating {len(camera_configs)} camera configurations")

        for cam_cfg in camera_configs:
            cam_idx = cam_cfg.get("index")
            if cam_idx is None:
                logger.error("Camera config missing 'index' field")
                raise ValueError("Each camera must have an 'index' field")

            strategy = cam_cfg.get("strategy")
            if strategy not in ["brownian_motion", "object_centric"]:
                logger.error(f"Invalid strategy '{strategy}' for camera {cam_idx}")
                raise ValueError(f"Strategy must be 'brownian_motion' or 'object_centric'")

        scaling_cfg = capture_cfg.get("adaptive_scaling", {})
        scaling_enabled = scaling_cfg.get("enabled", False)
        scaling_target_size = scaling_cfg.get("target_room_size", 5.0)
        scaling_min = scaling_cfg.get("min_scale_factor", 0.15)
        scaling_max = scaling_cfg.get("max_scale_factor", 1.0)

        logger.info(f"Creating new stage")
        omni.usd.get_context().new_stage()
        rep.orchestrator.set_capture_on_play(False)

        self.setup_cameras()
        self.scan_content_library()
        self.setup_distractors()
        self.setup_randomizers()
        self.setup_scene_lights()
        self.setup_writers()

        env_cycle = cycle(env_urls)
        capture_counter = 0

        while capture_counter < total_captures:
            env_url = next(env_cycle)
            logger.info(f"Loading environment: {env_url}")

            EnvironmentSetup.load_env(env_url, prim_path="/Environment")
            logger.info(f"Setting up environment")
            EnvironmentSetup.setup_env(root_path="/Environment", hide_top_walls=self.debug_mode)
            self.app.update()

            min_camera_height = EnvironmentAnalyzer.get_surface_height(
                "floor", camera_height_offset, add_offset=True, root_path="/Environment", default_value=camera_height_offset
            )
            max_camera_height = EnvironmentAnalyzer.get_surface_height(
                "ceiling", camera_height_offset, add_offset=False, root_path="/Environment", default_value=None
            )

            wall_data = EnvironmentAnalyzer.get_wall_data(root_path="/Environment")

            room_bounds = EnvironmentAnalyzer.calculate_room_bounds(
                wall_data, min_camera_height, max_camera_height,
                wall_clearance=capture_cfg.get("wall_clearance", 0.3)
            )

            actual_floor_height = min_camera_height - camera_height_offset

            self.spawn_assets()

            if scaling_enabled:
                scale_factor = AdaptiveScaling.calculate_scale_factor(
                    room_bounds, scaling_target_size, scaling_min, scaling_max
                )
                logger.info(f"\tApplying adaptive scale factor {scale_factor:.3f} to all objects...")
                for prim in self.target_assets + self.mesh_distractors + self.shape_distractors:
                    AdaptiveScaling.apply_scale_to_prim(prim, scale_factor)

            obj_placement_cfg = self.config.get("object_placement", {})
            furniture_surfaces = EnvironmentAnalyzer.find_furniture_surfaces(
                root_path="/Environment",
                furniture_keywords=obj_placement_cfg.get("furniture_keywords", ["Table", "Counter", "Shelf", "Desk"]),
                min_surface_area=obj_placement_cfg.get("min_surface_area", 0.1)
            )

            working_area_loc = EnvironmentAnalyzer.get_matching_prim_location(
                match_string="TableDining", root_path="/Environment"
            )

            if self.debug_mode:
                cam_loc = (working_area_loc[0], working_area_loc[1], working_area_loc[2] + 10)
                set_camera_view(eye=np.array(cam_loc), target=np.array(working_area_loc))

            logger.info("\tRandomizing assets and distractors...")
            surface_ratio = obj_placement_cfg.get("surface_placement_ratio", 0.3)
            surface_height_offset = obj_placement_cfg.get("surface_height_offset", 0.02)

            # Create floor_bounds with actual floor height (room_bounds uses camera height constraints)
            floor_bounds = (
                room_bounds[0], room_bounds[1], actual_floor_height,
                room_bounds[3], room_bounds[4], room_bounds[5]
            )

            ObjectPlacer.randomize_poses_with_surfaces(
                self.target_assets, floor_bounds, furniture_surfaces,
                surface_placement_ratio=surface_ratio,
                rotation_range=(0, 360), scale_range=(0.95, 1.15),
                surface_height_offset=surface_height_offset,
                description="target assets"
            )

            ObjectPlacer.randomize_poses_with_surfaces(
                self.mesh_distractors, floor_bounds, furniture_surfaces,
                surface_placement_ratio=surface_ratio * 0.5,
                rotation_range=(0, 360), scale_range=(0.3, 1.0),
                surface_height_offset=surface_height_offset,
                description="mesh distractors"
            )

            shape_range = ObjectPlacer.offset_range((-1.5, -1.5, 1, 1.5, 1.5, 2), working_area_loc)
            ObjectPlacer.randomize_poses(self.shape_distractors, shape_range, (0, 360), (0.01, 0.1))

            rep.utils.send_og_event(event_name="randomize_dome_lights")
            if self.shape_distractors:
                rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

            logger.info("\tSimulating physics (dropping objects)...")
            self.app.update()
            PhysicsEngine.run_simulation(num_frames=200, render=False)

            if disable_rp: self._toggle_updates(True)
            if use_path_tracing: self._set_renderer("PathTracing")

            brownian_cameras = []
            brownian_configs = []
            object_centric_cameras = []
            object_centric_configs = []

            for cam_cfg in camera_configs:
                cam_idx = cam_cfg["index"]
                cam = self.cameras[cam_idx]
                strategy = cam_cfg["strategy"]

                if strategy == "brownian_motion":
                    brownian_cameras.append(cam)
                    brownian_configs.append(cam_cfg)
                elif strategy == "object_centric":
                    object_centric_cameras.append(cam)
                    object_centric_configs.append(cam_cfg)

            logger.info(f"\tCamera split: {len(brownian_cameras)} Brownian, {len(object_centric_cameras)} object-centric")

            camera_paths = {}
            if brownian_cameras:
                logger.info(f"\tGenerating Brownian paths for {len(brownian_cameras)} camera(s) ({num_captures_per_env} steps each)...")
                for cam, cam_cfg in zip(brownian_cameras, brownian_configs):
                    bm_settings = cam_cfg.get("brownian_motion", {})

                    path = CameraSystem.generate_brownian_motion_path(
                        room_bounds=room_bounds,
                        num_steps=num_captures_per_env,
                        step_distance=bm_settings.get("step_distance", 0.1),
                        height=bm_settings.get("height", 0.6),
                        direction_change_range=bm_settings.get("direction_change_range", [-10, 10]),
                        wall_buffer=bm_settings.get("wall_buffer", 0.3),
                        bounce_angle_range=bm_settings.get("bounce_angle_range", [30, 60]),
                        initial_direction=bm_settings.get("initial_direction", "random"),
                        floor_height=actual_floor_height,
                        wall_data=wall_data
                    )
                    camera_paths[cam.GetPath()] = path
                    logger.info(f"\t  Brownian path for camera {cam_cfg['index']}: {len(path)} steps, start={path[0]['position'][:2]}, end={path[-1]['position'][:2]}")

            # Position scene lights once per environment (not per frame) for performance
            if self.scene_lights:
                Randomizers.position_lights_around_cameras(
                    lights=self.scene_lights,
                    cameras=self.cameras,
                    offset_range=(-2, -2, 0.5, 2, 2, 2.5),
                    intensity_range=(1000, 2500)
                )

            for capture_idx in range(num_captures_per_env):
                if capture_counter >= total_captures: break

                if brownian_cameras:
                    for cam, cam_cfg in zip(brownian_cameras, brownian_configs):
                        path = camera_paths[cam.GetPath()]
                        if capture_idx < len(path):
                            bm_settings = cam_cfg.get("brownian_motion", {})
                            CameraSystem.place_cameras_on_paths(
                                cameras=[cam],
                                paths=[path],
                                step_index=capture_idx,
                                look_ahead_distance=bm_settings.get("look_ahead_distance", 1.0),
                                pitch_angle=bm_settings.get("pitch_angle", -5)
                            )

                if object_centric_cameras:
                    for cam, cam_cfg in zip(object_centric_cameras, object_centric_configs):
                        obj_settings = cam_cfg.get("object_centric", {})

                        CameraSystem.randomize_object_centric_pose(
                            camera=cam,
                            targets=self.target_assets,
                            distance_range=tuple(obj_settings.get("distance_range", [1.25, 1.5])),
                            polar_angle_range=tuple(obj_settings.get("polar_angle_range", [0, 75])),
                            look_at_offset=tuple(obj_settings.get("look_at_offset", [-0.1, 0.1])),
                            keep_level=obj_settings.get("keep_level", False),
                            min_height=min_camera_height
                        )

                logger.info(f"\tCapturing frame {capture_idx+1}/{num_captures_per_env} (Total: {capture_counter+1})")
                rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
                capture_counter += 1

            if disable_rp: self._toggle_updates(False)
            if use_path_tracing: self._set_renderer("RayTracedLighting")

        rep.orchestrator.wait_until_complete()
        logger.info(f"Detaching writers")
        for w in self.writers: w.detach()
        logger.info(f"Destroying products")
        for rp in self.render_products: rp.destroy()
        logger.info(f"Finished. Captured {capture_counter * len(self.cameras)} frames.")

    def _toggle_updates(self, enabled: bool) -> None:
        """
        Enables or disables updates for all render products.
        
        Useful for optimization when running physics simulation steps 
        without needing to render frames.
        
        Args:
            enabled (bool): True to enable rendering, False to disable.
        """
        for rp in self.render_products:
            rp.hydra_texture.set_updates_enabled(enabled)

    def _set_renderer(self, mode: str) -> None:
        """
        Switches the Isaac Sim renderer mode (e.g., to PathTracing).
        
        Args:
            mode (str): Renderer mode string (e.g. "PathTracing", "RayTracedLighting").
        """
        carb.settings.get_settings().set("/rtx/rendermode", mode)
            
if __name__ == "__main__":
    debug_mode = config.get("debug_mode", False)

    if debug_mode:
        np.random.seed(10)
        random.seed(10)
        rep.set_global_seed(10)

    sdg = SceneBasedSDG(simulation_app, config)
    sdg.run()

    if args.keep_open:
        logger.info("Keeping application open for inspection. Close window or press Ctrl+C to exit.")
        while simulation_app.is_running():
            simulation_app.update()
        simulation_app.close()
    else:
        simulation_app.close()
