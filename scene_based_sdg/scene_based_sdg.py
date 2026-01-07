# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random
import re
import yaml
from itertools import cycle

from isaacsim import SimulationApp

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to the config file (json or yaml)")
parser.add_argument(
    "--close-on-completion", action="store_true", help="Ensure the app closes on completion even in debug mode"
)
args, unknown = parser.parse_known_args()

# Config Loading
config_file = args.config
if not os.path.isfile(config_file):
    print(f"[SDG] Error: Config file {config_file} does not exist.")
    exit(1)

with open(config_file, "r") as f:
    if config_file.endswith(".json"):
        config = json.load(f)
    elif config_file.endswith(".yaml"):
        config = yaml.safe_load(f)
    else:
        print(f"[SDG] Error: Config file {config_file} must be .json or .yaml")
        exit(1)

# Launch App
simulation_app = SimulationApp(launch_config={"headless": config.get("launch_config", {}).get("headless", True)})

# Imports after launch
import carb.settings
import numpy as np
import omni.client
import omni.kit.app
import omni.physx
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_labels, remove_labels
import scene_based_sdg_utils
from isaacsim.storage.native import get_assets_root_path

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
        
        # Setup settings
        carb.settings.get_settings().set("/physics/cooking/ujitsoCollisionCooking", False)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
        
        # State
        self.cameras = []
        self.render_products = []
        self.writers = []
        self.scene_lights = []
        
        # Lists of prims for randomization
        self.target_assets = [] 
        self.mesh_distractors = []
        self.shape_distractors = []
        
        self.unique_labels = set() # Track for Writer categories
        
        # Assets root path for resolution
        self.assets_root_path = get_assets_root_path()
        
        # Asset Library (All discovered assets)
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
        
        # Auto-Label (Scanned)
        if auto_cfg:
            scanned = scene_based_sdg_utils.scan_assets(
                folders=auto_cfg.get("folders", []),
                files=auto_cfg.get("files", []),
                regex_replace_pattern=auto_cfg.get("regex_replace_pattern", ""),
                regex_replace_repl=auto_cfg.get("regex_replace_repl", "")
            )
            count = auto_cfg.get("num", 1)
            gravity_chance = auto_cfg.get("gravity_disabled_chance", 0.0)
            if auto_cfg.get("floating", False): gravity_chance = 1.0 # Override
            scale_min_max = auto_cfg.get("scale_min_max", [1.0, 1.0])
            
            for asset in scanned:
                asset["count"] = count
                asset["gravity_disabled_chance"] = gravity_chance
                asset["scale_min_max"] = scale_min_max
                self.asset_library.append(asset)
                # Track label
                if "label" in asset:
                    self.unique_labels.add(asset["label"])
        
        # Manual Label
        manual_assets = labeled_cfg.get("manual_label", [])
        for item in manual_assets:
            # Normalize keys
            curr_item = item.copy()
            curr_item["count"] = item.get("num", item.get("count", 1))
            self.asset_library.append(curr_item)
            if "label" in curr_item:
                self.unique_labels.add(curr_item["label"])

        print(f"[SDG] Scanned {len(self.asset_library)} total asset definitions.")
        print(f"[SDG] Found {len(self.unique_labels)} unique labels.")

    def setup_cameras(self) -> None:
        """
        Creates cameras and render products based on configuration.
        
        Parses `capture` config to instantiate cameras with specific properties 
        (focal length, clipping range, etc.) and attaches RenderProducts to them.
        """
        capture_config = self.config.get("capture", {})
        num_cameras = capture_config.get("num_cameras", 1)
        camera_properties_kwargs = capture_config.get("camera_properties_kwargs", {})
        
        stage = omni.usd.get_context().get_stage()
        
        # Create Cameras
        self.cameras = []
        for i in range(num_cameras):
            cam_prim = stage.DefinePrim(f"/Cameras/cam_{i}", "Camera")
            for key, value in camera_properties_kwargs.items():
                if cam_prim.HasAttribute(key):
                    cam_prim.GetAttribute(key).Set(value)
            self.cameras.append(cam_prim)
        print(f"[SDG] Created {len(self.cameras)} cameras")

        # Create Render Products
        self.render_products = []
        resolution = capture_config.get("resolution", (1280, 720))
        disable_rp = capture_config.get("disable_render_products", False)
        
        for cam in self.cameras:
            rp = rep.create.render_product(cam.GetPath(), resolution, name=f"rp_{cam.GetName()}")
            if disable_rp:
                rp.hydra_texture.set_updates_enabled(False)
            self.render_products.append(rp)
        print(f"[SDG] Created {len(self.render_products)} render products")

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
                    print(f"\t {writer_cfg['type']} output: {writer_cfg.get('kwargs', {}).get('output_dir', '')}")
        print(f"[SDG] Created {len(self.writers)} writers")

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
            print("[SDG] No writer type specified. No writer will be used.")
            return None

        try:
            writer = rep.writers.get(writer_type)
        except Exception as e:
            print(f"[SDG] Writer type '{writer_type}' not found. No writer will be used. Error: {e}")
            return None

        writer_kwargs = config.get("kwargs", {})
        if out_dir := writer_kwargs.get("output_dir"):
            # If not an absolute path, make path relative to the current working directory
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.getcwd(), out_dir)
                writer_kwargs["output_dir"] = out_dir
        
            # Dynamic COCO Category Generation
            if writer_type == "CocoWriter":
                if "semantic_types" not in writer_kwargs:
                    writer_kwargs["semantic_types"] = ["class"]
                if "coco_categories" not in writer_kwargs:
                    print("[SDG] Auto-generating coco_categories...")
                    writer_kwargs["coco_categories"] = self._generate_coco_categories()

        writer.initialize(**writer_kwargs)
        return writer

    def _generate_coco_categories(self):
        """
        Generates the 'coco_categories' dictionary for the CocoWriter.
        """
        categories = {}
        ordered_labels = sorted(list(self.unique_labels))
        print(f"[SDG] Found {len(ordered_labels)} unique labels: {ordered_labels}")

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
        Spawns a random subset of assets from self.asset_library and distractors.
        Clears existing assets first to ensure a fresh set.

        This is called every time a new environment is loaded.
        """
        stage = omni.usd.get_context().get_stage()
        
        # Clear existing
        if stage.GetPrimAtPath("/Assets"):
            omni.kit.commands.execute("DeletePrims", paths=["/Assets"])
        if stage.GetPrimAtPath("/Distractors"):
            omni.kit.commands.execute("DeletePrims", paths=["/Distractors"])
            
        labeled_cfg = self.config.get("labeled_assets", {})
        distractors_cfg = self.config.get("distractors", {})
        auto_cfg = labeled_cfg.get("auto_label", {}) # For max types config
        
        # Sample Assets
        assets_to_spawn = self.asset_library.copy()
        
        max_types = auto_cfg.get("max_asset_types_to_load")
        if max_types and len(assets_to_spawn) > max_types:
            print(f"[SDG] Sub-sampling assets: selecting {max_types} from {len(assets_to_spawn)} available.")
            assets_to_spawn = random.sample(assets_to_spawn, max_types)
            
        self.target_assets = []
        
        # Spawn Loop
        print(f"[SDG] Spawning {len(assets_to_spawn)} unique labeled asset types...")
        for asset_def in assets_to_spawn:
            url = asset_def.get("url")
            label = asset_def.get("label")
            count = asset_def.get("count", 1)
            grav_chance = asset_def.get("gravity_disabled_chance", 0.0)
            scale_range = asset_def.get("scale_min_max", [1.0, 1.0])
            
            # Resolve URL if relative
            if not os.path.isabs(url) and "://" not in url:
                if not url.startswith("/"): # Check path relative to assets root
                     url = self.assets_root_path + url 

            for _ in range(count):
                disable_gravity = random.random() < grav_chance
                name_prefix = "floating_" if disable_gravity else "falling_"
                prim_path = omni.usd.get_stage_next_free_path(stage, f"/Assets/{name_prefix}{label}", False)
                try:
                    prim = add_reference_to_stage(usd_path=url, prim_path=prim_path)
                    
                    # Apply Random Scale
                    scale = random.uniform(scale_range[0], scale_range[1])
                    scene_based_sdg_utils.set_transform_attributes(prim, scale=(scale, scale, scale))

                    scene_based_sdg_utils.add_colliders_and_rigid_body_dynamics(prim, disable_gravity=disable_gravity)
                    remove_labels(prim, include_descendants=True)
                    add_labels(prim, labels=[label], instance_name="class")
                    self.target_assets.append(prim)
                except Exception as e:
                    print(f"[SDG] Failed to load asset {url}: {e}")

        print(f"[SDG] Total labeled assets spawned: {len(self.target_assets)}")

        # Shape Distractors
        s_float, s_fall = scene_based_sdg_utils.load_shape_distractors(distractors_cfg.get("shape_distractors", {}))
        self.shape_distractors = s_float + s_fall
        print(f"[SDG] Loaded {len(self.shape_distractors)} shape distractors")

        # Mesh Distractors
        m_float, m_fall = scene_based_sdg_utils.load_mesh_distractors(distractors_cfg.get("mesh_distractors", {}))
        self.mesh_distractors = m_float + m_fall
        print(f"[SDG] Loaded {len(self.mesh_distractors)} mesh distractors")
        
        # Scale fix
        scene_based_sdg_utils.resolve_scale_issues_with_metrics_assembler()

    def setup_scene_lights(self) -> None:
        """
        Creates dynamic scene lights to be randomized.
        
        Instantiates sphere lights in the stage, which are later manipulated 
        by `randomize_lights` during the simulation loop.
        """
        num_lights = self.config.get("capture", {}).get("num_scene_lights", 0)
        self.scene_lights = []
        stage = omni.usd.get_context().get_stage()
        for i in range(num_lights):
            light = stage.DefinePrim(f"/Lights/SphereLight_scene_{i}", "SphereLight")
            self.scene_lights.append(light)
        print(f"[SDG] Created {len(self.scene_lights)} scene lights")

    def setup_randomizers(self) -> None:
        """
        Registers Replicator randomizers.
        
        Sets up Replicator graph nodes for randomizing specific attributes 
        like dome light textures and distractor colors.
        """
        print(f"[SDG] Registering randomizers")
        scene_based_sdg_utils.register_dome_light_randomizer()
        if self.shape_distractors:
            scene_based_sdg_utils.register_shape_distractors_color_randomizer(self.shape_distractors)

    def run(self) -> None:
        """
        Main execution loop iterating over environments and capturing data.
        
        Flow:
            1. Scans for environment USD files.
            2. Configures capture parameters (cameras, subframes, etc.).
            3. Sets up static stage elements (cameras, lights, writers).
            4. Loads and spawns assets (labeled + distractors).
            5. Loop over environments:
                a. Loads environment USD.
                b. Configures environment (colliders, visibility).
                c. Finds working area (e.g. table).
                d. Randomizes asset and light poses/properties.
                e. Simulates physics for settling.
                f. Captures "Floating" scenarios (gravity disabled/flying).
                g. drops physics (gravity enabled).
                h. Captures "Dropped" scenarios (objects on table).
            6. Cleans up resources.
        """
        env_config = self.config.get("environments", {})
        env_urls = scene_based_sdg_utils.get_usd_paths(
            files=env_config.get("files", []), 
            folders=env_config.get("folders", []), 
            skip_folder_keywords=[".thumbs"]
        )
        
        capture_cfg = self.config.get("capture", {})
        total_captures = capture_cfg.get("total_captures", 0)
        num_floating = capture_cfg.get("num_floating_captures_per_env", 0)
        num_dropped = capture_cfg.get("num_dropped_captures_per_env", 0)
        rt_subframes = capture_cfg.get("rt_subframes", 3)
        use_path_tracing = capture_cfg.get("path_tracing", False)
        disable_rp = capture_cfg.get("disable_render_products", False)
        dist_range = capture_cfg.get("camera_distance_to_target_range", (0.5, 1.5))

        
        keep_level = capture_cfg.get("keep_camera_level", False)
        floating_angles = capture_cfg.get("floating_per_camera_angles", {})
        dropped_angles = capture_cfg.get("dropped_per_camera_angles", {})
        
        # Convert integer keys from string (YAML dict keys are sometimes strings if not careful, safe to cast)
        floating_angles = {int(k): v for k, v in floating_angles.items()}
        dropped_angles = {int(k): v for k, v in dropped_angles.items()}

        # Setup Stage
        print(f"[SDG] Creating new stage")
        omni.usd.get_context().new_stage()
        rep.orchestrator.set_capture_on_play(False)
        
        self.setup_cameras()
        self.scan_content_library() # Scan once to get all labels
        self.setup_writers()
        self.setup_scene_lights()

        env_cycle = cycle(env_urls)
        capture_counter = 0
        
        while capture_counter < total_captures:
            env_url = next(env_cycle)
            print(f"[SDG] Loading environment: {env_url}")
            
            # Load/Replace environment at /Environment
            scene_based_sdg_utils.load_env(env_url, prim_path="/Environment")
            print(f"[SDG] Setting up environment")
            scene_based_sdg_utils.setup_env(root_path="/Environment", hide_top_walls=self.debug_mode)
            self.app.update()
            
            # Spawn New Assets for this cycle
            self.spawn_assets()
            
            # Register Randomizers for new assets
            self.setup_randomizers()

            # Find working area
            working_area_loc = scene_based_sdg_utils.get_matching_prim_location(
                match_string="TableDining", root_path="/Environment"
            )
            
            if self.debug_mode:
                cam_loc = (working_area_loc[0], working_area_loc[1], working_area_loc[2] + 10)
                set_camera_view(eye=np.array(cam_loc), target=np.array(working_area_loc))

            # Randomize Assets & Distractors
            print(f"\tRandomizing assets and distractors...")
            target_range = scene_based_sdg_utils.offset_range((-0.5, -0.5, 1, 0.5, 0.5, 1.5), working_area_loc)
            scene_based_sdg_utils.randomize_poses(self.target_assets, target_range, (0, 360), (0.95, 1.15))

            mesh_range = scene_based_sdg_utils.offset_range((-1, -1, 1, 1, 1, 2), working_area_loc)
            scene_based_sdg_utils.randomize_poses(self.mesh_distractors, mesh_range, (0, 360), (0.3, 1.0))

            shape_range = scene_based_sdg_utils.offset_range((-1.5, -1.5, 1, 1.5, 1.5, 2), working_area_loc)
            scene_based_sdg_utils.randomize_poses(self.shape_distractors, shape_range, (0, 360), (0.01, 0.1))

            lights_range = scene_based_sdg_utils.offset_range((-2, -2, 1, 2, 2, 3), working_area_loc)
            scene_based_sdg_utils.randomize_lights(self.scene_lights, lights_range, (0.1, 0.9)*3, (500, 2500))

            rep.utils.send_og_event(event_name="randomize_dome_lights")
            if self.shape_distractors:
                rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

            # Physics Settle
            print(f"\tSimulating physics...")
            self.app.update()
            scene_based_sdg_utils.run_simulation(num_frames=4, render=True)

            if disable_rp: self._toggle_updates(True)
            if use_path_tracing: self._set_renderer("PathTracing")

            for i in range(num_floating):
                if capture_counter >= total_captures: break
                
                # Randomize Cameras
                keep_level_list = list(range(len(self.cameras))) if keep_level else []
                scene_based_sdg_utils.randomize_camera_poses(
                    self.cameras, self.target_assets, dist_range, (0, 75), 
                    per_camera_polar_angles=floating_angles, keep_level_cameras=keep_level_list
                )
                
                print(f"\tCapturing floating {i+1}/{num_floating} (Total: {capture_counter+1})")
                rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
                capture_counter += 1

            if disable_rp: self._toggle_updates(False)
            if use_path_tracing: self._set_renderer("RayTracedLighting")

            print(f"\tRunning drop simulation...")
            scene_based_sdg_utils.run_simulation(num_frames=200, render=False)

            if disable_rp: self._toggle_updates(True)
            if use_path_tracing: self._set_renderer("PathTracing")

            for i in range(num_dropped):
                if capture_counter >= total_captures: break
                
                # Randomize Cameras (Top-Down preference)
                keep_level_list = list(range(len(self.cameras))) if keep_level else []
                scene_based_sdg_utils.randomize_camera_poses(
                    self.cameras, self.target_assets, dist_range, (0, 45),
                    per_camera_polar_angles=dropped_angles, keep_level_cameras=keep_level_list
                )

                print(f"\tCapturing dropped {i+1}/{num_dropped} (Total: {capture_counter+1})")
                rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
                capture_counter += 1

            if disable_rp: self._toggle_updates(False)
            if use_path_tracing: self._set_renderer("RayTracedLighting")

        # Cleanup
        rep.orchestrator.wait_until_complete()
        print(f"[SDG] Detaching writers")
        for w in self.writers: w.detach()
        print(f"[SDG] Destroying products")
        for rp in self.render_products: rp.destroy()
        print(f"[SDG] Finished. Captured {capture_counter * len(self.cameras)} frames.")

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
    if args.close_on_completion:
        # Seed logic for reproducibility in debug
        if config.get("debug_mode", False):
            np.random.seed(10)
            random.seed(10)
            rep.set_global_seed(10)
            
        sdg = SceneBasedSDG(simulation_app, config)
        sdg.run()
        simulation_app.close()
    else:
        # Debug mode stay open
        if config.get("debug_mode", False):
            np.random.seed(10)
            random.seed(10)
            rep.set_global_seed(10)

        sdg = SceneBasedSDG(simulation_app, config)
        sdg.run()

        while simulation_app.is_running():
            simulation_app.update()
        simulation_app.close()
