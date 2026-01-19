# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import yaml
import carb
import random
import time
from itertools import chain
import carb.settings

# Isaac Sim specific imports
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()

if not args.config or not os.path.isfile(args.config):
    raise FileNotFoundError(f"Config file is REQUIRED but not found: {args.config}")

config = {}
with open(args.config, "r") as f:
    if args.config.endswith(".json"):
        config = json.load(f)
    elif args.config.endswith(".yaml"):
        config = yaml.safe_load(f)
    else:
        carb.log_warn(f"File {args.config} is not json or yaml")

print(f"[SDG] Using config:\n{config}")

launch_config = config.get("launch_config", {})
simulation_app = SimulationApp(launch_config=launch_config)

import omni.replicator.core as rep
import omni.timeline
import omni.usd
import usdrt
from isaacsim.core.utils.semantics import add_labels, remove_labels, upgrade_prim_semantics_to_labels
from isaacsim.storage.native import get_assets_root_path
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
import math
from pxr import UsdLux

import object_based_sdg_utils

class ObjectBasedSDG:
    """
    Main application class for Object-Based Synthetic Data Generation.

    This class encapsulates the entire SDG pipeline, including:
    - Environment setup (stage loading, lighting)
    - Asset loading and spawning (labeled objects and distractors)
    - Camera configuration and management
    - Physics simulation setup and callbacks
    - Randomization logic (lights, colors, poses)
    - Data writer initialization and execution
    """
    def __init__(self, app, config):
        """
        Initialize the Object-Based SDG environment.

        Args:
            app (SimulationApp): The running Isaac Sim application instance.
            config (dict): Configuration dictionary loaded from YAML/JSON containing
                           parameters for assets, cameras, writer, and simulation settings.
        """
        self.app = app
        self.config = config
        self.assets_root_path = get_assets_root_path()
        self.stage = None
        self.timeline = None
        
        # State containers
        self.cameras = []
        self.camera_colliders = []
        self.render_products = []
        
        self.labeled_prims = []
        self.floating_labeled_prims = []
        self.falling_labeled_prims = []
        
        self.shape_distractors = []
        self.floating_shape_distractors = []
        self.falling_shape_distractors = []
        
        self.mesh_distractors = []
        self.floating_mesh_distractors = []
        self.falling_mesh_distractors = []
        
        self.physx_sub = None
        self.unique_labels = set() # Track for Writer categories

        self.available_assets = []  # Pool of all scanned assets for random selection

        # Config extraction
        self.working_area_size = self.config.get("working_area_size", (3, 3, 3))
        self.working_area_min = (self.working_area_size[0] / -2, self.working_area_size[1] / -2, self.working_area_size[2] / -2)
        self.working_area_max = (self.working_area_size[0] / 2, self.working_area_size[1] / 2, self.working_area_size[2] / 2)
        
        self.num_frames = self.config.get("num_frames", 10)
        self.rt_subframes = self.config.get("rt_subframes", -1)
        self.sim_duration_between_captures = self.config.get("simulation_duration_between_captures", 0.025)
        self.disable_render_products_between_captures = self.config.get("disable_render_products_between_captures", True)

        # Setup Routine
        self.setup_environment()
        self.setup_physics()
        self.setup_assets()
        self.setup_distractors()
        self.setup_cameras()
        self.setup_writer()
        self.setup_randomizers()
        self.setup_physics_callbacks()

    def setup_environment(self):
        """
        Initializes the USD stage for the environment.

        It either loads an existing stage from a URL (e.g., an Omniverse Nucleus path)
        specified in the config or creates a new empty stage. 
        
        If a new stage is created, it automatically adds a default distant light
        and randomizes its intensity based on the configuration.
        """
        env_url = self.config.get("env_url", "")
        if env_url:
            env_path = env_url if env_url.startswith("omniverse://") else self.assets_root_path + env_url
            omni.usd.get_context().open_stage(env_path)
            self.stage = omni.usd.get_context().get_stage()
            # Remove any previous semantics
            for prim in self.stage.Traverse():
                upgrade_prim_semantics_to_labels(prim, include_descendants=True)
                remove_labels(prim, include_descendants=True)
        else:
            omni.usd.get_context().new_stage()
            self.stage = omni.usd.get_context().get_stage()
            # Add a distant light
            distant_light = self.stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
            
            # Randomize intensity
            light_cfg = self.config.get("light_config", {})
            dist_min, dist_max = light_cfg.get("distant_light_intensity_min_max", (300, 600))
            intensity = random.uniform(dist_min, dist_max)
            distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(intensity)
            
            if not distant_light.HasAttribute("xformOp:rotateXYZ"):
                UsdGeom.Xformable(distant_light).AddRotateXYZOp()
            distant_light.GetAttribute("xformOp:rotateXYZ").Set((0, 60, 0))

    def setup_physics(self):
        """
        Sets up the physics scene and collision boundaries.

        This method works by:
        1. Creating invisible collision walls around the working area to keep assets contained.
        2. Initializing or retrieving the `PhysxScene` to control simulation parameters like FPS.
        """
        # Create collision box
        object_based_sdg_utils.create_collision_box_walls(
            self.stage, "/World/CollisionWalls", 
            self.working_area_size[0], self.working_area_size[1], self.working_area_size[2]
        )

        usdrt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        physics_scenes = usdrt_stage.GetPrimsWithAppliedAPIName("PhysxSceneAPI")
        if physics_scenes:
            self.physics_scene = physics_scenes[0]
        else:
             # Just getting the name for querying, but creating USD prim if needed
            self.stage.DefinePrim("/PhysicsScene", "PhysicsScene")
            UsdPhysics.Scene.Define(self.stage, "/PhysicsScene")
            self.physx_scene = PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath("/PhysicsScene"))
            
        # We need the API object for setting FPS later
        self.physx_scene_api = PhysxSchema.PhysxSceneAPI(self.stage.GetPrimAtPath("/PhysicsScene")) \
            if self.stage.GetPrimAtPath("/PhysicsScene").IsValid() else None
            
        # Ensure default props if we just created it or even if loaded
        if self.physx_scene_api:
             self.physx_scene_api.GetTimeStepsPerSecondAttr().Set(60)

    def setup_assets(self):
        """
        Scans and stores available labeled assets, then spawns initial random selection.

        Assets are gathered from the `labeled_assets` configuration block, which supports:
        1. **Auto-scan (`auto_label`)**: Scans specified folders/files, applies regex for labeling.
        2. **Manual config (`manual_label`)**: Explicitly listed assets with overrides.

        The scanned assets are stored in `self.available_assets` as a pool for random selection.
        """
        self.available_assets = []

        # New "labeled_assets" schema
        la_config = self.config.get("labeled_assets", {})

        # 1. Process Auto-Label
        auto_cfg = la_config.get("auto_label", {})
        if auto_cfg:
            scanned = object_based_sdg_utils.scan_assets(
                folders=auto_cfg.get("folders", []),
                files=auto_cfg.get("files", []),
                recursive=auto_cfg.get("recursive", False),
                regex_replace_pattern=auto_cfg.get("regex_replace_pattern"),
                regex_replace_repl=auto_cfg.get("regex_replace_repl", "")
            )

            # Common properties for auto-labeled assets
            scale_min_max = auto_cfg.get("scale_min_max", (1, 1))
            # gravity_disabled_chance: probability [0.0-1.0] that an asset floats (gravity disabled)
            # 0.0 = all fall, 1.0 = all float, 0.5 = 50% float/50% fall
            gravity_disabled_chance = auto_cfg.get("gravity_disabled_chance", 0.0)

            for asset in scanned:
                asset["scale_min_max"] = scale_min_max
                asset["gravity_disabled_chance"] = gravity_disabled_chance
                self.available_assets.append(asset)

        # 2. Process Manual-Label
        manual_cfg = la_config.get("manual_label", [])
        if manual_cfg:
            for item in manual_cfg:
                self.available_assets.append(item)

        print(f"[SDG] Total available assets in pool: {len(self.available_assets)}")

        # Pre-populate unique_labels with ALL labels from the pool
        # This ensures the writer has all possible categories, even if not all are spawned initially
        for asset in self.available_assets:
            label = asset.get("label", "unknown")
            # Sanitize label for semantic system (no spaces allowed in Isaac Sim labeling)
            semantic_label = label.replace(" ", "_")
            self.unique_labels.add(semantic_label)
        print(f"[SDG] Total unique labels in pool: {len(self.unique_labels)}")

        # Spawn initial random selection
        self.spawn_random_labeled_assets()

    def spawn_random_labeled_assets(self):
        """
        Clears existing labeled assets and spawns a new random selection from the pool.

        This method:
        1. Removes any existing labeled prims from the stage
        2. Randomly selects N assets from `self.available_assets`
        3. Spawns each selected asset with random transforms

        Called during initial setup and whenever the environment changes.
        """
        # 1. Clear existing labeled prims
        for prim in self.labeled_prims:
            if prim.IsValid():
                self.stage.RemovePrim(prim.GetPath())
        self.labeled_prims = []
        self.floating_labeled_prims = []
        self.falling_labeled_prims = []

        # 2. Get count from config
        la_config = self.config.get("labeled_assets", {})
        auto_cfg = la_config.get("auto_label", {})
        count = auto_cfg.get("num", 1)

        # 3. Random selection from pool
        if not self.available_assets:
            print("[SDG] No assets available in pool")
            return

        if len(self.available_assets) <= count:
            selected = self.available_assets
        else:
            selected = random.sample(self.available_assets, count)

        print(f"[SDG] Spawning {len(selected)} randomly selected assets")

        # Use full working area centered at origin
        spawn_loc_min = self.working_area_min
        spawn_loc_max = self.working_area_max

        # 4. Spawn each selected asset
        for obj in selected:
            obj_url = obj.get("url", "")
            label = obj.get("label", "unknown")
            gravity_disabled_chance = obj.get("gravity_disabled_chance", 0.0)
            # Per-asset random decision: disable gravity if random() < chance
            floating = random.random() < gravity_disabled_chance
            scale_min_max = obj.get("scale_min_max", (1, 1))

            rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
                loc_min=spawn_loc_min, loc_max=spawn_loc_max, scale_min_max=scale_min_max
            )

            # Sanitize label for USD prim path (replace spaces and special chars with underscores)
            prim_name = re.sub(r'[^a-zA-Z0-9_]', '_', label).strip()
            prim_path = omni.usd.get_stage_next_free_path(self.stage, f"/World/Labeled/{prim_name}", False)
            prim = self.stage.DefinePrim(prim_path, "Xform")

            if obj_url.startswith("omniverse://") or os.path.isabs(obj_url):
                asset_path = obj_url
            else:
                asset_path = self.assets_root_path + obj_url

            prim.GetReferences().AddReference(asset_path)
            object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
            object_based_sdg_utils.add_colliders(prim)
            object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=floating)
            # Sanitize label for semantic system (no spaces allowed in Isaac Sim labeling)
            semantic_label = label.replace(" ", "_")
            add_labels(prim, labels=[semantic_label], instance_name="class")
            self.unique_labels.add(semantic_label)

            if floating:
                self.floating_labeled_prims.append(prim)
            else:
                self.falling_labeled_prims.append(prim)

        self.labeled_prims = self.floating_labeled_prims + self.falling_labeled_prims
        print(f"[SDG] Spawned: {len(self.floating_labeled_prims)} floating, {len(self.falling_labeled_prims)} falling")

    def setup_distractors(self):
        """
        Spawns shape and mesh distractors into the environment.

        Distractors are non-labeled objects used to add complexity to the scene.
        
        This method handles:
        - **Shape Distractors**: Primitive shapes (capsule, cone, etc.) generated procedurally.
          Some are set to float (gravity disabled), others fall.
        - **Mesh Distractors**: Loaded from USD files specified in the config.
        """
        # Shape Distractors
        shape_types = self.config.get("shape_distractors_types", ["capsule", "cone", "cylinder", "sphere", "cube"])
        shape_scale = self.config.get("shape_distractors_scale_min_max", (0.02, 0.2))
        shape_num = self.config.get("shape_distractors_num", 350)

        for _ in range(shape_num):
            rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
                loc_min=self.working_area_min, loc_max=self.working_area_max, scale_min_max=shape_scale
            )
            rand_shape = random.choice(shape_types)
            prim_path = omni.usd.get_stage_next_free_path(self.stage, f"/World/Distractors/{rand_shape}", False)
            prim = self.stage.DefinePrim(prim_path, rand_shape.capitalize())
            object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
            object_based_sdg_utils.add_colliders(prim)
            disable_gravity = random.choice([True, False])
            object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity)
            if disable_gravity:
                self.floating_shape_distractors.append(prim)
            else:
                self.falling_shape_distractors.append(prim)
            self.shape_distractors.append(prim)

        # Mesh Distractors
        mesh_urls = self.config.get("mesh_distractors_urls", [])
        mesh_scale = self.config.get("mesh_distractors_scale_min_max", (0.1, 2.0))
        mesh_num = self.config.get("mesh_distractors_num", 10)

        for _ in range(mesh_num):
            rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
                loc_min=self.working_area_min, loc_max=self.working_area_max, scale_min_max=mesh_scale
            )
            mesh_url = random.choice(mesh_urls)
            prim_name = os.path.basename(mesh_url).split(".")[0]
            prim_path = omni.usd.get_stage_next_free_path(self.stage, f"/World/Distractors/{prim_name}", False)
            prim = self.stage.DefinePrim(prim_path, "Xform")
            asset_path = mesh_url if mesh_url.startswith("omniverse://") else self.assets_root_path + mesh_url
            prim.GetReferences().AddReference(asset_path)
            object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
            object_based_sdg_utils.add_colliders(prim)
            disable_gravity = random.choice([True, False])
            object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=disable_gravity)
            if disable_gravity:
                self.floating_mesh_distractors.append(prim)
            else:
                self.falling_mesh_distractors.append(prim)
            self.mesh_distractors.append(prim)
            upgrade_prim_semantics_to_labels(prim, include_descendants=True)
            remove_labels(prim, include_descendants=True)

    def setup_cameras(self):
        """
        Creates cameras and render products for data capture.

        This method:
        1. Creates camera prims based on the `num_cameras` config.
        2. Applies camera properties (focal length, etc.).
        3. Optionally adds collision spheres to cameras to prevent assets from clipping through them.
        4. Initializes `render_products` for Replicator to capture images from these cameras.
        """
        rep.orchestrator.set_capture_on_play(False)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2) # Quality

        num_cameras = self.config.get("num_cameras", 1)
        camera_props = self.config.get("camera_properties_kwargs", {})

        # Get focal length from camera_positioning config (must match FOV calculation)
        camera_config = self.config.get("camera_positioning", {})
        focal_length = camera_config.get("focal_length", 35.0)  # mm
        print(f"[SDG] Creating {num_cameras} cameras with focal length: {focal_length}mm")

        for i in range(num_cameras):
            cam_prim = self.stage.DefinePrim(f"/World/Cameras/cam_{i}", "Camera")

            # Set focal length to match FOV calculations (critical for background plane sizing)
            if cam_prim.HasAttribute("focalLength"):
                cam_prim.GetAttribute("focalLength").Set(focal_length)

            for key, value in camera_props.items():
                if cam_prim.HasAttribute(key):
                    cam_prim.GetAttribute(key).Set(value)
                else:
                    print(f"Unknown camera attribute: {key}")
            self.cameras.append(cam_prim)

        # Camera Colliders
        collider_radius = self.config.get("camera_collider_radius", 0)
        if collider_radius > 0:
            for cam in self.cameras:
                cam_path = cam.GetPath()
                cam_collider = self.stage.DefinePrim(f"{cam_path}/CollisionSphere", "Sphere")
                cam_collider.GetAttribute("radius").Set(collider_radius)
                object_based_sdg_utils.add_colliders(cam_collider)
                collision_api = UsdPhysics.CollisionAPI(cam_collider)
                collision_api.GetCollisionEnabledAttr().Set(False)
                UsdGeom.Imageable(cam_collider).MakeInvisible()
                self.camera_colliders.append(cam_collider)

        self.app.update()
        
        # Render Products
        res = self.config.get("resolution", (640, 480))
        for cam in self.cameras:
            rp = rep.create.render_product(cam.GetPath(), res)
            self.render_products.append(rp)

        if self.disable_render_products_between_captures:
            object_based_sdg_utils.set_render_products_updates(self.render_products, False, include_viewport=False)

    def setup_writer(self):
        """
        Initializes the Replicator Writer.

        The writer (defaulting to "CocoWriter" as preferred) is configured with output paths
        and attached to the created render products to save the generated data.
        """
        writer_type = self.config.get("writer_type", "CocoWriter")
        writer_kwargs = self.config.get("writer_kwargs", {})
        
        if out_dir := writer_kwargs.get("output_dir"):
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.getcwd(), out_dir)
                writer_kwargs["output_dir"] = out_dir
            print(f"[SDG] Writing data to: {out_dir}")

        print(f"[SDG] Initializing {writer_type} Writer")
        if self.render_products:
            # Dynamic COCO Category Generation
            if writer_type == "CocoWriter":
                # Ensure semantic_types is present (fixes 'other' class issue)
                if "semantic_types" not in writer_kwargs:
                    writer_kwargs["semantic_types"] = ["class"]
                
                # Ensure coco_categories is present (fixes missing IDs)
                if "coco_categories" not in writer_kwargs:
                    print("[SDG] Auto-generating coco_categories...")
                    writer_kwargs["coco_categories"] = self._generate_coco_categories()

            self.writer = rep.writers.get(writer_type)
            self.writer.initialize(**writer_kwargs)
            self.writer.attach(self.render_products)

    def _generate_coco_categories(self):
        """
        Generates the 'coco_categories' dictionary for the CocoWriter.
        
        This matches the Labels found in the scene to unique IDs and Colors,
        satisfying the CocoWriter's requirement for specific category definitions.
        
        Returns:
            list[dict]: A List of category dictionaries (standard for some Writers) 
                        OR a Dict of Dicts depending on the specific Writer version. 
                        We conform to the Dict structure:
                        {
                           "name_key": { "id": int, "name": str, ... }
                        }
        """
        categories = {}
        # Use the set we populated during spawning (Robuster than querying USD attributes)
        ordered_labels = sorted(list(self.unique_labels))

        # Get supercategory from config (defaults to "object")
        la_config = self.config.get("labeled_assets", {})
        supercategory = la_config.get("supercategory", "object")

        print(f"[SDG] Found {len(ordered_labels)} unique labels: {ordered_labels}")

        for i, label in enumerate(ordered_labels):
            # ID 0 is usually reserved for background
            cat_id = i + 1

            # Generate a consistent color hash or random
            # Just using random for now effectively, but seeding could make it deterministic
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

            categories[label] = {
                "name": label.replace("_", " "),
                "id": cat_id,
                "supercategory": supercategory,
                "isthing": 1,
                "color": color
            }

        return categories

    def inject_all_categories_to_coco(self):
        """
        Post-processes COCO JSON to ensure ALL categories from unique_labels are included,
        even if they have no annotations in the generated images.

        Uses the SAME deterministic ID mapping from _generate_coco_categories() to ensure
        consistency - IDs are assigned based on alphabetical order of labels.
        """
        import json
        import glob

        writer_kwargs = self.config.get("writer_kwargs", {})
        output_dir = writer_kwargs.get("output_dir", "")

        # CocoWriter outputs with random suffix (e.g., coco_annotations_xncoejgm.json)
        # Find the most recent matching file
        coco_pattern = os.path.join(output_dir, "coco_annotations*.json")
        coco_files = glob.glob(coco_pattern)
        if not coco_files:
            print(f"[SDG] No COCO files found matching: {coco_pattern}")
            return

        # Use the most recently modified file
        coco_file = max(coco_files, key=os.path.getmtime)
        print(f"[SDG] Processing COCO file: {os.path.basename(coco_file)}")

        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        # Get existing category names (IDs are already correct from generation)
        existing_names = {cat['name'] for cat in coco_data.get('categories', [])}

        # Get the FULL category mapping with original deterministic IDs
        # (IDs are assigned alphabetically: sorted label at index 0 → ID 1, etc.)
        all_categories = self._generate_coco_categories()

        # Add missing categories with their ORIGINAL IDs
        added = 0
        for label, cat_info in all_categories.items():
            human_readable_name = label.replace("_", " ")
            if human_readable_name not in existing_names:
                coco_data['categories'].append({
                    "id": cat_info["id"],  # Use original deterministic ID
                    "name": human_readable_name,
                    "supercategory": cat_info.get("supercategory", "object")
                })
                added += 1

        # Sort categories by ID for cleaner output
        coco_data['categories'] = sorted(coco_data['categories'], key=lambda x: x['id'])

        if added > 0:
            with open(coco_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            print(f"[SDG] Injected {added} additional categories. Total: {len(coco_data['categories'])}")
        else:
            print(f"[SDG] All {len(coco_data['categories'])} categories already present")

    def _load_png_backgrounds(self):
        """
        Load PNG background images from config folder, sampling every Nth.

        Returns:
            list: List of absolute paths to sampled PNG images.
        """
        import glob

        bg_config = self.config.get("png_background", {})
        if not bg_config.get("enabled", False):
            return []

        folder = bg_config.get("folder", "")
        if not folder or not os.path.isdir(folder):
            print(f"[SDG] PNG background folder not found: {folder}")
            return []

        sample_n = bg_config.get("sample_every_n", 1)

        # Get all PNGs sorted by name
        pattern = os.path.join(folder, "*.png")
        all_images = sorted(glob.glob(pattern))

        if not all_images:
            print(f"[SDG] No PNG files found in: {folder}")
            return []

        # Sample every Nth image
        sampled = all_images[::sample_n]

        print(f"[SDG] PNG backgrounds: {len(sampled)} images (sampled every {sample_n} from {len(all_images)} total)")
        return sampled

    def _calculate_camera_fov(self):
        """
        Calculate camera horizontal and vertical FOV from focal length and resolution.

        Uses:
        - focal_length from camera_positioning config (default 35mm)
        - resolution from config for aspect ratio
        - 36mm sensor width (USD full-frame standard)

        Returns:
            tuple: (fov_horizontal, fov_vertical) in radians.
        """
        import math

        # Get focal length from config (default 35mm)
        camera_config = self.config.get("camera_positioning", {})
        focal_length = camera_config.get("focal_length", 35.0)  # mm

        # USD camera uses 36mm sensor width (full frame equivalent)
        sensor_width = 36.0  # mm

        # Get resolution for aspect ratio
        resolution = self.config.get("resolution", [640, 640])
        aspect_ratio = resolution[0] / resolution[1]  # width / height

        # Calculate horizontal FOV
        fov_h = 2 * math.atan(sensor_width / (2 * focal_length))

        # Calculate vertical FOV based on aspect ratio
        sensor_height = sensor_width / aspect_ratio
        fov_v = 2 * math.atan(sensor_height / (2 * focal_length))

        return fov_h, fov_v

    def create_png_background_plane(self, image_path: str):
        """
        Create a single flat backdrop plane behind the working area (studio-style setup).

        The plane is sized to match the camera's field of view at its distance,
        ensuring the entire background is visible in frame.
        Camera will be constrained to +Z side looking toward -Z (at the backdrop).

        Args:
            image_path: Path to PNG image file.

        Returns:
            Usd.Prim: The background plane prim.
        """

        prim_path = "/World/BackgroundPlane"

        # Get working area dimensions
        wx, wy, wz = self.working_area_size

        # Position plane behind working area (at -Z)
        # backdrop_distance controls how far behind the working area edge the plane is placed
        bg_config = self.config.get("png_background", {})
        backdrop_distance = bg_config.get("backdrop_distance", 1.0)
        plane_z = -(wz / 2 + backdrop_distance)

        # Calculate MAXIMUM distance from camera to plane
        # This ensures the plane fills the frame for the camera furthest away
        # In studio mode: camera at Z = distance, plane at Z = plane_z
        cam_dist_max = self.config.get("camera_distance_to_target_min_max", [0.75, 1.5])[1]
        # Camera at maximum distance from origin, looking at plane
        camera_to_plane_distance = cam_dist_max + abs(plane_z)

        # Calculate visible area using FOV and resolution
        fov_h, fov_v = self._calculate_camera_fov()
        visible_width = 2 * camera_to_plane_distance * math.tan(fov_h / 2)
        visible_height = 2 * camera_to_plane_distance * math.tan(fov_v / 2)

        # Size plane to fill frame for furthest camera, with margin for safety
        # Margin ensures full coverage even with slight camera angle variations
        plane_margin = bg_config.get("plane_margin", 1.0)  # 1.0 = exact fit at max camera distance
        plane_width = visible_width * plane_margin
        plane_height = visible_height * plane_margin

        print(f"[SDG] Background plane: {plane_width:.2f}x{plane_height:.2f}m at Z={plane_z:.1f} (camera-to-plane: {camera_to_plane_distance:.2f}m, margin: {plane_margin})")

        # Delete existing plane if present (ensures config changes take effect)
        plane_prim = self.stage.GetPrimAtPath(prim_path)
        if plane_prim.IsValid():
            self.stage.RemovePrim(prim_path)
            print(f"[SDG] Removed existing backdrop plane for recreation")

        plane_prim = self.stage.GetPrimAtPath(prim_path)
        if not plane_prim.IsValid():
            # Create single backdrop plane facing +Z (toward camera)
            plane_mesh = UsdGeom.Mesh.Define(self.stage, prim_path)

            hw, hh = plane_width / 2, plane_height / 2
            points = [
                (-hw, -hh, plane_z),  # Bottom-left
                (hw, -hh, plane_z),   # Bottom-right
                (hw, hh, plane_z),    # Top-right
                (-hw, hh, plane_z),   # Top-left
            ]

            # Simple 1:1 UV mapping (no tiling - preserves resolution)
            uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
            face_vertex_counts = [4]
            face_vertex_indices = [0, 1, 2, 3]

            plane_mesh.GetPointsAttr().Set(points)
            plane_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
            plane_mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

            # Set UVs
            primvars_api = UsdGeom.PrimvarsAPI(plane_mesh)
            texcoord_primvar = primvars_api.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            texcoord_primvar.Set(uvs)

            plane_mesh.GetDoubleSidedAttr().Set(True)
            plane_prim = plane_mesh.GetPrim()

            # Add collision so objects can't pass through the backdrop
            object_based_sdg_utils.add_colliders(plane_prim)

            print(f"[SDG] Created backdrop plane: {plane_width:.1f}x{plane_height:.1f}m at Z={plane_z:.1f}")

        # Set texture
        self._set_plane_texture(prim_path, image_path)

        # Create ambient dome light for scene illumination
        dome_path = "/World/AmbientDome"
        if not self.stage.GetPrimAtPath(dome_path).IsValid():
            dome = UsdLux.DomeLight.Define(self.stage, dome_path)
            dome.GetIntensityAttr().Set(500.0)
            print(f"[SDG] Created ambient dome light for scene illumination")

        return plane_prim

    def _set_plane_texture(self, plane_path: str, image_path: str):
        """
        Set or update the texture on the background plane.

        Args:
            plane_path: USD path to the plane prim.
            image_path: Path to PNG image file.
        """
        from pxr import UsdShade

        mat_path = f"{plane_path}/Material"
        tex_path = f"{mat_path}/DiffuseTexture"

        # Check if material exists
        mat_prim = self.stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            # Create material
            material = UsdShade.Material.Define(self.stage, mat_path)

            # Create primvar reader to get UV coordinates from mesh
            st_reader = UsdShade.Shader.Define(self.stage, f"{mat_path}/STReader")
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            # Create texture reader
            tex_reader = UsdShade.Shader.Define(self.stage, tex_path)
            tex_reader.CreateIdAttr("UsdUVTexture")
            tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(image_path)
            tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
            tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
            # Connect UV coordinates from primvar reader
            tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.GetOutput("result")
            )
            tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

            # Create shader - fully emissive so background glows regardless of lighting
            shader = UsdShade.Shader.Define(self.stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

            # Set diffuse to black so background is not affected by scene lighting
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))

            # Connect texture to emissive only - background is self-lit and unaffected by lights
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex_reader.GetOutput("rgb")
            )

            # Connect shader to material
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            # Bind material to plane
            plane_prim = self.stage.GetPrimAtPath(plane_path)
            UsdShade.MaterialBindingAPI(plane_prim).Bind(material)
        else:
            # Update texture file path
            tex_reader = UsdShade.Shader.Get(self.stage, tex_path)
            file_input = tex_reader.GetInput("file")
            if file_input:
                file_input.Set(image_path)

    def _update_skybox_texture(self, image_path: str):
        """
        Update texture on the backdrop plane.

        Args:
            image_path: Path to PNG image file.
        """
        prim_path = "/World/BackgroundPlane"
        if self.stage.GetPrimAtPath(prim_path).IsValid():
            self._set_plane_texture(prim_path, image_path)

    def setup_randomizers(self):
        """
        Defines Replicator Randomizers and Triggers.

        This sets up the randomizer graphs (Replicator logic) for:
        - **Shape Colors**: Randomizing the color of shape distractors.
        - **Lights**: Randomizing position, intensity, and temperature of scene lights.
        - **Dome Background**: Swapping HDRI textures for the environment background.
        - **Asset Appearance**: Randomizing the color of labeled assets (if enabled).

        These randomizers are triggered by custom events (e.g., `randomize_lights`) which are
        fired periodically in the main loop.
        """
        
        # Shape Colors (only if there are shape distractors)
        if self.floating_shape_distractors or self.falling_shape_distractors:
            with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
                paths = [p.GetPath() for p in chain(self.floating_shape_distractors, self.falling_shape_distractors)]
                group = rep.create.group(paths)
                with group:
                    rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

        # Lights
        with rep.trigger.on_custom_event(event_name="randomize_lights"):
            light_cfg = self.config.get("light_config", {})
            l_count = light_cfg.get("count", 3)
            l_type = light_cfg.get("type", "Sphere")
            l_int_min, l_int_max = light_cfg.get("intensity_min_max", (30000, 40000))
            l_temp_min, l_temp_max = light_cfg.get("temperature_min_max", (6000, 7000))
            l_color_min = tuple(light_cfg.get("color_min", (0, 0, 0)))
            l_color_max = tuple(light_cfg.get("color_max", (1, 1, 1)))

            rep.create.light(
                light_type=l_type,
                color=rep.distribution.uniform(l_color_min, l_color_max),
                temperature=rep.distribution.normal((l_temp_min + l_temp_max)/2, (l_temp_max - l_temp_min)/4),
                intensity=rep.distribution.normal((l_int_min + l_int_max)/2, (l_int_max - l_int_min)/4),
                position=rep.distribution.uniform(self.working_area_min, self.working_area_max),
                scale=rep.distribution.uniform(0.1, 1),
                count=l_count,
            )

        # PNG Background using DomeLight (if enabled)
        bg_config = self.config.get("png_background", {})
        png_enabled = bg_config.get("enabled", False)

        if png_enabled:
            # Load PNG images and create textured background plane
            self.png_background_images = self._load_png_backgrounds()
            if self.png_background_images:
                initial_image = random.choice(self.png_background_images)
                self.create_png_background_plane(initial_image)
                print(f"[SDG] Initial PNG background: {os.path.basename(initial_image)}")
        else:
            # Original HDRI dome behavior (only when PNG not enabled)
            with rep.trigger.on_custom_event(event_name="randomize_dome_background"):
                textures = [
                    self.assets_root_path + "/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                    self.assets_root_path + "/NVIDIA/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                    self.assets_root_path + "/NVIDIA/Assets/Skies/Indoor/hotel_room_4k.hdr",
                    self.assets_root_path + "/NVIDIA/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
                ]
                dome_light = rep.create.light(light_type="Dome")
                with dome_light:
                    rep.modify.attribute("inputs:texture:file", rep.distribution.choice(textures))
                    rep.randomizer.rotation()

        # Asset Appearance
        if self.config.get("assets", {}).get("properties", {}).get("randomize_color", False) and self.labeled_prims:
            with rep.trigger.on_custom_event(event_name="randomize_asset_appearance"):
                paths = [p.GetPath().pathString for p in self.labeled_prims]
                group = rep.create.group(paths)
                with group:
                    rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

    def _on_overlap_hit(self, hit):
        """
        Callback for physics overlap query.
        
        This makes objects 'bounce' if they overlap with the specified area.
        The callback is invoked for each object found in the query area.
        
        Args:
            hit: The hit information containing the rigid body path.
        Returns:
            bool: True to continue the query for other overlaps.
        """
        prim = self.stage.GetPrimAtPath(hit.rigid_body)
        if prim not in self.camera_colliders:
            rand_vel = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(4, 8))
            prim.GetAttribute("physics:velocity").Set(rand_vel)
        return True

    def _on_physics_step(self, dt: float):
        """
        Physics step callback to check for overlaps.

        Executes an overlap box query in the physics scene to detect objects near the floor/boundary
        and trigger the `_on_overlap_hit` callback to apply bouncing forces.

        Args:
            dt (float): Delta time (unused but required by signature).
        """
        overlap_area_thickness = 0.1
        overlap_origin = (0, 0, (-self.working_area_size[2] / 2) + (overlap_area_thickness / 2))
        overlap_extent = (
            self.working_area_size[0] / 2 * 0.99,
            self.working_area_size[1] / 2 * 0.99,
            overlap_area_thickness / 2 * 0.99,
        )
        
        get_physx_scene_query_interface().overlap_box(
            carb.Float3(overlap_extent),
            carb.Float3(overlap_origin),
            carb.Float4(0, 0, 0, 1),
            self._on_overlap_hit,
            False,
        )

    def setup_physics_callbacks(self):
        """Subscribes to physics step events to trigger `_on_physics_step` every frame."""
        self.physx_sub = get_physx_interface().subscribe_physics_step_events(self._on_physics_step)

    def randomize_camera_poses(self):
        """
        Randomizes camera positions looking at assets.

        Supports two modes:
        - "sphere" (default): Camera on random point on sphere around target
        - "studio": Camera constrained to +Z hemisphere facing backdrop (for PNG backgrounds)
        """
        import math

        min_dist, max_dist = self.config.get("camera_distance_to_target_min_max", (0.1, 0.5))
        offset = self.config.get("camera_look_at_target_offset", 0.2)

        # Check for studio mode (when PNG background is enabled)
        bg_config = self.config.get("png_background", {})
        camera_config = self.config.get("camera_positioning", {})
        studio_mode = bg_config.get("enabled", False) or camera_config.get("mode") == "studio"

        for cam in self.cameras:
            distance = random.uniform(min_dist, max_dist)

            if studio_mode:
                # Studio mode: camera at center (0, 0, distance) looking straight at backdrop
                # Camera is centered with background plane, looking down -Z axis
                cam_loc = Gf.Vec3f(0, 0, distance)

                # Camera looks straight down -Z axis (toward background)
                # No rotation needed - USD camera default forward is -Z
                # Set rotation to identity quaternion (no rotation)
                quat = Gf.Quatf(1, 0, 0, 0)  # Identity quaternion (w, x, y, z)

                object_based_sdg_utils.set_transform_attributes(cam, location=cam_loc, orientation=quat)
            else:
                # Default sphere mode - camera orbits around target
                target_asset = random.choice(self.labeled_prims)
                loc_offset = (
                    random.uniform(-offset, offset),
                    random.uniform(-offset, offset),
                    random.uniform(-offset, offset),
                )
                target_loc = target_asset.GetAttribute("xformOp:translate").Get() + loc_offset
                cam_loc, quat = object_based_sdg_utils.get_random_pose_on_sphere(origin=target_loc, radius=distance)
                object_based_sdg_utils.set_transform_attributes(cam, location=cam_loc, orientation=quat)

    def simulate_camera_collision(self, frames=1):
        """
        Pushes objects away from camera.
        
        Temporarily enables collision spheres around the cameras and runs a few simulation steps.
        This ensures that when a camera is placed, if it clips into an object, the object is
        physically pushed away to avoid a bad rendering.

        Args:
            frames (int, optional): Number of physics frames to simulate. Defaults to 1.
        """
        for col in self.camera_colliders:
            UsdPhysics.CollisionAPI(col).GetCollisionEnabledAttr().Set(True)
        
        if not self.timeline.is_playing():
            self.timeline.play()
            
        for _ in range(frames):
            self.app.update()
            
        for col in self.camera_colliders:
            UsdPhysics.CollisionAPI(col).GetCollisionEnabledAttr().Set(False)

    def capture_with_motion_blur(self, duration=0.05, num_samples=8, spp=64):
        """
        Advanced capture with pathtracing and motion blur.

        To achieve motion blur:
        1. Temporarily increases physics timestep frequency to allow for sub-steps.
        2. Configures Replicator/RTX settings for path-traced motion blur.
        3. Steps the simulation and renderer forward to capture effectively over a time duration.
        4. Restores original physics/render settings.

        Args:
            duration (float, optional): Duration of the capture (shutter speed equivalent) in seconds.
            num_samples (int, optional): Number of sub-samples for the blur.
            spp (int, optional): Sample per pixel for path tracing.
        """
        if not self.physx_scene_api:
            return

        orig_fps = self.physx_scene_api.GetTimeStepsPerSecondAttr().Get()
        target_fps = 1 / duration * num_samples
        if target_fps > orig_fps:
            self.physx_scene_api.GetTimeStepsPerSecondAttr().Set(target_fps)

        settings = carb.settings.get_settings()
        mb_enabled = settings.get("/omni/replicator/captureMotionBlur")
        if not mb_enabled:
             settings.set("/omni/replicator/captureMotionBlur", True)
        settings.set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)

        prev_mode = settings.get("/rtx/rendermode")
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", spp)
        settings.set("/rtx/pathtracing/totalSpp", spp)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", 0)

        if not self.timeline.is_playing():
            self.timeline.play()

        rep.orchestrator.step(delta_time=duration, pause_timeline=False)

        if target_fps > orig_fps:
            self.physx_scene_api.GetTimeStepsPerSecondAttr().Set(orig_fps)

        settings.set("/omni/replicator/captureMotionBlur", mb_enabled)
        settings.set("/rtx/rendermode", prev_mode)

    def run_simulation_loop(self, duration):
        """
        Simulates the world for a specific duration without capturing.
        
        Used to let the physics settle or evolve between captured frames.

        Args:
            duration (float): Time in seconds to simulate.
        """
        elapsed = 0.0
        prev_time = self.timeline.get_current_time()
        if not self.timeline.is_playing():
            self.timeline.play()
            
        while elapsed <= duration:
            self.app.update()
            curr = self.timeline.get_current_time()
            elapsed += curr - prev_time
            prev_time = curr

    def run(self):
        """
        Main execution loop.

        Steps:
        1. Triggers initial randomizations to settle the scene.
        2. Starts the timeline.
        3. Iterates through `num_frames` capture cycles:
           - Randomizes environment (cameras, lights, distractors) at set intervals.
           - Steps Replicator (capture) or performs Motion Blur capture.
           - Simulates physics between captures.
        4. Waits for the async writer to finish.
        5. Reports execution statistics (FPS, time).
        """
        # Initial Randomization triggers to ensure load
        rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
        rep.utils.send_og_event(event_name="randomize_dome_background")
        for _ in range(5):
            self.app.update()

        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.set_start_time(0)
        self.timeline.set_end_time(1000000)
        self.timeline.set_looping(False)
        self.timeline.play()
        self.timeline.commit()
        self.app.update()

        wall_start = time.perf_counter()

        # Get randomization intervals from config (0 = disabled)
        intervals = self.config.get("randomization_intervals", {})
        interval_camera = intervals.get("camera_poses", 3)
        interval_velocity = intervals.get("velocity_pull", 10)
        interval_lights = intervals.get("lights", 5)
        interval_shape_colors = intervals.get("shape_colors", 15)
        interval_dome = intervals.get("dome_background", 25)
        interval_png_bg = intervals.get("png_background", 25)
        interval_asset_reselection = intervals.get("asset_reselection", 25)
        interval_distractor_vel = intervals.get("distractor_velocities", 17)
        interval_motion_blur = intervals.get("motion_blur_capture", 5)

        # Camera collision and motion blur settings
        camera_collision_frames = self.config.get("camera_collision_frames", 4)
        motion_blur_cfg = self.config.get("motion_blur", {})
        mb_duration = motion_blur_cfg.get("duration", 0.025)
        mb_samples = motion_blur_cfg.get("num_samples", 8)
        mb_spp = motion_blur_cfg.get("spp", 128)

        for i in range(self.num_frames):
            print(f"[SDG] Processing Frame {i+1}/{self.num_frames}")

            # 1. Camera Randomization
            if interval_camera > 0 and i % interval_camera == 0:
                self.randomize_camera_poses()
                if self.camera_colliders:
                    self.simulate_camera_collision(frames=camera_collision_frames)

            # 2. Velocity Pull
            if interval_velocity > 0 and i % interval_velocity == 0:
                object_based_sdg_utils.apply_velocities_towards_target(
                    chain(self.labeled_prims, self.shape_distractors, self.mesh_distractors)
                )

            # 3. Lights
            if interval_lights > 0 and i % interval_lights == 0:
                rep.utils.send_og_event(event_name="randomize_lights")

            # 4. Shape Colors
            if interval_shape_colors > 0 and i % interval_shape_colors == 0:
                rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

            # 5. Dome Background
            if interval_dome > 0 and i % interval_dome == 0:
                rep.utils.send_og_event(event_name="randomize_dome_background")

            # 5b. PNG Background (if enabled)
            if interval_png_bg > 0 and i % interval_png_bg == 0:
                if hasattr(self, 'png_background_images') and self.png_background_images:
                    new_image = random.choice(self.png_background_images)
                    self._update_skybox_texture(new_image)
                    print(f"[SDG] Changed PNG background to: {os.path.basename(new_image)}")

            # 6. Asset Re-selection (can be tied to dome or independent)
            if interval_asset_reselection > 0 and i % interval_asset_reselection == 0:
                self.spawn_random_labeled_assets()

            # 7. Distractor Velocities
            if interval_distractor_vel > 0 and i % interval_distractor_vel == 0:
                object_based_sdg_utils.randomize_floating_distractor_velocities(
                    chain(self.floating_shape_distractors, self.floating_mesh_distractors)
                )

            # 8. Asset Colors (if configured)
            if self.config.get("assets", {}).get("properties", {}).get("randomize_color", False):
                rep.utils.send_og_event(event_name="randomize_asset_appearance")

            # Capture
            if self.disable_render_products_between_captures:
                object_based_sdg_utils.set_render_products_updates(self.render_products, True)

            if interval_motion_blur > 0 and i % interval_motion_blur == 0:
                self.capture_with_motion_blur(duration=mb_duration, num_samples=mb_samples, spp=mb_spp)
            else:
                rep.orchestrator.step(delta_time=0.0, rt_subframes=self.rt_subframes, pause_timeline=False)

            if self.disable_render_products_between_captures:
                object_based_sdg_utils.set_render_products_updates(self.render_products, False)

            # Sim between frames
            if self.sim_duration_between_captures > 0:
                self.run_simulation_loop(self.sim_duration_between_captures)
            else:
                self.app.update()

        rep.orchestrator.wait_until_complete()
        
        wall_end = time.perf_counter()
        print(f"[SDG] Finished. Total time: {wall_end - wall_start:.2f}s")

        # Post-process to ensure all categories are in COCO output
        if self.config.get("writer_type") == "CocoWriter":
            self.inject_all_categories_to_coco()

        if self.physx_sub:
            self.physx_sub.unsubscribe()
        
        self.app.update()
        self.timeline.stop()
        self.app.close()

if __name__ == "__main__":
    sdg = ObjectBasedSDG(simulation_app, config)
    sdg.run()
