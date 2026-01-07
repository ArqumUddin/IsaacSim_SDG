# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import yaml
import carb
import random
import time
from itertools import chain
import carb.settings
from pathlib import Path

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
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics

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
        Loads and spawns the labeled assets.

        Assets are gathered from the `labeled_assets` configuration block, which supports:
        1. **Auto-scan (`auto_label`)**: Scans specified folders/files, applies regex for labeling. (V2 Schema)
        2. **Manual config (`manual_label`)**: Explicitly listed assets with overrides.

        For each asset, multiple instances are spawned based on `num` (or `count`), with
        configurable gravity settings.
        """
        all_assets = []
        
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
            count = auto_cfg.get("num", 1)  # 'num' matches scene config style
            scale_min_max = auto_cfg.get("scale_min_max", (1, 1))
            floating = auto_cfg.get("floating", False)

            for asset in scanned:
                asset["count"] = count
                asset["scale_min_max"] = scale_min_max
                asset["floating"] = floating
                all_assets.append(asset)

        # 2. Process Manual-Label
        manual_cfg = la_config.get("manual_label", [])
        if manual_cfg:
            for item in manual_cfg:
                # Normalize keys (scene config uses 'num', our old one used 'count')
                item["count"] = item.get("num", item.get("count", 1))
                all_assets.append(item)

        print(f"[SDG] Total unique assets to spawn: {len(all_assets)}")

        for obj in all_assets:
            obj_url = obj.get("url", "")
            label = obj.get("label", "unknown")
            count = obj.get("count", 1)
            floating = obj.get("floating", False)
            scale_min_max = obj.get("scale_min_max", (1, 1))

            for _ in range(count):
                rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
                    loc_min=self.working_area_min, loc_max=self.working_area_max, scale_min_max=scale_min_max
                )
                prim_path = omni.usd.get_stage_next_free_path(self.stage, f"/World/Labeled/{label}", False)
                prim = self.stage.DefinePrim(prim_path, "Xform")
                
                if obj_url.startswith("omniverse://") or os.path.isabs(obj_url):
                    asset_path = obj_url
                else:
                    asset_path = self.assets_root_path + obj_url
                    
                prim.GetReferences().AddReference(asset_path)
                object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
                object_based_sdg_utils.add_colliders(prim)
                object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=floating)
                add_labels(prim, labels=[label], instance_name="class")
                self.unique_labels.add(label)

                if floating:
                    self.floating_labeled_prims.append(prim)
                else:
                    self.falling_labeled_prims.append(prim)
        
        self.labeled_prims = self.floating_labeled_prims + self.falling_labeled_prims

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
        
        for i in range(num_cameras):
            cam_prim = self.stage.DefinePrim(f"/World/Cameras/cam_{i}", "Camera")
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
        
        print(f"[SDG] Found {len(ordered_labels)} unique labels: {ordered_labels}")

        for i, label in enumerate(ordered_labels):
            # ID 0 is usually reserved for background
            cat_id = i + 1
            
            # Generate a consistent color hash or random
            # Just using random for now effectively, but seeding could make it deterministic
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            
            categories[label] = {
                "name": label,
                "id": cat_id,
                "supercategory": "ycb",
                "isthing": 1,
                "color": color
            }
            
        return categories

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
        
        # Shape Colors
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

            rep.create.light(
                light_type=l_type,
                color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
                temperature=rep.distribution.normal((l_temp_min + l_temp_max)/2, (l_temp_max - l_temp_min)/4),
                intensity=rep.distribution.normal((l_int_min + l_int_max)/2, (l_int_max - l_int_min)/4),
                position=rep.distribution.uniform(self.working_area_min, self.working_area_max),
                scale=rep.distribution.uniform(0.1, 1),
                count=l_count,
            )

        # Dome Background
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
        
        For each camera:
        1. Selects a random target asset (labeled prim).
        2. Applies a small random offset to the target point so it's not perfectly centered.
        3. Positions the camera at a random distance on a spherical surface around the target.
        """
        min_dist, max_dist = self.config.get("camera_distance_to_target_min_max", (0.1, 0.5))
        offset = self.config.get("camera_look_at_target_offset", 0.2)

        for cam in self.cameras:
            target_asset = random.choice(self.labeled_prims)
            loc_offset = (
                random.uniform(-offset, offset),
                random.uniform(-offset, offset),
                random.uniform(-offset, offset),
            )
            target_loc = target_asset.GetAttribute("xformOp:translate").Get() + loc_offset
            distance = random.uniform(min_dist, max_dist)
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

        for i in range(self.num_frames):
            print(f"[SDG] Processing Frame {i+1}/{self.num_frames}")

            # 1. Camera Randomization
            if i % 3 == 0:
                self.randomize_camera_poses()
                if self.camera_colliders:
                    self.simulate_camera_collision(frames=4)

            # 2. Velocity Pull
            if i % 10 == 0:
                object_based_sdg_utils.apply_velocities_towards_target(
                    chain(self.labeled_prims, self.shape_distractors, self.mesh_distractors)
                )

            # 3. Lights
            if i % 5 == 0:
                rep.utils.send_og_event(event_name="randomize_lights")

            # 4. Shape Colors
            if i % 15 == 0:
                rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

            # 5. Dome
            if i % 25 == 0:
                rep.utils.send_og_event(event_name="randomize_dome_background")

            # 6. Distractor Velocities
            if i % 17 == 0:
                object_based_sdg_utils.randomize_floating_distractor_velocities(
                    chain(self.floating_shape_distractors, self.floating_mesh_distractors)
                )

            # 7. Asset Colors (if configured)
            if self.config.get("assets", {}).get("properties", {}).get("randomize_color", False):
                rep.utils.send_og_event(event_name="randomize_asset_appearance")

            # Capture
            if self.disable_render_products_between_captures:
                object_based_sdg_utils.set_render_products_updates(self.render_products, True)

            if i % 5 == 0:
                self.capture_with_motion_blur(duration=0.025, num_samples=8, spp=128)
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

        if self.physx_sub:
            self.physx_sub.unsubscribe()
        
        self.app.update()
        self.timeline.stop()
        self.app.close()

if __name__ == "__main__":
    sdg = ObjectBasedSDG(simulation_app, config)
    sdg.run()
