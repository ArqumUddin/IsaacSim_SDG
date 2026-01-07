# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import random
import re
from itertools import chain

import omni.kit.app
import omni.kit.commands
import omni.physx
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


def set_transform_attributes(
    prim: Usd.Prim,
    location: Gf.Vec3d | None = None,
    orientation: Gf.Quatf | None = None,
    rotation: Gf.Vec3f | None = None,
    scale: Gf.Vec3f | None = None,
) -> None:
    """
    Sets transformation attributes for a given USD Prim.

    Checks for existing transform operations and adds them if missing.
    
    Args:
        prim (Usd.Prim): The target USD prim.
        location (Gf.Vec3d, optional): Translation vector.
        orientation (Gf.Quatf, optional): Quaternion orientation.
        rotation (Gf.Vec3f, optional): Euler rotation (XYZ) in degrees.
        scale (Gf.Vec3f, optional): Scale vector.
    """
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if orientation is not None:
        if not prim.HasAttribute("xformOp:orient"):
            UsdGeom.Xformable(prim).AddOrientOp()
        prim.GetAttribute("xformOp:orient").Set(orientation)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            UsdGeom.Xformable(prim).AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)


def add_colliders(root_prim: Usd.Prim, approximation_type: str = "convexHull") -> None:
    """
    Adds collision attributes to mesh and geometry descendants of a root prim.
    
    Args:
        root_prim (Usd.Prim): Root prim to traverse.
        approximation_type (str, optional): PhysX collision approximation type (e.g. "convexHull", "boundingCube"). Defaults to "convexHull".
    """
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Gprim):
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)

        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set(approximation_type)


def has_colliders(root_prim: Usd.Prim) -> bool:
    """Check if any descendant prims under the root prim have collision attributes."""
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False


def add_rigid_body_dynamics(prim: Usd.Prim, disable_gravity: bool = False) -> None:
    """
    Applies rigid body dynamics to a prim.
    
    Requires the prim to have colliders. Adds both standard RigidBodyAPI and PhysX schema.
    
    Args:
        prim (Usd.Prim): The target prim.
        disable_gravity (bool, optional): If True, gravity force is ignored for this body.
    """
    if has_colliders(prim):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        else:
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)

        # Apply PhysX rigid body dynamics
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        else:
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
    else:
        print(
            f"[SDG-Infinigen] Prim '{prim.GetPath()}' has no colliders. Skipping adding rigid body dynamics properties."
        )


def add_colliders_and_rigid_body_dynamics(prim: Usd.Prim, disable_gravity: bool = False) -> None:
    """Add colliders and rigid body dynamics properties to a prim, with optional gravity setting."""
    add_colliders(prim)
    add_rigid_body_dynamics(prim, disable_gravity)


def get_random_pose_on_sphere(
    origin: tuple[float, float, float],
    radius_range: tuple[float, float],
    polar_angle_range: tuple[float, float],
    camera_forward_axis: tuple[float, float, float] = (0, 0, -1),
    keep_level: bool = False,
) -> tuple[Gf.Vec3d, Gf.Quatf]:
    """
    Calculates a random position and orientation on a spherical shell looking at an origin.

    Args:
        origin (tuple): Center target point (x, y, z).
        radius_range (tuple): Min and max distance from origin.
        polar_angle_range (tuple): Min and max polar angle (theta) in degrees (0 = top, 180 = bottom).
        camera_forward_axis (tuple, optional): Camera's forward axis vector.
        keep_level (bool, optional): If True, computes orientation that keeps the camera level (no roll/pitch tilt relative to horizon), rotating only yaw.

    Returns:
        tuple[Gf.Vec3d, Gf.Quatf]: Position and Orientation quaternion.
    """
    # https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html
    # Convert degrees to radians for polar angles (theta)
    polar_angle_min_rad = math.radians(polar_angle_range[0])
    polar_angle_max_rad = math.radians(polar_angle_range[1])

    # Generate random spherical coordinates
    radius = random.uniform(radius_range[0], radius_range[1])
    polar_angle = random.uniform(polar_angle_min_rad, polar_angle_max_rad)
    azimuthal_angle = random.uniform(0, 2 * math.pi)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * math.sin(polar_angle) * math.cos(azimuthal_angle)
    y = radius * math.sin(polar_angle) * math.sin(azimuthal_angle)
    z = radius * math.cos(polar_angle)

    # Calculate the location in 3D space
    location = Gf.Vec3d(origin[0] + x, origin[1] + y, origin[2] + z)
    # Calculate direction vector from camera to look_at point
    direction = Gf.Vec3d(origin) - location
    
    if keep_level:
        # Use LookAt to ensure Up vector is preserved (Level Horizon)
        # SetLookAt creates World->Camera (View Matrix) looking down -Z with +Y Up (by default) or specified Up
        # We want Camera -> World
        view_mat = Gf.Matrix4d()
        view_mat.SetLookAt(location, origin, Gf.Vec3d(0, 0, 1))
        model_mat = view_mat.GetInverse()
        rotation = model_mat.ExtractRotation()
    else:
        direction_normalized = direction.GetNormalized()
        # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
        # This shortest-arc rotation does not constrain the up-vector, allowing 'roll'
        rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), direction_normalized)
        
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation

def randomize_camera_poses(
    cameras: list[Usd.Prim],
    targets: list[Usd.Prim],
    distance_range: tuple[float, float],
    polar_angle_range: tuple[float, float] = (0, 180),
    look_at_offset: tuple[float, float] = (-0.1, 0.1),
    per_camera_polar_angles: dict[int, tuple[float, float]] | None = None,
    keep_level_cameras: list[int] | None = None,
) -> None:
    """
    Randomizes a list of cameras to look at selected targets.

    Each camera picks a random target from the list and is placed on a spherical shell around it.

    Args:
        cameras (list[Usd.Prim]): List of camera prims to update.
        targets (list[Usd.Prim]): List of potential target prims to look at.
        distance_range (tuple): (min, max) distance from target.
        polar_angle_range (tuple): (min, max) polar angle in degrees.
        look_at_offset (tuple): Random (min, max) jitter added to the look-at point.
        per_camera_polar_angles (dict, optional): Overrides polar angle range for specific camera indices.
        keep_level_cameras (list, optional): List of indices for cameras that should remain level.
    """
    for i, cam in enumerate(cameras):
        # Get a random target asset to look at
        target_asset = random.choice(targets)

        # Add a look_at offset so the target is not always in the center of the camera view
        target_loc = target_asset.GetAttribute("xformOp:translate").Get()
        target_loc = (
            target_loc[0] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[1] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[2] + random.uniform(look_at_offset[0], look_at_offset[1]),
        )

        # Determine polar angle range for this camera
        camera_polar_angle_range = polar_angle_range
        if per_camera_polar_angles is not None and i in per_camera_polar_angles:
            camera_polar_angle_range = per_camera_polar_angles[i]

        # Determine if this camera should keep level orientation
        keep_level = keep_level_cameras is not None and i in keep_level_cameras

        # Generate random camera pose
        loc, quat = get_random_pose_on_sphere(target_loc, distance_range, camera_polar_angle_range, keep_level=keep_level)

        # Set the camera's transform attributes to the generated location and orientation
        set_transform_attributes(cam, location=loc, orientation=quat)

def get_usd_paths_from_folder(
    folder_path: str, recursive: bool = True, usd_paths: list[str] = None, skip_keywords: list[str] = None
) -> list[str]:
    """
    Retrieve USD file paths from a folder, optionally searching recursively and filtering by keywords.
    
    Args:
        folder_path (str): Directory to search.
        recursive (bool): If True, searches subdirectories.
        usd_paths (list[str], optional): Accumulator list for results.
        skip_keywords (list[str], optional): List of substrings to exclude if found in path.
        
    Returns:
        list[str]: list of absolute USD file paths.
    """
    if usd_paths is None:
        usd_paths = []
    skip_keywords = skip_keywords or []

    # Make sure the omni.client extension is enabled
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.client"):
        ext_manager.set_extension_enabled_immediate("omni.client", True)
    import omni.client

    result, entries = omni.client.list(folder_path)
    if result != omni.client.Result.OK:
        print(f"[SDG-Infinigen] Could not list assets in path: {folder_path}")
        return usd_paths

    for entry in entries:
        if any(keyword.lower() in entry.relative_path.lower() for keyword in skip_keywords):
            continue
        _, ext = os.path.splitext(entry.relative_path)
        if ext in [".usd", ".usda", ".usdc"]:
            path_posix = os.path.join(folder_path, entry.relative_path).replace("\\", "/")
            usd_paths.append(path_posix)
        elif recursive and entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            sub_folder = os.path.join(folder_path, entry.relative_path).replace("\\", "/")
            get_usd_paths_from_folder(sub_folder, recursive=recursive, usd_paths=usd_paths, skip_keywords=skip_keywords)

    return usd_paths

def get_usd_paths(
    files: list[str] = None, folders: list[str] = None, skip_folder_keywords: list[str] = None
) -> list[str]:
    """
    Aggregates USD paths from a list of individual files and folders.

    Resolves paths relative to Isaac Sim assets root if they are not absolute.
    
    Args:
        files (list[str], optional): List of specific file paths/URLs.
        folders (list[str], optional): List of folder paths to scan.
        skip_folder_keywords (list[str], optional): Keywords to use for filtering folder contents.
        
    Returns:
        list[str]: Combined list of found USD asset paths.
    """
    files = files or []
    folders = folders or []
    skip_folder_keywords = skip_folder_keywords or []

    assets_root_path = get_assets_root_path()
    env_paths = []

    for file_path in files:
        file_path = (
            file_path
            if file_path.startswith(("omniverse://", "http://", "https://", "file://")) or os.path.exists(file_path)
            else assets_root_path + file_path
        )
        env_paths.append(file_path)

    for folder_path in folders:
        folder_path = (
            folder_path
            if folder_path.startswith(("omniverse://", "http://", "https://", "file://")) or os.path.exists(folder_path)
            else assets_root_path + folder_path
        )
        env_paths.extend(get_usd_paths_from_folder(folder_path, recursive=True, skip_keywords=skip_folder_keywords))

    return env_paths

def load_env(usd_path: str, prim_path: str, remove_existing: bool = True) -> Usd.Prim:
    """
    Loads an environment into the stage.
    
    Args:
        usd_path (str): Path to the USD environment file.
        prim_path (str): Stage path where the environment should be loaded (e.g. "/Environment").
        remove_existing (bool): If True, deletes any existing prim at prim_path before loading.
        
    Returns:
        Usd.Prim: The loaded root prim.
    """
    stage = omni.usd.get_context().get_stage()

    # Remove existing prim if specified
    if remove_existing and stage.GetPrimAtPath(prim_path):
        omni.kit.commands.execute("DeletePrimsCommand", paths=[prim_path])

    root_prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    return root_prim

def add_colliders_to_env(root_path: str | None = None, approximation_type: str = "none") -> None:
    """
    Adds collision properties to all meshes within an environment.
    
    Args:
        root_path (str, optional): Root stage path to traverse. If None, uses PseudoRoot.
        approximation_type (str): Collision approximation for meshes (default "none" uses mesh geometry, effectively).
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

    for prim in Usd.PrimRange(prim):
        if prim.IsA(UsdGeom.Mesh):
            add_colliders(prim, approximation_type)

def find_matching_prims(
    match_strings: list[str], root_path: str | None = None, prim_type: str | None = None, first_match_only: bool = False
) -> Usd.Prim | list[Usd.Prim] | None:
    """
    Finds prims in the stage that match given substrings in their path.
    
    Args:
        match_strings (list[str]): List of distinct strings to check for in prim paths.
        root_path (str, optional): Root prim to start search from.
        prim_type (str, optional): USD type name to filter by (e.g. "Xform", "Mesh").
        first_match_only (bool): If True, returns the first match found instead of a list.
        
    Returns:
        Usd.Prim | list[Usd.Prim] | None: Found prim(s) or None.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

    matching_prims = []
    for prim in Usd.PrimRange(root_prim):
        if any(match in str(prim.GetPath()) for match in match_strings):
            if prim_type is None or prim.GetTypeName() == prim_type:
                if first_match_only:
                    return prim
                matching_prims.append(prim)

    return matching_prims if not first_match_only else None

def hide_matching_prims(match_strings: list[str], root_path: str | None = None, prim_type: str | None = None) -> None:
    """
    Sets prims matching the given criteria to be invisible.
    
    Args:
        match_strings (list[str]): Strings to match in prim paths.
        root_path (str, optional): Root search path.
        prim_type (str, optional): Type of prim to target.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

    for prim in Usd.PrimRange(root_prim):
        if prim_type is None or prim.GetTypeName() == prim_type:
            if any(match in str(prim.GetPath()) for match in match_strings):
                prim.GetAttribute("visibility").Set("invisible")

def setup_env(root_path: str | None = None, approximation_type: str = "none", hide_top_walls: bool = False) -> None:
    """
    Configures a loaded environment for simulation.
    
    Performs specific fixes for Infinigen environments (e.g. ceiling lights, table collisions) 
    and adds general collision properties.
    
    Args:
        root_path (str, optional): Path to environment root.
        approximation_type (str): Collision approximation type.
        hide_top_walls (bool): If True, hides ceiling/exterior for debugging.
    """
    # Fix ceiling lights: meshes are blocking the light and need to be set to invisible
    ceiling_light_meshes = find_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")
    for light_mesh in ceiling_light_meshes:
        light_mesh.GetAttribute("visibility").Set("invisible")

    # Hide ceiling light meshes for lighting fix
    hide_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")

    # Hide top walls for better debug view, if specified
    if hide_top_walls:
        hide_matching_prims(["_exterior", "_ceiling"], root_path)

    # Add colliders to the environment
    add_colliders_to_env(root_path, approximation_type)

    # Fix dining table collision by setting it to a bounding cube approximation
    table_prim = find_matching_prims(
        match_strings=["TableDining"], root_path=root_path, prim_type="Xform", first_match_only=True
    )
    if table_prim is not None:
        add_colliders(table_prim, approximation_type="boundingCube")
    else:
        print("[SDG-Infinigen] Could not find dining table prim in the environment.")

def create_shape_distractors(
    num_distractors: int, shape_types: list[str], root_path: str, gravity_disabled_chance: float
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """
    Creates geometric shape distractors (cube, sphere, etc.).
    
    Args:
        num_distractors (int): Number of shapes to spawn.
        shape_types (list[str]): List of available shapes choices.
        root_path (str): Parent path for the new prims.
        gravity_disabled_chance (float): Probability (0.0-1.0) that a shape has gravity disabled.
        
    Returns:
        tuple[list, list]: (floating_shapes, falling_shapes)
    """
    stage = omni.usd.get_context().get_stage()
    floating_shapes = []
    falling_shapes = []
    for _ in range(num_distractors):
        rand_shape = random.choice(shape_types)
        disable_gravity = random.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        prim_path = omni.usd.get_stage_next_free_path(stage, f"{root_path}/{name_prefix}{rand_shape}", False)
        prim = stage.DefinePrim(prim_path, rand_shape.capitalize())
        add_colliders_and_rigid_body_dynamics(prim, disable_gravity=disable_gravity)
        (floating_shapes if disable_gravity else falling_shapes).append(prim)
    return floating_shapes, falling_shapes

def load_shape_distractors(shape_distractors_config: dict) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """
    Wrapper to load shape distractors from a config dictionary.
    
    Args:
        shape_distractors_config (dict): Configuration containing num, types, etc.
    """
    num_shapes = shape_distractors_config.get("num", 0)
    shape_types = shape_distractors_config.get("shape_types", ["capsule", "cone", "cylinder", "sphere", "cube"])
    shape_gravity_disabled_chance = shape_distractors_config.get("gravity_disabled_chance", 0.0)
    return create_shape_distractors(num_shapes, shape_types, "/Distractors", shape_gravity_disabled_chance)

def create_mesh_distractors(
    num_distractors: int, mesh_urls: list[str], root_path: str, gravity_disabled_chance: float
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """
    Creates distractors from existing USD mesh assets.
    
    Args:
        num_distractors (int): Count to spawn.
        mesh_urls (list[str]): List of USD asset paths to pick from.
        root_path (str): Parent path.
        gravity_disabled_chance (float): Probability of floating.
        
    Returns:
        tuple[list, list]: (floating_meshes, falling_meshes)
    """
    stage = omni.usd.get_context().get_stage()
    floating_meshes = []
    falling_meshes = []
    for _ in range(num_distractors):
        rand_mesh_url = random.choice(mesh_urls)
        disable_gravity = random.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        prim_name = os.path.basename(rand_mesh_url).split(".")[0]
        prim_path = omni.usd.get_stage_next_free_path(stage, f"{root_path}/{name_prefix}{prim_name}", False)
        try:
            prim = add_reference_to_stage(usd_path=rand_mesh_url, prim_path=prim_path)
        except Exception as e:
            print(f"[SDG-Infinigen] Failed to load mesh distractor reference {rand_mesh_url} with exception: {e}")
            continue
        add_colliders_and_rigid_body_dynamics(prim, disable_gravity=disable_gravity)
        (floating_meshes if disable_gravity else falling_meshes).append(prim)
    return floating_meshes, falling_meshes

def load_mesh_distractors(mesh_distractors_config: dict) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """
    Wrapper to load mesh distractors from config.
    
    Scans folders/files specified in config to find assets.
    
    Args:
        mesh_distractors_config (dict): Config dict.
    """
    num_meshes = mesh_distractors_config.get("num", 0)
    mesh_gravity_disabled_chance = mesh_distractors_config.get("gravity_disabled_chance", 0.0)
    mesh_folders = mesh_distractors_config.get("folders", [])
    mesh_files = mesh_distractors_config.get("files", [])
    mesh_urls = get_usd_paths(
        files=mesh_files, folders=mesh_folders, skip_folder_keywords=["material", "texture", ".thumbs"]
    )
    floating_meshes, falling_meshes = create_mesh_distractors(
        num_meshes, mesh_urls, "/Distractors", mesh_gravity_disabled_chance
    )
    for prim in chain(floating_meshes, falling_meshes):
        remove_labels(prim, include_descendants=True)
    return floating_meshes, falling_meshes

def scan_assets(
    folders: list[str],
    files: list[str],
    regex_replace_pattern: str = "",
    regex_replace_repl: str = "",
    recursive: bool = True
) -> list[dict]:
    """
    Scans directories and files for USD assets, applying regex to generate labels.

    Args:
        folders (list[str]): List of folders to scan.
        files (list[str]): List of specific files to include.
        regex_replace_pattern (str): Regex pattern to clean filenames for labels.
        regex_replace_repl (str): Replacement string.
        recursive (bool): Whether to scan folders recursively.

    Returns:
        list[dict]: List of dictionaries containing 'path' and 'label' for each found asset.
    """
    found_assets = []
    
    # helper to process a single path
    def process_path(usd_path):
        basename = os.path.basename(usd_path)
        name_without_ext = os.path.splitext(basename)[0]
        label = name_without_ext
        if regex_replace_pattern:
            label = re.sub(regex_replace_pattern, regex_replace_repl, name_without_ext)
        return {"url": usd_path, "label": label}

    # Process explicit files
    resolved_files = get_usd_paths(files=files)
    for f in resolved_files:
        found_assets.append(process_path(f))

    # Process folders
    resolved_folder_assets = get_usd_paths(folders=folders, skip_folder_keywords=["material", "texture", ".thumbs"])
    for f in resolved_folder_assets:
        # Avoid duplicates if file was already listed
        if not any(a["url"] == f for a in found_assets):
            found_assets.append(process_path(f))
            
    return found_assets

def resolve_scale_issues_with_metrics_assembler() -> None:
    """
    Uses the Metrics Assembler extension to fix scale discrepancies in the stage.
    """
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.usd.metrics.assembler"):
        ext_manager.set_extension_enabled_immediate("omni.usd.metrics.assembler", True)
    from omni.metrics.assembler.core import get_metrics_assembler_interface

    stage_id = omni.usd.get_context().get_stage_id()
    get_metrics_assembler_interface().resolve_stage(stage_id)

def get_matching_prim_location(match_string, root_path=None):
    prim = find_matching_prims(
        match_strings=[match_string], root_path=root_path, prim_type="Xform", first_match_only=True
    )
    if prim is None:
        print(f"[SDG-Infinigen] Could not find matching prim, returning (0, 0, 0)")
        return (0, 0, 0)
    if prim.HasAttribute("xformOp:translate"):
        return prim.GetAttribute("xformOp:translate").Get()
    elif prim.HasAttribute("xformOp:transform"):
        return prim.GetAttribute("xformOp:transform").Get().ExtractTranslation()
    else:
        print(f"[SDG-Infinigen] Could not find location attribute for '{prim.GetPath()}', returning (0, 0, 0)")
        return (0, 0, 0)

def offset_range(
    range_coords: tuple[float, float, float, float, float, float], offset: tuple[float, float, float]
) -> tuple[float, float, float, float, float, float]:
    """Offset the min and max coordinates of a range by the specified offset."""
    return (
        range_coords[0] + offset[0],  # min_x
        range_coords[1] + offset[1],  # min_y
        range_coords[2] + offset[2],  # min_z
        range_coords[3] + offset[0],  # max_x
        range_coords[4] + offset[1],  # max_y
        range_coords[5] + offset[2],  # max_z
    )

def randomize_poses(
    prims: list[Usd.Prim],
    location_range: tuple[float, float, float, float, float, float],
    rotation_range: tuple[float, float],
    scale_range: tuple[float, float],
) -> None:
    """
    Randomizes location, rotation, and scale for a list of prims.
    
    Args:
        prims (list[Usd.Prim]): List of prims to modify.
        location_range (tuple): (min_x, min_y, min_z, max_x, max_y, max_z).
        rotation_range (tuple): (min_angle, max_angle) applied to all axes.
        scale_range (tuple): (min, max) uniform scale factor.
    """
    for prim in prims:
        rand_loc = (
            random.uniform(location_range[0], location_range[3]),
            random.uniform(location_range[1], location_range[4]),
            random.uniform(location_range[2], location_range[5]),
        )
        rand_rot = (
            random.uniform(rotation_range[0], rotation_range[1]),
            random.uniform(rotation_range[0], rotation_range[1]),
            random.uniform(rotation_range[0], rotation_range[1]),
        )
        rand_scale = random.uniform(scale_range[0], scale_range[1])
        set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=(rand_scale, rand_scale, rand_scale))

def run_simulation(num_frames: int, render: bool = True) -> None:
    """
    Steps the physics simulation.
    
    Args:
        num_frames (int): Number of frames to advance.
        render (bool): If True, updates the application (rendering). If False, only steps physics engine.
    """
    if render:
        # Start the timeline and advance the app, this will render the physics simulation results every frame
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0)
        timeline.set_end_time(1000000)
        timeline.set_looping(False)
        timeline.play()
        for _ in range(num_frames):
            omni.kit.app.get_app().update()
        timeline.pause()
    else:
        # Run the physics simulation steps without advancing the app
        stage = omni.usd.get_context().get_stage()
        physx_scene = None

        # Search for or create a physics scene
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
                break

        if physx_scene is None:
            physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

        # Get simulation parameters
        physx_dt = 1 / physx_scene.GetTimeStepsPerSecondAttr().Get()
        physx_sim_interface = omni.physx.get_physx_simulation_interface()

        # Run physics simulation for each frame
        for _ in range(num_frames):
            physx_sim_interface.simulate(physx_dt, 0)
            physx_sim_interface.fetch_results()

def register_dome_light_randomizer() -> None:
    """
    Registers a Replicator randomizer for dome lights.
    
    Selects from a predefined list of HDR sky textures.
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

def register_shape_distractors_color_randomizer(shape_distractors: list[Usd.Prim]) -> None:
    """
    Registers a Replicator randomizer for shape colors.
    
    Args:
        shape_distractors (list[Usd.Prim]): List of prims to randomize colors for.
    """
    with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
        shape_distractors_paths = [prim.GetPath() for prim in shape_distractors]
        shape_distractors_group = rep.create.group(shape_distractors_paths)
        with shape_distractors_group:
            rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

def randomize_lights(
    lights: list[Usd.Prim],
    location_range: tuple[float, float, float, float, float, float] | None = None,
    color_range: tuple[float, float, float, float, float, float] | None = None,
    intensity_range: tuple[float, float] | None = None,
) -> None:
    """
    Randomizes properties of point lights.
    
    Args:
        lights (list[Usd.Prim]): List of light prims.
        location_range (tuple, optional): (min_x, ... max_z) bounds.
        color_range (tuple, optional): (min_r, ... max_b) bounds.
        intensity_range (tuple, optional): (min, max) intensity.
    """
    for light in lights:
        # Randomize the location of the light
        if location_range is not None:
            rand_loc = (
                random.uniform(location_range[0], location_range[3]),
                random.uniform(location_range[1], location_range[4]),
                random.uniform(location_range[2], location_range[5]),
            )
            set_transform_attributes(light, location=rand_loc)

        # Randomize the color of the light
        if color_range is not None:
            rand_color = (
                random.uniform(color_range[0], color_range[3]),
                random.uniform(color_range[1], color_range[4]),
                random.uniform(color_range[2], color_range[5]),
            )
            light.GetAttribute("inputs:color").Set(rand_color)

        # Randomize the intensity of the light
        if intensity_range is not None:
            rand_intensity = random.uniform(intensity_range[0], intensity_range[1])
            light.GetAttribute("inputs:intensity").Set(rand_intensity)


