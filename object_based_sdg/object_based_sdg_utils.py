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

import random
import re
from pathlib import Path
import carb

import numpy as np
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

def set_transform_attributes(prim, location=None, orientation=None, rotation=None, scale=None):
    """
    Sets transformation attributes for a given USD Prim.
    
    This function checks if the specific translation, orientation, rotation, or scale operations
    exist on the prim. If not, it adds them using the UsdGeom.Xformable API.

    Args:
        prim (Usd.Prim): The target USD prim to modify.
        location (tuple or list, optional): (x, y, z) translation coordinates.
        orientation (tuple or list, optional): (r, i, j, k) quaternion orientation.
        rotation (tuple or list, optional): (x, y, z) rotation in degrees (Euler XYZ).
        scale (tuple or list, optional): (x, y, z) scale factors.
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
        
        # Respect existing scale from referenced assets (Weak Layer)
        # by multiplying the new random scale (Strong Layer) with it.
        existing_scale = prim.GetAttribute("xformOp:scale").Get()
        
        if existing_scale:
            # Perform component-wise multiplication
            # Ensure we handle Gf.Vec3f or tuple types correctly by explicit component access
            final_scale = Gf.Vec3f(
                scale[0] * existing_scale[0],
                scale[1] * existing_scale[1],
                scale[2] * existing_scale[2]
            )
        else:
            final_scale = scale
            
        prim.GetAttribute("xformOp:scale").Set(final_scale)

def add_colliders(root_prim):
    """
    Enables collisions with the asset (without rigid body dynamics the asset will be static)
    Adds collision properties to a prim and its descendants.

    Iterates through the prim's hierarchy and applies:
    - **CollisionAPI**: To enable general collisions.
    - **PhysxCollisionAPI**: To set PhysX-specific contact/rest offsets.
    - **MeshCollisionAPI**: Only for meshes, setting approximation to "convexHull".

    Args:
        root_prim (Usd.Prim): The root prim to start traversing from.
    """
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            # PhysX
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            # Set PhysX specific properties
            physx_collision_api.CreateContactOffsetAttr(0.001)
            physx_collision_api.CreateRestOffsetAttr(0.0)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            # Add mesh collision properties to the mesh (e.g. collider aproximation type)
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")

def has_colliders(root_prim):
    """
    Checks if a prim or any of its descendants has a CollisionAPI applied.

    Args:
        root_prim (Usd.Prim): The root prim to search from.

    Returns:
        bool: True if collision API is found, False otherwise.
    """
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False

def add_rigid_body_dynamics(prim, disable_gravity=False, angular_damping=None):
    """
    Applies rigid body dynamics properties to a prim.
    
    This function adds both generic UsdPhysics RigidBodyAPI and PhysX-specific PhysxRigidBodyAPI.
    Be aware: It will print a warning if the prim does not have colliders, as rigid bodies need collisions
    to interact physically with the world (unless static kinemtatics are desired, which is not the default here).

    Args:
        prim (Usd.Prim): The target USD prim.
        disable_gravity (bool, optional): If True, the object will not be affected by gravity.
        angular_damping (float, optional): Damping factor for angular velocity.
    """
    if has_colliders(prim):
        # Physics
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        else:
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        # PhysX
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        else:
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
        if angular_damping is not None:
            physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)
    else:
        # Note: Silent continuation preferred in some pipelines, but keeping print for debug clarity
        print(f"Prim '{prim.GetPath()}' has no colliders. Skipping rigid body dynamics properties.")

def add_colliders_and_rigid_body_dynamics(prim, disable_gravity=False):
    """
    Convenience function to add both colliders and rigid body dynamics to a prim.
    
    This essentially calls `add_colliders` followed by `add_rigid_body_dynamics`.

    Args:
        prim (Usd.Prim): The target prim.
        disable_gravity (bool, optional): If True, gravity is disabled for this rigid body.
    """
    # Add colliders to mesh or primitive types of the descendants of the prim (including root)
    add_colliders(prim)
    # Add rigid body dynamics properties (to the root only) only if it has colliders
    add_rigid_body_dynamics(prim, disable_gravity=disable_gravity)

def create_collision_box_walls(stage, path, width, depth, height, thickness=0.5, visible=False):
    """
    Creates a box of 6 walls (floor, ceiling, 4 sides) to serve as a container or boundary.

    The walls are created as static colliders (no rigid body API, just collision API).
    
    Args:
        stage (Usd.Stage): The active stage.
        path (str): The USD path where the wall prims will be grouped.
        width (float): internal width (x-axis) of the box.
        depth (float): internal depth (y-axis) of the box.
        height (float): internal height (z-axis) of the box.
        thickness (float, optional): Thickness of the walls. Defaults to 0.5.
        visible (bool, optional): If True, walls are rendered; otherwise invisible. Defaults to False.
    """
    # Define the walls (name, location, size) with thickness towards outside of the working area
    walls = [
        ("floor", (0, 0, (height + thickness) / -2.0), (width, depth, thickness)),
        ("ceiling", (0, 0, (height + thickness) / 2.0), (width, depth, thickness)),
        ("left_wall", ((width + thickness) / -2.0, 0, 0), (thickness, depth, height)),
        ("right_wall", ((width + thickness) / 2.0, 0, 0), (thickness, depth, height)),
        ("front_wall", (0, (depth + thickness) / 2.0, 0), (width, thickness, height)),
        ("back_wall", (0, (depth + thickness) / -2.0, 0), (width, thickness, height)),
    ]
    for name, location, size in walls:
        prim = stage.DefinePrim(f"{path}/{name}", "Cube")
        scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        set_transform_attributes(prim, location=location, scale=scale)
        add_colliders(prim)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()

def get_random_transform_values(
    loc_min=(0, 0, 0), loc_max=(1, 1, 1), rot_min=(0, 0, 0), rot_max=(360, 360, 360), scale_min_max=(0.1, 1.0)
):
    """
    Generates random 3D transformation values within specified ranges.

    Args:
        loc_min (tuple): Minimum (x, y, z) for location.
        loc_max (tuple): Maximum (x, y, z) for location.
        rot_min (tuple): Minimum (x, y, z) for rotation (degrees).
        rot_max (tuple): Maximum (x, y, z) for rotation (degrees).
        scale_min_max (tuple): Min and Max scalar for uniform scaling.

    Returns:
        tuple: (location, rotation, scale) where each is a Gf type for proper USD compatibility.
    """
    location = Gf.Vec3d(
        random.uniform(loc_min[0], loc_max[0]),
        random.uniform(loc_min[1], loc_max[1]),
        random.uniform(loc_min[2], loc_max[2]),
    )
    rotation = Gf.Vec3f(
        random.uniform(rot_min[0], rot_max[0]),
        random.uniform(rot_min[1], rot_max[1]),
        random.uniform(rot_min[2], rot_max[2]),
    )
    scale_val = random.uniform(scale_min_max[0], scale_min_max[1])
    scale = Gf.Vec3f(scale_val, scale_val, scale_val)
    return location, rotation, scale

def get_random_pose_on_sphere(origin, radius, camera_forward_axis=(0, 0, -1)):
    """
    Calculates a random position on a sphere surface looking at the sphere center.

    This is commonly used for randomizing camera positions around a target object.
    
    Args:
        origin (tuple or pxr.Gf.Vec3f): Center of the sphere (target point).
        radius (float): Radius of the sphere (distance from target).
        camera_forward_axis (tuple, optional): The axis that represents 'forward' for the camera. 
                                              Defaults to (0, 0, -1) which is standard for USD cameras.

    Returns:
        tuple: (location, orientation) where 
               location is Gf.Vec3f position
               orientation is Gf.Quatf rotation
    """
    origin = Gf.Vec3f(origin)
    camera_forward_axis = Gf.Vec3f(camera_forward_axis)

    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arcsin(np.random.uniform(-1, 1))

    # Spherical to Cartesian conversion
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(phi)
    z = radius * np.sin(theta) * np.cos(phi)

    location = origin + Gf.Vec3f(x, y, z)

    # Calculate direction vector from camera to look_at point
    direction = origin - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), Gf.Vec3d(direction_normalized))
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation

def set_render_products_updates(render_products, enabled, include_viewport=False):
    """
    Enables or disables updates for a list of render products.
    
    Disabling updates can significantly improve performance during physics simulation phases where
    rendering is not needed.

    Args:
        render_products (list): List of render product objects.
        enabled (bool): True to enable updates, False to disable.
        include_viewport (bool, optional): If True, also toggles the active viewport updates.
    """
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(enabled)
    if include_viewport:
        get_active_viewport().updates_enabled = enabled

def scan_assets(folders=None, files=None, recursive=False, regex_replace_pattern=None, regex_replace_repl=""):
    """
    Scans folders and lists files for USD assets, applying regex label cleaning.

    Args:
        folders (list of str, optional): List of directories to scan.
        files (list of str, optional): List of specific file paths to include.
        recursive (bool, optional): If True, scans folders recursively ("**/*.usd"). Defaults to False.
        regex_replace_pattern (str, optional): Regex pattern to find in filenames for label cleaning.
        regex_replace_repl (str, optional): Replacement string for the regex pattern.

    Returns:
        list[dict]: List of {"url": str, "label": str} dicts.
    """
    folders = folders or []
    files = files or []
    loaded_assets = []
    seen_urls = set()

    # 1. Scan folders
    for folder in folders:
        source_path = Path(folder)
        if not source_path.exists():
            carb.log_warn(f"Asset source directory not found: {folder}")
            continue

        pattern = "**/*.usd" if recursive else "*.usd"
        usd_files = sorted(list(source_path.glob(pattern)))
        
        print(f"[SDG] Found {len(usd_files)} assets in {folder}")
        
        for f in usd_files:
            url = str(f.resolve())
            if url in seen_urls:
                continue
            
            raw_name = f.stem
            # Apply regex replacement if provided
            if regex_replace_pattern:
                clean_label = re.sub(regex_replace_pattern, regex_replace_repl, raw_name)
            else:
                clean_label = raw_name
            
            loaded_assets.append({"url": url, "label": clean_label})
            seen_urls.add(url)

    # 2. Add specific files
    for file_path in files:
        f = Path(file_path)
        # If explicit file path is not absolute and doesn't exist, might be an Omniverse URL or relative
        # We can mostly assume explicit files might be local absolute paths or Nucleus paths.
        # For label generation, we need the filename.
        
        url = str(f)
        if hasattr(f, "resolve") and f.exists():
             url = str(f.resolve())

        if url in seen_urls:
            continue
            
        # Extract filename for label
        raw_name = f.stem
        if regex_replace_pattern:
            clean_label = re.sub(regex_replace_pattern, regex_replace_repl, raw_name)
        else:
            clean_label = raw_name
            
        loaded_assets.append({"url": url, "label": clean_label})
        seen_urls.add(url)

    return loaded_assets

def apply_velocities_towards_target(assets, target=(0, 0, 0)):
    """
    Applies a velocity vector to assets directing them towards a target point.

    This is useful for keeping floating objects from drifting too far away from the center
    of the workspace. The velocity magnitude is randomized between 0.1 and 1.0.

    Args:
        assets (iterable): An iterable of USD Prims to apply velocity to.
        target (tuple, optional): (x, y, z) coordinates of the target point. Defaults to (0, 0, 0).
    """
    for prim in assets:
        loc = prim.GetAttribute("xformOp:translate").Get()
        strength = random.uniform(0.1, 1.0)
        pull_vel = ((target[0] - loc[0]) * strength, (target[1] - loc[1]) * strength, (target[2] - loc[2]) * strength)
        prim.GetAttribute("physics:velocity").Set(pull_vel)

def randomize_floating_distractor_velocities(assets):
    """
    Applies random linear and angular velocities to infinite-mass (floating) objects.

    Since floating objects (with gravity disabled) might be stationary or drift slowly,
    this function injects random movement to create dynamic scene variations.
    
    Args:
        assets (iterable): An iterable of USD Prims to randomize.
    """
    for prim in assets:
        lin_vel = (random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5))
        ang_vel = (random.uniform(-45, 45), random.uniform(-45, 45), random.uniform(-45, 45))
        prim.GetAttribute("physics:velocity").Set(lin_vel)
        prim.GetAttribute("physics:angularVelocity").Set(ang_vel)
