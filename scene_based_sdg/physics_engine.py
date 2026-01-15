# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Physics engine utilities for PhysX simulation.
Handles colliders, rigid body dynamics, and simulation stepping.
"""

import logging

import omni.kit.app
import omni.kit.commands
import omni.physx
import omni.timeline
import omni.usd
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

logger = logging.getLogger(__name__)


class PhysicsEngine:
    """Manages PhysX physics setup and simulation."""

    @staticmethod
    def add_colliders(root_prim: Usd.Prim, approximation_type: str = "convexHull") -> None:
        """
        Apply PhysX collision shapes to all geometric primitives.

        Traverses hierarchy, applies CollisionAPI (Gprim) and MeshCollisionAPI (Mesh) with specified approximation.
        Approximations: convexHull (fast), convexDecomposition (accurate), boundingCube/Sphere (fastest), none (slow).

        Args:
            root_prim: Root USD prim to start traversal from
                      All descendant Gprim and Mesh prims will receive collision properties
            approximation_type: PhysX collision approximation mode for mesh geometry
                               Recommended: "convexHull" (fast, accurate for most objects)
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

    @staticmethod
    def has_colliders(root_prim: Usd.Prim) -> bool:
        """
        Check if prim or descendants have collision shapes.

        Searches hierarchy for CollisionAPI (early exit on first match). Rigid bodies require colliders.

        Args:
            root_prim: USD prim to search from (includes the prim itself and all descendants)

        Returns:
            True if root_prim or any descendant has UsdPhysics.CollisionAPI
            False if no collision shapes found in the entire hierarchy
        """
        for desc_prim in Usd.PrimRange(root_prim):
            if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                return True
        return False

    @staticmethod
    def add_rigid_body_dynamics(prim: Usd.Prim, disable_gravity: bool = False) -> None:
        """
        Enable rigid body dynamics for physics simulation.

        Applies RigidBodyAPI and PhysxRigidBodyAPI. Requires colliders (checked first).
        disable_gravity useful for floating objects, ceiling mounts, or controlled motion.

        Args:
            prim: USD prim to make dynamic (must have collision shapes)
            disable_gravity: If True, object will float/hover (not affected by gravity)
                           If False, object will fall naturally (default physics behavior)
        """
        if PhysicsEngine.has_colliders(prim):
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
            logger.info(f"Prim '{prim.GetPath()}' has no colliders. Skipping adding rigid body dynamics properties.")

    @staticmethod
    def add_colliders_and_dynamics(prim: Usd.Prim, disable_gravity: bool = False) -> None:
        """
        Add both colliders and rigid body dynamics in one call.

        Combines add_colliders (convexHull) + add_rigid_body_dynamics. Most common workflow.

        Args:
            prim: USD prim to make into a physics object
            disable_gravity: If True, object will float (zero gravity)
                           If False, object will fall naturally (normal gravity)
        """
        PhysicsEngine.add_colliders(prim)
        PhysicsEngine.add_rigid_body_dynamics(prim, disable_gravity)

    @staticmethod
    def add_collision_sphere_to_camera(
        cam_prim: Usd.Prim,
        radius: float = 0.15,
        disable_gravity: bool = True,
        visible: bool = False
    ) -> Usd.Prim:
        """
        Attach collision sphere to camera for physics-based wall avoidance.

        Creates child sphere with PhysX collision/dynamics. Complements raycasting with continuous collision detection.
        Typically invisible with no gravity. Radius should be < wall_buffer.

        Args:
            cam_prim: Camera prim to attach collision sphere to
            radius: Collision sphere radius in meters (recommended: 0.10-0.20m)
                   Larger radius = more conservative (harder to get close to walls)
                   Smaller radius = more aggressive (can get closer to walls, more clipping risk)
            disable_gravity: Should be True to prevent camera from falling
            visible: If False, sphere is invisible in renders (recommended for data generation)
                    If True, sphere is visible (useful for debug visualization)

        Returns:
            The created sphere prim (child of cam_prim at "collision_sphere" path)
        """
        stage = omni.usd.get_context().get_stage()

        sphere_path = str(cam_prim.GetPath()) + "/collision_sphere"
        sphere_prim = stage.DefinePrim(sphere_path, "Sphere")

        if not sphere_prim.HasAttribute("radius"):
            sphere_prim.CreateAttribute("radius", Sdf.ValueTypeNames.Double)
        sphere_prim.GetAttribute("radius").Set(radius)

        if not visible:
            imageable = UsdGeom.Imageable(sphere_prim)
            imageable.MakeInvisible()
        PhysicsEngine.add_colliders_and_dynamics(sphere_prim, disable_gravity=disable_gravity)

        logger.debug(f"Added collision sphere (r={radius}m) to camera {cam_prim.GetPath()}")

        return sphere_prim

    @staticmethod
    def run_simulation(num_frames: int, render: bool = True) -> None:
        """
        Step PhysX simulation forward for objects to settle.

        render=True: timeline mode with app updates (slower, visualizable).
        render=False: headless mode with direct simulate+fetch (faster, production).

        Args:
            num_frames: Number of physics steps to simulate
                       Typical values: 60-120 frames (1-2 seconds of physics time)
                       More frames = more settling time but slower execution
            render: Execution mode selection
                   True = timeline mode with rendering (slower, visualizable)
                   False = headless mode without rendering (faster, production)
        """
        if render:
            timeline = omni.timeline.get_timeline_interface()
            timeline.set_start_time(0)
            timeline.set_end_time(1000000)
            timeline.set_looping(False)
            timeline.play()
            for _ in range(num_frames):
                omni.kit.app.get_app().update()
            timeline.pause()
        else:
            stage = omni.usd.get_context().get_stage()
            physx_scene = None

            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
                    break

            if physx_scene is None:
                physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

            physx_dt = 1 / physx_scene.GetTimeStepsPerSecondAttr().Get()
            physx_sim_interface = omni.physx.get_physx_simulation_interface()

            for _ in range(num_frames):
                physx_sim_interface.simulate(physx_dt, 0)
                physx_sim_interface.fetch_results()
