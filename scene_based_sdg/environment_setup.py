# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Environment setup and loading utilities.
Handles environment loading, collision setup, and prim searching.
"""

import logging

import omni.kit.commands
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom

from .physics_engine import PhysicsEngine

logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """Manages environment loading and configuration."""

    @staticmethod
    def load_env(usd_path: str, prim_path: str, remove_existing: bool = True) -> Usd.Prim:
        """
        Load a USD environment file into the stage as a reference.

        Imports an external USD file (typically a room, building, or scene) into the current
        stage at the specified prim path using USD's reference composition arc. This creates
        a non-destructive link to the external file rather than copying its contents, allowing
        multiple environments to be loaded/unloaded efficiently.

        The remove_existing parameter enables environment cycling: when generating synthetic
        data across multiple environments, the pipeline can delete the previous environment
        prim and load a new one at the same path. This keeps the stage hierarchy clean and
        prevents prim path conflicts.

        USD references maintain the original file's structure and can be edited non-destructively.
        Changes to the referenced USD file automatically propagate to all stages referencing it.
        This is ideal for synthetic data generation where the same base environments are used
        across multiple data collection runs.

        Args:
            usd_path: Path to USD environment file (supports omniverse://, file://, or local paths)
                     Examples: "omniverse://server/Environments/room.usd"
                              "/Isaac/Samples/Replicator/Infinigen/dining_rooms/dining_room_0.usd"
            prim_path: Stage path where environment will be loaded (typically "/Environment")
                      Must be a valid USD path (starts with /, no spaces)
            remove_existing: If True, deletes existing prim at prim_path before loading new environment
                           Used for environment cycling in multi-environment data generation

        Returns:
            Root prim of the loaded environment reference
            This prim serves as the parent for all environment geometry (walls, floor, furniture, etc.)

        Example:
            # Load Infinigen dining room
            env = EnvironmentSetup.load_env(
                usd_path="/Isaac/Samples/Replicator/Infinigen/dining_rooms/dining_room_0.usd",
                prim_path="/Environment",
                remove_existing=True
            )
        """
        stage = omni.usd.get_context().get_stage()

        if remove_existing and stage.GetPrimAtPath(prim_path):
            omni.kit.commands.execute("DeletePrimsCommand", paths=[prim_path])

        root_prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        return root_prim

    @staticmethod
    def add_colliders_to_env(root_path: str | None = None, approximation_type: str = "none") -> None:
        """
        Apply PhysX collision shapes to all mesh geometry in the environment.

        Traverses the USD hierarchy and adds PhysX collision properties to every UsdGeom.Mesh
        prim found. This enables physics simulation: objects can rest on floors, collide with
        walls, and sit on furniture surfaces. Without colliders, objects would fall through
        geometry as if it were a ghost.

        The approximation_type parameter controls the collision shape complexity:
        - "none": Uses exact mesh geometry (slow but accurate - not recommended for complex meshes)
        - "convexHull": Wraps mesh in convex hull (fast, good for most objects)
        - "convexDecomposition": Splits concave meshes into convex pieces (better accuracy)
        - "boundingCube": Simple box (fastest, good for simple rectangular objects)
        - "boundingSphere": Simple sphere (fastest for round objects)

        For environments with thousands of mesh prims (walls, floor tiles, furniture details),
        this can be expensive. In production, environments should ideally have pre-authored
        collision shapes, but this method provides automatic setup for procedurally generated
        or imported environments without physics metadata.

        Args:
            root_path: Stage path to search under (e.g., "/Environment")
                      If None, searches entire stage from PseudoRoot (includes all loaded assets)
            approximation_type: PhysX collision approximation mode
                               "none" = exact mesh (slow)
                               "convexHull" = convex wrapper (recommended default)
                               "convexDecomposition" = multiple convex pieces (accurate for concave)
                               "boundingCube" / "boundingSphere" = simple primitives (fastest)

        Example:
            # Add convex hull colliders to entire environment
            EnvironmentSetup.add_colliders_to_env("/Environment", approximation_type="convexHull")

            # Add exact mesh colliders (slow, not recommended for complex environments)
            EnvironmentSetup.add_colliders_to_env("/Environment", approximation_type="none")
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

        for prim in Usd.PrimRange(prim):
            if prim.IsA(UsdGeom.Mesh):
                PhysicsEngine.add_colliders(prim, approximation_type)

    @staticmethod
    def find_matching_prims(
        match_strings: list[str],
        root_path: str | None = None,
        prim_type: str | None = None,
        first_match_only: bool = False
    ) -> Usd.Prim | list[Usd.Prim] | None:
        """
        Search USD stage hierarchy for prims matching name substrings and type criteria.

        Performs depth-first traversal of the USD scene graph, testing each prim's path
        against the provided substring patterns and optionally filtering by USD type.
        This is the primary search utility used throughout the codebase to locate specific
        scene elements (walls, floors, furniture, lights, etc.) without hard-coding paths.

        The search is case-sensitive and uses simple substring matching (not regex). Each
        prim path is tested against ALL match_strings - if ANY substring matches, the prim
        passes the name filter. The prim_type filter is applied as an AND condition after
        the name filter passes.

        Example matching logic:
        - match_strings=["wall", "Wall"] → matches "/Environment/wall_001" AND "/Room/Wall"
        - prim_type="Xform" → only returns Xform prims (excludes Mesh, Camera, etc.)

        first_match_only mode short-circuits the search after finding the first qualifying
        prim, useful for unique landmarks like "spawn_point" or "floor" where multiple
        matches are unexpected. Returns the prim directly instead of a single-item list.

        Args:
            match_strings: List of case-sensitive substrings to search for in prim paths
                          Uses OR logic (any match qualifies the prim)
                          Examples: ["Table", "table"] matches both "/Furniture/Table" and "/props/table_01"
            root_path: Optional root path to narrow search scope (e.g., "/Environment")
                      If None, searches entire stage from PseudoRoot (includes all prims)
            prim_type: Optional USD type name filter (e.g., "Xform", "Mesh", "Camera")
                      Applied after name matching (AND logic)
                      If None, all types are included
            first_match_only: If True, returns first matching prim instead of list
                            Returns None if no match found (not empty list)
                            Useful for finding unique objects (floor, ceiling, spawn point)

        Returns:
            - If first_match_only=True: Single Usd.Prim or None
            - If first_match_only=False: List of Usd.Prim (may be empty)
        """
        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

        logger.debug(f"find_matching_prims called with:")
        logger.debug(f"  match_strings: {match_strings}")
        logger.debug(f"  root_path: {root_path}")
        logger.debug(f"  prim_type: {prim_type}")
        logger.debug(f"  root_prim path: {root_prim.GetPath()}")
        logger.debug(f"  root_prim is valid: {root_prim.IsValid()}")

        matching_prims = []
        prim_count = 0
        matched_substring_count = 0

        for prim in Usd.PrimRange(root_prim):
            prim_count += 1
            prim_path = str(prim.GetPath())
            prim_type_name = prim.GetTypeName()

            substring_match = any(match in prim_path for match in match_strings)
            if substring_match:
                matched_substring_count += 1

            type_match = prim_type is None or prim_type_name == prim_type

            if substring_match and matched_substring_count <= 10:
                logger.debug(f"  Prim with substring match: {prim_path} (type: {prim_type_name}, type_match: {type_match})")

            if substring_match and type_match:
                if first_match_only:
                    return prim
                matching_prims.append(prim)

        logger.debug(f"Traversal complete:")
        logger.debug(f"  Total prims checked: {prim_count}")
        logger.debug(f"  Prims with substring match: {matched_substring_count}")
        logger.debug(f"  Prims matching both substring AND type: {len(matching_prims)}")

        return matching_prims if not first_match_only else None

    @staticmethod
    def hide_matching_prims(
        match_strings: list[str],
        root_path: str | None = None,
        prim_type: str | None = None
    ) -> None:
        """
        Make prims invisible by setting their USD visibility attribute.

        Searches for prims matching the given name substrings and optionally type filter,
        then sets their visibility attribute to "invisible". This hides the geometry from
        rendering without deleting it from the stage, useful for:
        - Debug visualization (hiding ceilings/exterior walls for top-down view)
        - Fixing rendering artifacts (hiding problematic mesh elements)
        - Optimizing performance (culling unnecessary visual elements)

        The visibility attribute is inherited through the USD hierarchy: hiding a parent
        Xform also hides all child meshes. This makes it efficient to hide entire objects
        (e.g., hiding "/Environment/Ceiling" hides all ceiling tiles and light meshes).

        Unlike deleting prims, hiding them preserves collision shapes and scene structure.
        Objects still physically exist for physics simulation but are excluded from rendering
        and camera captures. This is critical when hiding ceilings - objects can still be
        placed on furniture, but the ceiling doesn't occlude the view.

        Args:
            match_strings: List of case-sensitive substrings to match in prim paths
                          Uses OR logic (any match hides the prim)
                          Examples: ["_exterior", "_ceiling"] hides both exterior and ceiling geometry
            root_path: Optional root path to narrow search (e.g., "/Environment")
                      If None, searches entire stage from PseudoRoot
            prim_type: Optional USD type name filter (e.g., "Xform", "Mesh")
                      If None, all types are included
        """
        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

        for prim in Usd.PrimRange(root_prim):
            if prim_type is None or prim.GetTypeName() == prim_type:
                if any(match in str(prim.GetPath()) for match in match_strings):
                    prim.GetAttribute("visibility").Set("invisible")

    @staticmethod
    def setup_env(
        root_path: str | None = None,
        approximation_type: str = "none",
        hide_top_walls: bool = False
    ) -> None:
        """
        Configure loaded environment for physics simulation and rendering.

        Applies environment-specific fixes and general setup steps to make a raw USD environment
        ready for synthetic data generation. Handles both generic setup (collision shapes) and
        environment-specific workarounds (Infinigen lighting issues, table collision accuracy).

        Setup steps performed:
        1. Fix Infinigen ceiling lights: Glass mesh blocks light emission, set to invisible
        2. Optionally hide ceiling/exterior walls for debug top-down visualization
        3. Add PhysX colliders to all meshes (walls, floor, furniture)
        4. Fix dining table collision with bounding cube (better than convex hull for tables)

        The Infinigen-specific fixes address known issues in procedurally generated environments:
        - Ceiling lights have "001_SPLIT_GLA" glass meshes that block light rays → make invisible
        - Dining tables have complex concave geometry that causes physics instability → use simpler cube

        These fixes are environment-specific (Infinigen procedural generation artifacts) but are
        applied unconditionally. If the target prims don't exist (non-Infinigen environments),
        the fixes gracefully skip with informational logs.

        The hide_top_walls parameter is useful for debug mode: hiding ceiling and exterior walls
        gives a clear top-down view of the room interior for verifying camera paths, object
        placement, and scene layout without occluding geometry.

        Args:
            root_path: Path to environment root prim (typically "/Environment")
                      If None, applies setup to entire stage from PseudoRoot
            approximation_type: PhysX collision approximation mode for mesh colliders
                               "none" = exact mesh (slow)
                               "convexHull" = convex wrapper (default)
                               "convexDecomposition" = accurate concave (slower)
            hide_top_walls: If True, hides ceiling and exterior walls for debug visualization
                          Useful for top-down viewport inspection of room layout
        """
        ceiling_light_meshes = EnvironmentSetup.find_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")
        for light_mesh in ceiling_light_meshes:
            light_mesh.GetAttribute("visibility").Set("invisible")

        if hide_top_walls:
            EnvironmentSetup.hide_matching_prims(["_exterior", "_ceiling"], root_path)

        EnvironmentSetup.add_colliders_to_env(root_path, approximation_type)

        table_prim = EnvironmentSetup.find_matching_prims(
            match_strings=["TableDining"], root_path=root_path, prim_type="Xform", first_match_only=True
        )
        if table_prim is not None:
            PhysicsEngine.add_colliders(table_prim, approximation_type="boundingCube")
        else:
            logger.info("Could not find dining table prim in the environment.")
