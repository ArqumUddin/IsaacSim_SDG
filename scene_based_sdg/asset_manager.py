# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Asset management for loading and scanning USD files.
Handles asset discovery, distractor creation, and scale resolution.
"""

import logging
import os
import random
import re
from itertools import chain

import omni.client
import omni.kit.app
import omni.usd
from omni.metrics.assembler.core import get_metrics_assembler_interface
from isaacsim.core.utils.semantics import remove_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Usd
from .physics_engine import PhysicsEngine

logger = logging.getLogger(__name__)

class AssetManager:
    """Manages asset loading, scanning, and distractor creation."""

    @staticmethod
    def _get_usd_paths_from_folder(
        folder_path: str, recursive: bool = True, usd_paths: list[str] = None, skip_keywords: list[str] = None
    ) -> list[str]:
        """
        Retrieve USD file paths from a folder using Omniverse client.

        Recursively scans a folder for USD files (.usd, .usda, .usdc) using omni.client,
        which supports both local paths and Omniverse Nucleus server paths. Filters out
        files/folders matching skip_keywords (case-insensitive).

        Args:
            folder_path: Path to scan (supports omniverse://, file://, or local paths)
            recursive: Whether to recurse into subdirectories
            usd_paths: Accumulator list for recursive calls (created if None)
            skip_keywords: List of keywords to filter out (e.g., ["test", "backup"])

        Returns:
            List of USD file paths found in the folder
        """
        if usd_paths is None:
            usd_paths = []
        skip_keywords = skip_keywords or []

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        if not ext_manager.is_extension_enabled("omni.client"):
            ext_manager.set_extension_enabled_immediate("omni.client", True)

        result, entries = omni.client.list(folder_path)
        if result != omni.client.Result.OK:
            logger.info(f"Could not list assets in path: {folder_path}")
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
                AssetManager._get_usd_paths_from_folder(sub_folder, recursive=recursive, usd_paths=usd_paths, skip_keywords=skip_keywords)

        return usd_paths

    @staticmethod
    def get_usd_paths(
        files: list[str] = None, folders: list[str] = None, skip_folder_keywords: list[str] = None
    ) -> list[str]:
        """
        Aggregate USD paths from files and folders with protocol handling.

        Processes both individual file paths and folder paths. For paths without a protocol
        (omniverse://, http://, https://, file://), prepends the Isaac Sim assets root path.
        This allows using shorthand paths for built-in Isaac Sim assets while supporting
        full paths for custom assets.

        Args:
            files: List of individual USD file paths
            folders: List of folder paths to scan recursively
            skip_folder_keywords: Keywords to filter out when scanning folders

        Returns:
            Combined list of all USD file paths
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
            env_paths.extend(AssetManager._get_usd_paths_from_folder(folder_path, recursive=True, skip_keywords=skip_folder_keywords))

        return env_paths

    @staticmethod
    def _process_asset_path(usd_path: str, regex_pattern: str, regex_repl: str) -> dict:
        """
        Extract label from USD asset path using regex substitution.

        Extracts the filename (without extension) from a USD path and optionally applies
        regex pattern replacement to generate a clean label. Used for auto-labeling assets
        based on filename conventions (e.g., "002_banana.usd" -> "banana").

        Args:
            usd_path: Full path to the USD asset
            regex_pattern: Regex pattern to match in the filename (empty string = no replacement)
            regex_repl: Replacement string for matched pattern

        Returns:
            Dictionary with 'url' (full path) and 'label' (processed name) keys
        """
        basename = os.path.basename(usd_path)
        name_without_ext = os.path.splitext(basename)[0]
        label = re.sub(regex_pattern, regex_repl, name_without_ext) if regex_pattern else name_without_ext
        return {"url": usd_path, "label": label}

    @staticmethod
    def scan_assets(
        folders: list[str],
        files: list[str],
        regex_replace_pattern: str = "",
        regex_replace_repl: str = "",
        recursive: bool = True
    ) -> list[dict]:
        """
        Scan directories for USD assets and generate labels via regex.

        Collects all USD assets from the specified files and folders, applies regex-based
        label generation to extract clean names from filenames. Ensures files explicitly
        listed are not duplicated when also found in folder scans.

        Args:
            folders: List of folder paths to scan
            files: List of individual file paths
            regex_replace_pattern: Regex pattern to match in filenames (e.g., "^[0-9_]+")
            regex_replace_repl: Replacement string (e.g., "" to remove matched pattern)
            recursive: Whether to scan folders recursively (currently unused)

        Returns:
            List of dicts with 'url' and 'label' keys for each asset found
        """
        found_assets = []

        resolved_files = AssetManager.get_usd_paths(files=files)
        for f in resolved_files:
            found_assets.append(AssetManager._process_asset_path(f, regex_replace_pattern, regex_replace_repl))

        resolved_folder_assets = AssetManager.get_usd_paths(folders=folders, skip_folder_keywords=["material", "texture", ".thumbs"])
        for f in resolved_folder_assets:
            if not any(a["url"] == f for a in found_assets):
                found_assets.append(AssetManager._process_asset_path(f, regex_replace_pattern, regex_replace_repl))

        return found_assets

    @staticmethod
    def _create_shape_distractors(
        num_distractors: int, shape_types: list[str], root_path: str, gravity_disabled_chance: float
    ) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
        """
        Create geometric shape distractors (spheres, cubes, cones, etc.).

        Generates simple USD primitive shapes at random, adds collision and physics properties,
        and randomly disables gravity on a percentage of them (for floating vs falling objects).
        Each shape gets a unique path using get_stage_next_free_path.

        Args:
            num_distractors: Number of shape distractors to create
            shape_types: List of shape type names (e.g., ["sphere", "cube", "cylinder"])
            root_path: USD path prefix for all distractors (e.g., "/Distractors")
            gravity_disabled_chance: Probability (0.0-1.0) that gravity is disabled on each shape

        Returns:
            Tuple of (floating_shapes, falling_shapes) prim lists
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
            PhysicsEngine.add_colliders_and_dynamics(prim, disable_gravity=disable_gravity)
            (floating_shapes if disable_gravity else falling_shapes).append(prim)

        return floating_shapes, falling_shapes

    @staticmethod
    def load_shape_distractors(config: dict) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
        """
        Load shape distractors from config dictionary.

        Convenience wrapper around _create_shape_distractors that extracts parameters
        from a config dict with sensible defaults.

        Args:
            config: Dictionary with keys: 'num', 'types', 'gravity_disabled_chance'

        Returns:
            Tuple of (floating_shapes, falling_shapes) prim lists
        """
        return AssetManager._create_shape_distractors(
            config.get("num", 0),
            config.get("types", ["capsule", "cone", "cylinder", "sphere", "cube"]),
            "/Distractors",
            config.get("gravity_disabled_chance", 0.0)
        )

    @staticmethod
    def _create_mesh_distractors(
        num_distractors: int, mesh_urls: list[str], root_path: str, gravity_disabled_chance: float
    ) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
        """
        Create mesh-based distractors by loading USD files.

        Loads pre-made 3D mesh assets from USD files (e.g., books, props, furniture),
        adds collision and physics properties, and randomly disables gravity on a percentage.
        Uses add_reference_to_stage to load external USD files into the scene.

        Args:
            num_distractors: Number of mesh distractors to create
            mesh_urls: List of USD file paths to randomly choose from
            root_path: USD path prefix for all distractors (e.g., "/Distractors")
            gravity_disabled_chance: Probability (0.0-1.0) that gravity is disabled on each mesh

        Returns:
            Tuple of (floating_meshes, falling_meshes) prim lists
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
                logger.info(f"Failed to load mesh distractor {rand_mesh_url}: {e}")
                continue

            PhysicsEngine.add_colliders_and_dynamics(prim, disable_gravity=disable_gravity)
            (floating_meshes if disable_gravity else falling_meshes).append(prim)

        return floating_meshes, falling_meshes

    @staticmethod
    def load_mesh_distractors(config: dict) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
        """
        Load mesh distractors from config dictionary.

        Resolves USD paths from config, creates mesh distractors, and strips any semantic
        labels from them (since distractors should not appear in annotations). Pre-made
        mesh assets sometimes have labels baked in, so we remove them to keep distractors
        unlabeled in the final dataset.

        Args:
            config: Dictionary with keys: 'num', 'files', 'folders', 'gravity_disabled_chance'

        Returns:
            Tuple of (floating_meshes, falling_meshes) prim lists
        """
        mesh_urls = AssetManager.get_usd_paths(
            files=config.get("files", []),
            folders=config.get("folders", []),
            skip_folder_keywords=["material", "texture", ".thumbs"]
        )

        floating, falling = AssetManager._create_mesh_distractors(
            config.get("num", 0),
            mesh_urls,
            "/Distractors",
            config.get("gravity_disabled_chance", 0.0)
        )

        for prim in chain(floating, falling):
            remove_labels(prim, include_descendants=True)

        return floating, falling

    @staticmethod
    def resolve_scale_issues():
        """
        Fix scale discrepancies across assets using Metrics Assembler.

        USD assets from different sources may be authored with different units (meters,
        centimeters, inches). This can cause scale mismatches when loading mixed assets
        (e.g., one object appears 100x smaller than expected). The Metrics Assembler scans
        all prims in the stage, detects unit metadata mismatches, and applies corrective
        scale transforms to make everything consistent with the stage's metersPerUnit setting.

        Should be called after loading all assets but before positioning or physics simulation.
        """
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        if not ext_manager.is_extension_enabled("omni.usd.metrics.assembler"):
            ext_manager.set_extension_enabled_immediate("omni.usd.metrics.assembler", True)

        stage_id = omni.usd.get_context().get_stage_id()
        get_metrics_assembler_interface().resolve_stage(stage_id)
