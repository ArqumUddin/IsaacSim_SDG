# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Transform utilities for USD prims.
Handles position, rotation, scale operations and adaptive scaling.
"""

import logging

from pxr import Gf, Usd, UsdGeom

logger = logging.getLogger(__name__)


class TransformUtils:
    """Utilities for manipulating USD prim transforms."""

    @staticmethod
    def set_transform_attributes(
        prim: Usd.Prim,
        location: Gf.Vec3d | None = None,
        orientation: Gf.Quatf | None = None,
        rotation: Gf.Vec3f | None = None,
        scale: Gf.Vec3f | None = None
    ) -> None:
        """
        Apply transform operations to USD prim via xformOp attributes.

        Sets translation, rotation (quaternion or Euler), and scale independently (only non-None params applied).
        Transform order: translate → orient/rotate → scale. Creates missing xformOp attributes automatically.
        Use orientation (quaternion) for cameras; rotation (Euler) for user-friendly angle specification.

        Args:
            prim: USD prim to transform (must be UsdGeom.Xformable)
            location: Translation vector (x, y, z) in meters or None to skip
            orientation: Quaternion orientation (w, x, y, z) or None to skip
                        Preferred for cameras and smooth interpolation
            rotation: Euler angle rotation (X, Y, Z) in degrees or None to skip
                     Applied in XYZ order (roll, pitch, yaw)
            scale: Scale factors (x, y, z) where 1.0 = original size, or None to skip
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

    @staticmethod
    def get_prim_bounding_box(prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d] | None:
        """
        Compute world-space axis-aligned bounding box (AABB) for a prim.

        Uses UsdGeom.BBoxCache to compute AABB including all descendant geometry and parent transforms.
        Essential for object placement, collision avoidance, surface detection, and room bounds calculation.
        Returns None for prims without geometry, empty/invalid bboxes, or zero-volume (degenerate) boxes.

        Args:
            prim: USD prim to compute bounding box for
                 Can be any prim type, but only geometric prims have meaningful boxes

        Returns:
            Tuple of (bbox_min, bbox_max) as Gf.Vec3d world coordinates, or None if:
            - Prim has no geometry
            - Bounding box is empty or invalid
            - Bounding box has zero volume (degenerate)
            - Computation raises an exception
        """
        try:
            prim_path = str(prim.GetPath())
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim)

            if not bbox or bbox.GetRange().IsEmpty():
                logger.debug(f"bbox: Empty or invalid bounding box: {prim_path}")
                return None

            bbox_range = bbox.ComputeAlignedRange()
            bbox_min = bbox_range.GetMin()
            bbox_max = bbox_range.GetMax()

            if bbox_min == bbox_max:
                logger.debug(f"bbox: Degenerate bbox (zero volume): {prim_path}")
                return None

            logger.debug(f"bbox: Success for {prim_path}: min={bbox_min}, max={bbox_max}")

            return (bbox_min, bbox_max)
        except Exception as e:
            logger.info(f"Warning: Failed to compute bounding box for '{prim.GetPath()}': {e}")
            return None

    @staticmethod
    def get_prim_world_transform(prim: Usd.Prim) -> Gf.Matrix4d | None:
        """
        Compute cumulative world-space 4x4 transform matrix for a prim.

        Composes all transforms from prim to stage root (own xformOps + parent hierarchy).
        Encodes translation, rotation, scale; used to extract world-space position, orientation, and normal vectors.
        Returns None if prim is not xformable or computation fails.

        Args:
            prim: USD prim to compute world transform for
                 Must be UsdGeom.Xformable or have transform stack

        Returns:
            4x4 transformation matrix (Gf.Matrix4d) in world space, or None if:
            - Prim is not xformable
            - Transform computation fails
        """
        try:
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                return None
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return world_transform
        except Exception as e:
            logger.debug(f"Could not get world transform for {prim.GetPath()}: {e}")
            return None

    @staticmethod
    def extract_normal_from_transform(transform: Gf.Matrix4d, axis: int = 0) -> Gf.Vec3d:
        """
        Extract axis direction vector from 4x4 transform matrix rotation component.

        Converts rotation to 3x3 matrix and extracts column vector for specified axis (0=X, 1=Y, 2=Z).
        Critical for wall collision detection: Y-axis typically gives wall normal for realistic camera bouncing.
        Returns normalized world-space direction vector.

        Args:
            transform: 4x4 transformation matrix (typically from get_prim_world_transform)
            axis: Which local axis to extract as world-space direction
                 0 = X-axis (left/right)
                 1 = Y-axis (forward/back)
                 2 = Z-axis (up/down)

        Returns:
            Normalized 3D vector representing the chosen axis direction in world space
        """
        rotation = transform.ExtractRotation()
        quat = rotation.GetQuat()
        matrix3 = Gf.Matrix3d(quat)
        if axis == 0:
            normal = Gf.Vec3d(matrix3[0][0], matrix3[1][0], matrix3[2][0])
        elif axis == 1:
            normal = Gf.Vec3d(matrix3[0][1], matrix3[1][1], matrix3[2][1])
        else:
            normal = Gf.Vec3d(matrix3[0][2], matrix3[1][2], matrix3[2][2])
        return normal.GetNormalized()


class AdaptiveScaling:
    """Adaptive scaling to make small rooms feel bigger."""

    @staticmethod
    def calculate_scale_factor(
        room_bounds: tuple[float, float, float, float, float, float],
        target_size: float,
        min_scale: float,
        max_scale: float
    ) -> float:
        """
        Compute adaptive object scaling to make small rooms feel spacious.

        Calculates scale_factor = min(width, depth) / target_size, clamped to [min_scale, max_scale].
        Uses smaller dimension to handle asymmetric rooms conservatively (e.g., narrow corridors).
        Shrinks objects in small procedural rooms to maintain realistic object-to-room size ratios.

        Args:
            room_bounds: Room AABB as (min_x, min_y, min_z, max_x, max_y, max_z) in meters
            target_size: Reference room size in meters (e.g., 5.0 = typical room size)
                        Rooms smaller than this get scaled-down objects
            min_scale: Minimum allowed scale factor (e.g., 0.15 = don't go below 15%)
            max_scale: Maximum allowed scale factor (e.g., 1.0 = never upscale)

        Returns:
            Scale multiplier to apply to object scales
            1.0 = no scaling, 0.5 = half size, 0.2 = miniature (1/5 scale)
        """
        min_x, min_y, min_z, max_x, max_y, max_z = room_bounds

        actual_width = max_x - min_x
        actual_depth = max_y - min_y

        actual_size = min(actual_width, actual_depth)

        scale_factor = actual_size / target_size
        scale_factor = max(min_scale, min(scale_factor, max_scale))

        logger.info(f"Room size: {actual_width:.2f}m × {actual_depth:.2f}m")
        logger.info(f"Adaptive scale factor: {scale_factor:.3f} (objects scaled to {scale_factor*100:.1f}%)")

        return scale_factor

    @staticmethod
    def apply_scale_to_prim(prim: Usd.Prim, scale_factor: float) -> None:
        """
        Apply uniform scale multiplier to prim while preserving other transforms.

        Multiplies existing scale by scale_factor (compounds rather than replaces).
        Preserves transform order by clearing xformOpOrder and reapplying translate → orient/rotate → scale.
        Use case: Apply room-aware adaptive scaling after initial random placement with randomize_poses_with_surfaces().

        Args:
            prim: USD prim to scale (must be UsdGeom.Xformable)
            scale_factor: Multiplier to apply to existing scale
                         1.0 = no change, 0.5 = shrink to half, 2.0 = double size
        """
        xformable = UsdGeom.Xformable(prim)
        existing_scale = Gf.Vec3f(1, 1, 1)
        for xform_op in xformable.GetOrderedXformOps():
            if xform_op.GetOpType() == UsdGeom.XformOp.TypeScale:
                existing_scale = xform_op.Get()
                break

        new_scale = Gf.Vec3f(
            existing_scale[0] * scale_factor,
            existing_scale[1] * scale_factor,
            existing_scale[2] * scale_factor
        )

        xformable.ClearXformOpOrder()
        if prim.HasAttribute("xformOp:translate"):
            loc = prim.GetAttribute("xformOp:translate").Get()
            xformable.AddTranslateOp().Set(loc)
        if prim.HasAttribute("xformOp:orient"):
            orient = prim.GetAttribute("xformOp:orient").Get()
            xformable.AddOrientOp().Set(orient)
        if prim.HasAttribute("xformOp:rotateXYZ"):
            rot = prim.GetAttribute("xformOp:rotateXYZ").Get()
            xformable.AddRotateXYZOp().Set(rot)

        xformable.AddScaleOp().Set(new_scale)
