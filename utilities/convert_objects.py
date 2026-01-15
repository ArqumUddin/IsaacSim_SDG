import os
import argparse
import yaml
import shutil
import glob
import asyncio
import re
from pathlib import Path

# Note: Omniverse kits must be launched before importing omni modules
from isaacsim import SimulationApp

# Launch in Headless mode with Asset Converter extension enabled
simulation_app = SimulationApp({
    "headless": True,
    "enable_extensions": ["omni.kit.asset_converter"]
})

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.prims import SingleRigidPrim as RigidPrim, SingleGeometryPrim as GeometryPrim
from pxr import UsdGeom, UsdPhysics, Gf, Usd, UsdShade, Sdf
import omni.usd
import omni.kit.asset_converter


def load_config(config_path):
    """
    Loads the config file.

    Args:
        config_path (str): The path to the config file.
    """
    if not os.path.exists(config_path):
        # Fallback to local dir
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(curr_dir, config_path)
        if os.path.exists(relative_path):
            config_path = relative_path
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

async def convert_asset_native(source_path, output_path):
    """
    Uses omni.kit.asset_converter to convert mesh to USD.

    Args:
        source_path (str): The path to the source mesh.
        output_path (str): The path to the output USD.
    """
    converter = omni.kit.asset_converter.get_instance()
    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.use_meter_as_world_unit = True # Ensure typical Isaac Sim scale
    context.merge_all_meshes = True
    context.ignore_animations = True
    context.export_preview_surface = False  # Use original materials
    
    print(f"  Native Conversion: {source_path} -> {output_path}")
    task = converter.create_converter_task(source_path, output_path, None, context)
    
    # Wait for the async task to complete
    success = await task.wait_until_finished()
    if not success:
        error_msg = task.get_error_message()
        print(f"  ERROR: Conversion failed for {source_path}: {error_msg}")
        return False
    return True

def parse_mtl(mtl_path):
    """
    Parses a simple MTL file for basic material properties.
    Returns a dict with 'Kd', 'Ks', 'Ns', 'd', 'Ni', 'map_Kd'.
    """
    props = {
        'Kd': (1.0, 1.0, 1.0), # Diffuse (Default White to ensure map_Kd works)
        'Ks': None,            # Specular (Default None -> Let USD decide, usually 0)
        'Ns': None,            # Shininess (Default None -> Let USD decide, usually 0.5 rough)
        'd': 1.0,              # Opacity
        'Ni': None,            # IOR
        'map_Kd': None         # Texture map
    }
    
    if not os.path.exists(mtl_path):
        return props

    try:
        with open(mtl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                key = parts[0]
                # Handle potential comments
                if key.startswith("#"):
                    continue
                    
                if key == 'Ks':
                    props['Ks'] = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif key == 'Ns':
                    props['Ns'] = float(parts[1])
                elif key == 'd':
                    props['d'] = float(parts[1])
                elif key == 'Kd':
                    props['Kd'] = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif key == 'Ni':
                    props['Ni'] = float(parts[1])
                elif key == 'map_Kd':
                    # Can be multiple parts if spaces in filename, but usually last part
                    # Or take the rest of the line ensuring to strip comments if any
                    content = line.strip().split(' ', 1)[1]
                    props['map_Kd'] = content.strip()
    except Exception as e:
        print(f"    Warning: Failed to parse MTL {mtl_path}: {e}")
        
    return props

def texture_application(stage, root_prim_path, source_obj_path):
    """
    Manually binds a texture to a prim using Strict Asset Separation.

    1. Uses source_obj_path to find the local MTL file.
    2. Parses MTL for properties (Ns, Ks, Kd, etc.) and Texture (map_Kd).
    3. Applies these STRICTLY from the source folder (node cross-folder borrowing).
    """
    source_dir = os.path.dirname(source_obj_path)
    obj_name_stem = Path(source_obj_path).stem
    
    # Check for OBJ-name-based MTL
    mtl_candidates = [
        os.path.join(source_dir, f"{obj_name_stem}.mtl"),
        os.path.join(source_dir, "textured.mtl")
    ]
    
    source_mtl_path = None
    for c in mtl_candidates:
        if os.path.exists(c):
            source_mtl_path = c
            break
            
    if not source_mtl_path:
        # No MTL found, cannot determine texture strictly from MTL
        return

    # Parse ONLY the local MTL
    props = parse_mtl(source_mtl_path)
    
    # Locate Texture from map_Kd
    texture_path = None
    if props['map_Kd']:
        # Try finding map_Kd relative to source dir
        t_path = os.path.join(source_dir, props['map_Kd'])
        if os.path.exists(t_path):
            texture_path = t_path
    
    # Fallback to "textured.png" ONLY if MTL didn't specify one or it wasn't found
    if not texture_path:
        fallback = os.path.join(source_dir, "textured.png")
        if os.path.exists(fallback):
            texture_path = fallback

    if not texture_path:
        return

    print(f"    [Hybrid] Applying Material (Strict). Source: {source_dir}. Texture: {os.path.basename(texture_path)}")

    # Create the Material Container
    mat_path = f"{root_prim_path}/Looks/VerifiedMaterial"
    material = UsdShade.Material.Define(stage, mat_path)

    # Create the Surface Shader (UsdPreviewSurface)
    shader_path = f"{mat_path}/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    
    # Diffuse Color
    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse_input.Set(Gf.Vec3f(*props['Kd']))

    # Specular Color (Ks)
    if props['Ks'] is not None:
        spec_input = shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f)
        spec_input.Set(Gf.Vec3f(*props['Ks']))

    # Roughness
    if props['Ns'] is not None:
        ns = props['Ns']
        roughness = 0.5 
        if ns > 0:
            # Roughness = sqrt(2 / (Ns + 2))
            roughness = (2.0 / (ns + 2.0)) ** 0.5
        
        roughness_input = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        roughness_input.Set(roughness)

    # Opacity (d)
    opacity_input = shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)
    opacity_input.Set(props['d'])

    # IOR (Ni)
    if props['Ni'] is not None:
        ior_input = shader.CreateInput("ior", Sdf.ValueTypeNames.Float)
        ior_input.Set(props['Ni'])

    # Create the Texture Sampler (UsdUVTexture)
    sampler_path = f"{mat_path}/DiffuseSampler"
    sampler = UsdShade.Shader.Define(stage, sampler_path)
    sampler.CreateIdAttr("UsdUVTexture")
    sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
    sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    # Create Primvar Reader
    reader_path = f"{mat_path}/stReader"
    reader = UsdShade.Shader.Define(stage, reader_path)
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    # Connect Reader -> Sampler
    sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")

    # Connect Sampler -> Surface Shader
    diffuse_input.ConnectToSource(sampler.ConnectableAPI(), "rgb")

    # Connect Surface Shader -> Material
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Apply to the Root Prim
    root_prim = stage.GetPrimAtPath(root_prim_path)
    UsdShade.MaterialBindingAPI(root_prim).Bind(
        material, 
        bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    
    print(f"    [Hybrid] Forced texture binding: {texture_path} (StrongerThanDescendants)")

def scale_mesh_vertices(root_prim, scale_factor):
    """
    Directly scale mesh vertex positions by the given factor.

    This is a more aggressive approach that modifies the actual mesh data
    rather than relying on transforms, which may not propagate correctly
    in some USD hierarchies.

    Args:
        root_prim: The root USD prim containing meshes.
        scale_factor: The uniform scale factor to apply to all vertices.
    """
    for descendant in Usd.PrimRange(root_prim):
        if descendant.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(descendant)
            points_attr = mesh.GetPointsAttr()
            if points_attr and points_attr.HasValue():
                points = points_attr.Get()
                if points:
                    # Scale each vertex position
                    scaled_points = [Gf.Vec3f(p[0] * scale_factor, p[1] * scale_factor, p[2] * scale_factor) for p in points]
                    points_attr.Set(scaled_points)

                    # Also update the extent attribute if it exists
                    extent_attr = mesh.GetExtentAttr()
                    if extent_attr and extent_attr.HasValue():
                        extent = extent_attr.Get()
                        if extent and len(extent) == 2:
                            scaled_extent = [
                                Gf.Vec3f(extent[0][0] * scale_factor, extent[0][1] * scale_factor, extent[0][2] * scale_factor),
                                Gf.Vec3f(extent[1][0] * scale_factor, extent[1][1] * scale_factor, extent[1][2] * scale_factor)
                            ]
                            extent_attr.Set(scaled_extent)

                    print(f"    Scaled mesh vertices: {descendant.GetPath()}")


def compute_raw_geometry_extent(root_prim):
    """
    Compute the raw untransformed extent of geometry under a prim.

    This bypasses any transform issues by using ComputeUntransformedBound
    and falling back to reading mesh extent attributes directly.

    Args:
        root_prim: The root USD prim to compute extent for.

    Returns:
        float: Maximum dimension of untransformed geometry, or 0.0 if no geometry found.
    """
    # Method 1: Use USD's ComputeUntransformedBound
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    untransformed_bound = bbox_cache.ComputeUntransformedBound(root_prim)
    aligned_range = untransformed_bound.ComputeAlignedRange()

    if not aligned_range.IsEmpty():
        size = aligned_range.GetSize()
        max_dim = max(size[0], size[1], size[2])
        if max_dim > 0:
            return max_dim

    # Method 2: Fallback to reading mesh extent attributes directly
    overall_max_dim = 0.0
    for descendant in Usd.PrimRange(root_prim):
        if descendant.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(descendant)
            extent_attr = mesh.GetExtentAttr()
            if extent_attr and extent_attr.HasValue():
                extent = extent_attr.Get()
                if extent and len(extent) == 2:
                    size = [extent[1][i] - extent[0][i] for i in range(3)]
                    overall_max_dim = max(overall_max_dim, max(size))

    return overall_max_dim


def validate_and_correct_scale(stage, root_prim, threshold, raw_size=None, applied_scale=None):
    """
    Validate final world size after vertex scaling and apply correction if needed.

    Since we now directly scale mesh vertices, we can compute the actual world
    bound (vertices have been modified, not just transforms).

    Args:
        stage: The USD stage.
        root_prim: The root prim to validate.
        threshold: Maximum allowed world size.
        raw_size: The original raw geometry size (before scaling).
        applied_scale: The scale factor that was applied to vertices.

    Returns:
        tuple: (final_max_dim, correction_applied) where correction_applied is True
               if additional scaling was needed.
    """
    # Compute actual world size from geometry extent (vertices have been modified)
    actual_world_size = compute_raw_geometry_extent(root_prim)

    # If we have the expected values, also compute expected for comparison
    if raw_size is not None and applied_scale is not None:
        expected_world_size = raw_size * applied_scale
        print(f"    [VALIDATION] Actual world size: {actual_world_size:.3f}m, Expected: {expected_world_size:.3f}m")
    else:
        print(f"    [VALIDATION] Actual world size: {actual_world_size:.3f}m")

    # Check if already within threshold
    if actual_world_size <= threshold:
        print(f"    [VALIDATION] Size {actual_world_size:.3f}m <= {threshold}m, OK")
        return (actual_world_size, False)

    # Need additional correction - scale vertices again
    correction = (threshold / actual_world_size) * 0.95
    print(f"    [VALIDATION] Size {actual_world_size:.3f}m > {threshold}m, applying vertex correction {correction:.6f}")

    scale_mesh_vertices(root_prim, correction)

    # Calculate final world size after correction
    final_world_size = actual_world_size * correction
    print(f"    [VALIDATION] Final size after correction: {final_world_size:.3f}m")

    return (final_world_size, True)


def normalize_to_target_size(root_prim, target_min=0.05, target_max=1.5):
    """
    Normalize object to fit within target size range by scaling mesh vertices.

    This function ensures all objects end up at reasonable real-world sizes,
    regardless of what units the source file used. Objects smaller than target_min
    are scaled up, objects larger than target_max are scaled down.

    Uses vertex scaling (not transform ops) to avoid double-scaling issues when
    the USD is loaded elsewhere.

    Args:
        root_prim: Root prim of the object
        target_min: Minimum acceptable size in meters (default 0.05m = 5cm)
        target_max: Maximum acceptable size in meters (default 1.5m)

    Returns:
        tuple: (final_size, scale_was_applied)
    """
    # Get current world size
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = bbox_cache.ComputeWorldBound(root_prim)
    size = bound.GetRange().GetSize()
    max_dim = max(size[0], size[1], size[2])

    if max_dim <= 0:
        print(f"    [NORMALIZE] Warning: Zero or negative size detected")
        return (0, False)

    # Determine if scaling needed
    if target_min <= max_dim <= target_max:
        print(f"    [NORMALIZE] Size {max_dim:.3f}m within target range [{target_min}, {target_max}]")
        return (max_dim, False)

    # Calculate scale factor
    if max_dim < target_min:
        # Object too small - scale up to minimum
        target = target_min
        scale_factor = target / max_dim
        action = "Scale UP"
    else:
        # Object too large - scale down to maximum
        target = target_max
        scale_factor = target / max_dim
        action = "Scale DOWN"

    print(f"    [NORMALIZE] {action}: {max_dim:.3f}m -> {target:.3f}m (factor: {scale_factor:.4f})")

    # Scale mesh vertices directly (not transform ops) to avoid double-scaling
    scale_mesh_vertices(root_prim, scale_factor)

    return (target, True)


def post_process_usd(stage_path, name, scale, semantic_label, source_path, auto_detect_units=False, threshold=50.0,
                     normalize_size=False, target_size_min=0.05, target_size_max=1.5):
    """
    Opens the converted USD and adds Physics, Collision, Semantics, Scale, and Material Fallback.

    Args:
        stage_path (str): The path to the USD stage.
        name (str): The name of the asset.
        scale (float): The scale factor for the asset.
        semantic_label (str): The semantic label for the asset.
        source_path (str): Path to original source asset (for material lookup).
    """
    # Open the stage
    stage_utils.open_stage(stage_path)
    stage = omni.usd.get_context().get_stage()

    # The converter usually creates a Root Node or just Mesh children.
    root_prim = stage.GetDefaultPrim()
    # If no default prim, find the first valid root Xform or define one
    if not root_prim:
        usd_safe_name = name.replace(" ", "_").replace(".", "_")
        root_path = f"/{usd_safe_name}"
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            children = [p for p in stage.GetPseudoRoot().GetChildren()]
            if children:
                root_prim = children[0]
    root_prim_path = root_prim.GetPath()

    action_taken = "None"
    detected_size = 0.0

    if auto_detect_units:
        # Use compute_raw_geometry_extent to get the TRUE raw mesh size
        # This bypasses any transforms the native converter may have applied
        raw_size = compute_raw_geometry_extent(root_prim)
        detected_size = raw_size

        print(f"  [AUTO-SCALE] Raw untransformed size: {raw_size:.3f} (Threshold: {threshold})")

        if raw_size > threshold:
            action_taken = "Iterative Normalization"
            # Iteratively reduce by power of 10 until within threshold
            # This handles mm, cm, or even smaller units (micrometers)
            temp_scale = 1.0
            reduced_dim = raw_size
            while reduced_dim > threshold:
                temp_scale *= 0.1
                reduced_dim = raw_size * temp_scale

            scale = scale * temp_scale
            action_taken += f" (x{temp_scale:.6f})"
        else:
            action_taken = "Kept Original"

    # Apply Material Strategy based on source format
    # OBJ: Uses external MTL + texture files, needs manual texture_application()
    # GLB/GLTF: Materials and textures are embedded, converter handles automatically
    source_ext = os.path.splitext(source_path)[1].lower()
    if source_ext == ".obj":
        texture_application(stage, root_prim_path, source_path)

    # Apply Scale by directly modifying mesh vertices
    # Transform-based scaling doesn't work reliably because the native converter
    # creates USD hierarchies where parent transforms don't propagate to geometry
    if scale != 1.0:
        print(f"    Scaling mesh vertices by factor: {scale}")
        scale_mesh_vertices(root_prim, scale)

    # Set scale op to (1,1,1) since we already scaled the vertices
    # This prevents double-scaling when the USD is loaded elsewhere
    # (e.g., SDG multiplies existing scale by random scale)
    final_scale = (1.0, 1.0, 1.0)
    xform = UsdGeom.Xformable(root_prim)
    scale_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if not scale_op:
        scale_op = xform.AddScaleOp()
    scale_op.Set(final_scale)

    # Validate and correct if final world size is still too large
    # This handles cases where the native converter applied internal transforms
    # that aren't captured by the raw geometry extent computation
    final_world_size = 0.0
    validation_applied = False
    if auto_detect_units:
        # Pass the raw size and the scale we just applied so validation can compute
        # expected world size mathematically (avoiding stale BBoxCache issues)
        final_world_size, validation_applied = validate_and_correct_scale(
            stage, root_prim, threshold, detected_size, final_scale[0]
        )
        if validation_applied:
            action_taken += " + Validation Correction"
            # Get the updated scale op value after validation
            for op in UsdGeom.Xformable(root_prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    final_scale = op.Get()
                    break

    # Target Size Normalization (runs AFTER auto_detect to ensure final size is reasonable)
    # This catches objects that ended up too small or too large after initial scaling
    normalize_applied = False
    if normalize_size:
        final_world_size, normalize_applied = normalize_to_target_size(
            root_prim, target_size_min, target_size_max
        )
        if normalize_applied:
            action_taken = "Size Normalization" if action_taken == "None" else action_taken + " + Size Normalization"
            # Scale op remains (1,1,1) since we scaled vertices directly

    # Explicit Report for User
    print(f"  [RESULT] Asset: {name}")
    if auto_detect_units:
        print(f"           Detected Size: {detected_size:.3f} (Threshold: {threshold})")
    print(f"           Action: {action_taken}")
    print(f"           Final Scale: ({final_scale[0]:.6f}, {final_scale[1]:.6f}, {final_scale[2]:.6f})")
    if auto_detect_units or normalize_size:
        print(f"           Final World Size: {final_world_size:.3f}m")
    if normalize_size:
        print(f"           Target Range: [{target_size_min}, {target_size_max}]m")
    print(f"           Post-process complete.\n")

    # Apply Semantics
    label = semantic_label if semantic_label else name
    add_labels(root_prim, labels=[label], instance_name="class")

    # Apply Physics
    rigid_prim = RigidPrim(prim_path=str(root_prim_path), mass=0.1)
    rigid_prim.enable_rigid_body_physics()

    # Apply Collision
    try:
        geometry_prim = GeometryPrim(prim_path=str(root_prim_path), collision=True)
        geometry_prim.set_collision_approximation("convexHull")
    except Exception as e:
        print(f"    Warning: Convex Hull failed: {e}. Trying bound cube.")
        try:
             geometry_prim = GeometryPrim(prim_path=str(root_prim_path), collision=True)
             geometry_prim.set_collision_approximation("boundingCube")
        except:
             print(f"    Error: Failed to add collision.")

    # Save changes
    stage.GetRootLayer().Save()
    print(f"    Post-process complete for {name}")

async def process_asset_chain(name, source_path, dest_folder, scale, semantic_label,
                              auto_detect_units=False, threshold=50.0,
                              normalize_size=False, target_size_min=0.05, target_size_max=1.5):
    """
    Process a single asset through native conversion and post-processing.

    Args:
        name (str): The name of the asset.
        source_path (str): The path to the source asset.
        dest_folder (str): The destination folder for the converted asset.
        scale (float): The scale factor for the asset.
        semantic_label (str): The semantic label for the asset.
        auto_detect_units (bool): Enable auto unit detection and scaling.
        threshold (float): Threshold for auto unit detection.
        normalize_size (bool): Enable target size normalization.
        target_size_min (float): Minimum target size in meters.
        target_size_max (float): Maximum target size in meters.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    usd_safe_name = name.replace(" ", "_").replace(".", "_")
    output_path = os.path.join(dest_folder, f"{usd_safe_name}.usd")

    # Native Conversion
    abs_source = os.path.abspath(str(source_path))
    abs_dest = os.path.abspath(output_path)

    success = await convert_asset_native(abs_source, abs_dest)
    if success:
        post_process_usd(abs_dest, usd_safe_name, scale, semantic_label, abs_source,
                        auto_detect_units, threshold, normalize_size, target_size_min, target_size_max)
    else:
        print(f"Skipping post-process for {name} due to conversion failure.")

def clean_label(raw_name, regex_pattern=None, regex_repl=""):
    """
    Clean a raw asset name into a semantic label.

    Args:
        raw_name (str): The raw filename/folder name.
        regex_pattern (str, optional): Regex pattern to match and remove/replace.
        regex_repl (str, optional): Replacement string for regex matches.

    Returns:
        str: Cleaned label suitable for semantic segmentation.
    """
    label = raw_name

    # Apply regex replacement if provided
    if regex_pattern:
        label = re.sub(regex_pattern, regex_repl, label)

    # Replace common separators with spaces
    label = label.replace("_", " ").replace("-", " ")

    # Collapse multiple spaces and strip
    label = " ".join(label.split())

    return label


async def run_conversion_async(config):
    """
    Run the conversion process for a list of assets.

    Args:
        config (dict): The configuration dictionary.
    """
    destination_root = config.get("destination_dir", "./converted_assets")
    assets_config = config.get("assets", [])

    if not assets_config:
        print("No assets defined.")
        return

    print(f"Starting Native Conversion. Output: {destination_root}")

    for group in assets_config:
        group_name = group.get("name", "generic")
        group_type = group.get("type", "directory")
        group_dest = os.path.join(destination_root, group_name)
        base_scale = group.get("scale", 1.0)
        auto_detect_units = group.get("auto_detect_units", False)
        threshold = group.get("auto_detection_threshold", 50.0)
        normalize_size = group.get("normalize_size", False)
        target_size_min = group.get("target_size_min", 0.05)
        target_size_max = group.get("target_size_max", 1.5)

        print(f"  Group '{group_name}' using scale: {base_scale}. Auto-Detect Units: {auto_detect_units}")
        if normalize_size:
            print(f"    Target Size Normalization: [{target_size_min}, {target_size_max}]m")

        # Supported mesh formats for conversion
        supported_extensions = [".obj", ".glb", ".gltf", ".fbx"]

        # Get regex options for label cleaning
        regex_pattern = group.get("regex_replace_pattern")
        regex_repl = group.get("regex_replace_repl", "")

        targets = []
        if group_type == "glob":
            search_pattern = group.get("search_pattern")
            name_depth = group.get("name_depth", 1)
            all_files = sorted(glob.glob(search_pattern, recursive=True))
            # Filter to only supported mesh formats
            files = [f for f in all_files if os.path.splitext(f)[1].lower() in supported_extensions]
            for file_path in files:
                path_obj = Path(file_path)
                try:
                    if name_depth < 1:
                        asset_name = path_obj.stem
                    else:
                        asset_name = path_obj.parents[name_depth - 1].name
                except:
                    asset_name = path_obj.stem

                targets.append({
                    "name": asset_name,
                    "path": file_path,
                    "label": clean_label(asset_name, regex_pattern, regex_repl)
                })
        elif group_type == "directory":
            src_path = group.get("path")
            mesh_rel_raw = group.get("mesh_relative_path", "")

            candidate_rels = []
            if isinstance(mesh_rel_raw, list):
                candidate_rels = mesh_rel_raw
            else:
                # Default hardcoded preference if not specified, or just use the single provided one
                if not mesh_rel_raw:
                    # If config doesn't specify, default to our YCB logic + common GLB patterns
                    candidate_rels = [
                        "google_512k/textured.obj",
                        "tsdf/textured.obj",
                        "model.glb",  # Common GLB naming conventions
                        "mesh.glb",
                        "scene.gltf",
                    ]
                else:
                    candidate_rels = [mesh_rel_raw]

            root = Path(src_path)
            if root.exists():
                for folder in sorted([f for f in root.iterdir() if f.is_dir()]):
                    # Check candidates in order
                    found_mesh = None
                    for rel in candidate_rels:
                        mesh_file = folder / rel
                        if mesh_file.exists():
                            found_mesh = mesh_file
                            break
                    
                    if found_mesh:
                        targets.append({
                            "name": folder.name,
                            "path": str(found_mesh.absolute()),
                            "label": clean_label(folder.name, regex_pattern, regex_repl)
                        })
                    else:
                         print(f"    Warning: No valid mesh found for {folder.name} in candidates: {candidate_rels}")
        elif group_type == "list":
            items = group.get("items", [])
            for item in items:
                item_name = item.get("name", "unknown")
                # Use explicit label if provided, otherwise clean the name
                item_label = item.get("label") or clean_label(item_name, regex_pattern, regex_repl)
                targets.append({
                    "name": item_name,
                    "path": item.get("path"),
                    "label": item_label,
                    "scale": item.get("scale", base_scale)  # Override scale
                })

        # Launch Tasks
        for t in targets:
            safe_name = t["name"].replace(" ", "_").replace(".", "_").replace("-", "_")
            if safe_name and safe_name[0].isdigit():
                safe_name = f"_{safe_name}"
            s = t.get("scale", base_scale)
            await process_asset_chain(safe_name, t["path"], group_dest, s, t["label"],
                                      auto_detect_units, threshold, normalize_size, target_size_min, target_size_max)
            
    print("All conversions completed.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert OBJ/GLB/GLTF/FBX meshes to USD with physics and semantics.",
        epilog="Example configs: config/obj_assets_config.yaml, config/glb_assets_config.yaml"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config/obj_assets_config.yaml")
    parser.add_argument("--config", type=str, default=default_config,
                        help="Path to config YAML (default: config/obj_assets_config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Run Async Loop within SimulationApp
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_conversion_async(config))
    
    simulation_app.close()

if __name__ == "__main__":
    main()