import os
import argparse
import yaml
import shutil
import glob
import asyncio
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
    
    print(f"  Native Conversion: {source_path} -> {output_path}")
    task = converter.create_converter_task(source_path, output_path, None, context)
    
    # Wait for the async task to complete
    success = await task.wait_until_finished()
    if not success:
        error_msg = task.get_error_message()
        print(f"  ERROR: Conversion failed for {source_path}: {error_msg}")
        return False
    return True

def texture_application(stage, root_prim_path, source_obj_path):
    """
    Manually binds a texture to a prim when standard MTL referencing fails.

    This function attempts to locate a `textured.png` file in the same directory as the source OBJ.
    If found, it explicitly creates a UsdPreviewSurface material graph:
    1.  **Sampler**: Creates a UsdUVTexture sampler referencing the image file.
    2.  **Shader**: Creates a UsdPreviewSurface shader.
    3.  **Connection**: Connects Sampler (RGB) -> Shader (DiffuseColor).
    4.  **Binding**: Binds the material to the root prim of the asset.

    Args:
        stage (Usd.Stage): The active USD stage.
        root_prim_path (str): The path to the root prim of the asset being converted.
        source_obj_path (str): The absolute filesystem path to the source OBJ file.
    """
    # Locate the texture file on disk
    source_dir = os.path.dirname(source_obj_path)
    # Common names for YCB textures. Add others if yours differ.
    texture_file_name = "textured.png"
    
    texture_path = os.path.join(source_dir, texture_file_name)
    if not os.path.exists(texture_path):
        # Silent return if no texture manual override needed/found
        return

    print(f"    [Hybrid] Found texture: {texture_path}. Applying explicit material.")

    # Create the Material Container
    mat_path = f"{root_prim_path}/Looks/VerifiedMaterial"
    material = UsdShade.Material.Define(stage, mat_path)

    # Create the Surface Shader (UsdPreviewSurface)
    shader_path = f"{mat_path}/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    
    # Create the inputs on the Surface Shader
    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse_input.Set(Gf.Vec3f(1.0, 1.0, 1.0)) # Fallback white

    # Create the Texture Sampler (UsdUVTexture)
    sampler_path = f"{mat_path}/DiffuseSampler"
    sampler = UsdShade.Shader.Define(stage, sampler_path)
    sampler.CreateIdAttr("UsdUVTexture")
    sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
    sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    # Create Primvar Reader (Essential for UV Mapping)
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

def post_process_usd(stage_path, name, scale, semantic_label, source_path):
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

    # Apply Hybrid Material Strategy
    texture_application(stage, root_prim_path, source_path)

    # Apply Scale
    # Find or create scale op
    xform = UsdGeom.Xformable(root_prim)
    scale_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if not scale_op:
        scale_op = xform.AddScaleOp()
    scale_op.Set((scale, scale, scale))

    # Apply Semantics
    label = semantic_label if semantic_label else name
    add_labels(root_prim, labels=[label], instance_name="class")

    # Apply Physics
    rigid_prim = RigidPrim(prim_path=str(root_prim_path), mass=0.1)
    rigid_prim.enable_rigid_body_physics()

    # Apply Collision
    # Note: Collision might be tricky on the root Xform if meshes are children.
    # RigidPrim handles mass on root. Collision might need to be on the mesh prims or convex hull on root.
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

async def process_asset_chain(name, source_path, dest_folder, scale, semantic_label):
    """
    Process a single asset through native conversion and post-processing.

    Args:
        name (str): The name of the asset.
        source_path (str): The path to the source asset.
        dest_folder (str): The destination folder for the converted asset.
        scale (float): The scale factor for the asset.
        semantic_label (str): The semantic label for the asset.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    usd_safe_name = name.replace(" ", "_").replace(".", "_")
    output_path = os.path.join(dest_folder, f"{usd_safe_name}.usd")
    
    # 1. Native Conversion
    abs_source = os.path.abspath(str(source_path))
    abs_dest = os.path.abspath(output_path)
    
    success = await convert_asset_native(abs_source, abs_dest)
    if success:
        post_process_usd(abs_dest, usd_safe_name, scale, semantic_label, abs_source)
    else:
        print(f"Skipping post-process for {name} due to conversion failure.")

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
        
        targets = []
        if group_type == "glob":
            search_pattern = group.get("search_pattern")
            name_depth = group.get("name_depth", 1)
            files = sorted(glob.glob(search_pattern, recursive=True))
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
                    "label": asset_name.replace("_", " ")
                })
        elif group_type == "directory":
            src_path = group.get("path")
            mesh_rel = group.get("mesh_relative_path", "")
            root = Path(src_path)
            if root.exists():
                for folder in sorted([f for f in root.iterdir() if f.is_dir()]):
                    mesh_file = folder / mesh_rel
                    if mesh_file.exists():
                        targets.append({
                            "name": folder.name, 
                            "path": str(mesh_file.absolute()), 
                            "label": folder.name
                        })
        elif group_type == "list":
             items = group.get("items", [])
             for item in items:
                 targets.append({
                     "name": item.get("name", "unknown"),
                     "path": item.get("path"),
                     "label": item.get("name"),
                     "scale": item.get("scale", base_scale) # Override scale
                 })

        # Launch Tasks
        for t in targets:
            safe_name = t["name"].replace(" ", "_").replace(".", "_").replace("-", "_")
            if safe_name and safe_name[0].isdigit():
                safe_name = f"_{safe_name}"
            s = t.get("scale", base_scale)
            await process_asset_chain(safe_name, t["path"], group_dest, s, t["label"])
            
    print("All conversions completed.")

def main():
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config/assets_config.yaml")
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Run Async Loop within SimulationApp
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_conversion_async(config))
    
    simulation_app.close()

if __name__ == "__main__":
    main()