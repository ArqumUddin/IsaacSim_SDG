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

    # 1. Parse ONLY the local MTL
    props = parse_mtl(source_mtl_path)
    
    # 2. Locate Texture from map_Kd
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
    
    # Native Conversion
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
            mesh_rel_raw = group.get("mesh_relative_path", "")
            
            candidate_rels = []
            if isinstance(mesh_rel_raw, list):
                candidate_rels = mesh_rel_raw
            else:
                # Default hardcoded preference if not specified, or just use the single provided one
                if not mesh_rel_raw:
                     # If config doesn't specify, default to our YCB logic
                     candidate_rels = ["google_512k/textured.obj", "tsdf/textured.obj"]
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
                            "label": folder.name
                        })
                    else:
                         print(f"    Warning: No valid mesh found for {folder.name} in candidates: {candidate_rels}")
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