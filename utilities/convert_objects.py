import os
import argparse
import yaml
import shutil
import glob
from pathlib import Path

# Note: Omniverse kits must be launched before importing omni modules
from isaacsim import SimulationApp

# Launch in Headless mode by default for batch processing
# You can change to headless=False if you want to see the process (slower)
simulation_app = SimulationApp({"headless": True})

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.prims import SingleRigidPrim as RigidPrim, SingleGeometryPrim as GeometryPrim
from isaacsim.core.api.materials import PhysicsMaterial
from pxr import UsdGeom, UsdPhysics, Gf

def load_config(config_path):
    if not os.path.exists(config_path):
        # Fallback to local dir
        script_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(script_dir, config_path)
        if os.path.exists(relative_path):
            config_path = relative_path
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_asset(name, source_path, dest_folder, scale, semantic_label=None):
    """
    Converts a single asset to USD with physics and semantics.
    1. Creates a new stage.
    2. References the source mesh.
    3. Adds Physics & Semantics.
    4. Saves to dest_folder/name.usd.
    """
    usd_filename = f"{name}.usd"
    output_path = os.path.join(dest_folder, usd_filename)
    
    # Ensure destination directory exists
    os.makedirs(dest_folder, exist_ok=True)
    
    print(f"  Processing: {name} -> {output_path}")

    # Create a fresh stage
    stage_utils.create_new_stage()
    
    # Define a root Xform for the object
    # It's good practice to have a single root prim e.g. /Object
    root_prim_path = f"/{name}"
    
    # We reference the original OBJ/USD. 
    # Note: If it's an OBJ, Isaac Sim handles the reading via the reference.
    prim_utils.create_prim(root_prim_path, "Xform")
    stage_utils.add_reference_to_stage(usd_path=str(source_path), prim_path=root_prim_path)

    # We apply scale to the root prim so everything scales together
    # Robustly handle scaling: Check if op exists, else add it.
    prim = prim_utils.get_prim_at_path(root_prim_path)
    xform = UsdGeom.Xformable(prim)
    
    scale_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
            
    if not scale_op:
        scale_op = xform.AddScaleOp()
        
    scale_op.Set((scale, scale, scale))
    
    # Semantics
    label = semantic_label if semantic_label else name
    prim = prim_utils.get_prim_at_path(root_prim_path)
    add_labels(prim, labels=[label], instance_name="class")
    
    # Physics
    # Add Rigid Body API
    rigid_prim = RigidPrim(prim_path=root_prim_path, mass=0.1)
    rigid_prim.enable_rigid_body_physics()
    
    # Add Collision Approximation (Convex Hull is standard for generic objects)
    # We need to apply this to the geometry. 
    # If the referenced file is a mesh, we might need to find the mesh prim inside.
    # However, applying collision to the Xform (RigidBody) and letting Isaac resolve headers often works,
    # but strictly, collisions belong on meshes or proxies.
    # For robust "auto-collision", we use GeometryPrim to apply collision to the prim or its children.
    
    # Find the first Mesh prim under the root, or treat the root as the collision holder if it was a mesh
    # Since we added a reference to an Xform, the Mesh is a child.
    # We'll use GeometryPrim which can recurse or we just set collision on the Root which implies "Compound".
    # Just enabling collision on the RigidPrim XFORM usually implies a convex hull or bounding box if not specified, 
    # but usually you want explicit collision API on the mesh.
    
    # Simple approach for utility: Enable Convex Hull collision on the Root
    # (Isaac Core's GeometryPrim can automagically help here)
    try:
        geometry_prim = GeometryPrim(prim_path=root_prim_path, collision=True)
        geometry_prim.set_collision_approximation("convexHull")
    except Exception as e:
        print(f"    Warning: Could not auto-configure collision: {e}")

    stage_utils.save_stage(output_path)
    print(f"    Saved.")

def run_conversion(config):
    destination_root = config.get("destination_dir", "./converted_assets")
    assets_config = config.get("assets", [])
    
    if not assets_config:
        print("No assets defined in config.")
        return

    print(f"Starting Conversion. Output Dir: {destination_root}")
    
    for group in assets_config:
        group_name = group.get("name", "generic")
        group_type = group.get("type", "directory")
        group_dest = os.path.join(destination_root, group_name)
        
        print(f"\nGroup: {group_name}")
        
        # Pull parameters
        base_scale = group.get("scale", 1.0)
        
        if group_type == "glob":
            search_pattern = group.get("search_pattern")
            name_depth = group.get("name_depth", 1)
            
            if not search_pattern:
                print(f"  Error: 'search_pattern' required for glob type.")
                continue

            print(f"  Searching pattern: {search_pattern}")
            files = sorted(glob.glob(search_pattern, recursive=True))
            print(f"  Found {len(files)} files.")
            
            for file_path in files:
                path_obj = Path(file_path)
                # Extract name based on depth
                try:
                    if name_depth < 1:
                        asset_name = path_obj.stem
                    else:
                        asset_name = path_obj.parents[name_depth - 1].name
                except IndexError:
                    print(f"  Warning: Path too shallow for depth {name_depth}: {file_path}")
                    asset_name = path_obj.stem

                safe_name = asset_name.replace(" ", "_").replace(".", "_").replace("-", "_")
                # Ensure the name starts with a valid character for USD (cannot start with a digit)
                if safe_name and safe_name[0].isdigit():
                    safe_name = f"_{safe_name}"
                process_asset(
                    name=safe_name,
                    source_path=file_path,
                    dest_folder=group_dest,
                    scale=base_scale,
                    semantic_label=safe_name.replace("_", " ")
                )
        elif group_type == "directory":
            src_path = group.get("path")
            mesh_rel = group.get("mesh_relative_path", "")
            
            if not src_path or not os.path.exists(src_path):
                print(f"  Skipping invalid path: {src_path}")
                continue

            root = Path(src_path)
            # Scan folders
            for folder in sorted([f for f in root.iterdir() if f.is_dir()]):
                mesh_file = folder / mesh_rel
                if mesh_file.exists():
                    safe_name = folder.name.replace(" ", "_")
                    process_asset(
                        name=safe_name,
                        source_path=str(mesh_file.absolute()),
                        dest_folder=group_dest,
                        scale=base_scale,
                        semantic_label=folder.name
                    )     
        elif group_type == "list":
            items = group.get("items", [])
            for item in items:
                name = item.get("name", "unknown")
                path = item.get("path")
                item_scale = item.get("scale", base_scale)
                
                if path and os.path.exists(path):
                    safe_name = name.replace(" ", "_")
                    process_asset(
                        name=safe_name,
                        source_path=path,
                        dest_folder=group_dest,
                        scale=item_scale,
                        semantic_label=name
                    )

def main():
    parser = argparse.ArgumentParser(description="Convert assets to Isaac Sim USD format properly.")
    # Default to loading config from the same dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config/assets_config.yaml")
    
    parser.add_argument("--config", type=str, default=default_config, help="Path to assets_config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_conversion(config)
        print("\nAll Done. Simulation App closing.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()