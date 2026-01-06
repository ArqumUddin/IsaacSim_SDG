# Note: Omniverse kits must be launched before importing omni modules
from isaacsim import SimulationApp

# Launch in Headless mode with Asset Converter extension enabled
simulation_app = SimulationApp({
    "headless": True
})

import os
import argparse
from pxr import Usd, UsdShade, UsdGeom, Sdf

def inspect_material(usd_path):
    """
    Inspects the material graph of a USD file.

    Args:
        usd_path (str): The path to the USD file.
    """
    if not os.path.exists(usd_path):
        print(f"Error: File not found: {usd_path}")
        return

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"Error: Could not open stage: {usd_path}")
        return

    print(f"Inspecting: {usd_path}")
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        print("  Warning: No default prim found. Using root pseudo-children[0].")
        children = [p for p in stage.GetPseudoRoot().GetChildren()]
        if children:
            root_prim = children[0]
        else:
            print("  Error: Empty stage?")
            return

    print(f"  Root Prim: {root_prim.GetPath()}")
    
    # Check Binding
    binding_api = UsdShade.MaterialBindingAPI(root_prim)
    direct_binding = binding_api.GetDirectBinding()
    material = direct_binding.GetMaterial()
    
    if material:
        print(f"  Root Material Binding: {material.GetPath()}")
    else:
        print("  Root Material Binding: None")

    # Walk Hierarchy to find Materials and Shaders
    print("\n  --- Material Graph Dump ---")
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            print(f"  Material: {prim.GetPath()}")
        elif prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            id_attr = shader.GetIdAttr().Get()
            print(f"    Shader: {prim.GetPath()} (ID: {id_attr})")
            
            # Check inputs
            for input_ in shader.GetInputs():
                val = input_.Get()
                print(f"      Input: {input_.GetBaseName()} = {val} ({input_.GetTypeName()})")
                if input_.HasConnectedSource():
                    source, source_name, _ = input_.GetConnectedSource()
                    print(f"        -> Connected to: {source.GetPath()}.{source_name}")

    print("\n  --- Mesh Primvars Dump ---")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            print(f"  Mesh: {prim.GetPath()}")
            primvars_api = UsdGeom.PrimvarsAPI(prim)
            st = primvars_api.GetPrimvar("st")
            if st.HasValue():
                print(f"    Has 'st' UVs: Yes ({len(st.Get())} points)")
            else:
                tex_coord = primvars_api.GetPrimvar("texCoord")
                if tex_coord.HasValue():
                     print(f"    Has 'texCoord' UVs: Yes ({len(tex_coord.Get())} points)")
                else:
                    print("    Has 'st' UVs: NO (Texture will not map!)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect USD Material Graph")
    parser.add_argument("--usd_path", type=str, required=True, help="Path to USD file")
    args = parser.parse_args()

    inspect_material(args.usd_path)
