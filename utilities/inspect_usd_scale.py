from isaacsim import SimulationApp

# Launch in Headless mode
simulation_app = SimulationApp({"headless": True})

import omni.usd
import isaacsim.core.utils.stage as stage_utils
from pxr import Usd, UsdGeom
import glob
import os
import argparse

def inspect_usd(usd_path):
    print(f"\nScanning: {os.path.basename(usd_path)}")
    stage_utils.open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()
    
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        print("  ERROR: No default prim found.")
        return

    print(f"  Root Prim: {root_prim.GetPath()}")
    
    # Check Scale Op
    xform = UsdGeom.Xformable(root_prim)
    scale_op = None
    scale_val = (1.0, 1.0, 1.0)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_val = op.Get()
            scale_op = op
            break
            
    print(f"  Applied Scale Op: {scale_val}")
    
    # Check Dimensions
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = bbox_cache.ComputeWorldBound(root_prim)
    range_box = bound.GetRange()
    size = range_box.GetSize()
    
    print(f"  World Size (with scale): {size}")
    print(f"  Max Dimension: {max(size[0], size[1], size[2]):.3f}")
    
    # Check MetersPerUnit
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"  Stage MetersPerUnit: {mpu}")

def main():
    parser = argparse.ArgumentParser(description="Inspect USD scale and dimensions.")
    parser.add_argument("--dir", type=str, default="/home/ubuntu/Workspace/data/glb_converted/my_glb_models", help="Directory to scan")
    parser.add_argument("--limit", type=int, default=10, help="Max files to scan")
    args = parser.parse_args()
    
    files = sorted(glob.glob(os.path.join(args.dir, "*.usd")))
    if not files:
        print(f"No USD files found in {args.dir}")
        return

    print(f"Found {len(files)} files. Scanning first {args.limit}...")
    
    for f in files[:args.limit]:
        inspect_usd(f)
        
    simulation_app.close()

if __name__ == "__main__":
    main()
