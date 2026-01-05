# IsaacSim_SDG: Synthetic Data Generation Toolkit

A comprehensive synthetic data generation (SDG) toolkit built on NVIDIA Isaac Sim and Replicator. This project provides production-ready pipelines for generating photorealistic synthetic training data for computer vision and robotics applications, including object detection, instance segmentation, depth estimation, and pose estimation.

## Overview

IsaacSim_SDG offers two complementary approaches to synthetic data generation:

1. **Object-Based SDG**: Generates data by spawning objects in simple/empty environments with heavy physics-based randomization (objects floating, falling, and bouncing)
2. **Scene-Based SDG**: Places objects naturally within complex, photorealistic environments (e.g., Infinigen-generated dining rooms and kitchens)

Both approaches leverage Isaac Sim's advanced rendering capabilities (ray tracing, path tracing, DLSS) and physics simulation to create realistic, diverse datasets with automatic annotations.

## Features

- **Physics-Based Realism**: Rigid body dynamics, gravity, collisions, settling simulations
- **Advanced Rendering**: Ray-traced lighting, path tracing, motion blur, HDR environments
- **Flexible Randomization**: Camera poses, lighting, object properties, materials, and backgrounds
- **Automatic Annotations**: COCO format, bounding boxes, semantic segmentation, depth maps
- **Data Augmentation**: GPU-accelerated augmentations using Warp (Gaussian noise, color transforms)
- **Asset Pipeline**: Tools to convert custom OBJ/FBX models to Isaac Sim USD format with physics and semantics
- **Auto-Labeling**: Regex-based automatic semantic labeling from file/folder names

## Project Structure

```
IsaacSim_SDG/
├── augmentation/              # Data augmentation examples
│   ├── writer_augmentation.py        # Augmentation at writer level
│   └── annotator_augmentation.py     # Augmentation at annotator level
├── object_based_sdg/          # Object-based SDG pipeline
│   ├── object_based_sdg.py           # Main entry point
│   ├── object_based_sdg_utils.py     # Helper utilities
│   └── config/
│       └── object_based_sdg_config.yaml
├── scene_based_sdg/           # Scene-based SDG pipeline
│   ├── scene_based_sdg.py            # Main entry point
│   ├── scene_based_sdg_utils.py      # Helper utilities
│   └── config/
│       └── scene_multi_writers_pt.yaml
└── utilities/                 # Asset conversion utilities
    ├── convert_objects.py            # OBJ to USD converter
    └── config/
        └── assets_config.yaml
```

## Prerequisites

### Required Software

- **NVIDIA Isaac Sim** (2024.1 or later)
  - Download from: https://developer.nvidia.com/isaac-sim
  - Includes Omni Replicator for SDG
- **NVIDIA GPU** with RTX support (recommended: RTX 3080 or better)
  - Required for ray tracing and path tracing
- **Ubuntu 20.04/22.04** or Windows 10/11
- **Python 3.10+** (bundled with Isaac Sim)

### Python Dependencies

The following dependencies are included with Isaac Sim:

- `isaacsim` - Isaac Sim core library
- `omni.replicator.core` - Replicator for SDG
- `warp` - GPU-accelerated computing
- `numpy` - Numerical computing
- `PIL/Pillow` - Image processing
- `PyYAML` - YAML configuration parsing
- `pxr` (USD) - Universal Scene Description

## Installation

1. **Install Isaac Sim**

   ```bash
   # Download and install from NVIDIA's website
   # https://developer.nvidia.com/isaac-sim
   ```

2. **Clone this repository**

   ```bash
   # Make sure the repository is cloned in the same directory as Isaac Sim
   git clone git@github.com:ArqumUddin/IsaacSim_SDG.git
   cd IsaacSim_SDG
   ```

3. **Set up Isaac Sim Python environment**

   ```bash
   # Locate your Isaac Sim installation (example path)
   export ISAAC_SIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.0.0"

   # Use Isaac Sim's Python interpreter
   alias omni_python="${ISAAC_SIM_PATH}/python.sh"
   ```

4. **Verify installation**
   ```bash
   omni_python -c "from isaacsim import SimulationApp; print('Isaac Sim ready!')"
   ```

## Quick Start

### 1. Object-Based SDG

Generate synthetic data with objects floating/falling in a simple environment:

```bash
cd object_based_sdg
omni_python object_based_sdg.py --config config/object_based_sdg_config.yaml
```

**What it does:**

- Spawns 5 labeled YCB objects + 350 shape distractors + 75 mesh distractors
- Randomizes object poses, rotations, velocities
- Places 2 cameras at random positions around objects
- Generates 10 frames with COCO annotations
- Output: `_out_coco_data/` (RGB images, semantic masks, COCO JSON)

### 2. Scene-Based SDG

Generate data with objects placed in realistic Infinigen environments:

```bash
cd scene_based_sdg
omni_python scene_based_sdg.py --config config/scene_multi_writers_pt.yaml
```

**What it does:**

- Loads Infinigen dining room environments
- Spawns labeled objects on tables/surfaces
- Captures "floating" frames (objects in air) and "dropped" frames (after physics simulation)
- Uses path tracing for photorealism
- Dual writers: BasicWriter (segmentation) + DataVisualizationWriter (bounding boxes)
- Output: `_out_infinigen_basicwriter_pt/` and `_out_infinigen_dataviswriter_pt/`

### 3. Convert Custom Assets

Convert your OBJ files to Isaac Sim USD format with physics and semantics:

```bash
cd utilities

# Edit config/assets_config.yaml to point to your OBJ files
omni_python convert_objects.py --config config/assets_config.yaml
```

### 4. Data Augmentation Examples

```bash
cd augmentation

# Writer-level augmentation (applies to all data)
omni_python writer_augmentation.py --num_frames 25 --use_warp

# Annotator-level augmentation (per-annotator control)
omni_python annotator_augmentation.py --num_frames 25 --use_warp
```

## Configuration

All pipelines use YAML configuration files. Key parameters:

### Object-Based Config (`object_based_sdg/config/object_based_sdg_config.yaml`)

```yaml
launch_config:
  renderer: RaytracedLighting # or PathTracing
  headless: false # Set true for servers

num_frames: 10 # Number of captures
num_cameras: 2 # Cameras per capture
resolution: [640, 480] # Image resolution

working_area_size: [4, 4, 3] # XYZ bounds (meters)

labeled_assets:
  auto_label:
    num: 5 # Number of labeled objects
    folders:
      - /path/to/ycb/objects # Folder to scan
    regex_replace_pattern: "^\\d+[-_]*" # Auto-labeling regex

shape_distractors_num: 350 # Background shapes
mesh_distractors_num: 75 # Background meshes

writer_type: CocoWriter
writer_kwargs:
  output_dir: "_out_coco_data"
  rgb: true
  semantic_segmentation: true
  bounding_box_2d_tight: true
```

### Scene-Based Config (`scene_based_sdg/config/scene_multi_writers_pt.yaml`)

```yaml
environments:
  folders:
    - /Isaac/Samples/Replicator/Infinigen/dining_rooms/

capture:
  total_captures: 12
  num_floating_captures_per_env: 2 # Captures before physics
  num_dropped_captures_per_env: 3 # Captures after physics
  num_cameras: 2
  resolution: [640, 480]
  path_tracing: true # High-quality rendering
  rt_subframes: 8 # Reduce temporal artifacts

writers:
  - type: BasicWriter
    kwargs:
      output_dir: "_out_infinigen_basicwriter_pt"
      semantic_segmentation: true
  - type: DataVisualizationWriter
    kwargs:
      output_dir: "_out_infinigen_dataviswriter_pt"
      bounding_box_2d_tight: true
      bounding_box_3d: true

labeled_assets:
  auto_label:
    num: 5
    folders:
      - /Isaac/Props/YCB/Axis_Aligned/
```

### Asset Conversion Config (`utilities/config/assets_config.yaml`)

```yaml
destination_dir: "/path/to/output"

assets:
  - name: "ycb_objects"
    type: "glob"
    search_pattern: "/path/to/ycb/**/textured.obj"
    name_depth: 2 # Folders up for object name
    scale: 0.001 # OBJ to USD scale

  - name: "custom"
    type: "list"
    items:
      - name: "my_cube"
        path: "/path/to/cube.obj"
        scale: 1.0
```

## Output Data

### Object-Based SDG (CocoWriter)

```
_out_coco_data/
├── coco_annotations.json      # COCO format annotations
├── rgb_0000.png              # RGB images
├── rgb_0001.png
├── semantic_segmentation_0000.png  # Segmentation masks
└── semantic_segmentation_0001.png
```

### Scene-Based SDG (BasicWriter + DataVisualizationWriter)

```
_out_infinigen_basicwriter_pt/
├── rgb/
│   ├── rgb_0000.png
│   └── ...
└── semantic_segmentation/
    ├── semantic_segmentation_0000.png
    └── ...

_out_infinigen_dataviswriter_pt/
├── rgb_annotated_0000.png    # RGB with bounding boxes overlaid
└── normals_annotated_0000.png  # Normals with 3D boxes
```

## Advanced Usage

### Custom Randomization

Modify `object_based_sdg.py` or `scene_based_sdg.py` to add custom randomizers:

```python
# In your rep.trigger.on_frame() block:
with rep.trigger.on_frame():
    # Randomize object materials
    with labeled_objects:
        rep.randomizer.materials(materials_list)

    # Randomize camera parameters
    with cameras:
        rep.modify.attribute("focalLength", rep.distribution.uniform(18, 85))
```

### GPU-Accelerated Augmentations

Use Warp kernels for fast augmentations:

```python
import warp as wp

@wp.kernel
def custom_augmentation(data_in: wp.array3d(dtype=wp.uint8),
                        data_out: wp.array3d(dtype=wp.uint8)):
    i, j = wp.tid()
    # Your augmentation logic
    data_out[i, j, 0] = data_in[i, j, 0] * 0.8  # Example

# Register augmentation
rep.AnnotatorRegistry.register_augmentation(
    "my_augmentation",
    rep.annotators.Augmentation.from_function(custom_augmentation)
)
```

### Headless Rendering (for servers)

```yaml
# In your config YAML:
launch_config:
  headless: true
  renderer: RaytracedLighting
```

```bash
# Run without display
DISPLAY= omni_python object_based_sdg.py --config config.yaml
```

### Multiple Writers

```yaml
# In scene-based config:
writers:
  - type: BasicWriter
    kwargs: { ... }
  - type: CocoWriter
    kwargs: { ... }
  - type: DataVisualizationWriter
    kwargs: { ... }
```

## Performance Tips

1. **Use Warp for augmentations**: 10-50x faster than NumPy on GPU
2. **Disable render products between captures**: Set `disable_render_products: true`
3. **Reduce rt_subframes for faster iteration**: Trade quality for speed
4. **Use RaytracedLighting instead of PathTracing**: Faster, still high quality
5. **Batch processing**: Generate multiple datasets in parallel on different GPUs

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'isaacsim'`

- **Solution**: Use Isaac Sim's Python interpreter (`omni_python` or `${ISAAC_SIM_PATH}/python.sh`)

**Issue**: Config file not found

- **Solution**: Use absolute paths or ensure you're in the correct directory

**Issue**: Assets not found at `/Isaac/Props/YCB/...`

- **Solution**: Download YCB dataset or modify config to point to your assets
- YCB download: http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/

**Issue**: Out of GPU memory

- **Solution**: Reduce `resolution`, `num_cameras`, or number of distractors

**Issue**: Rendering artifacts or ghosting

- **Solution**: Increase `rt_subframes` (e.g., 8, 16, 32)

**Issue**: Black/dark images

- **Solution**: Check light configuration, increase `intensity_min_max`

## Examples and Use Cases

### Use Case 1: Training Object Detectors

1. Convert your CAD models using `utilities/convert_objects.py`
2. Configure `object_based_sdg_config.yaml` with your objects
3. Generate 10,000+ images with COCO annotations
4. Train YOLOv8/Faster R-CNN on synthetic data

### Use Case 2: Domain Randomization for Sim2Real

1. Use scene-based SDG with diverse Infinigen environments
2. Heavy randomization of lighting, materials, camera poses
3. Add distractor objects for robustness
4. Train policies that transfer to real robots

### Use Case 3: Data Augmentation

1. Generate base dataset with object-based SDG
2. Apply Warp-based augmentations (noise, blur, color jitter)
3. Mix with real-world data for hybrid training

## Contributing

This is an official NVIDIA project. Please report issues or suggestions through the appropriate channels.

## License

```
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
```

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## References

- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [Omni Replicator Documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html)
- [YCB Dataset](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/)
- [Infinigen: Infinite Photorealistic Worlds](https://infinigen.org/)
- [NVIDIA Warp](https://nvidia.github.io/warp/)

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{isaacim_sdg,
  title = {IsaacSim_SDG: Synthetic Data Generation Toolkit},
  author = {NVIDIA Corporation},
  year = {2025},
  url = {https://github.com/nvidia/IsaacSim_SDG}
}
```

## Support

For issues and questions:

- Isaac Sim: https://forums.developer.nvidia.com/c/omniverse/simulation/69
- Documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/
