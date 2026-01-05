# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Writer-level augmentation example for synthetic data generation.

This module demonstrates how to apply augmentations at the writer level,
which affects all data before it's written to disk. It showcases:
- Gaussian noise on RGB data (NumPy and Warp implementations)
- Gaussian noise on depth data with configurable sigma
- Composing multiple augmentations (RGB->HSV->Noise->HSV->RGB)
- Performance benchmarking of CPU vs GPU augmentations
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import carb.settings
import numpy as np
import warp as wp
import yaml

# Isaac Sim imports must come after SimulationApp initialization
from isaacsim import SimulationApp

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"[INFO] Loaded configuration from: {config_path}")
    return config

def get_default_config_path() -> str:
    """Get the default configuration file path."""
    script_dir = Path(__file__).parent
    return str(script_dir / "config" / "writer_augmentation_config.yaml")

def gaussian_noise_rgb_np(data_in: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """
    Add Gaussian noise to RGBA data using NumPy (CPU).

    Applies independent Gaussian noise to each RGB channel while preserving alpha.

    Args:
        data_in: Input RGBA image array of shape (H, W, 4)
        sigma: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        RGBA array with added Gaussian noise, clipped to [0, 255]
    """
    np.random.seed(seed)
    data_in = data_in.astype(np.float32)

    # Apply noise to each RGB channel independently
    for channel in range(3):
        data_in[:, :, channel] += np.random.randn(*data_in.shape[:-1]) * sigma

    # Clip to valid range and convert back to uint8
    return np.clip(data_in, 0, 255).astype(np.uint8)

def gaussian_noise_depth_np(data_in: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """
    Add Gaussian noise to depth data using NumPy (CPU).

    Args:
        data_in: Input depth array
        sigma: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Depth array with added Gaussian noise, clipped to non-negative values
    """
    np.random.seed(seed)
    result = data_in.astype(np.float32) + np.random.randn(*data_in.shape) * sigma
    return np.clip(result, 0, None).astype(data_in.dtype)

@wp.kernel
def gaussian_noise_rgb_wp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    sigma: float,
    seed: int,
):
    """
    Add Gaussian noise to RGBA data using Warp (GPU).

    Applies independent Gaussian noise to each RGB channel with unique random seeds
    to ensure independent noise patterns across channels.

    Args:
        data_in: Input RGBA image array
        data_out: Output RGBA image array with added noise
        sigma: Standard deviation of the Gaussian noise
        seed: Base random seed for reproducibility
    """
    # Get thread coordinates and image dimensions
    i, j = wp.tid()
    dim_i = data_in.shape[0]
    dim_j = data_in.shape[1]
    pixel_id = i * dim_i + j

    # Use pixel_id as offset to create unique seeds for each pixel and channel
    # This ensures independent noise patterns across R, G, B channels
    state_r = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
    state_g = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
    state_b = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))

    # Apply noise to each channel independently
    data_out[i, j, 0] = wp.uint8(wp.int32(data_in[i, j, 0]) + wp.int32(sigma * wp.randn(state_r)))
    data_out[i, j, 1] = wp.uint8(wp.int32(data_in[i, j, 1]) + wp.int32(sigma * wp.randn(state_g)))
    data_out[i, j, 2] = wp.uint8(wp.int32(data_in[i, j, 2]) + wp.int32(sigma * wp.randn(state_b)))
    data_out[i, j, 3] = data_in[i, j, 3]  # Preserve alpha channel

@wp.kernel
def gaussian_noise_depth_wp(
    data_in: wp.array2d(dtype=wp.float32),
    data_out: wp.array2d(dtype=wp.float32),
    sigma: float,
    seed: int,
):
    """
    Add Gaussian noise to depth data using Warp (GPU).

    Args:
        data_in: Input depth array
        data_out: Output depth array with added noise
        sigma: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility
    """
    i, j = wp.tid()
    # Generate unique ID for random seed per pixel
    scalar_pixel_id = i * data_in.shape[1] + j
    state = wp.rand_init(seed, scalar_pixel_id)
    data_out[i, j] = data_in[i, j] + sigma * wp.randn(state)

def configure_settings(config: Dict) -> None:
    """
    Configure Isaac Sim settings for optimal SDG performance.

    Args:
        config: Configuration dictionary
    """
    settings = carb.settings.get_settings()

    # Enable script nodes
    settings.set_bool("/app/omni.graph.scriptnode/opt_in", True)

    # Set DLSS mode
    dlss_mode = config["rendering"]["dlss_mode"]
    settings.set("rtx/post/dlss/execMode", dlss_mode)

    # Disable capture on play and async rendering for deterministic behavior
    settings.set("/omni/replicator/captureOnPlay", False)
    settings.set("/omni/replicator/asyncRendering", False)
    settings.set("/app/asyncRendering", False)

def setup_scene(app: SimulationApp, config: Dict):
    """
    Set up the simulation scene with environment, objects, and camera.

    Args:
        app: The SimulationApp instance
        config: Configuration dictionary

    Returns:
        Tuple of (red_cube, render_product) for later randomization
    """
    import omni.replicator.core as rep
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.storage.native import get_assets_root_path

    scene_config = config["scene"]
    rendering_config = config["rendering"]
    data_config = config["data_generation"]

    # Load environment
    assets_root_path = get_assets_root_path()
    open_stage(assets_root_path + scene_config["env_url"])

    # Create scene objects
    cube_color = tuple(scene_config["cube"]["color"])
    cube_position = tuple(scene_config["cube"]["position"])
    red_mat = rep.create.material_omnipbr(diffuse=cube_color)
    red_cube = rep.create.cube(position=cube_position, material=red_mat)

    cam_position = tuple(scene_config["camera"]["position"])
    cam_lookat = tuple(scene_config["camera"]["look_at"])
    cam = rep.create.camera(position=cam_position, look_at=cam_lookat)

    resolution = tuple(rendering_config["resolution"])
    rp = rep.create.render_product(cam, resolution)

    # Update app to fully load textures/materials
    for _ in range(data_config["material_load_updates"]):
        app.update()

    return red_cube, rp

def register_augmentations(config: Dict) -> None:
    """
    Register augmentation functions with Replicator registry.

    This makes augmentations available for use with writers and annotators.

    Args:
        config: Configuration dictionary
    """
    import omni.replicator.core as rep

    augmentation_config = config["augmentation"]
    use_warp = augmentation_config["backend"] == "warp"
    depth_sigma = augmentation_config["gaussian_noise"]["depth_sigma"]

    if use_warp:
        # Register Warp-based depth augmentation
        rep.AnnotatorRegistry.register_augmentation(
            "gn_depth_wp",
            rep.annotators.Augmentation.from_function(
                gaussian_noise_depth_wp,
                sigma=depth_sigma,
                seed=None,
            ),
        )
    else:
        # Register NumPy-based depth augmentation
        rep.AnnotatorRegistry.register_augmentation(
            "gn_depth_np",
            rep.annotators.Augmentation.from_function(
                gaussian_noise_depth_np,
                sigma=depth_sigma,
                seed=None,
            ),
        )

def create_augmentations(config: Dict):
    """
    Create augmentation objects for RGB and depth data.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (rgb_to_hsv_augm, hsv_to_rgb_augm, gn_rgb_augm, gn_depth_augm)
    """
    import omni.replicator.core as rep

    augmentation_config = config["augmentation"]
    use_warp = augmentation_config["backend"] == "warp"
    rgb_sigma = augmentation_config["gaussian_noise"]["rgb_sigma"]

    # Get built-in color space conversion augmentations
    rgb_to_hsv_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_rgb_to_hsv)
    hsv_to_rgb_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_hsv_to_rgb)

    # Create custom augmentations based on backend
    if use_warp:
        gn_rgb_augm = rep.annotators.Augmentation.from_function(
            gaussian_noise_rgb_wp,
            sigma=rgb_sigma,
            seed=None,
        )
        gn_depth_augm = rep.AnnotatorRegistry.get_augmentation("gn_depth_wp")
    else:
        gn_rgb_augm = rep.annotators.Augmentation.from_function(
            gaussian_noise_rgb_np,
            sigma=rgb_sigma,
            seed=None,
        )
        gn_depth_augm = rep.AnnotatorRegistry.get_augmentation("gn_depth_np")

    return rgb_to_hsv_augm, hsv_to_rgb_augm, gn_rgb_augm, gn_depth_augm

def setup_writer(output_dir: str, render_product, augmentations: Tuple):
    """
    Create and configure a BasicWriter with augmented annotators.

    Args:
        output_dir: Directory to save output data
        render_product: The render product to attach
        augmentations: Tuple of (rgb_to_hsv, hsv_to_rgb, gn_rgb, gn_depth)

    Returns:
        Configured writer instance
    """
    import omni.replicator.core as rep

    rgb_to_hsv_augm, hsv_to_rgb_augm, gn_rgb_augm, gn_depth_augm = augmentations

    # Create writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=output_dir, rgb=True, distance_to_camera=True)

    # Create composed RGB augmentation: RGB -> HSV -> Add Noise -> HSV -> RGB
    # This demonstrates chaining multiple augmentations together
    augmented_rgb_annot = rep.annotators.get("rgb").augment_compose(
        [rgb_to_hsv_augm, gn_rgb_augm, hsv_to_rgb_augm],
        name="rgb",
    )
    writer.add_annotator(augmented_rgb_annot)

    # Augment depth annotator
    writer.augment_annotator("distance_to_camera", gn_depth_augm)

    # Attach render product to writer
    writer.attach([render_product])

    return writer

def setup_randomization(red_cube):
    """
    Configure randomization graph for the scene.

    Args:
        red_cube: The cube object to randomize
    """
    import omni.replicator.core as rep

    with rep.trigger.on_frame():
        with red_cube:
            rep.randomizer.rotation()

def generate_data(num_frames: int, config: Dict) -> float:
    """
    Generate synthetic data with augmentations applied by the writer.

    Args:
        num_frames: Number of frames to capture
        config: Configuration dictionary

    Returns:
        Total time taken for data generation in seconds
    """
    import omni.replicator.core as rep

    rt_subframes = config["rendering"]["rt_subframes"]

    # Start data generation
    rep.orchestrator.preview()

    # Capture frames
    start_time = time.time()
    for _ in range(num_frames):
        rep.orchestrator.step(rt_subframes=rt_subframes)

    return time.time() - start_time

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate augmented synthetic data with writer-level augmentations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=get_default_config_path(),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to capture (overrides config)",
    )
    parser.add_argument(
        "--use_warp",
        action="store_true",
        help="Use GPU-accelerated Warp augmentations (overrides config)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated data (overrides config)",
    )
    return parser.parse_args()

def main() -> None:
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    if args.headless:
        config["simulation"]["headless"] = True
    if args.use_warp:
        config["augmentation"]["backend"] = "warp"
    if args.num_frames is not None:
        config["data_generation"]["num_frames"] = args.num_frames
    if args.output_dir is not None:
        config["data_generation"]["output_dir"] = args.output_dir

    # Initialize Isaac Sim
    headless = config["simulation"]["headless"]
    simulation_app = SimulationApp(launch_config={"headless": headless})

    try:
        # Configure settings
        configure_settings(config)

        # Setup scene
        print("[INFO] Setting up scene...")
        red_cube, render_product = setup_scene(simulation_app, config)

        # Register augmentations in the registry
        backend = config["augmentation"]["backend"]
        print(f"[INFO] Registering augmentations (backend: {backend.upper()})...")
        register_augmentations(config)

        # Create augmentation objects
        print("[INFO] Creating augmentation pipeline...")
        augmentations = create_augmentations(config)

        # Setup writer with augmentations
        output_dir = os.path.join(os.getcwd(), config["data_generation"]["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Setting up writer with augmented annotators...")
        print(f"[INFO] Output directory: {output_dir}")
        writer = setup_writer(output_dir, render_product, augmentations)

        # Setup randomization
        print("[INFO] Configuring randomization...")
        setup_randomization(red_cube)

        # Generate data
        num_frames = config["data_generation"]["num_frames"]
        print(f"[INFO] Generating {num_frames} frames...")
        duration = generate_data(num_frames, config)

        # Print results
        avg_time = duration / num_frames
        backend_display = "Warp (GPU)" if backend == "warp" else "NumPy (CPU)"
        print(f"[RESULTS] Data generation complete!")
        print(f"  Backend:       {backend_display}")
        print(f"  Total frames:  {num_frames}")
        print(f"  Total time:    {duration:.4f} seconds")
        print(f"  Average time:  {avg_time:.4f} seconds per frame")
        print(f"  Output dir:    {output_dir}")

    finally:
        # Clean up
        simulation_app.close()

if __name__ == "__main__":
    main()
