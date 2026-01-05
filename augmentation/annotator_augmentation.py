# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Annotator-level augmentation example for synthetic data generation.

This module demonstrates how to apply augmentations at the annotator level,
allowing for fine-grained control over individual annotators. It showcases:
- RGB to BGR channel swapping (NumPy and Warp implementations)
- Gaussian noise on depth data with configurable sigma
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
from PIL import Image

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
    return str(script_dir / "config" / "annotator_augmentation_config.yaml")

def rgb_to_bgr_np(data_in: np.ndarray) -> np.ndarray:
    """
    Swap red and blue channels in RGB data using NumPy (CPU).

    Args:
        data_in: Input RGBA image array of shape (H, W, 4)

    Returns:
        Modified array with R and B channels swapped
    """
    data_in[:, :, [0, 2]] = data_in[:, :, [2, 0]]
    return data_in

def gaussian_noise_depth_np(data_in: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """
    Add Gaussian noise to depth data using NumPy (CPU).

    Args:
        data_in: Input depth array
        sigma: Standard deviation of the Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Depth array with added Gaussian noise
    """
    np.random.seed(seed)
    result = data_in.astype(np.float32) + np.random.randn(*data_in.shape) * sigma
    return np.clip(result, 0, None).astype(data_in.dtype)

@wp.kernel
def rgb_to_bgr_wp(data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array3d(dtype=wp.uint8)):
    """
    Swap red and blue channels in RGB data using Warp (GPU).

    Args:
        data_in: Input RGBA image array
        data_out: Output RGBA image array with swapped channels
    """
    i, j = wp.tid()
    data_out[i, j, 0] = data_in[i, j, 2]  # R <- B
    data_out[i, j, 1] = data_in[i, j, 1]  # G <- G
    data_out[i, j, 2] = data_in[i, j, 0]  # B <- R
    data_out[i, j, 3] = data_in[i, j, 3]  # A <- A

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

def write_rgb(data: np.ndarray, path: str) -> None:
    """
    Write RGB data to a PNG file.

    Args:
        data: RGB/RGBA image array
        path: Output file path (without extension)
    """
    rgb_img = Image.fromarray(data).convert("RGBA")
    rgb_img.save(f"{path}.png")

def write_depth(data, path: str) -> None:
    """
    Write depth data to a normalized PNG file.

    Handles Warp arrays, normalizes to 0-255 range, and handles invalid values.

    Args:
        data: Depth array (NumPy or Warp)
        path: Output file path (without extension)
    """
    # Convert to numpy if needed
    if isinstance(data, wp.array):
        data = data.numpy()

    # Replace inf values with nan, then replace nan with mean
    data[np.isinf(data)] = np.nan
    data = np.nan_to_num(data, nan=np.nanmean(data), copy=False)

    # Normalize to 0-255 range
    normalized_array = (data - np.min(data)) / (np.max(data) - np.min(data))
    integer_array = (normalized_array * 255).astype(np.uint8)

    depth_img = Image.fromarray(integer_array).convert("L")
    depth_img.save(f"{path}.png")

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


def register_augmentations(config: Dict):
    """
    Register augmentation functions with Replicator.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (rgb_augmentation, depth_augmentation)
    """
    import omni.replicator.core as rep

    augmentation_config = config["augmentation"]
    use_warp = augmentation_config["backend"] == "warp"
    sigma_default = augmentation_config["gaussian_noise"]["depth_sigma_default"]

    if use_warp:
        # Register Warp-based augmentations
        rep.AnnotatorRegistry.register_augmentation(
            "gn_depth_wp",
            rep.annotators.Augmentation.from_function(
                gaussian_noise_depth_wp, sigma=sigma_default, seed=None
            ),
        )
        rgb_augm = rep.annotators.Augmentation.from_function(rgb_to_bgr_wp)
        depth_augm = rep.AnnotatorRegistry.get_augmentation("gn_depth_wp")
    else:
        # Register NumPy-based augmentations
        rep.AnnotatorRegistry.register_augmentation(
            "gn_depth_np",
            rep.annotators.Augmentation.from_function(
                gaussian_noise_depth_np, sigma=sigma_default, seed=None
            ),
        )
        rgb_augm = rep.annotators.Augmentation.from_function(rgb_to_bgr_np)
        depth_augm = rep.AnnotatorRegistry.get_augmentation("gn_depth_np")

    return rgb_augm, depth_augm


def setup_annotators(render_product, rgb_augmentation, depth_augmentation, config: Dict):
    """
    Create and configure annotators with augmentations.

    Args:
        render_product: The render product to attach annotators to
        rgb_augmentation: RGB augmentation to apply
        depth_augmentation: Depth augmentation to apply
        config: Configuration dictionary

    Returns:
        Tuple of (rgb_annotator, depth_annotator_1, depth_annotator_2)
    """
    import omni.replicator.core as rep

    augmentation_config = config["augmentation"]
    sigma_high = augmentation_config["gaussian_noise"]["depth_sigma_high"]

    # Register RGB annotator with augmentation
    rep.annotators.register(
        name="rgb_to_bgr_augm",
        annotator=rep.annotators.augment(
            source_annotator=rep.AnnotatorRegistry.get_annotator("rgb"),
            augmentation=rgb_augmentation,
        ),
    )

    # Create annotators
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb_to_bgr_augm")
    depth_annotator_1 = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    depth_annotator_2 = rep.AnnotatorRegistry.get_annotator("distance_to_camera")

    # Augment depth annotators with different sigma values
    depth_annotator_1.augment(depth_augmentation)
    depth_annotator_2.augment(depth_augmentation, sigma=sigma_high)

    # Attach annotators to render product
    rgb_annotator.attach(render_product)
    depth_annotator_1.attach(render_product)
    depth_annotator_2.attach(render_product)

    return rgb_annotator, depth_annotator_1, depth_annotator_2


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


def generate_data(num_frames: int, output_dir: str, annotators: Tuple, config: Dict) -> float:
    """
    Generate synthetic data with augmentations.

    Args:
        num_frames: Number of frames to capture
        output_dir: Directory to save output images
        annotators: Tuple of (rgb_annotator, depth_annotator_1, depth_annotator_2)
        config: Configuration dictionary

    Returns:
        Total time taken for data generation in seconds
    """
    import omni.replicator.core as rep

    rgb_annotator, depth_annotator_1, depth_annotator_2 = annotators
    rt_subframes = config["rendering"]["rt_subframes"]

    # Start data generation
    rep.orchestrator.preview()

    start_time = time.time()
    for i in range(num_frames):
        rep.orchestrator.step(rt_subframes=rt_subframes)

        # Retrieve data from annotators
        rgb_data = rgb_annotator.get_data()
        depth_data_1 = depth_annotator_1.get_data()
        depth_data_2 = depth_annotator_2.get_data()

        # Write data to disk
        write_rgb(rgb_data, os.path.join(output_dir, f"annot_rgb_{i}"))
        write_depth(depth_data_1, os.path.join(output_dir, f"annot_depth_1_{i}"))
        write_depth(depth_data_2, os.path.join(output_dir, f"annot_depth_2_{i}"))

    return time.time() - start_time

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate augmented synthetic data with annotator-level augmentations",
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

    # Import Replicator (must be after SimulationApp initialization)
    import omni.replicator.core as rep

    try:
        # Configure settings
        configure_settings(config)

        # Setup scene
        print("[INFO] Setting up scene...")
        red_cube, render_product = setup_scene(simulation_app, config)

        # Register and get augmentations
        backend = config["augmentation"]["backend"]
        print(f"[INFO] Registering augmentations (backend: {backend.upper()})...")
        rgb_augmentation, depth_augmentation = register_augmentations(config)

        # Setup annotators
        print("[INFO] Setting up annotators with augmentations...")
        annotators = setup_annotators(render_product, rgb_augmentation, depth_augmentation, config)

        # Setup randomization
        print("[INFO] Configuring randomization...")
        setup_randomization(red_cube)

        # Prepare output directory
        output_dir = os.path.join(os.getcwd(), config["data_generation"]["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Writing data to: {output_dir}")

        # Generate data
        num_frames = config["data_generation"]["num_frames"]
        print(f"[INFO] Generating {num_frames} frames...")
        duration = generate_data(num_frames, output_dir, annotators, config)

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
