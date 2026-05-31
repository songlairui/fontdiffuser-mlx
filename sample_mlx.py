#!/usr/bin/env python3
"""FontDiffuser MLX sampling script."""

import argparse
import sys
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlx_fd.model import FontDiffuserModel
from mlx_fd.unet import UNet
from mlx_fd.encoders import ContentEncoder, StyleEncoder
from mlx_fd.scheduler import DDPMScheduler, DPMSolverPipeline


def load_weights(model, weights_dir):
    """Load MLX weights from npz files."""
    import numpy as np
    
    print(f"Loading weights from {weights_dir}...")
    
    # Load and convert numpy arrays to mx.array
    content_encoder_weights = {
        k: mx.array(v) for k, v in np.load(weights_dir / "content_encoder.npz").items()
    }
    style_encoder_weights = {
        k: mx.array(v) for k, v in np.load(weights_dir / "style_encoder.npz").items()
    }
    unet_weights = {
        k: mx.array(v) for k, v in np.load(weights_dir / "unet.npz").items()
    }
    
    model.content_encoder.load_weights(content_encoder_weights)
    model.style_encoder.load_weights(style_encoder_weights, strict=False)
    model.unet.load_weights(unet_weights)
    
    print("  ✓ Weights loaded successfully\n")


def load_image(image_path, size=96):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [-1, 1]
    img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
    
    # Convert to MLX array with shape [1, H, W, C]
    return mx.array(img_array)[None]


def save_image(image_array, output_path):
    """Save an image from MLX array."""
    # Convert from [-1, 1] to [0, 255]
    image_array = ((image_array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(image_array)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="FontDiffuser MLX Sampling")
    
    # Model config
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to MLX weights directory")
    parser.add_argument("--content_image_size", type=int, default=96)
    parser.add_argument("--style_image_size", type=int, default=96)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--unet_channels", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--cross_attention_dim", type=int, default=1024)
    parser.add_argument("--attention_head_dim", type=int, default=1)
    parser.add_argument("--content_encoder_downsample_size", type=int, default=3)
    parser.add_argument("--content_start_channel", type=int, default=64)
    parser.add_argument("--style_start_channel", type=int, default=64)
    
    # Sampling config
    parser.add_argument("--content_image", type=str, required=True, help="Path to content image")
    parser.add_argument("--style_image", type=str, required=True, help="Path to style image")
    parser.add_argument("--output_path", type=str, default="output.png", help="Output image path")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--solver", type=str, default="ddpm", choices=["ddpm", "dpm_solver"], help="Sampler type")
    
    args = parser.parse_args()
    
    # Set random seed
    mx.random.seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("Creating model...")
    content_encoder = ContentEncoder(
        G_ch=args.content_start_channel,
        resolution=args.content_image_size,
    )
    style_encoder = StyleEncoder(
        G_ch=args.style_start_channel,
        resolution=args.style_image_size,
    )
    unet = UNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        block_out_channels=tuple(args.unet_channels),
        layers_per_block=2,
        cross_attention_dim=args.cross_attention_dim,
        attention_head_dim=args.attention_head_dim,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
    )
    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
    )
    print("  ✓ Model created\n")
    
    # Load weights
    weights_dir = Path(args.weights_dir)
    load_weights(model, weights_dir)
    
    # Load images
    print("Loading images...")
    content_img = load_image(args.content_image, args.content_image_size)
    style_img = load_image(args.style_image, args.style_image_size)
    print(f"  ✓ Content image: {content_img.shape}")
    print(f"  ✓ Style image: {style_img.shape}\n")
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="scaled_linear",
    )
    
    if args.solver == "dpm_solver":
        from mlx_fd.model import FontDiffuserModelDPM
        model_dpm = FontDiffuserModelDPM(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder,
        )
        pipeline = DPMSolverPipeline(
            model=model_dpm,
            ddpm_train_scheduler=scheduler,
            guidance_type="classifier-free",
            guidance_scale=args.guidance_scale,
        )
        
        print(f"Sampling with DPM-Solver++ ({args.num_inference_steps} steps)...")
        start_time = time.time()
        
        result = pipeline.generate(
            content_images=content_img,
            style_images=style_img,
            batch_size=1,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            dm_size=(args.resolution, args.resolution),
        )
        
        end_time = time.time()
        x_t = result
    else:
        # DDPM sampling
        print(f"Sampling with DDPM ({args.num_inference_steps} steps)...")
        start_time = time.time()
        
        # Initialize from random noise
        x_t = mx.random.normal((1, args.resolution, args.resolution, 3))
        
        # Create timestep schedule
        timesteps = mx.linspace(999, 0, args.num_inference_steps + 1).astype(mx.int32)
        
        for i in range(args.num_inference_steps):
            t = int(timesteps[i])
            
            # Predict noise
            noise_pred, _ = model(
                x_t,
                mx.array([t]),
                style_images=style_img,
                content_images=content_img,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
            )
            
            # Scheduler step
            x_t = scheduler.step(noise_pred, t, x_t)
            
            if (i + 1) % 5 == 0:
                print(f"  Step {i + 1}/{args.num_inference_steps}")
        
        end_time = time.time()
    print(f"\n  ✓ Sampling complete in {end_time - start_time:.2f}s\n")
    
    # Save output
    print(f"Saving output to {output_path}...")
    save_image(np.array(x_t[0]), output_path)
    print("  ✓ Output saved\n")
    
    print("=" * 50)
    print("✓✓✓ Sampling complete! ✓✓✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
