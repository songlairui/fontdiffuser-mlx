#!/usr/bin/env python3
"""Convert FontDiffuser PyTorch weights to MLX format."""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlx_fd.weight_converter import (
    load_pytorch_checkpoint,
    convert_content_encoder_weights,
    convert_style_encoder_weights,
    convert_unet_weights,
)


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to MLX format")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint directory containing content_encoder.pth, style_encoder.pth, unet.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mlx_weights",
        help="Output directory for MLX weights (default: ./mlx_weights)",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)

    print(f"Converting PyTorch weights from {ckpt_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load PyTorch weights
    print("Loading PyTorch weights...")
    pt_weights = load_pytorch_checkpoint(ckpt_dir)
    print(f"  ✓ Content encoder: {len(pt_weights['content_encoder'])} tensors")
    print(f"  ✓ Style encoder: {len(pt_weights['style_encoder'])} tensors")
    print(f"  ✓ UNet: {len(pt_weights['unet'])} tensors\n")

    # Convert weights
    print("Converting weights...")
    content_encoder_mlx = convert_content_encoder_weights(pt_weights["content_encoder"])
    print(f"  ✓ Content encoder converted: {len(content_encoder_mlx)} tensors")

    style_encoder_mlx = convert_style_encoder_weights(pt_weights["style_encoder"])
    print(f"  ✓ Style encoder converted: {len(style_encoder_mlx)} tensors")

    unet_mlx = convert_unet_weights(pt_weights["unet"])
    print(f"  ✓ UNet converted: {len(unet_mlx)} tensors\n")

    # Save MLX weights
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving MLX weights to {output_dir}...")
    
    import numpy as np
    for name, weights in [
        ("content_encoder", content_encoder_mlx),
        ("style_encoder", style_encoder_mlx),
        ("unet", unet_mlx),
    ]:
        output_path = output_dir / f"{name}.npz"
        np.savez(output_path, **weights)
        print(f"  ✓ Saved {name}.npz ({len(weights)} tensors)")

    print("\n✓✓✓ Weight conversion complete! ✓✓✓")
    print(f"\nYou can now use these weights with sample_mlx.py:")
    print(f"  python sample_mlx.py --weights_dir {output_dir}")


if __name__ == "__main__":
    main()
