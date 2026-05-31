#!/usr/bin/env python3
"""FontDiffuser MLX training script."""

import argparse
import sys
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlx_fd.model import FontDiffuserModel
from mlx_fd.unet import UNet
from mlx_fd.encoders import ContentEncoder, StyleEncoder
from mlx_fd.scheduler import DDPMScheduler


class FontDataset:
    """Font training dataset supporting real images or synthetic fallback."""

    def __init__(self, data_root, content_size=96, style_size=96, resolution=96):
        self.data_root = Path(data_root)
        self.content_size = content_size
        self.style_size = style_size
        self.resolution = resolution

        content_dir = self.data_root / "content"
        style_dir = self.data_root / "style"
        target_dir = self.data_root / "target"

        if content_dir.is_dir() and style_dir.is_dir() and target_dir.is_dir():
            content_stems = {p.stem for p in content_dir.glob("*") if p.is_file()}
            style_stems = {p.stem for p in style_dir.glob("*") if p.is_file()}
            target_stems = {p.stem for p in target_dir.glob("*") if p.is_file()}
            shared = sorted(content_stems & style_stems & target_stems)
            self.samples = [
                {
                    "content": content_dir / f"{stem}.png",
                    "style": style_dir / f"{stem}.png",
                    "target": target_dir / f"{stem}.png",
                }
                for stem in shared
            ]
            self.synthetic = False
        else:
            # Fallback: keep runnable for quick smoke-tests without real data
            self.samples = [None] * 32
            self.synthetic = True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.synthetic:
            return None

        sample = self.samples[idx]
        return {
            "content": load_image(sample["content"], self.content_size),
            "style": load_image(sample["style"], self.style_size),
            "target": load_image(sample["target"], self.resolution),
        }


def load_image(image_path, size):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
    return mx.array(img_array)


def train_step(model, scheduler, content_imgs, style_imgs, target_imgs, optimizer, args):
    """Single training step."""
    batch_size = content_imgs.shape[0]
    
    # Sample random timesteps
    timesteps = mx.random.randint(0, 1000, (batch_size,))
    
    # Sample noise
    noise = mx.random.normal(target_imgs.shape)
    
    # Add noise to targets
    noisy_targets = scheduler.add_noise(target_imgs, noise, timesteps)
    
    # Define loss function
    def loss_fn(model):
        # Forward pass
        noise_pred, offset_out_sum = model(
            noisy_targets,
            timesteps,
            style_images=style_imgs,
            content_images=content_imgs,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
        )
        
        # Diffusion loss (MSE)
        diffusion_loss = mx.mean((noise_pred - noise) ** 2)
        
        # Offset loss (regularization for deformable conv)
        offset_loss = offset_out_sum
        
        # Total loss
        total_loss = diffusion_loss + args.offset_coefficient * offset_loss
        
        return total_loss
    
    # Compute loss and gradients
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    
    # Update weights
    optimizer.update(model, grads)
    
    return loss


def save_checkpoint(model, output_dir, step_or_label, checkpoint_format):
    """Save checkpoint in the requested format."""
    import numpy as np
    
    if checkpoint_format == "npz":
        weights = {
            "content_encoder": {k: np.array(v) for k, v in model.content_encoder.parameters().items()},
            "style_encoder": {k: np.array(v) for k, v in model.style_encoder.parameters().items()},
            "unet": {k: np.array(v) for k, v in model.unet.parameters().items()},
        }
        
        for name, params in weights.items():
            path = output_dir / f"{name}.npz"
            np.savez(path, **params)
        
        print(f"  ✓ Saved NPZ checkpoint at step {step_or_label}")
    else:
        path = output_dir / f"checkpoint_{step_or_label}.safetensors"
        model.save_weights(str(path))
        print(f"  ✓ Saved Safetensors checkpoint at {path}")


def main():
    parser = argparse.ArgumentParser(description="FontDiffuser MLX Training")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    # Model config
    parser.add_argument("--content_image_size", type=int, default=96)
    parser.add_argument("--style_image_size", type=int, default=96)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--unet_channels", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--cross_attention_dim", type=int, default=1024)
    parser.add_argument("--attention_head_dim", type=int, default=1)
    parser.add_argument("--content_encoder_downsample_size", type=int, default=3)
    parser.add_argument("--content_start_channel", type=int, default=64)
    parser.add_argument("--style_start_channel", type=int, default=64)
    
    # Training config
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--offset_coefficient", type=float, default=0.5)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    
    # Output
    parser.add_argument("--checkpoint_format", type=str, default="npz", choices=["npz", "safetensors"], help="Checkpoint format: npz for sample_mlx compatibility")
    
    args = parser.parse_args()
    
    # Set random seed
    mx.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="scaled_linear",
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    
    # Create dataset
    print("Loading dataset...")
    dataset = FontDataset(
        args.data_root,
        content_size=args.content_image_size,
        style_size=args.style_image_size,
        resolution=args.resolution,
    )
    if dataset.synthetic:
        print("  ⚠ Real dataset not found, using synthetic smoke-test data.")
    print(f"  ✓ Dataset loaded: {len(dataset)} samples\n")
    
    # Training loop
    print("Starting training...")
    print(f"  Max steps: {args.max_train_steps}")
    print(f"  Batch size: {args.train_batch_size}")
    print(f"  Learning rate: {args.learning_rate}\n")
    
    global_step = 0
    start_time = time.time()
    
    while global_step < args.max_train_steps:
        batch_size = args.train_batch_size

        if dataset.synthetic:
            content_imgs = mx.random.normal((batch_size, args.content_image_size, args.content_image_size, 3))
            style_imgs = mx.random.normal((batch_size, args.style_image_size, args.style_image_size, 3))
            target_imgs = mx.random.normal((batch_size, args.resolution, args.resolution, 3))
        else:
            content_imgs = mx.stack([dataset[i]["content"] for i in range(batch_size)])
            style_imgs = mx.stack([dataset[i]["style"] for i in range(batch_size)])
            target_imgs = mx.stack([dataset[i]["target"] for i in range(batch_size)])
        
        # Training step
        loss = train_step(
            model, scheduler, content_imgs, style_imgs, target_imgs, optimizer, args
        )
        
        # Evaluate and log
        mx.eval(loss)
        
        if global_step % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Step {global_step}/{args.max_train_steps} | Loss: {loss.item():.6f} | Time: {elapsed:.1f}s")
        
        # Save checkpoint
        if global_step % args.save_interval == 0 and global_step > 0:
            save_checkpoint(model, output_dir, global_step, args.checkpoint_format)
            
        global_step += 1
    
    save_checkpoint(model, output_dir, "final", args.checkpoint_format)
    
    print("\n" + "=" * 50)
    print("✓✓✓ Training complete! ✓✓✓")
    print("=" * 50)
    
    if args.checkpoint_format == "npz":
        print(f"\nYou can now use the checkpoint with sample_mlx.py:")
        print(f"  python sample_mlx.py --weights_dir {output_dir}")
    
    print("\n" + "=" * 50)
    print("✓✓✓ Training complete! ✓✓✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
