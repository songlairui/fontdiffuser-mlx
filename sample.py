#!/usr/bin/env python3
"""FontDiffuser MLX — 推理脚本。

用法：
    cd fontdiffuser-mlx
    source ../fontdiffuser/.venv/bin/activate
    python sample.py --char 你
"""

import argparse
import os
import sys
import time

import mlx.core as mx
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# 只把 fontdiffuser-mlx/ 加入 sys.path（避免 fontdiffuser 的 configs 模块冲突）
MLX_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MLX_DIR)

from src.unet import UNet
from src.encoders import ContentEncoder, StyleEncoder
from src.scheduler import scaled_linear_beta_schedule
from weights import load_all_weights, apply_weights


CONTENT_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def render_content(char: str, size: int = 96) -> mx.array:
    """渲染标准字形为 MLX 数组 [1, H, W, 3]，归一化到 [-1, 1]。"""
    font = ImageFont.truetype(CONTENT_FONT, size=size)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(char)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), char, fill=(0, 0, 0), font=font)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return mx.array(arr)[None, :]


def load_style_ref(path: str, size: int = 96) -> mx.array:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return mx.array(arr)[None, :]


def main():
    parser = argparse.ArgumentParser(description="FontDiffuser MLX 推理")
    parser.add_argument("--char", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="../fontdiffuser/ckpt/ckpt")
    parser.add_argument("--style", type=str, default="../fontdiffuser/data_hw/sampling/example_style.jpg")
    parser.add_argument("--output", type=str, default="/tmp/mlx_gen.png")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    print(f"Loading weights from {args.ckpt_dir}...")
    t0 = time.time()
    weights = load_all_weights(args.ckpt_dir)
    print(f"Weights loaded in {time.time() - t0:.1f}s")

    # Create models
    unet = UNet()
    content_enc = ContentEncoder()
    style_enc = StyleEncoder()

    # Apply weights
    apply_weights(unet, weights["unet"])
    apply_weights(content_enc, weights["content_encoder"])
    apply_weights(style_enc, weights["style_encoder"])

    # Prepare inputs
    content_img = render_content(args.char)
    style_img = load_style_ref(args.style)

    print(f"Generating '{args.char}' with {args.steps} steps...")

    # Extract features
    style_feat, _, _ = style_enc(style_img)
    B, H, W, C = style_feat.shape
    style_hidden = style_feat.reshape(B, H * W, C)

    content_feat, content_res = content_enc(content_img)
    content_res.append(content_feat)

    style_content_feat, _, style_content_res = style_enc(style_img)
    style_content_res.append(style_content_feat)

    # DDPM sampling loop
    shape = (1, 96, 96, 3)
    x = mx.random.normal(shape)

    betas = scaled_linear_beta_schedule(1000)
    alphas = 1.0 - betas
    alphas_cumprod = mx.cumprod(alphas)
    timesteps = mx.linspace(999, 0, args.steps).astype(mx.int32)

    t_total = time.time()
    for i in range(len(timesteps) - 1):
        t = int(timesteps[i])
        t_next = int(timesteps[i + 1])

        noise_pred = unet(x, mx.array([t]), style_hidden, content_res, style_content_res)

        alpha_t = alphas_cumprod[t]
        alpha_next = alphas_cumprod[t_next]

        x0 = (x - mx.sqrt(1 - alpha_t) * noise_pred) / mx.sqrt(alpha_t)
        x0 = mx.clip(x0, -1, 1)
        x = mx.sqrt(alpha_next) * x0 + mx.sqrt(1 - alpha_next) * noise_pred

        mx.eval(x)
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{args.steps}")

    print(f"Total: {time.time() - t_total:.1f}s")

    # Save result
    result = ((np.array(x[0]) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(result).save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
