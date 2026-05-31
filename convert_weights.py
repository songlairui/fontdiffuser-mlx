"""FontDiffuser MLX — PyTorch 权重转 MLX 数组。

用法：
    from convert_weights import load_fontdiffuser_weights
    model_weights = load_fontdiffuser_weights("ckpt/ckpt/")
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path


def convert_conv2d_weight(w: np.ndarray) -> np.ndarray:
    """PyTorch Conv2d [O,I,H,W] → MLX Conv2d [O,H,W,I]。"""
    if w.ndim == 4:
        return w.transpose(0, 2, 3, 1)
    return w


def convert_linear_weight(w: np.ndarray) -> np.ndarray:
    """Linear 权重直接可用（MLX Linear 也是 [out, in]）。"""
    return w


def load_state_dict(path: str | Path) -> dict[str, np.ndarray]:
    """加载 PyTorch state_dict 并转为 numpy。"""
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k: v.cpu().numpy() for k, v in sd.items()}


def convert_unet_weights(sd: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """转换 UNet 权重命名和布局。"""
    out = {}
    for k, v in sd.items():
        # Skip spectral norm buffers (u, sv) — not needed for inference/fine-tuning
        if ".sn_" in k or k.endswith(".u") or k.endswith(".sv"):
            continue

        # Conv2d weights: transpose
        if v.ndim == 4:
            v = convert_conv2d_weight(v)

        out[k] = v
    return out


def convert_encoder_weights(sd: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """转换 Encoder 权重（ContentEncoder / StyleEncoder）。

    SNConv2d 的权重已经是 normalized 的（推理时 SN 不影响）。
    跳过 spectral norm 的 u/sv buffers。
    """
    out = {}
    for k, v in sd.items():
        if ".u" in k and v.ndim == 1 and v.shape[0] <= 2:
            continue
        if ".sv" in k and v.ndim == 1 and v.shape[0] <= 2:
            continue

        if v.ndim == 4:
            v = convert_conv2d_weight(v)

        out[k] = v
    return out


def load_fontdiffuser_weights(ckpt_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """加载 FontDiffuser 全部权重并转换格式。

    Returns:
        {"unet": {...}, "style_encoder": {...}, "content_encoder": {...}}
    """
    ckpt_dir = Path(ckpt_dir)
    unet_sd = load_state_dict(ckpt_dir / "unet.pth")
    style_sd = load_state_dict(ckpt_dir / "style_encoder.pth")
    content_sd = load_state_dict(ckpt_dir / "content_encoder.pth")

    return {
        "unet": convert_unet_weights(unet_sd),
        "style_encoder": convert_encoder_weights(style_sd),
        "content_encoder": convert_encoder_weights(content_sd),
    }
