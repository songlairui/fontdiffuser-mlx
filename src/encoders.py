"""FontDiffuser MLX — ContentEncoder & StyleEncoder。

CG-GAN 风格编码器，DBlock 堆叠 + SNConv2d（推理时用普通 Conv2d）。
"""

import mlx.core as mx
import mlx.nn as nn


class DBlock(nn.Module):
    """DBlock: Conv → GroupNorm → SiLU → Conv → ChannelAttn → Downsample."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(32, out_ch, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if downsample else None
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.silu(self.conv1(x))
        h = nn.silu(self.norm(self.conv2(h)))
        if self.shortcut is not None:
            x = self.shortcut(x)
        x = x + h
        residual = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, residual


class ContentEncoder(nn.Module):
    """ContentEncoder: DBlock × 3, 输出多尺度特征金字塔。

    Returns: (final_feature, [residual_0, residual_1, ...])
    """

    def __init__(self, in_ch: int = 3, start_ch: int = 64, num_blocks: int = 3):
        super().__init__()
        self.blocks = []
        for i in range(num_blocks):
            ic = in_ch if i == 0 else start_ch * (2 ** (i - 1))
            oc = start_ch * (2 ** i)
            self.blocks.append(DBlock(ic, oc, downsample=True))

    def __call__(self, x: mx.array) -> tuple[mx.array, list[mx.array]]:
        residuals = []
        for block in self.blocks:
            x, res = block(x)
            residuals.append(res)
        return x, residuals


class StyleEncoder(nn.Module):
    """StyleEncoder: DBlock × 3, 输出风格特征和残差。

    Returns: (style_feature, style_global, [residual_0, residual_1, ...])
    """

    def __init__(self, in_ch: int = 3, start_ch: int = 64, num_blocks: int = 3):
        super().__init__()
        self.blocks = []
        for i in range(num_blocks):
            ic = in_ch if i == 0 else start_ch * (2 ** (i - 1))
            oc = start_ch * (2 ** i)
            self.blocks.append(DBlock(ic, oc, downsample=True))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, list[mx.array]]:
        residuals = []
        for block in self.blocks:
            x, res = block(x)
            residuals.append(res)
        # Global average pooling for style vector
        style_global = mx.mean(x, axis=(1, 2))
        return x, style_global, residuals
