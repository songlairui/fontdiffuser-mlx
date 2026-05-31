"""Content and Style Encoders for FontDiffuser.

These encoders use spectral-normalized convolutions (SNConv2d) and DBlock residual blocks.
For inference, SN weights are pre-normalized during weight loading.
For training, SN is applied on-the-fly.

All tensors are in NHWC format: [B, H, W, C].
"""

import functools
import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Optional
from .snconv import SNConv2d


class DBlock(nn.Module):
    """Residual block with optional downsampling.
    
    Uses SNConv2d for spectral normalization during training.
    For inference, weights are pre-normalized.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wide: bool = True,
        preactivation: bool = False,
        downsample: bool = True,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = out_channels if wide else in_channels
        self.preactivation = preactivation
        self.downsample = downsample
        
        # Conv layers (SNConv2d with dynamic spectral normalization)
        self.conv1 = SNConv2d(
            in_channels, self.hidden_channels, kernel_size, padding=padding
        )
        self.conv2 = SNConv2d(
            self.hidden_channels, out_channels, kernel_size, padding=padding
        )
        
        # Shortcut
        self.learnable_sc = (in_channels != out_channels) or downsample
        if self.learnable_sc:
            self.conv_sc = SNConv2d(
                in_channels, out_channels, 1, padding=0
            )
    
    def shortcut(self, x: mx.array) -> mx.array:
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = nn.AvgPool2d(2)(x)
        else:
            if self.downsample:
                x = nn.AvgPool2d(2)(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, H, W, C]
        
        Returns:
            Output tensor [B, H/2, W/2, C_out] if downsample
        """
        if self.preactivation:
            h = nn.relu(x)
        else:
            h = x
        
        h = self.conv1(h)
        h = nn.relu(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = nn.AvgPool2d(2)(h)
        
        return h + self.shortcut(x)


def content_encoder_arch(ch: int = 64, input_nc: int = 3, resolution: int = 96):
    """Architecture definition for ContentEncoder."""
    if resolution == 96:
        return {
            'in_channels': [input_nc] + [ch * item for item in [1, 2]],
            'out_channels': [item * ch for item in [1, 2, 4]],
            'resolution': [48, 24, 12],
        }
    elif resolution == 128:
        return {
            'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
            'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
            'resolution': [64, 32, 16, 8, 4],
        }
    elif resolution == 256:
        return {
            'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8]],
            'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16]],
            'resolution': [128, 64, 32, 16, 8, 4],
        }
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


def style_encoder_arch(ch: int = 64, input_nc: int = 3, resolution: int = 96):
    """Architecture definition for StyleEncoder."""
    if resolution == 96:
        return {
            'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
            'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
            'resolution': [48, 24, 12, 6, 3],
        }
    elif resolution == 128:
        return {
            'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
            'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
            'resolution': [64, 32, 16, 8, 4],
        }
    elif resolution == 256:
        return {
            'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8]],
            'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16]],
            'resolution': [128, 64, 32, 16, 8, 4],
        }
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


class ContentEncoder(nn.Module):
    """Content image encoder with spectral normalization.
    
    Produces multi-scale features for the UNet's content attention.
    All tensors in NHWC format.
    """
    
    def __init__(
        self,
        G_ch: int = 64,
        G_wide: bool = True,
        resolution: int = 96,
        input_nc: int = 3,
    ):
        super().__init__()
        self.ch = G_ch
        self.resolution = resolution
        
        if resolution == 96:
            self.save_features = [0, 1, 2, 3, 4]
        elif resolution == 128:
            self.save_features = [0, 1, 2, 3, 4]
        elif resolution == 256:
            self.save_features = [0, 1, 2, 3, 4, 5]
        
        self.arch = content_encoder_arch(G_ch, input_nc, resolution)
        
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks.append(
                DBlock(
                    in_channels=self.arch['in_channels'][index],
                    out_channels=self.arch['out_channels'][index],
                    wide=G_wide,
                    preactivation=(index > 0),
                    downsample=True,
                )
            )
    
    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """
        Args:
            x: Input image [B, H, W, C] in NHWC format, normalized to [-1, 1]
        
        Returns:
            (h, residual_features) where:
                h: Final encoded feature [B, H', W', C']
                residual_features: List of intermediate features for skip connections
        """
        h = x
        residual_features = []
        residual_features.append(h)  # Save input
        
        for index, block in enumerate(self.blocks):
            h = block(h)
            if index in self.save_features[:-1]:
                residual_features.append(h)
        
        return h, residual_features


class StyleEncoder(nn.Module):
    """Style image encoder with spectral normalization.
    
    Produces style embedding and multi-scale features.
    All tensors in NHWC format.
    """
    
    def __init__(
        self,
        G_ch: int = 64,
        G_wide: bool = True,
        resolution: int = 96,
        input_nc: int = 3,
    ):
        super().__init__()
        self.ch = G_ch
        self.resolution = resolution
        
        if resolution == 96:
            self.save_features = [0, 1, 2, 3, 4]
        elif resolution == 128:
            self.save_features = [0, 1, 2, 3, 4]
        elif resolution == 256:
            self.save_features = [0, 1, 2, 3, 4, 5]
        
        self.arch = style_encoder_arch(G_ch, input_nc, resolution)
        
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks.append(
                DBlock(
                    in_channels=self.arch['in_channels'][index],
                    out_channels=self.arch['out_channels'][index],
                    wide=G_wide,
                    preactivation=(index > 0),
                    downsample=True,
                )
            )
        
        # Last layer: InstanceNorm + ReLU + Conv1x1 (matching upstream)
        # PyTorch InstanceNorm2d uses biased variance (correction=0) and eps inside sqrt.
        # We implement manually to guarantee exact numerical parity.
        self.last_norm = None
        self.last_conv = nn.Conv2d(self.arch['out_channels'][-1], self.arch['out_channels'][-1], 1, padding=0)
    
    def __call__(
        self, x: mx.array
    ) -> Tuple[mx.array, mx.array, List[mx.array]]:
        """
        Args:
            x: Input image [B, H, W, C] in NHWC format, normalized to [-1, 1]
        
        Returns:
            (style_emd, style_vector, residual_features) where:
                style_emd: Style embedding [B, H', W', C']
                style_vector: Global style vector [B, C']
                residual_features: List of intermediate features
        """
        h = x
        residual_features = []
        residual_features.append(h)
        
        for index, block in enumerate(self.blocks):
            h = block(h)
            if index in self.save_features[:-1]:
                residual_features.append(h)
        
        # Last layer: apply twice to match upstream forward implementation
        for _ in range(2):
            axes = (1, 2)
            mean = mx.mean(h, axis=axes, keepdims=True)
            var = mx.mean((h - mean) ** 2, axis=axes, keepdims=True)
            h = (h - mean) / mx.sqrt(var + 1e-5)
            h = nn.relu(h)
            h = self.last_conv(h)
        style_emd = h
        
        # Global average pooling: [B, H, W, C] -> [B, C]
        style_vector = mx.mean(h, axis=(1, 2))
        
        return style_emd, style_vector, residual_features
