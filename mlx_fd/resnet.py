"""ResNet blocks and up/downsampling layers."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class Mish(nn.Module):
    """Mish activation function."""
    
    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.tanh(nn.softplus(x))


class Downsample2D(nn.Module):
    """Downsampling layer with optional convolution."""
    
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.name = name
        
        if use_conv:
            self.conv = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=2, padding=padding
            )
        else:
            # Will use average pooling in forward
            self.conv = None
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Args:
            hidden_states: [B, H, W, C] in MLX format
        
        Returns:
            Downsampled tensor [B, H/2, W/2, C]
        """
        assert hidden_states.shape[-1] == self.channels
        
        if self.use_conv:
            if self.padding == 0:
                # Pad manually: (left, right, top, bottom)
                hidden_states = mx.pad(
                    hidden_states, [(0, 0), (0, 1), (0, 1), (0, 0)]
                )
            hidden_states = self.conv(hidden_states)
        else:
            # Average pooling with stride 2
            hidden_states = nn.AvgPool2d(kernel_size=2, stride=2)(hidden_states)
        
        return hidden_states


class Upsample2D(nn.Module):
    """Upsampling layer with optional convolution."""
    
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        
        if use_conv_transpose:
            self.conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
        else:
            self.conv = None
    
    def __call__(
        self, hidden_states: mx.array, output_size: Optional[tuple] = None
    ) -> mx.array:
        """
        Args:
            hidden_states: [B, H, W, C] in MLX format
            output_size: optional target size
        
        Returns:
            Upsampled tensor [B, H*2, W*2, C]
        """
        assert hidden_states.shape[-1] == self.channels
        
        if self.use_conv_transpose:
            return self.conv(hidden_states)
        
        # Nearest neighbor upsampling
        B, H, W, C = hidden_states.shape
        if output_size is None:
            new_H, new_W = H * 2, W * 2
        else:
            new_H, new_W = output_size
        
        # Manual nearest neighbor upsampling
        # Repeat each pixel 2x2
        hidden_states = mx.repeat(hidden_states, 2, axis=1)  # Repeat along H
        hidden_states = mx.repeat(hidden_states, 2, axis=2)  # Repeat along W
        
        # Apply convolution if needed
        if self.use_conv and self.conv is not None:
            hidden_states = self.conv(hidden_states)
        
        return hidden_states


class ResnetBlock2D(nn.Module):
    """ResNet block with optional time embedding."""
    
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "silu",
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor
        
        if groups_out is None:
            groups_out = groups
        
        self.norm1 = nn.GroupNorm(num_groups=groups, dims=in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, self.out_channels)
        else:
            self.time_emb_proj = None
        
        self.norm2 = nn.GroupNorm(
            num_groups=groups_out, dims=self.out_channels, eps=eps
        )
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        
        if non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        else:
            self.nonlinearity = nn.SiLU()
        
        # Shortcut connection
        self.use_in_shortcut = (
            self.in_channels != self.out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )
        
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, self.out_channels, 1, padding=0
            )
        else:
            self.conv_shortcut = None
    
    def __call__(self, input_tensor: mx.array, temb: Optional[mx.array]) -> mx.array:
        """
        Args:
            input_tensor: [B, H, W, C]
            temb: [B, temb_channels] time embedding
        
        Returns:
            Output tensor [B, H, W, C_out]
        """
        hidden_states = input_tensor
        
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        
        if temb is not None and self.time_emb_proj is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))
            # Add time embedding: [B, temb_dim] -> [B, 1, 1, out_channels]
            hidden_states = hidden_states + temb[:, None, None, :]
        
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        
        if self.dropout_layer is not None:
            hidden_states = self.dropout_layer(hidden_states)
        
        hidden_states = self.conv2(hidden_states)
        
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        
        return output_tensor
