"""Deformable Convolution for MLX.

This is a pure MLX implementation of DeformConv2d from torchvision.
The key operation is deformable convolution where kernel sampling locations
are offset by learned offsets.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Union


class DeformConv2d(nn.Module):
    """Deformable Convolution v2 (without modulation mask).
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        dilation: Spacing between kernel elements
        bias: If True, adds a learnable bias to the output
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle tuple/int for kernel_size, stride, padding, dilation
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)
        
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = tuple(dilation)
        
        # Initialize weight: [out_channels, kH, kW, in_channels] for MLX
        kH, kW = self.kernel_size
        scale = (kH * kW * in_channels) ** -0.5
        self.weight = mx.random.normal(
            (out_channels, kH, kW, in_channels)
        ) * scale
        
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None
    
    def __call__(self, x: mx.array, offset: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor [B, H, W, C_in] in MLX NHWC format
            offset: Offset tensor [B, H_out, W_out, 2*kH*kW]
                   Organized as [dy_00, dx_00, dy_01, dx_01, ...]
        
        Returns:
            Output tensor [B, H_out, W_out, C_out]
        """
        B, H, W, C_in = x.shape
        C_out, kH, kW, _ = self.weight.shape
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        
        # Compute output dimensions
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        
        # Pad input if needed
        if pH > 0 or pW > 0:
            x_padded = mx.pad(x, [(0, 0), (pH, pH), (pW, pW), (0, 0)])
        else:
            x_padded = x
        
        H_pad, W_pad = H + 2 * pH, W + 2 * pW
        
        # Generate base grid for output positions
        oh = mx.arange(H_out) * sH
        ow = mx.arange(W_out) * sW
        grid_y, grid_x = mx.meshgrid(oh, ow, indexing='ij')
        # grid_y, grid_x: [H_out, W_out]
        
        # Generate kernel offsets
        kh = mx.arange(kH) * dH
        kw = mx.arange(kW) * dW
        kernel_y, kernel_x = mx.meshgrid(kh, kw, indexing='ij')
        # kernel_y, kernel_x: [kH, kW]
        
        # Compute base sampling locations
        # For each output position (oh, ow) and kernel position (kh, kw):
        # base_y = oh + kh, base_x = ow + kw
        base_y = grid_y[:, :, None, None] + kernel_y[None, None, :, :]
        base_x = grid_x[:, :, None, None] + kernel_x[None, None, :, :]
        # base_y, base_x: [H_out, W_out, kH, kW]
        
        # Reshape offsets: [B, H_out, W_out, 2*kH*kW] -> [B, H_out, W_out, kH, kW, 2]
        offset_reshaped = offset.reshape(B, H_out, W_out, kH, kW, 2)
        offset_y = offset_reshaped[..., 0]  # [B, H_out, W_out, kH, kW]
        offset_x = offset_reshaped[..., 1]  # [B, H_out, W_out, kH, kW]
        
        # Add offsets to base locations
        sample_y = base_y[None, :, :, :, :] + offset_y
        sample_x = base_x[None, :, :, :, :] + offset_x
        # sample_y, sample_x: [B, H_out, W_out, kH, kW]
        
        # Clip to valid range for bilinear interpolation
        sample_y = mx.clip(sample_y, 0, H_pad - 1)
        sample_x = mx.clip(sample_x, 0, W_pad - 1)
        
        # Bilinear interpolation
        y0 = mx.floor(sample_y).astype(mx.int32)
        y1 = y0 + 1
        x0 = mx.floor(sample_x).astype(mx.int32)
        x1 = x0 + 1
        
        y1 = mx.clip(y1, 0, H_pad - 1)
        x1 = mx.clip(x1, 0, W_pad - 1)
        
        # Compute interpolation weights
        wa = (y1 - sample_y) * (x1 - sample_x)
        wb = (y1 - sample_y) * (sample_x - x0)
        wc = (sample_y - y0) * (x1 - sample_x)
        wd = (sample_y - y0) * (sample_x - x0)
        # wa, wb, wc, wd: [B, H_out, W_out, kH, kW]
        
        # Sample input at four corners
        # x_padded: [B, H_pad, W_pad, C_in]
        # Need to index: x_padded[b, y, x, :]
        
        # Create batch indices
        batch_idx = mx.arange(B)[:, None, None, None, None]
        batch_idx = mx.broadcast_to(batch_idx, (B, H_out, W_out, kH, kW))
        
        # Sample at four corners
        Ia = x_padded[batch_idx, y0, x0]  # [B, H_out, W_out, kH, kW, C_in]
        Ib = x_padded[batch_idx, y0, x1]
        Ic = x_padded[batch_idx, y1, x0]
        Id = x_padded[batch_idx, y1, x1]
        
        # Interpolate
        wa_exp = wa[..., None]  # [B, H_out, W_out, kH, kW, 1]
        wb_exp = wb[..., None]
        wc_exp = wc[..., None]
        wd_exp = wd[..., None]
        
        sampled = wa_exp * Ia + wb_exp * Ib + wc_exp * Ic + wd_exp * Id
        # sampled: [B, H_out, W_out, kH, kW, C_in]
        
        # Apply convolution weights
        # weight: [C_out, kH, kW, C_in]
        # sampled: [B, H_out, W_out, kH, kW, C_in]
        # output: [B, H_out, W_out, C_out]
        
        # Reshape for matrix multiplication
        sampled_flat = sampled.reshape(B * H_out * W_out, kH * kW * C_in)
        weight_flat = self.weight.reshape(C_out, kH * kW * C_in)
        
        output = sampled_flat @ weight_flat.T  # [B*H_out*W_out, C_out]
        output = output.reshape(B, H_out, W_out, C_out)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
