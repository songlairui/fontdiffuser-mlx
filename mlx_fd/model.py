"""FontDiffuser model wrapper for MLX."""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional

from .unet import UNet
from .encoders import ContentEncoder, StyleEncoder


class FontDiffuserModel(nn.Module):
    """FontDiffuser model combining UNet, StyleEncoder, and ContentEncoder.
    
    All tensors in NHWC format: [B, H, W, C].
    """
    
    def __init__(
        self,
        unet: UNet,
        style_encoder: StyleEncoder,
        content_encoder: ContentEncoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
    
    def __call__(
        self,
        x_t: mx.array,
        timesteps: mx.array,
        style_images: mx.array,
        content_images: mx.array,
        content_encoder_downsample_size: int = 3,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x_t: Noisy input [B, H, W, C] in NHWC
            timesteps: Diffusion timesteps [B]
            style_images: Style reference images [B, H, W, C]
            content_images: Content images [B, H, W, C]
            content_encoder_downsample_size: Content encoder downsample size
        
        Returns:
            (noise_pred, offset_out_sum)
        """
        # Encode style image
        style_img_feature, _, _ = self.style_encoder(style_images)
        
        # Style hidden states for cross-attention: [B, H*W, C]
        batch, height, width, channel = style_img_feature.shape
        style_hidden_states = style_img_feature.reshape(batch, height * width, channel)
        
        # Encode content image
        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)
        
        # Encode style image through content encoder for structure features
        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)
        
        # Build encoder hidden states
        encoder_hidden_states = [
            style_img_feature,  # [B, H, W, C] - style image feature
            content_residual_features,  # List of content features
            style_hidden_states,  # [B, H*W, C] - for cross-attention
            style_content_res_features,  # List of style structure features
        ]
        
        # Run UNet
        noise_pred, offset_out_sum = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        
        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(nn.Module):
    """FontDiffuser model for DPM-Solver sampling."""
    
    def __init__(
        self,
        unet: UNet,
        style_encoder: StyleEncoder,
        content_encoder: ContentEncoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
    
    def __call__(
        self,
        x_t: mx.array,
        timesteps: mx.array,
        cond: Tuple[mx.array, mx.array],
        content_encoder_downsample_size: int = 3,
    ) -> mx.array:
        """
        Args:
            x_t: Noisy input [B, H, W, C]
            timesteps: Diffusion timesteps [B]
            cond: (content_images, style_images)
            content_encoder_downsample_size: Content encoder downsample size
        
        Returns:
            noise_pred [B, H, W, C]
        """
        content_images = cond[0]
        style_images = cond[1]
        
        # Encode style
        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        
        batch, height, width, channel = style_img_feature.shape
        style_hidden_states = style_img_feature.reshape(batch, height * width, channel)
        
        # Encode content
        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)
        
        # Encode style through content encoder
        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)
        
        encoder_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
        ]
        
        noise_pred, _ = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        
        return noise_pred
