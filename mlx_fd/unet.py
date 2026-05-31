"""UNet model for FontDiffuser."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Union

from .embeddings import Timesteps, TimestepEmbedding
from .unet_blocks import (
    DownBlock2D,
    MCADownBlock2D,
    UNetMidMCABlock2D,
    UpBlock2D,
    StyleRSIUpBlock2D,
)


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    attn_num_head_channels: int,
    resnet_groups: int = 32,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: int = 1,
    channel_attn: bool = False,
    content_channel: int = 32,
    reduction: int = 32,
):
    """Factory function for down blocks."""
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
        )
    elif down_block_type == "MCADownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim required for MCADownBlock2D")
        return MCADownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            channel_attn=channel_attn,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            content_channel=content_channel,
            reduction=reduction,
        )
    else:
        raise ValueError(f"Unknown down block type: {down_block_type}")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    attn_num_head_channels: int,
    upblock_index: int,
    resnet_groups: int = 32,
    cross_attention_dim: Optional[int] = None,
    structure_feature_begin: int = 64,
):
    """Factory function for up blocks."""
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
        )
    elif up_block_type == "StyleRSIUpBlock2D":
        return StyleRSIUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            structure_feature_begin=structure_feature_begin,
            upblock_index=upblock_index,
        )
    else:
        raise ValueError(f"Unknown up block type: {up_block_type}")


class UNet(nn.Module):
    """FontDiffuser UNet with MCA (Multi-level Content Attention).
    
    All tensors are in NHWC format: [B, H, W, C].
    """
    
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "DownBlock2D",
            "MCADownBlock2D",
            "MCADownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "StyleRSIUpBlock2D",
            "StyleRSIUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int] = (64, 128, 256, 512),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1024,
        attention_head_dim: int = 1,
        channel_attn: bool = True,
        content_encoder_downsample_size: int = 3,
        content_start_channel: int = 64,
        reduction: int = 32,
    ):
        super().__init__()
        
        self.content_encoder_downsample_size = content_encoder_downsample_size
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        
        # Time embeddings
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(
            block_out_channels[0], time_embed_dim
        )
        
        # Down blocks
        self.down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            if i != 0:
                content_channel = content_start_channel * (2 ** (i - 1))
            else:
                content_channel = 0
            
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                content_channel=content_channel,
                reduction=reduction,
                channel_attn=channel_attn,
            )
            self.down_blocks.append(down_block)
        
        # Mid block
        self.mid_block = UNetMidMCABlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            channel_attn=channel_attn,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            content_channel=content_start_channel
            * (2 ** (content_encoder_downsample_size - 1)),
            reduction=reduction,
        )
        
        # Count upsamplers
        self.num_upsamplers = 0
        
        # Up blocks
        reversed_block_out_channels = list(reversed(block_out_channels))
        self.up_blocks = []
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False
            
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                upblock_index=i,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        
        # Output
        self.conv_norm_out = nn.GroupNorm(
            dims=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)
    
    def __call__(
        self,
        sample: mx.array,
        timestep: Union[mx.array, float, int],
        encoder_hidden_states: list,
        content_encoder_downsample_size: int = 3,
        return_dict: bool = False,
    ):
        """
        Args:
            sample: Noisy input [B, H, W, C] in NHWC format
            timestep: Diffusion timestep [B] or scalar
            encoder_hidden_states: [style_img_feature, content_residual_features, 
                                    style_hidden_states, style_content_res_features]
            content_encoder_downsample_size: Content encoder downsample size
            return_dict: Whether to return a dict
        
        Returns:
            (noise_pred, offset_out_sum) or UNetOutput
        """
        # 1. Time embedding
        timesteps = timestep
        if not isinstance(timesteps, mx.array):
            timesteps = mx.array([timesteps], dtype=mx.int32)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        
        # Broadcast to batch
        timesteps = mx.broadcast_to(timesteps, (sample.shape[0],))
        
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.astype(sample.dtype)
        emb = self.time_embedding(t_emb)
        
        # 2. Pre-process
        sample = self.conv_in(sample)
        
        # 3. Down
        down_block_res_samples = (sample,)
        for index, down_block in enumerate(self.down_blocks):
            if isinstance(down_block, MCADownBlock2D):
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    index=index,
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=emb)
            
            down_block_res_samples += res_samples
        
        # 4. Mid
        sample = self.mid_block(
            sample,
            emb,
            index=content_encoder_downsample_size,
            encoder_hidden_states=encoder_hidden_states,
        )
        
        # 5. Up
        offset_out_sum = 0
        for i, up_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]
            
            if isinstance(up_block, StyleRSIUpBlock2D):
                sample, offset_out = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    style_structure_features=encoder_hidden_states[3],
                    encoder_hidden_states=encoder_hidden_states[2],
                )
                offset_out_sum += offset_out
            else:
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )
        
        # 6. Post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        return sample, offset_out_sum
