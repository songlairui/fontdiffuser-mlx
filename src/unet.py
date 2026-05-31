"""FontDiffuser MLX — UNet 主模型。

废弃原型：当前结构与上游 FontDiffuser 不等价，不应作为主线继续修补。
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional

from .blocks import (
    ResBlock, CrossAttention, SpatialTransformer, ChannelAttnBlock,
    Downsample, Upsample, DeformConv2dSimple,
    TimestepEmbedding, get_timestep_embedding,
)


class DownBlock(nn.Module):
    """简单 DownBlock: ResBlock × N + optional Downsample."""

    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, num_layers: int = 1,
                 add_downsample: bool = True):
        super().__init__()
        self.resnets = [ResBlock(in_ch if i == 0 else out_ch, out_ch, temb_ch) for i in range(num_layers)]
        self.downsample = Downsample(out_ch) if add_downsample else None

    def __call__(self, x: mx.array, temb: mx.array):
        outputs = []
        for resnet in self.resnets:
            x = resnet(x, temb)
            outputs.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
            outputs.append(x)
        return x, outputs


class MCADownBlock(nn.Module):
    """MCADownBlock: ResBlock + CrossAttn + ChannelAttn + optional Downsample."""

    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, context_dim: int,
                 content_ch: int = 0, num_layers: int = 1, add_downsample: bool = True,
                 channel_attn: bool = False, reduction: int = 32):
        super().__init__()
        self.resnets = []
        self.attentions = []
        self.content_attentions = []
        self.channel_attns = []

        for i in range(num_layers):
            ic = in_ch if i == 0 else out_ch
            self.resnets.append(ResBlock(ic, out_ch, temb_ch))
            self.attentions.append(SpatialTransformer(out_ch, context_dim, heads=8, dim_head=out_ch // 8))
            if content_ch > 0:
                self.content_attentions.append(SpatialTransformer(out_ch, content_ch, heads=8, dim_head=out_ch // 8))
            else:
                self.content_attentions.append(None)
            if channel_attn:
                self.channel_attns.append(ChannelAttnBlock(out_ch, reduction))
            else:
                self.channel_attns.append(None)

        self.downsample = Downsample(out_ch) if add_downsample else None

    def __call__(self, x: mx.array, temb: mx.array,
                 style_hidden: mx.array, content_feat: mx.array | None = None):
        outputs = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            x = self.attentions[i](x, style_hidden)
            if self.content_attentions[i] is not None and content_feat is not None:
                B, H, W, C = x.shape
                ctx = content_feat.reshape(B, -1, content_feat.shape[-1])
                x = self.content_attentions[i](x, ctx)
            if self.channel_attns[i] is not None:
                x = self.channel_attns[i](x)
            outputs.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
            outputs.append(x)
        return x, outputs


class MidBlock(nn.Module):
    """UNetMidMCABlock2D: ResBlock + CrossAttn + ResBlock."""

    def __init__(self, in_ch: int, temb_ch: int, context_dim: int,
                 content_ch: int = 0, channel_attn: bool = False, reduction: int = 32):
        super().__init__()
        self.resnet1 = ResBlock(in_ch, in_ch, temb_ch)
        self.attn = SpatialTransformer(in_ch, context_dim, heads=8, dim_head=in_ch // 8)
        self.content_attn = SpatialTransformer(in_ch, content_ch, heads=8, dim_head=in_ch // 8) if content_ch > 0 else None
        self.channel_attn = ChannelAttnBlock(in_ch, reduction) if channel_attn else None
        self.resnet2 = ResBlock(in_ch, in_ch, temb_ch)

    def __call__(self, x: mx.array, temb: mx.array,
                 style_hidden: mx.array, content_feat: mx.array | None = None):
        x = self.resnet1(x, temb)
        x = self.attn(x, style_hidden)
        if self.content_attn is not None and content_feat is not None:
            B, H, W, C = x.shape
            ctx = content_feat.reshape(B, -1, content_feat.shape[-1])
            x = self.content_attn(x, ctx)
        if self.channel_attn is not None:
            x = self.channel_attn(x)
        x = self.resnet2(x, temb)
        return x


class StyleRSIUpBlock(nn.Module):
    """StyleRSIUpBlock: ResBlock + CrossAttn + DeformConv + optional Upsample.

    使用标准 Conv2d 近似 DeformConv2d（微调时冻结此层）。
    """

    def __init__(self, in_ch: int, out_ch: int, prev_out_ch: int, temb_ch: int,
                 context_dim: int, num_layers: int = 2, add_upsample: bool = True):
        super().__init__()
        self.resnets = []
        self.attentions = []
        self.dcn_deforms = []

        for i in range(num_layers):
            ic = (in_ch + out_ch) if i == 0 else out_ch
            if i > 0:
                ic = (prev_out_ch + out_ch) if i == 1 else out_ch
            self.resnets.append(ResBlock(ic, out_ch, temb_ch))
            self.attentions.append(SpatialTransformer(out_ch, context_dim, heads=8, dim_head=out_ch // 8))
            # DeformConv approximated as standard Conv
            self.dcn_deforms.append(DeformConv2dSimple(out_ch, out_ch))

        self.upsample = Upsample(out_ch) if add_upsample else None

    def __call__(self, x: mx.array, temb: mx.array,
                 res_samples: list[mx.array], style_struct_feats: mx.array):
        offset_out = mx.array(0.0)
        for i, resnet in enumerate(self.resnets):
            # Concatenate skip connection
            res = res_samples[i]
            x = mx.concatenate([x, res], axis=-1)
            x = resnet(x, temb)
            # Cross-attention with style structure features
            B, H, W, C = x.shape
            style_ctx = style_struct_feats.reshape(B, -1, style_struct_feats.shape[-1])
            x = self.attentions[i](x, style_ctx)
            # Deformable conv (approximated)
            x = self.dcn_deforms[i](x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x, offset_out


class UpBlock(nn.Module):
    """简单 UpBlock: ResBlock × N + optional Upsample."""

    def __init__(self, in_ch: int, out_ch: int, prev_out_ch: int, temb_ch: int,
                 num_layers: int = 2, add_upsample: bool = True):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            if i == 0:
                ic = in_ch + out_ch
            elif i == 1:
                ic = prev_out_ch + out_ch
            else:
                ic = out_ch + out_ch
            self.resnets.append(ResBlock(ic, out_ch, temb_ch))
        self.upsample = Upsample(out_ch) if add_upsample else None

    def __call__(self, x: mx.array, temb: mx.array, res_samples: list[mx.array]):
        for i, resnet in enumerate(self.resnets):
            res = res_samples[i]
            x = mx.concatenate([x, res], axis=-1)
            x = resnet(x, temb)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


# ── UNet ────────────────────────────────────────────────────────

class UNet(nn.Module):
    """FontDiffuser UNet (MLX, NHWC layout)。

    block_out_channels = (64, 128, 256, 512)
    down_blocks: DownBlock2D, MCADownBlock2D × 2, DownBlock2D
    up_blocks: UpBlock2D, StyleRSIUpBlock2D × 2, UpBlock2D
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 block_out_channels: tuple = (64, 128, 256, 512),
                 cross_attention_dim: int = 1024,
                 content_start_channel: int = 64,
                 content_encoder_downsample_size: int = 3,
                 channel_attn: bool = False, reduction: int = 32):
        super().__init__()
        time_embed_dim = block_out_channels[0] * 4  # 256

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # Down blocks: DownBlock, MCADown×2, DownBlock
        self.down_blocks = [
            DownBlock(block_out_channels[0], block_out_channels[0], time_embed_dim, 1, True),
            MCADownBlock(block_out_channels[0], block_out_channels[1], time_embed_dim,
                         cross_attention_dim, content_start_channel, 1, True, channel_attn, reduction),
            MCADownBlock(block_out_channels[1], block_out_channels[2], time_embed_dim,
                         cross_attention_dim, content_start_channel * 2, 1, True, channel_attn, reduction),
            DownBlock(block_out_channels[2], block_out_channels[3], time_embed_dim, 1, False),
        ]

        # Mid block
        mid_content_ch = content_start_channel * (2 ** (content_encoder_downsample_size - 1))
        self.mid_block = MidBlock(block_out_channels[3], time_embed_dim, cross_attention_dim,
                                  mid_content_ch, channel_attn, reduction)

        # Up blocks: UpBlock, StyleRSI×2, UpBlock
        reversed_ch = list(reversed(block_out_channels))  # (512, 256, 128, 64)
        self.up_blocks = [
            UpBlock(reversed_ch[0], reversed_ch[0], reversed_ch[0], time_embed_dim, 2, True),
            StyleRSIUpBlock(reversed_ch[1], reversed_ch[1], reversed_ch[0], time_embed_dim,
                            cross_attention_dim, 2, True),
            StyleRSIUpBlock(reversed_ch[2], reversed_ch[2], reversed_ch[1], time_embed_dim,
                            cross_attention_dim, 2, True),
            UpBlock(reversed_ch[3], reversed_ch[3], reversed_ch[2], time_embed_dim, 2, False),
        ]

        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[0], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def __call__(self, x: mx.array, timestep: mx.array,
                 style_hidden: mx.array, content_residuals: list[mx.array],
                 style_content_residuals: list[mx.array]):
        """
        Args:
            x: [B, H, W, 3] noised image (NHWC)
            timestep: [B] timestep indices
            style_hidden: [B, HW, C] style features for cross-attention
            content_residuals: list of [B, H, W, C] content features at different scales
            style_content_residuals: list of [B, H, W, C] style-content features
        Returns:
            noise_pred: [B, H, W, 3]
        """
        # 1. Time embedding
        t_emb = get_timestep_embedding(timestep, self.time_embedding.linear_1.weight.shape[1])
        emb = self.time_embedding(t_emb)

        # 2. Conv in
        x = self.conv_in(x)

        # 3. Down
        down_samples = [x]
        content_idx = 0
        for i, block in enumerate(self.down_blocks):
            if isinstance(block, MCADownBlock):
                # Get content features for this scale
                cf = content_residuals[content_idx] if content_idx < len(content_residuals) else None
                x, res = block(x, emb, style_hidden, cf)
                content_idx += 1
            else:
                x, res = block(x, emb)
            down_samples.extend(res)

        # 4. Mid
        mid_cf = content_residuals[-1] if content_residuals else None
        x = self.mid_block(x, emb, style_hidden, mid_cf)

        # 5. Up
        offset_sum = mx.array(0.0)
        for i, block in enumerate(self.up_blocks):
            n_res = len(block.resnets)
            res = down_samples[-n_res:]
            down_samples = down_samples[:-n_res]

            if isinstance(block, StyleRSIUpBlock):
                # Use style_content_residuals for deformable conv
                sci = len(style_content_residuals) - 1 - (i - 1)  # index mapping
                sci = max(0, min(sci, len(style_content_residuals) - 1))
                x, off = block(x, emb, res, style_content_residuals[sci])
                offset_sum = offset_sum + off
            else:
                x = block(x, emb, res)

        # 6. Conv out
        x = nn.silu(self.conv_norm_out(x))
        x = self.conv_out(x)
        return x
