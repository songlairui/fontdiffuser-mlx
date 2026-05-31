"""UNet building blocks: Down/Mid/Up blocks."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple

from .resnet import ResnetBlock2D, Downsample2D, Upsample2D
from .attention import SpatialTransformer, ChannelAttnBlock, OffsetRefStrucInter
from .deform_conv import DeformConv2d


class DownBlock2D(nn.Module):
    """Basic downsampling block with ResNet layers."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                )
            )
        
        self.resnets = resnets
        
        if add_downsample:
            self.downsamplers = [
                Downsample2D(
                    out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
            ]
        else:
            self.downsamplers = None
    
    def __call__(
        self, hidden_states: mx.array, temb: Optional[mx.array] = None
    ) -> Tuple[mx.array, Tuple[mx.array, ...]]:
        """
        Args:
            hidden_states: [B, H, W, C]
            temb: [B, temb_channels]
        
        Returns:
            (hidden_states, output_states) where output_states contains intermediate features
        """
        output_states = []
        
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)
        
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)
        
        return hidden_states, tuple(output_states)


class MCADownBlock2D(nn.Module):
    """Downsampling block with Multi-level Content Attention (MCA)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        channel_attn: bool = False,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        attn_num_head_channels: int = 1,
        cross_attention_dim: int = 1280,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        content_channel: int = 16,
        reduction: int = 32,
    ):
        super().__init__()
        content_attentions = []
        resnets = []
        style_attentions = []
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=in_ch + content_channel,
                    out_channels=in_ch,
                    groups=resnet_groups,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                )
            )
            
            style_attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )
            )
        
        self.content_attentions = content_attentions
        self.style_attentions = style_attentions
        self.resnets = resnets
        
        if num_layers == 1:
            in_ch = out_channels
        
        if add_downsample:
            self.downsamplers = [
                Downsample2D(
                    in_ch,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
            ]
        else:
            self.downsamplers = None
    
    def __call__(
        self,
        hidden_states: mx.array,
        index: int,
        temb: Optional[mx.array] = None,
        encoder_hidden_states: Optional[list] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, ...]]:
        """
        Args:
            hidden_states: [B, H, W, C]
            index: Content feature index
            temb: [B, temb_channels]
            encoder_hidden_states: [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]
        
        Returns:
            (hidden_states, output_states)
        """
        output_states = []
        
        for content_attn, resnet, style_attn in zip(
            self.content_attentions, self.resnets, self.style_attentions
        ):
            # Content attention
            current_content_feature = encoder_hidden_states[1][index]
            hidden_states = content_attn(hidden_states, current_content_feature)
            
            # ResNet with time embedding
            hidden_states = resnet(hidden_states, temb)
            
            # Style attention
            current_style_feature = encoder_hidden_states[0]
            batch, height, width, channel = current_style_feature.shape
            current_style_feature = current_style_feature.reshape(
                batch, height * width, channel
            )
            hidden_states = style_attn(hidden_states, context=current_style_feature)
            
            output_states.append(hidden_states)
        
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)
        
        return hidden_states, tuple(output_states)


class UNetMidMCABlock2D(nn.Module):
    """UNet middle block with MCA."""
    
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        channel_attn: bool = False,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        attn_num_head_channels: int = 1,
        cross_attention_dim: int = 1280,
        content_channel: int = 256,
        reduction: int = 32,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
            )
        ]
        
        content_attentions = []
        style_attentions = []
        
        for _ in range(num_layers):
            content_attentions.append(
                ChannelAttnBlock(
                    in_channels=in_channels + content_channel,
                    out_channels=in_channels,
                    non_linearity=resnet_act_fn,
                    channel_attn=channel_attn,
                    reduction=reduction,
                )
            )
            
            style_attentions.append(
                SpatialTransformer(
                    in_channels,
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )
            )
            
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                )
            )
        
        self.content_attentions = content_attentions
        self.style_attentions = style_attentions
        self.resnets = resnets
    
    def __call__(
        self,
        hidden_states: mx.array,
        temb: Optional[mx.array] = None,
        encoder_hidden_states: Optional[list] = None,
        index: Optional[int] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: [B, H, W, C]
            temb: [B, temb_channels]
            encoder_hidden_states: [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]
            index: Content feature index
        
        Returns:
            Output tensor [B, H, W, C]
        """
        hidden_states = self.resnets[0](hidden_states, temb)
        
        for content_attn, style_attn, resnet in zip(
            self.content_attentions, self.style_attentions, self.resnets[1:]
        ):
            # Content attention
            current_content_feature = encoder_hidden_states[1][index]
            hidden_states = content_attn(hidden_states, current_content_feature)
            
            # ResNet
            hidden_states = resnet(hidden_states, temb)
            
            # Style attention
            current_style_feature = encoder_hidden_states[0]
            batch, height, width, channel = current_style_feature.shape
            current_style_feature = current_style_feature.reshape(
                batch, height * width, channel
            )
            hidden_states = style_attn(hidden_states, context=current_style_feature)
        
        return hidden_states


class UpBlock2D(nn.Module):
    """Basic upsampling block with ResNet layers."""
    
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []
        
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                )
            )
        
        self.resnets = resnets
        
        if add_upsample:
            self.upsamplers = [
                Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ]
        else:
            self.upsamplers = None
    
    def __call__(
        self,
        hidden_states: mx.array,
        res_hidden_states_tuple: Tuple[mx.array, ...],
        temb: Optional[mx.array] = None,
        upsample_size: Optional[Tuple[int, int]] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: [B, H, W, C]
            res_hidden_states_tuple: Tuple of residual hidden states from encoder
            temb: [B, temb_channels]
            upsample_size: Optional target size for upsampling
        
        Returns:
            Output tensor [B, H*2, W*2, C] if upsampling, else [B, H, W, C]
        """
        for resnet in self.resnets:
            # Pop residual hidden state
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            
            # Concatenate along channel dimension
            hidden_states = mx.concatenate([hidden_states, res_hidden_states], axis=-1)
            hidden_states = resnet(hidden_states, temb)
        
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        
        return hidden_states


class StyleRSIUpBlock2D(nn.Module):
    """Upsampling block with Style Reference Structure Interpreter (RSI) and Deformable Conv."""
    
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        attn_num_head_channels: int = 1,
        cross_attention_dim: int = 1280,
        add_upsample: bool = True,
        structure_feature_begin: int = 64,
        upblock_index: int = 1,
    ):
        super().__init__()
        resnets = []
        attentions = []
        sc_interpreter_offsets = []
        dcn_deforms = []
        
        self.upblock_index = upblock_index
        self.num_layers = num_layers
        
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            
            sc_interpreter_offsets.append(
                OffsetRefStrucInter(
                    res_in_channels=res_skip_channels,
                    style_feat_in_channels=int(structure_feature_begin * 2 / upblock_index),
                    n_heads=attn_num_head_channels,
                    num_groups=resnet_groups,
                )
            )
            
            dcn_deforms.append(
                DeformConv2d(
                    in_channels=res_skip_channels,
                    out_channels=res_skip_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                )
            )
            
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                )
            )
            
            attentions.append(
                SpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                )
            )
        
        self.sc_interpreter_offsets = sc_interpreter_offsets
        self.dcn_deforms = dcn_deforms
        self.attentions = attentions
        self.resnets = resnets
        
        if add_upsample:
            self.upsamplers = [
                Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ]
        else:
            self.upsamplers = None
    
    def __call__(
        self,
        hidden_states: mx.array,
        res_hidden_states_tuple: Tuple[mx.array, ...],
        style_structure_features: list,
        temb: Optional[mx.array] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        upsample_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            hidden_states: [B, H, W, C]
            res_hidden_states_tuple: Tuple of residual hidden states
            style_structure_features: Style structure features from encoder
            temb: [B, temb_channels]
            encoder_hidden_states: [B, seq_len, context_dim]
            upsample_size: Optional target size
        
        Returns:
            (hidden_states, offset_out) where offset_out is the mean absolute offset
        """
        total_offset = 0
        
        # Get style content feature for this upblock
        style_content_feat = style_structure_features[-self.upblock_index - 2]
        
        for i, (sc_inter_offset, dcn_deform, resnet, attn) in enumerate(
            zip(self.sc_interpreter_offsets, self.dcn_deforms, self.resnets, self.attentions)
        ):
            # Pop residual hidden state
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            
            # Compute offset using Style Content Interpreter
            offset = sc_inter_offset(res_hidden_states, style_content_feat)
            
            # Compute offset sum for regularization
            offset_sum = mx.mean(mx.abs(offset))
            total_offset = total_offset + offset_sum
            
            # Apply deformable convolution
            res_hidden_states = dcn_deform(res_hidden_states, offset)
            
            # Concatenate and pass through resnet
            hidden_states = mx.concatenate([hidden_states, res_hidden_states], axis=-1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=encoder_hidden_states)
        
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        
        offset_out = total_offset / self.num_layers
        
        return hidden_states, offset_out
