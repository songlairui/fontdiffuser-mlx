"""Attention modules: SpatialTransformer, CrossAttention, ChannelAttnBlock, etc."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation."""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states, gate = mx.split(self.proj(hidden_states), 2, axis=-1)
        return hidden_states * nn.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network with optional GEGLU."""
    
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        
        self.net = []
        self.net.append(GEGLU(dim, inner_dim))
        if dropout > 0:
            self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))
    
    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class CrossAttention(nn.Module):
    """Multi-head cross-attention layer."""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = [
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout) if dropout > 0 else None,
        ]
    
    def reshape_heads_to_batch_dim(self, tensor: mx.array) -> mx.array:
        """[B, seq_len, dim] -> [B*heads, seq_len, dim_head]"""
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.heads, dim // self.heads)
        tensor = tensor.transpose(0, 2, 1, 3).reshape(
            batch_size * self.heads, seq_len, dim // self.heads
        )
        return tensor
    
    def reshape_batch_dim_to_heads(self, tensor: mx.array) -> mx.array:
        """[B*heads, seq_len, dim_head] -> [B, seq_len, dim]"""
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(
            batch_size // self.heads, self.heads, seq_len, dim
        )
        tensor = tensor.transpose(0, 2, 1, 3).reshape(
            batch_size // self.heads, seq_len, dim * self.heads
        )
        return tensor
    
    def __call__(
        self,
        hidden_states: mx.array,
        context: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: [B, seq_len, query_dim]
            context: [B, context_len, context_dim] or None for self-attention
            mask: optional attention mask
        
        Returns:
            Output tensor [B, seq_len, query_dim]
        """
        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)
        
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        
        # Attention
        attention_scores = (query @ key.transpose(0, 2, 1)) * self.scale
        attention_probs = mx.softmax(attention_scores, axis=-1)
        hidden_states = attention_probs @ value
        
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        
        for layer in self.to_out:
            if layer is not None:
                hidden_states = layer(hidden_states)
        
        return hidden_states


class BasicTransformerBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and FFN."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def __call__(
        self, hidden_states: mx.array, context: Optional[mx.array] = None
    ) -> mx.array:
        # Self-attention
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        # Cross-attention
        hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
        # FFN
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class SpatialTransformer(nn.Module):
    """Transformer block for image-like data."""
    
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        num_groups: int = 32,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        
        self.norm = nn.GroupNorm(num_groups=num_groups, dims=in_channels, eps=1e-6)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, 1, padding=0)
        
        self.transformer_blocks = [
            BasicTransformerBlock(
                inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
            )
            for _ in range(depth)
        ]
        
        self.proj_out = nn.Conv2d(inner_dim, in_channels, 1, padding=0)
    
    def __call__(
        self, hidden_states: mx.array, context: Optional[mx.array] = None
    ) -> mx.array:
        """
        Args:
            hidden_states: [B, H, W, C] in MLX format
            context: [B, seq_len, context_dim]
        
        Returns:
            Output tensor [B, H, W, C]
        """
        batch, height, width, channel = hidden_states.shape
        residual = hidden_states
        
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[-1]
        
        # Reshape to sequence: [B, H, W, C] -> [B, H*W, C]
        hidden_states = hidden_states.reshape(batch, height * width, inner_dim)
        
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)
        
        # Reshape back: [B, H*W, C] -> [B, H, W, C]
        hidden_states = hidden_states.reshape(batch, height, width, inner_dim)
        hidden_states = self.proj_out(hidden_states)
        
        return hidden_states + residual


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""
    
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.fc = [
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() if hasattr(nn, 'Sigmoid') else (lambda x: 1 / (1 + mx.exp(-x))),
        ]
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, H, W, C]
        
        Returns:
            Scaled tensor [B, H, W, C]
        """
        b, h, w, c = x.shape
        # Global average pooling: [B, H, W, C] -> [B, C]
        y = mx.mean(x, axis=(1, 2))
        
        for layer in self.fc:
            y = layer(y)
        
        # Reshape and expand: [B, C] -> [B, 1, 1, C] -> [B, H, W, C]
        y = y[:, None, None, :]
        return x * y


class ChannelAttnBlock(nn.Module):
    """Channel attention block for MCA (Multi-level Content Attention)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "silu",
        channel_attn: bool = False,
        reduction: int = 32,
    ):
        super().__init__()
        
        if groups_out is None:
            groups_out = groups
        
        self.norm1 = nn.GroupNorm(num_groups=groups, dims=in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, padding=0)
        
        if non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        else:
            self.nonlinearity = nn.SiLU()
        
        self.channel_attn = channel_attn
        if self.channel_attn:
            self.se_channel_attn = SELayer(channel=in_channels, reduction=reduction)
        
        self.norm3 = nn.GroupNorm(num_groups=groups, dims=in_channels, eps=eps)
        self.down_channel = nn.Conv2d(in_channels, out_channels, 1, padding=0)
    
    def __call__(self, input_tensor: mx.array, content_feature: mx.array) -> mx.array:
        """
        Args:
            input_tensor: [B, H, W, C1]
            content_feature: [B, H, W, C2]
        
        Returns:
            Output tensor [B, H, W, out_channels]
        """
        # Concatenate along channel dimension
        concat_feature = mx.concatenate([input_tensor, content_feature], axis=-1)
        hidden_states = concat_feature
        
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        
        if self.channel_attn:
            hidden_states = self.se_channel_attn(hidden_states)
            hidden_states = hidden_states + concat_feature
        
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.down_channel(hidden_states)
        
        return hidden_states


class OffsetRefStrucInter(nn.Module):
    """Offset-based Reference Structure Interpreter for deformable convolution."""
    
    def __init__(
        self,
        res_in_channels: int,
        style_feat_in_channels: int,
        n_heads: int,
        num_groups: int = 32,
        dropout: float = 0.0,
        gated_ff: bool = True,
    ):
        super().__init__()
        # Style feature projector
        self.style_proj_in = nn.Conv2d(
            style_feat_in_channels, style_feat_in_channels, 1, padding=0
        )
        self.gnorm_s = nn.GroupNorm(
            num_groups=num_groups, dims=style_feat_in_channels, eps=1e-6
        )
        self.ln_s = nn.LayerNorm(style_feat_in_channels)
        
        # Content feature projector
        self.content_proj_in = nn.Conv2d(
            res_in_channels, res_in_channels, 1, padding=0
        )
        self.gnorm_c = nn.GroupNorm(
            num_groups=num_groups, dims=res_in_channels, eps=1e-6
        )
        self.ln_c = nn.LayerNorm(res_in_channels)
        
        # Cross-attention
        self.cross_attention = CrossAttention(
            query_dim=style_feat_in_channels,
            context_dim=res_in_channels,
            heads=n_heads,
            dim_head=res_in_channels,
            dropout=dropout,
        )
        
        # FFN
        self.ff = FeedForward(style_feat_in_channels, dropout=dropout, glu=gated_ff)
        self.ln_ff = nn.LayerNorm(style_feat_in_channels)
        
        self.gnorm_out = nn.GroupNorm(
            num_groups=num_groups, dims=style_feat_in_channels, eps=1e-6
        )
        # Output: 1*2*3*3 = 18 channels for 3x3 deformable conv offsets
        self.proj_out = nn.Conv2d(style_feat_in_channels, 18, 1, padding=0)
    
    def __call__(
        self, res_hidden_states: mx.array, style_content_hidden_states: mx.array
    ) -> mx.array:
        """
        Args:
            res_hidden_states: [B, H, W, C_res]
            style_content_hidden_states: [B, H, W, C_style]
        
        Returns:
            offset: [B, H, W, 18] for 3x3 deformable conv
        """
        batch, height, width, c_channel = res_hidden_states.shape
        _, _, _, s_channel = style_content_hidden_states.shape
        
        # Style projector
        style_content_hidden_states = self.gnorm_s(style_content_hidden_states)
        style_content_hidden_states = self.style_proj_in(style_content_hidden_states)
        style_content_hidden_states = style_content_hidden_states.reshape(
            batch, height * width, s_channel
        )
        style_content_hidden_states = self.ln_s(style_content_hidden_states)
        
        # Content projector
        res_hidden_states = self.gnorm_c(res_hidden_states)
        res_hidden_states = self.content_proj_in(res_hidden_states)
        res_hidden_states = res_hidden_states.reshape(
            batch, height * width, c_channel
        )
        res_hidden_states = self.ln_c(res_hidden_states)
        
        # Cross-attention: style queries content
        hidden_states = self.cross_attention(
            style_content_hidden_states, context=res_hidden_states
        )
        
        # FFN
        hidden_states = self.ff(self.ln_ff(hidden_states)) + hidden_states
        
        # Reshape back to image: [B, H*W, C] -> [B, H, W, C]
        _, _, c = hidden_states.shape
        reshape_out = hidden_states.reshape(batch, height, width, c)
        
        # Project to offset
        reshape_out = self.gnorm_out(reshape_out)
        offset_out = self.proj_out(reshape_out)
        
        return offset_out
