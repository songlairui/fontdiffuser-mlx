"""FontDiffuser MLX — 核心模块。

NHWC 数据布局，MLX 原生。
"""

import mlx.core as mx
import mlx.nn as nn
import math


# ── 基础工具 ────────────────────────────────────────────────────

def upsample_nearest(x: mx.array, scale: int = 2) -> mx.array:
    """最近邻上采样（MLX 无 F.interpolate）。"""
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    return x.reshape(B, H * scale, W * scale, C)


# ── 时间步嵌入 ──────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """正弦时间步编码 → MLP 投影。"""

    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, t_emb: mx.array) -> mx.array:
        t_emb = self.linear_1(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = self.linear_2(t_emb)
        return t_emb


def get_timestep_embedding(timesteps: mx.array, dim: int, flip_sin_to_cos: bool = True) -> mx.array:
    """生成正弦位置编码。timesteps: [B], 返回 [B, dim]。"""
    half_dim = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(half_dim).astype(mx.float32) / half_dim)
    args = timesteps[:, None].astype(mx.float32) * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    return emb


# ── ResBlock ────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """ResnetBlock2D with timestep conditioning. NHWC layout."""

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int,
                 groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, pytorch_compatible=True, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, pytorch_compatible=True, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(temb_channels, out_channels)
        self.use_shortcut = in_channels != out_channels
        if self.use_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        # Timestep conditioning
        h = h + nn.silu(self.time_proj(temb))[:, None, None, :]
        h = nn.silu(self.norm2(h))
        h = self.conv2(h)
        if self.use_shortcut:
            x = self.conv_shortcut(x)
        return x + h


# ── 注意力 ──────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """Cross-attention for UNet. Input: NHWC [B, H, W, C] → flatten → attn → reshape."""

    def __init__(self, query_dim: int, context_dim: int | None = None,
                 heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def __call__(self, x: mx.array, context: mx.array | None = None) -> mx.array:
        """x: [B, N, C], context: [B, S, C_ctx] or None (self-attn)."""
        context = context if context is not None else x
        B, N, _ = x.shape
        h = self.heads
        d = x.shape[-1] // h

        q = self.to_q(x).reshape(B, N, h, d).transpose(0, 2, 1, 3)
        S = context.shape[1]
        k = self.to_k(context).reshape(B, S, h, d).transpose(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, S, h, d).transpose(0, 2, 1, 3)

        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * self.scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    """GEGLU FeedForward."""

    def __init__(self, dim: int, dim_out: int | None = None):
        super().__init__()
        dim_out = dim_out or dim
        self.proj = nn.Linear(dim, dim_out * 2)
        self.out = nn.Linear(dim_out, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        x = x * nn.gelu(gate)
        return self.out(x)


class TransformerBlock(nn.Module):
    """Self-attn + Cross-attn + FFN."""

    def __init__(self, dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, heads=heads, dim_head=dim_head)  # self-attn
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads=heads, dim_head=dim_head)  # cross-attn
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    """Conv → Transformer → Conv. 输入 NHWC [B, H, W, C]."""

    def __init__(self, channels: int, context_dim: int, heads: int = 8, dim_head: int = 64, depth: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, pytorch_compatible=True)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = [TransformerBlock(channels, context_dim, heads, dim_head) for _ in range(depth)]
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        B, H, W, C = x.shape
        h = self.norm(x)
        h = self.proj_in(h)
        h = h.reshape(B, H * W, C)
        for block in self.transformer_blocks:
            h = block(h, context)
        h = h.reshape(B, H, W, C)
        h = self.proj_out(h)
        return x + h


# ── 下采样 / 上采样 ────────────────────────────────────────────

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(upsample_nearest(x))


# ── Channel Attention ──────────────────────────────────────────

class ChannelAttnBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        w = mx.mean(x, axis=(1, 2))  # [B, C]
        w = nn.silu(self.fc1(w))
        w = mx.sigmoid(self.fc2(w))
        return x * w[:, None, None, :]


# ── Deformable Conv (简化版：标准 Conv + 冻结) ─────────────────

class DeformConv2dSimple(nn.Module):
    """标准 Conv2d 近似 DeformConv2d。

    微调时冻结此层（保留预训练权重），只训练注意力和编码器。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)
