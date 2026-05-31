"""Timestep embeddings for diffusion models."""

import math
import mlx.core as mx
import mlx.nn as nn


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> mx.array:
    """Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1-D array of N indices
        embedding_dim: dimension of the output
        flip_sin_to_cos: whether to flip sine and cosine
        downscale_freq_shift: frequency shift
        scale: scaling factor
        max_period: controls minimum frequency
    
    Returns:
        Tensor of shape [N, embedding_dim]
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    
    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    
    # Scale embeddings
    emb = scale * emb
    
    # Concat sine and cosine
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    
    # Flip if needed
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    
    # Zero pad if odd dimension
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    
    return emb


class Timesteps(nn.Module):
    """Timestep embedding layer."""
    
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
    
    def __call__(self, timesteps: mx.array) -> mx.array:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding."""
    
    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else None
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
    
    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample
