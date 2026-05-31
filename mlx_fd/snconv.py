import math
import mlx.core as mx
import mlx.nn as nn


def _power_iteration_sigma(weight_2d: mx.array, u: mx.array, num_itrs: int = 1, eps: float = 1e-12) -> mx.array:
    """weight_2d: [out, in], u: [out]

    Matches upstream SNConv2d.W_() behavior: start from the stored `u0` vector,
    compute `v = W^T u`, normalize, then iterate if requested.
    """
    v = weight_2d.T @ u
    v = v / (mx.linalg.norm(v) + eps)
    for _ in range(num_itrs):
        u = weight_2d @ v
        u = u / (mx.linalg.norm(u) + eps)
        v = weight_2d.T @ u
        v = v / (mx.linalg.norm(v) + eps)
    return u @ (weight_2d @ v)


class SNConv2d(nn.Module):
    """MLX Conv2d with dynamic spectral normalization matching upstream W_()."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        num_itrs: int = 1,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.num_itrs = num_itrs
        self.eps = eps

        if isinstance(kernel_size, tuple):
            kH, kW = kernel_size
        else:
            kH = kernel_size
            kW = kernel_size
        scale = 1.0 / math.sqrt(in_channels * kH * kW)
        self.weight = mx.random.normal((out_channels, kH, kW, in_channels)).astype(mx.float32) * scale
        self.bias = mx.zeros((out_channels,)).astype(mx.float32) if bias else None
        # SN buffer: u0 is populated from checkpoint
        self.u0 = mx.zeros((out_channels,)).astype(mx.float32)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def _normalized_weight(self) -> mx.array:
        # weight: [O, kH, kW, I]
        O, kH, kW, I = self.weight.shape
        W = self.weight.reshape(O, -1).astype(mx.float32)
        sigma = _power_iteration_sigma(W, self.u0.astype(mx.float32), self.num_itrs, self.eps)
        return (self.weight / sigma).astype(self.weight.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        w = self._normalized_weight()
        return mx.conv2d(x, w, self.stride, self.padding, self.dilation, self.groups) + (self.bias if self.bias is not None else 0)
