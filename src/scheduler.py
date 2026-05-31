"""FontDiffuser MLX — DDPM Scheduler + DPM-Solver++ Sampler.

废弃原型：当前采样器没有等价复刻上游 DPM-Solver++ 推理链路。
"""

import mlx.core as mx
import math


def linear_beta_schedule(num_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> mx.array:
    """线性 beta schedule。"""
    return mx.linspace(beta_start, beta_end, num_steps)


def scaled_linear_beta_schedule(num_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> mx.array:
    """Scaled linear beta schedule (FontDiffuser 默认)。"""
    return mx.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2


class DDPMScheduler:
    """DDPM 前向加噪 + 反向去噪。"""

    def __init__(self, num_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
                 schedule: str = "scaled_linear"):
        if schedule == "scaled_linear":
            self.betas = scaled_linear_beta_schedule(num_steps, beta_start, beta_end)
        else:
            self.betas = linear_beta_schedule(num_steps, beta_start, beta_end)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)
        self.num_steps = num_steps

    def add_noise(self, x: mx.array, noise: mx.array, timesteps: mx.array) -> mx.array:
        """前向扩散：x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * noise。"""
        sqrt_alpha = mx.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha = mx.sqrt(1.0 - self.alphas_cumprod[timesteps])
        # [B] → [B, 1, 1, 1] for NHWC broadcast
        while len(sqrt_alpha.shape) < len(x.shape):
            sqrt_alpha = sqrt_alpha[:, None]
            sqrt_one_minus_alpha = sqrt_one_minus_alpha[:, None]
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise

    def step(self, model_output: mx.array, timestep: int, sample: mx.array) -> mx.array:
        """单步去噪：x_{t-1} = (x_t - β_t/√(1-ᾱ_t) * ε_θ) / √α_t + σ_t * z。"""
        beta_t = self.betas[timestep]
        alpha_t = self.alphas[timestep]
        alpha_bar_t = self.alphas_cumprod[timestep]

        pred_x0 = (sample - mx.sqrt(1 - alpha_bar_t) * model_output) / mx.sqrt(alpha_bar_t)
        pred_x0 = mx.clip(pred_x0, -1, 1)

        sigma_t = mx.sqrt(beta_t)
        noise = mx.random.normal(sample.shape)
        return mx.sqrt(alpha_t) * pred_x0 + sigma_t * noise


class DPMSolverPlusPlus:
    """DPM-Solver++ 采样器（二阶 multistep）。"""

    def __init__(self, num_steps: int = 1000, num_inference_steps: int = 20,
                 beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_steps = num_steps
        self.num_inference_steps = num_inference_steps
        betas = scaled_linear_beta_schedule(num_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        self.alphas_cumprod = mx.cumprod(alphas)
        # timesteps for inference (均匀间隔)
        self.timesteps = mx.linspace(num_steps - 1, 0, num_inference_steps).astype(mx.int32)

    def sample(self, model_fn, shape: tuple, cond, guidance_scale: float = 1.0) -> mx.array:
        """运行 DPM-Solver++ 采样。

        Args:
            model_fn: callable(x_t, t, cond) → noise_pred
            shape: (B, H, W, C)
            cond: 条件输入
        Returns:
            x_0: [B, H, W, C]
        """
        x = mx.random.normal(shape)

        for i in range(len(self.timesteps) - 1):
            t = int(self.timesteps[i])
            t_next = int(self.timesteps[i + 1])

            noise_pred = model_fn(x, mx.array([t] * shape[0]), cond)

            # DPM-Solver++ 二阶更新
            alpha_t = self.alphas_cumprod[t]
            alpha_t_next = self.alphas_cumprod[t_next]

            x0_pred = (x - mx.sqrt(1 - alpha_t) * noise_pred) / mx.sqrt(alpha_t)
            x0_pred = mx.clip(x0_pred, -1, 1)

            # 一阶更新
            x = mx.sqrt(alpha_t_next) * x0_pred + mx.sqrt(1 - alpha_t_next) * noise_pred

        return x
