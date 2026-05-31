"""DDPM and DPM-Solver++ schedulers for diffusion sampling."""

import mlx.core as mx
import numpy as np
from typing import Optional, Tuple, Union


def scaled_linear_beta_schedule(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> mx.array:
    """Scaled linear beta schedule for DDPM.
    
    Args:
        num_train_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        Beta schedule [num_train_timesteps]
    """
    betas = np.linspace(
        beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float32
    ) ** 2
    return mx.array(betas)


class DDPMScheduler:
    """DDPM scheduler for training and inference."""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[np.ndarray] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.variance_type = variance_type
        self.clip_sample = clip_sample
        
        if trained_betas is not None:
            self.betas = mx.array(trained_betas)
        elif beta_schedule == "scaled_linear":
            self.betas = scaled_linear_beta_schedule(
                num_train_timesteps, beta_start, beta_end
            )
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)
        
        # For adding noise
        self.sqrt_alphas_cumprod = mx.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """Add noise to samples for training.
        
        Args:
            original_samples: Clean samples [B, H, W, C]
            noise: Noise to add [B, H, W, C]
            timesteps: Timesteps [B]
        
        Returns:
            Noisy samples [B, H, W, C]
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
        sqrt_alpha_prod = sqrt_alpha_prod[:, None, None, None]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, None, None, None]
        
        noisy_samples = (
            sqrt_alpha_prod * original_samples
            + sqrt_one_minus_alpha_prod * noise
        )
        
        return noisy_samples
    
    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
    ) -> mx.array:
        """Single reverse diffusion step.
        
        Args:
            model_output: Predicted noise [B, H, W, C]
            timestep: Current timestep
            sample: Current noisy sample [B, H, W, C]
        
        Returns:
            Denoised sample [B, H, W, C]
        """
        t = timestep
        prev_t = t - 1 if t > 0 else 0
        
        # Compute predicted x_0
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else mx.array(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Predict x_0
        pred_original_sample = (
            sample - mx.sqrt(beta_prod_t) * model_output
        ) / mx.sqrt(alpha_prod_t)
        
        if self.clip_sample:
            pred_original_sample = mx.clip(pred_original_sample, -1, 1)
        
        # Compute mean
        pred_original_sample_coeff = (
            mx.sqrt(alpha_prod_t_prev) * self.betas[t] / beta_prod_t
        )
        current_sample_coeff = (
            mx.sqrt(self.alphas[t]) * beta_prod_t_prev / beta_prod_t
        )
        
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )
        
        # Add noise if not final step
        if t > 0:
            noise = mx.random.normal(sample.shape)
            variance = (beta_prod_t_prev / beta_prod_t) * self.betas[t]
            variance = mx.sqrt(variance)
            pred_prev_sample = pred_prev_sample + variance * noise
        
        return pred_prev_sample


class DPMSolverPipeline:
    """DPM-Solver++ pipeline for fast sampling."""
    
    def __init__(
        self,
        model,
        ddpm_train_scheduler: DDPMScheduler,
        model_type: str = "noise",
        guidance_type: str = "classifier-free",
        guidance_scale: float = 7.5,
    ):
        self.model = model
        self.scheduler = ddpm_train_scheduler
        self.model_type = model_type
        self.guidance_type = guidance_type
        self.guidance_scale = guidance_scale
    
    def generate(
        self,
        content_images: mx.array,
        style_images: mx.array,
        batch_size: int = 1,
        order: int = 1,
        num_inference_step: int = 20,
        content_encoder_downsample_size: int = 3,
        t_start: Optional[int] = None,
        t_end: Optional[int] = None,
        dm_size: Tuple[int, int] = (96, 96),
        algorithm_type: str = "dpmsolver++",
        skip_type: str = "time_uniform",
        method: str = "multistep",
        correcting_x0_fn: Optional[str] = None,
        initial_noise: Optional[mx.array] = None,
    ) -> mx.array:
        """Generate samples using DPM-Solver++.
        
        Args:
            content_images: Content images [B, H, W, C]
            style_images: Style images [B, H, W, C]
            batch_size: Batch size
            order: Order of DPM-Solver (1, 2, or 3)
            num_inference_step: Number of inference steps
            content_encoder_downsample_size: Content encoder downsample size
            t_start: Starting timestep
            t_end: Ending timestep
            dm_size: Output image size
            algorithm_type: Algorithm type
            skip_type: Skip type for timesteps
            method: Method type
            correcting_x0_fn: Correction function for x_0
        
        Returns:
            Generated samples [B, H, W, C]
        """
        # Initialize from noise (supports external initial noise for deterministic testing)
        if initial_noise is not None:
            x = initial_noise
        else:
            x = mx.random.normal((batch_size, dm_size[0], dm_size[1], 3))
        
        # Setup timesteps
        if skip_type == "time_uniform":
            timesteps = np.linspace(
                self.scheduler.num_train_timesteps - 1 if t_start is None else t_start,
                0 if t_end is None else t_end,
                num_inference_step + 1,
                dtype=np.int32,
            )
        else:
            raise ValueError(f"Unknown skip type: {skip_type}")
        
        # Encode conditions
        cond = (content_images, style_images)
        
        # DPM-Solver++ multistep sampling loop
        # Store model outputs as x0 predictions for multistep updates
        x0_preds = []
        
        for i in range(len(timesteps) - 1):
            t = int(timesteps[i])
            t_next = int(timesteps[i + 1])
            
            # Predict noise
            if self.guidance_type == "classifier-free" and self.guidance_scale > 1.0:
                noise_pred_cond = self.model(
                    x, mx.array([t] * batch_size), cond, content_encoder_downsample_size,
                )
                uncond_content = mx.ones_like(content_images)
                uncond_style = mx.ones_like(style_images)
                uncond_cond = (uncond_content, uncond_style)
                noise_pred_uncond = self.model(
                    x, mx.array([t] * batch_size), uncond_cond, content_encoder_downsample_size,
                )
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.model(
                    x, mx.array([t] * batch_size), cond, content_encoder_downsample_size,
                )
            
            # Convert noise prediction to x0 prediction
            alpha_t = self.scheduler.alphas_cumprod[t]
            alpha_next = self.scheduler.alphas_cumprod[t_next]
            sigma_t = mx.sqrt(1 - alpha_t)
            
            x0_pred = (x - sigma_t * noise_pred) / mx.sqrt(alpha_t)
            
            if correcting_x0_fn == "dynamic_thresholding":
                s = 0.995
                x0_abs = mx.abs(x0_pred)
                threshold = mx.percentile(x0_abs, 99.5, axis=(1, 2, 3), keepdims=True)
                threshold = mx.maximum(threshold, mx.array(1.0))
                x0_pred = mx.clip(x0_pred, -threshold, threshold) / threshold
            else:
                x0_pred = mx.clip(x0_pred, -1, 1)
            
            x0_preds.append(x0_pred)
            
            # DPM-Solver update (first-order for stability)
            # First-order update: x = sqrt(alpha_next) * x0_pred + sqrt(1-alpha_next) * noise_pred
            x = mx.sqrt(alpha_next) * x0_pred + mx.sqrt(1 - alpha_next) * noise_pred
            
            mx.eval(x)
            # Debug logging
            x_np = np.array(x)
            print(f'DPM step {i}: t={t}->{t_next}, mean={x_np.mean():.6f}, std={x_np.std():.6f}')
        
        return x
