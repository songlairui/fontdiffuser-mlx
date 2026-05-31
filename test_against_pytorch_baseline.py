#!/usr/bin/env python3
"""验证 MLX 推理输出与 PyTorch 基线的一致性。

用法：
    cd fontdiffuser-mlx
    .venv/bin/python test_against_pytorch_baseline.py

测试数据在 _inbox/testdata/ 中，由 PyTorch 原版生成。
"""

import sys
import os
import numpy as np
import mlx.core as mx
from PIL import Image
mx.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlx_fd.model import FontDiffuserModel, FontDiffuserModelDPM
from mlx_fd.unet import UNet
from mlx_fd.encoders import ContentEncoder, StyleEncoder
from mlx_fd.scheduler import DDPMScheduler, DPMSolverPipeline

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "_inbox", "testdata")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "mlx_weights")
TOLERANCE = 0.1  # max_abs_diff 阈值


def nchw_to_nhwc(arr: np.ndarray) -> np.ndarray:
    """[B, C, H, W] → [B, H, W, C]"""
    return np.transpose(arr, (0, 2, 3, 1))


def nhwc_to_nchw(arr: np.ndarray) -> np.ndarray:
    """[B, H, W, C] → [B, C, H, W]"""
    return np.transpose(arr, (0, 3, 1, 2))


def compare(name: str, pt_arr: np.ndarray, mlx_arr: np.ndarray, tolerance: float = TOLERANCE) -> bool:
    """对比 PyTorch 和 MLX 输出，返回是否通过。"""
    mlx_arr = np.array(mlx_arr)
    # MLX 输出是 NHWC，PyTorch 基线是 NCHW，统一到 NCHW
    if mlx_arr.ndim == 4 and pt_arr.ndim == 4 and mlx_arr.shape != pt_arr.shape:
        mlx_nchw = nhwc_to_nchw(mlx_arr)
        if mlx_nchw.shape == pt_arr.shape:
            mlx_arr = mlx_nchw
    
    diff = np.abs(pt_arr - mlx_arr)
    max_diff = diff.max()
    mean_diff = diff.mean()
    corr = np.corrcoef(pt_arr.flatten(), mlx_arr.flatten())[0, 1]
    
    passed = max_diff < tolerance
    status = "✓" if passed else "✗"
    
    print(f"  {status} {name}:")
    print(f"    PT:  mean={pt_arr.mean():.6f}, std={pt_arr.std():.6f}")
    print(f"    MLX: mean={mlx_arr.mean():.6f}, std={mlx_arr.std():.6f}")
    print(f"    diff: max={max_diff:.6f}, mean={mean_diff:.6f}, corr={corr:.6f}")
    
    if not passed:
        print(f"    ⚠️  max_diff {max_diff:.6f} > tolerance {tolerance}")
    
    return passed


def main():
    print("=" * 60)
    print("FontDiffuser MLX vs PyTorch 基线验证")
    print("=" * 60)
    
    # 检查测试数据
    if not os.path.exists(TESTDATA_DIR):
        print(f"✗ 测试数据目录不存在: {TESTDATA_DIR}")
        print("  请从 _inbox/testdata/ 获取")
        sys.exit(1)
    
    results = []
    
    # 1. 加载模型
    print("\n--- 加载模型 ---")
    content_enc = ContentEncoder(G_ch=64, resolution=96)
    style_enc = StyleEncoder(G_ch=64, resolution=96)
    unet = UNet(
        sample_size=96, in_channels=3, out_channels=3,
        block_out_channels=(64, 128, 256, 512), layers_per_block=2,
        cross_attention_dim=1024, attention_head_dim=1,
        content_encoder_downsample_size=3, content_start_channel=64,
    )
    model = FontDiffuserModel(unet=unet, style_encoder=style_enc, content_encoder=content_enc)
    
    # 加载权重
    content_w = {k: mx.array(v) for k, v in np.load(os.path.join(WEIGHTS_DIR, "content_encoder.npz")).items()}
    style_w = {k: mx.array(v) for k, v in np.load(os.path.join(WEIGHTS_DIR, "style_encoder.npz")).items()}
    unet_w = {k: mx.array(v) for k, v in np.load(os.path.join(WEIGHTS_DIR, "unet.npz")).items()}
    content_enc.load_weights(content_w)
    style_enc.load_weights(style_w, strict=False)
    unet.load_weights(unet_w)
    print("  ✓ 模型和权重加载完成")
    
    # 2. 加载测试输入
    print("\n--- 加载测试输入 ---")
    pt_content = np.load(os.path.join(TESTDATA_DIR, "content_img_nchw.npy"))
    pt_style = np.load(os.path.join(TESTDATA_DIR, "style_img_nchw.npy"))
    pt_xt = np.load(os.path.join(TESTDATA_DIR, "x_t_nchw.npy"))
    
    content_img = mx.array(nchw_to_nhwc(pt_content))
    style_img = mx.array(nchw_to_nhwc(pt_style))
    x_t = mx.array(nchw_to_nhwc(pt_xt))
    print(f"  content_img: {content_img.shape}")
    print(f"  style_img: {style_img.shape}")
    print(f"  x_t: {x_t.shape}")
    
    # 3. 对比 Encoder 输出
    print("\n--- Encoder 输出对比 ---")
    
    # StyleEncoder
    style_feat, _, _ = style_enc(style_img)
    mx.eval(style_feat)
    pt_sf = np.load(os.path.join(TESTDATA_DIR, "style_feat_nchw.npy"))
    results.append(compare("style_feat", pt_sf, np.array(style_feat), tolerance=0.1))
    
    # ContentEncoder
    content_feat, content_res = content_enc(content_img)
    mx.eval(content_feat)
    pt_cf = np.load(os.path.join(TESTDATA_DIR, "content_feat_nchw.npy"))
    results.append(compare("content_feat", pt_cf, np.array(content_feat)))
    
    for i, cr in enumerate(content_res):
        pt_cr = np.load(os.path.join(TESTDATA_DIR, f"content_res_{i}_nchw.npy"))
        results.append(compare(f"content_res[{i}]", pt_cr, np.array(cr)))
    
    # 4. 对比 UNet 单步输出
    print("\n--- UNet noise_pred@t=999 对比 ---")
    b, h, w, c = style_feat.shape
    style_hidden = style_feat.reshape(b, h * w, c)
    cr_list = list(content_res) + [content_feat]
    style_content_feat, style_content_res = content_enc(style_img)
    scr_list = list(style_content_res) + [style_content_feat]
    
    encoder_hidden_states = [style_feat, cr_list, style_hidden, scr_list]
    noise_pred, _ = unet(
        x_t, mx.array([999]),
        encoder_hidden_states=encoder_hidden_states,
        content_encoder_downsample_size=3,
    )
    mx.eval(noise_pred)
    pt_np = np.load(os.path.join(TESTDATA_DIR, "noise_pred_t999_nchw.npy"))
    results.append(compare("noise_pred@t=999", pt_np, np.array(noise_pred), tolerance=0.5))
    
    # 5. 对比完整推理输出
    print("\n--- 完整推理输出对比 ---")
    scheduler = DDPMScheduler(1000, 0.0001, 0.02, "scaled_linear")
    model_dpm = FontDiffuserModelDPM(unet=unet, style_encoder=style_enc, content_encoder=content_enc)
    pipeline = DPMSolverPipeline(
        model=model_dpm, ddpm_train_scheduler=scheduler,
        guidance_type="classifier-free", guidance_scale=1.0,
    )
    
    # Use fixed initial noise for deterministic comparison
    initial_noise = mx.array(np.load(os.path.join(TESTDATA_DIR, 'x_t_nchw.npy')).transpose(0,2,3,1).astype(np.float32))
    result = pipeline.generate(
        content_images=content_img, style_images=style_img,
        batch_size=1, num_inference_step=20,
        content_encoder_downsample_size=3, dm_size=(96, 96),
        initial_noise=initial_noise,
    )
    mx.eval(result)
    
    # 转为图片检查 fg ratio
    arr = np.array(result[0])
    arr_uint8 = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    fg = (arr_uint8 < 128).any(axis=2).sum() / (96 * 96)
    
    pt_dpm = np.load(os.path.join(TESTDATA_DIR, "dpm_output_nchw.npy"))
    results.append(compare("dpm_output", pt_dpm, np.array(result), tolerance=0.25))
    
    print(f"\n  MLX output fg_ratio: {fg:.3f}")
    expected_img = np.array(Image.open(os.path.join(TESTDATA_DIR, "expected_output.png")).convert("L"))
    expected_fg = (expected_img < 128).sum() / expected_img.size
    print(f"  Expected fg_ratio:   {expected_fg:.3f}")
    
    fg_ok = 0.08 < fg < 0.20
    results.append(fg_ok)
    print(f"  {'✓' if fg_ok else '✗'} fg_ratio in [0.08, 0.20]")
    
    # 汇总
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"总计: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 全部通过！MLX 实现与 PyTorch 基线一致。")
    else:
        print(f"⚠️  {total - passed} 项失败，需要修复。")
        sys.exit(1)


if __name__ == "__main__":
    main()
