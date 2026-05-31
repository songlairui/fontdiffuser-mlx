#!/usr/bin/env python3
"""Test script to validate FontDiffuser MLX implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def test_content_encoder():
    """测试 ContentEncoder"""
    print("\n" + "="*60)
    print("测试 1: ContentEncoder")
    print("="*60)
    
    from mlx_fd.encoders import ContentEncoder
    
    encoder = ContentEncoder(
        G_ch=64,
        resolution=96,
    )
    
    # 创建测试输入
    x = mx.random.normal((1, 96, 96, 3))
    
    # 前向传播
    print("输入形状:", x.shape)
    h, residual_features = encoder(x)
    
    print("输出特征形状:", h.shape)
    print("残差特征数量:", len(residual_features))
    for i, feat in enumerate(residual_features):
        print(f"  残差特征 {i}: {feat.shape}")
    
    print("✓ ContentEncoder 测试通过")
    return True


def test_style_encoder():
    """测试 StyleEncoder"""
    print("\n" + "="*60)
    print("测试 2: StyleEncoder")
    print("="*60)
    
    from mlx_fd.encoders import StyleEncoder
    
    encoder = StyleEncoder(
        G_ch=64,
        resolution=96,
    )
    
    # 创建测试输入
    x = mx.random.normal((1, 96, 96, 3))
    
    # 前向传播
    print("输入形状:", x.shape)
    style_img_feature, _, style_residual_features = encoder(x)
    
    print("风格特征形状:", style_img_feature.shape)
    print("残差特征数量:", len(style_residual_features))
    for i, feat in enumerate(style_residual_features):
        print(f"  残差特征 {i}: {feat.shape}")
    
    # 测试重塑为序列格式
    b, h, w, c = style_img_feature.shape
    style_hidden_states = style_img_feature.reshape(b, h * w, c)
    print("风格隐藏状态（序列格式）:", style_hidden_states.shape)
    
    print("✓ StyleEncoder 测试通过")
    return True


def test_unet():
    """测试 UNet"""
    print("\n" + "="*60)
    print("测试 3: UNet")
    print("="*60)
    
    from mlx_fd.unet import UNet
    from mlx_fd.encoders import ContentEncoder, StyleEncoder
    
    # 创建编码器以获取正确的特征维度
    content_enc = ContentEncoder(G_ch=64, resolution=96)
    style_enc = StyleEncoder(G_ch=64, resolution=96)
    
    unet = UNet(
        sample_size=96,
        in_channels=3,
        out_channels=3,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        cross_attention_dim=1024,
        attention_head_dim=1,
        content_encoder_downsample_size=3,
        content_start_channel=64,
    )
    
    # 创建测试输入
    x_t = mx.random.normal((1, 96, 96, 3))
    timesteps = mx.array([500])
    content_img = mx.random.normal((1, 96, 96, 3))
    style_img = mx.random.normal((1, 96, 96, 3))
    
    # 使用编码器生成正确的特征
    style_feat, _, _ = style_enc(style_img)
    b, h, w, c = style_feat.shape
    style_hidden = style_feat.reshape(b, h*w, c)
    
    content_feat, content_res = content_enc(content_img)
    content_res.append(content_feat)
    
    style_content_feat, style_content_res = content_enc(style_img)
    style_content_res.append(style_content_feat)
    
    encoder_hidden_states = [style_feat, content_res, style_hidden, style_content_res]
    
    print("输入 x_t 形状:", x_t.shape)
    print("时间步:", timesteps)
    print("编码器隐藏状态:")
    print(f"  style_feat: {style_feat.shape}")
    print(f"  content_res: {[r.shape for r in content_res]}")
    print(f"  style_hidden: {style_hidden.shape}")
    print(f"  style_content_res: {[r.shape for r in style_content_res]}")
    
    # 前向传播
    noise_pred, offset_out = unet(
        x_t,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        content_encoder_downsample_size=3,
    )
    
    print("\n输出 noise_pred 形状:", noise_pred.shape)
    print("偏移损失:", offset_out)
    
    print("✓ UNet 测试通过")
    return True


def test_full_model():
    """测试完整模型"""
    print("\n" + "="*60)
    print("测试 4: 完整 FontDiffuserModel")
    print("="*60)
    
    from mlx_fd.model import FontDiffuserModel
    from mlx_fd.unet import UNet
    from mlx_fd.encoders import ContentEncoder, StyleEncoder
    
    # 创建模型
    content_encoder = ContentEncoder(G_ch=64, resolution=96)
    style_encoder = StyleEncoder(G_ch=64, resolution=96)
    unet = UNet(
        sample_size=96,
        in_channels=3,
        out_channels=3,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        cross_attention_dim=1024,
        attention_head_dim=1,
        content_encoder_downsample_size=3,
        content_start_channel=64,
    )
    
    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
    )
    
    # 创建测试输入
    x_t = mx.random.normal((1, 96, 96, 3))
    timesteps = mx.array([500])
    content_img = mx.random.normal((1, 96, 96, 3))
    style_img = mx.random.normal((1, 96, 96, 3))
    
    print("输入:")
    print(f"  x_t: {x_t.shape}")
    print(f"  timesteps: {timesteps}")
    print(f"  content_img: {content_img.shape}")
    print(f"  style_img: {style_img.shape}")
    
    # 前向传播
    noise_pred, offset_out = model(
        x_t,
        timesteps,
        style_images=style_img,
        content_images=content_img,
        content_encoder_downsample_size=3,
    )
    
    print("\n输出:")
    print(f"  noise_pred: {noise_pred.shape}")
    print(f"  offset_out: {offset_out}")
    
    print("✓ 完整模型测试通过")
    return True


def test_scheduler():
    """测试调度器"""
    print("\n" + "="*60)
    print("测试 5: DDPM 调度器")
    print("="*60)
    
    from mlx_fd.scheduler import DDPMScheduler
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="scaled_linear",
    )
    
    # 测试加噪
    original = mx.random.normal((1, 96, 96, 3))
    noise = mx.random.normal((1, 96, 96, 3))
    timesteps = mx.array([100, 200, 300])
    
    print("原始样本形状:", original.shape)
    print("噪声形状:", noise.shape)
    print("时间步:", timesteps)
    
    noisy = scheduler.add_noise(original, noise, timesteps)
    
    print("加噪后形状:", noisy.shape)
    print("✓ 调度器测试通过")
    return True


def test_deform_conv():
    """测试可变形卷积"""
    print("\n" + "="*60)
    print("测试 6: DeformConv2d")
    print("="*60)
    
    from mlx_fd.deform_conv import DeformConv2d
    
    dcn = DeformConv2d(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        padding=1,
    )
    
    # 创建测试输入
    x = mx.random.normal((1, 24, 24, 64))
    offset = mx.random.normal((1, 24, 24, 18))  # 2 * 3 * 3 = 18
    
    print("输入形状:", x.shape)
    print("偏移形状:", offset.shape)
    
    # 前向传播
    output = dcn(x, offset)
    
    print("输出形状:", output.shape)
    print("✓ DeformConv2d 测试通过")
    return True


def create_test_images():
    """创建测试图像"""
    print("\n" + "="*60)
    print("测试 7: 创建测试图像")
    print("="*60)
    
    # 创建内容图像（标准字体）
    content_img = Image.new('RGB', (96, 96), color='white')
    draw = ImageDraw.Draw(content_img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 64)
    except:
        font = ImageFont.load_default()
    
    draw.text((16, 16), "永", fill='black', font=font)
    content_img.save('test_content.png')
    
    # 创建风格图像（模拟书法）
    style_img = Image.new('RGB', (96, 96), color='white')
    draw = ImageDraw.Draw(style_img)
    draw.text((16, 16), "永", fill='black', font=font)
    style_img.save('test_style.png')
    
    print("✓ 测试图像已创建: test_content.png, test_style.png")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("FontDiffuser MLX 测试套件")
    print("="*60)
    
    tests = [
        test_content_encoder,
        test_style_encoder,
        test_unet,
        test_full_model,
        test_scheduler,
        test_deform_conv,
        create_test_images,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n✗ {test.__name__} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！FontDiffuser MLX 实现完整可用。")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查上述错误。")


if __name__ == "__main__":
    main()
