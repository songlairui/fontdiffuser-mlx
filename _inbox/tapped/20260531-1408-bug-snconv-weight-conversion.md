---
id: 20260531-1408
title: SNConv2d 权重谱归一化缺失导致推理输出发散
type: bug
tags: [weight-conversion, snconv, spectral-norm, encoder, inference]
created: 2026-05-31T14:08:00+08:00
status: tapped
intent_root: MLX 推理输出为纯噪点，PyTorch 原版正常；根因是权重转换时未对 SNConv2d 权重做谱归一化
---

## 问题描述

MLX 推理输出为纯噪点（fg≈50%），PyTorch 原版输出正常字符（fg≈15%）。逐层对比发现 ContentEncoder 第三层输出已完全发散：

| 张量 | PyTorch mean | MLX mean | 倍率 |
|------|-------------|----------|------|
| content_res[0] (input) | 0.772 | 0.772 | 1.0x |
| content_res[1] (DBlock 0) | -0.057 | -0.067 | 1.2x |
| content_res[2] (DBlock 1) | -0.012 | -0.070 | 5.8x |
| content_feat (DBlock 2) | -0.007 | 1.903 | **288x** |
| style_feat | 0.045 | 0.047 | 1.0x (但 std 1.47x) |
| noise_pred@t=999 | -0.027 | -0.006 | — |
| 最终输出 fg | 0.151 | 0.455 | 纯噪点 |

**相关性 0.995** 说明模型结构正确，但数值尺度错误。

## 根因

`weight_converter.py` 的 `convert_content_encoder_weights()` / `convert_style_encoder_weights()` **跳过了 SN buffers（`.u0`, `.sv0`）但未对权重做谱归一化**。

PyTorch 的 `SNConv2d` 在前向传播时：
```python
W_normalized = W / sigma(W)
```
其中 `sigma(W)` 是从 `.sv0` buffer 读取的谱范数。保存的权重是**未归一化的原始权重**。

MLX 版本直接加载原始权重，未除以 sigma，导致每层输出被放大 sigma 倍。3 层累积后发散 100+ 倍。

**实测 sigma 值：**
- ContentEncoder conv1: σ=0.618, conv2: σ=1.961, conv_sc: σ=0.456
- StyleEncoder conv1: σ=0.504, conv2: σ=1.453
- 深层 σ 更大（1.4~3.1），多层累积指数级放大

## 修复方案

在 `weight_converter.py` 的 `convert_content_encoder_weights()` 和 `convert_style_encoder_weights()` 中，对每个 SNConv2d 的 `.weight` 除以其对应的 `.sv0`：

```python
# 在跳过 .sv0 之前，用它来归一化对应的 weight
for key, value in pt_state_dict.items():
    if key.endswith('.sv0'):
        # 找到对应的 weight key
        weight_key = key.replace('.sv0', '.weight')
        if weight_key in pt_state_dict:
            sigma = value[0]  # scalar
            pt_state_dict[weight_key] = pt_state_dict[weight_key] / sigma
        continue  # 不保存 sv0 到 MLX weights
    ...
```

或者在加载后、保存前统一处理。

## 待验证

1. ContentEncoder 9 个 conv 层全部需要归一化（blocks.{0,1,2}.{conv1,conv2,conv_sc}）
2. StyleEncoder 15 个 conv 层（blocks.{0,1,2,3,4}.{conv1,conv2,conv_sc} + last_conv）
3. UNet 不使用 SNConv2d（diffusers 标准 Conv2d），无需处理
4. 修复后逐层 mean/std 应与 PyTorch 基线在 1e-3 以内

## 测试数据

已提供 PyTorch 基线数据到 `_inbox/testdata/`：

| 文件 | 说明 |
|------|------|
| `content_万.png` | 测试内容图（标准字体渲染 "万"） |
| `style_永.png` | 测试风格图（标准字体渲染 "永"） |
| `expected_output.png` | PyTorch 推理预期输出（fg=0.151） |
| `content_img_nchw.npy` | 输入 [1,3,96,96] NCHW |
| `style_img_nchw.npy` | 输入 [1,3,96,96] NCHW |
| `x_t_nchw.npy` | 初始噪声 [1,3,96,96] seed=42 |
| `style_feat_nchw.npy` | StyleEncoder 输出 [1,1024,3,3] |
| `content_feat_nchw.npy` | ContentEncoder 最终输出 [1,256,12,12] |
| `content_res_{0,1,2,3}_nchw.npy` | ContentEncoder 中间残差 |
| `noise_pred_t999_nchw.npy` | UNet 在 t=999 的噪声预测 [1,3,96,96] |
| `dpm_output_nchw.npy` | DPM-Solver++ 20 步最终输出 |

**注意：** `.npy` 文件为 NCHW 格式（PyTorch 原生），MLX 读取时需转置为 NHWC。

## 验收标准

1. 修复后 `noise_pred@t=999` 与 PyTorch 基线的 max_abs_diff < 0.1
2. 修复后 `expected_output.png` 的 fg 在 0.10~0.20 范围内
3. `test_model.py` 全部 7 项测试仍然通过
4. 提供回归测试脚本：加载基线 .npy → MLX 推理 → 逐层对比

## 关联

- GOAL.md：「权重加载必须是硬门槛」「MLX 推理能生成与原版同分布的结果」
- 当前 mlx_weights/ 中的 .npz 文件需要重新生成

<details>
  <summary>原文</summary>

personal-handwriting 项目使用 fontdiffuser-mlx 做字体推理时发现 MLX 输出为纯噪点。对比 PyTorch 原版确认模型结构和权重加载正确，但 ContentEncoder 第三层输出已发散 288 倍。根因是 weight_converter.py 跳过了 SNConv2d 的谱归一化 buffers 但未对权重做归一化处理。

</details>
