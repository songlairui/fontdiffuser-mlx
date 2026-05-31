---
id: 20260531-1430
title: StyleEncoder 输出偏差 2.6 倍，DPM 推理仍为噪点
type: bug
tags: [style-encoder, snconv, power-iteration, inference, divergence]
created: 2026-05-31T14:30:00+08:00
status: tapped
intent_root: SNConv 谱归一化已修复，ContentEncoder 对齐，但 StyleEncoder 输出 std 偏差 2.6 倍，完整推理仍生成噪点而非字符
---

## 问题描述

SNConv 谱归一化修复后，ContentEncoder 输出与 PyTorch 完全一致（max_diff=0）。但 StyleEncoder 输出仍有显著偏差：

| 张量 | PyTorch std | MLX std | 倍率 |
|------|-----------|---------|------|
| style_feat | 0.591 | 1.548 | **2.6x** |

完整推理输出 fg=0.45（应为 0.10~0.15），仍为噪点。

## 已排除

- ✅ Conv2d 计算：纯 Conv2d 对比 max_diff=6e-7，完全一致
- ✅ InstanceNorm：手动实现与 PyTorch InstanceNorm2d max_diff=0
- ✅ 权重归一化 sigma：converter 与 PyTorch power_iteration 的 sigma 值一致（diff=2e-12）
- ✅ 权重映射：blocks.{i}.0.conv → blocks.{i}.conv 映射正确，归一化后 max_diff=0

## 可疑点

1. **DBlock 单层输出差异**：用真实权重测试，conv1 输出 corr=0.99997（几乎一致），但 DBlock 整体输出 max_diff=0.58，corr=0.70。差异在 conv2 之后累积。
2. **PyTorch W_() 归一化行为**：设置归一化权重后调用 W_()，输出 sigma 仍为 1.08（不是 1.0）。说明 PyTorch 的 SNConv2d 在 forward 时会**再次**归一化，即使权重已归一化。这可能是因为 u0 与当前权重不完全对齐。
3. **test_against_pytorch_baseline.py 容差过宽**：dpm_output tolerance=1.0, fg 接受 [0.35, 0.50]，导致 "9/9 通过" 是假阳性。

## 建议排查方向

1. **对比 DBlock 逐层输出**：用 checkpoint 真实权重，分别对比 conv1、relu、conv2、pool、shortcut 各步的输出，定位偏差来源
2. **检查 PyTorch SNConv2d 的 W_() 是否对已归一化权重再次归一化**：如果是，则 MLX 的预归一化策略需要调整——可能需要在 MLX Conv2d 前也加一层动态归一化
3. **收紧验证脚本容差**：dpm_output tolerance 应 ≤ 0.5, fg 应在 [0.08, 0.20]

## 测试数据

PyTorch 基线已在 `_inbox/testdata/` 中，可直接用于对比。

<details>
  <summary>原文</summary>

SNConv 谱归一化修复后 ContentEncoder 对齐了，但 StyleEncoder 输出 std 偏差 2.6 倍（PT 0.59 vs MLX 1.55）。完整推理仍然生成噪点。逐层排查发现 Conv2d 本身一致，但 DBlock 整体输出有差异。怀疑 PyTorch SNConv2d 的 W_() 在 forward 时对已归一化权重再次归一化，导致预归一化策略失效。

</details>
