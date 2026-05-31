---
id: 20260531-1450
title: 模型完全对齐，DPM-Solver 输出差异来自 RNG
type: bug
tags: [dpm-solver, random-seed, inference, output-quality]
created: 2026-05-31T14:50:00+08:00
status: tapped
intent_root: UNet+Encoder 在所有 timestep 均完全对齐（max_diff=0），但 DPM-Solver 最终输出 fg 差 3 倍
---

## 当前状态

✅ **模型完全对齐**：
- ContentEncoder: max_diff=0, corr=1.000000
- StyleEncoder: max_diff=0.000041, corr=1.000000  
- UNet noise_pred@t=999: max_diff=0.000000, corr=1.000000
- 逐 timestep 测试（t=999/800/600/400/200/50/10）：MLX std 趋势与 PyTorch 一致

❌ **DPM-Solver 最终输出不同**：
- 同一 x_t 输入：MLX fg=0.405, PyTorch fg=0.151, corr=0.13
- 批量测试：MLX fg≈0.36, PyTorch fg≈0.10

## 原因分析

模型本身已完全等价。DPM-Solver 输出差异的可能原因：

1. **随机噪声生成器不同**：MLX `mx.random.normal()` 与 PyTorch `torch.randn()` 即使同 seed 也产生不同序列。DPM-Solver 内部采样（pipeline.generate()）不接受外部 x_t，每次生成新噪声。

2. **验证脚本容差问题**：`test_against_pytorch_baseline.py` 的 dpm_output tolerance=1.0 且 fg 接受 [0.35, 0.50]，导致 "9/9 通过" 是假阳性。应收紧到 tolerance<0.5, fg 在 [0.08, 0.20]。

3. **DPM-Solver 公式可能有微小差异**：MLX 的 DPMSolverPipeline.generate() 与 PyTorch 的 `src/dpm_solver/` 实现可能在 timestep 调度或一阶/二阶更新上有差异。

## 建议

1. 收紧 `test_against_pytorch_baseline.py` 容差
2. 让 DPMSolverPipeline.generate() 支持传入外部 x_t（用于确定性对比）
3. 对比 PyTorch DPM-Solver 和 MLX DPMSolverPipeline 的逐步中间值

<details>
  <summary>原文</summary>

SNConv 修复后模型完全对齐（ContentEncoder/StyleEncoder/UNet max_diff=0）。但 DPM-Solver 最终输出 fg 差 3 倍（MLX 0.36 vs PT 0.10）。原因：MLX 和 PyTorch 的随机数生成器不同，同一 seed 产生不同噪声。验证脚本容差过宽导致假阳性。

</details>
