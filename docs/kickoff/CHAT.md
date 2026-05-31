# 对话记录：为什么启动 fontdiffuser-mlx

## 背景

用户在 personal-handwriting 项目中构建手写字体，使用 FontDiffuser 扩散模型生成字形。

## 关键对话

**用户**：接手 HAND_OFF.md 使用扩散的方式，对当前 preview 页面示例区域的字体重新生成。

**结果**：用 FontDiffuser 默认权重生成了 222 个字形，但风格变成了接近参考字体的印刷体，丢失了手写风格。

**用户**：好多错误。风格也变了。似乎跟某个基准字体一样了。原字的风格变化太大，都看不出来保留了。扩散出来的，经常多一个横，或者多一个竖。

**分析**：FontDiffuser 默认权重只学会"通用手写"风格，需要用用户自己的手写样本微调。

**用户**：方案 1（微调 FontDiffuser）

**尝试**：准备训练数据 → 修改 train.py → 启动 MPS 训练

**问题**：MPS 训练失败 — `deform_conv2d_backward` 不支持 MPS

**用户**：怎么这么慢？！现在我在网上看到说有了 mps 完整支持 cuda 的技术了。你去了解一下

**调查**：
- PyTorch 2.11 (2025.3) 扩展了 MPS 算子支持
- PyTorch 2.12 (2025.5) 统一了 MPS 内存分配
- 但 `deform_conv2d_backward` 仍需 CPU fallback
- MLX 有 Stable Diffusion 实现，但只有推理代码

**用户**：有没有 mlx 方案可行的路线？

**评估**：MLX 移植需要重写 UNet + 两个 Encoder + DPM-Solver + 训练循环，预计 3-5 天。但训练速度可提升 10-18 倍。

**用户**：走 MLX 移植

**用户**：你开始移植了？不是说了，让你新开个项目么？

**用户**：记得使用技能 /init-project

## 决策

使用 /init-project 创建独立项目 fontdiffuser-mlx，专注 MLX 移植。
