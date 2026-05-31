# 为什么要做 FontDiffuser MLX 移植

## 触发点

在 personal-handwriting 项目中，我用 FontDiffuser（PyTorch 扩散模型）从 6 张手写照片生成 237 个字形。默认权重生成的字形丢失了手写风格，需要用用户自己的样本微调。

**卡住的地方：** 微调在 Apple M4 Pro 上跑不动。
- MPS（Metal GPU）：`deform_conv2d_backward` 算子不支持，每步卡死
- MPS + CPU fallback：数据传输开销太大，第一步就超时
- CPU：1.78s/step，3000 步 ≈ 89 分钟——勉强能用但太慢

**想通的事：** MLX 是 Apple 自己的 ML 框架，统一内存架构，没有 MPS↔CPU 传输瓶颈。M4 Pro 的 16 核 GPU 在 MLX 下应该能跑到 0.1-0.2s/step，比 CPU 快 10-18 倍。

## 目标

1. 把 FontDiffuser 的 UNet + StyleEncoder + ContentEncoder + DPM-Solver 完整移植到 MLX
2. 实现推理（生成字形）和训练（微调风格）
3. 用用户手写样本微调，生成保留个人笔迹风格的字体

## 技术挑战

FontDiffuser 不是一个标准的 Stable Diffusion，它有三个自定义组件：
- **MCADownBlock2D**：多模态交叉注意力下采样块
- **StyleRSIUpBlock2D**：风格感知可变形卷积上采样块
- **SCInterpreterOffsets**：风格-内容偏移预测器

这些组件在 MLX 中没有现成实现，需要从零移植。PyTorch 模型有 782 个权重 key，需要精确映射到 MLX 模块。

## 参考

- 原始项目：yeungchenwa/FontDiffuser
- MLX 参考：ml-explore/mlx-examples/stable_diffusion
- 设备：Apple M4 Pro, 24GB RAM
