# FontDiffuser MLX

FontDiffuser 的 Apple MLX 移植版本，支持在 Apple Silicon 上进行推理和训练。

## 项目状态

✅ **基础架构完成** - 所有核心模块已实现并通过测试  
🔄 **权重转换** - 转换脚本已就绪，需要 PyTorch 权重文件  
⏳ **推理测试** - 待获取权重后进行  
⏳ **训练测试** - 待数据集准备后进行

## 架构概述

本项目将 FontDiffuser 从 PyTorch 完整移植到 MLX，保持以下特性：

- **ContentEncoder**: 内容编码器，提取字形结构特征
- **StyleEncoder**: 风格编码器，提取书法风格特征  
- **UNet**: 核心去噪网络，包含 MCA（Multi-level Content Attention）
- **DeformConv2d**: 可变形卷积，用于风格内容对齐

所有模块使用 NHWC 格式（MLX 原生），权重自动从 PyTorch 的 NCHW 格式转换。

## 安装

```bash
cd /Users/larysong/repo/projects/fontdiffuser-mlx
source .venv/bin/activate
```

依赖已在 `.venv` 中安装：
- mlx (0.31.2)
- mlx-metal (0.31.2)
- numpy
- pillow
- scipy

## 使用方法

### 1. 权重转换

从 PyTorch 检查点转换为 MLX 格式：

```bash
python convert_weights.py \
  --ckpt_dir /path/to/pytorch/ckpt \
  --output_dir ./mlx_weights
```

输入目录应包含：
- `content_encoder.pth`
- `style_encoder.pth`
- `unet.pth`

输出：
- `mlx_weights/content_encoder.npz`
- `mlx_weights/style_encoder.npz`
- `mlx_weights/unet.npz`

### 2. 推理

使用转换后的 MLX 权重生成字体图像：

```bash
python sample_mlx.py \
  --weights_dir ./mlx_weights \
  --content_image ./examples/content.png \
  --style_image ./examples/style.png \
  --output_path ./output.png \
  --num_inference_steps 20 \
  --solver ddpm
```

参数说明：
- `--weights_dir`: MLX 权重目录
- `--content_image`: 内容图像（标准字体）
- `--style_image`: 风格图像（目标书法风格）
- `--output_path`: 输出图像路径
- `--num_inference_steps`: 推理步数（默认 20）
- `--solver`: 采样器类型（ddpm 或 dpm_solver）
- `--guidance_scale`: 分类器自由引导强度（默认 7.5）
- `--seed`: 随机种子（默认 123）

### 3. 训练

在自定义数据集上训练：

```bash
python train_mlx.py \
  --data_root /path/to/dataset \
  --output_dir ./output \
  --train_batch_size 4 \
  --max_train_steps 100000 \
  --learning_rate 1e-4
```

## 项目结构

```
fontdiffuser-mlx/
├── mlx_fd/                    # MLX 模型实现
│   ├── __init__.py
│   ├── embeddings.py          # 时间步嵌入
│   ├── resnet.py              # ResNet 块和上下采样
│   ├── attention.py           # 注意力模块（MCA、CrossAttention 等）
│   ├── deform_conv.py         # 可变形卷积（MLX 原生实现）
│   ├── unet_blocks.py         # UNet 构建块
│   ├── unet.py                # 完整 UNet 模型
│   ├── encoders.py            # ContentEncoder 和 StyleEncoder
│   ├── model.py               # FontDiffuserModel 封装
│   ├── scheduler.py           # DDPM 和 DPM-Solver 调度器
│   └── weight_converter.py    # PyTorch → MLX 权重转换
├── sample_mlx.py              # 推理脚本
├── train_mlx.py               # 训练脚本
├── convert_weights.py         # 权重转换工具
├── GOAL.md                    # 项目目标文档
├── CLAUDE.md                  # AI 助手指南
└── README.md                  # 本文档
```

## 技术细节

### 权重转换

PyTorch 使用 NCHW 格式，MLX 使用 NHWC 格式。转换规则：

1. **Conv2d 权重**: `[O, I, kH, kW]` → `[O, kH, kW, I]`
2. **Linear 权重**: 保持不变 `[O, I]`
3. **GroupNorm/LayerNorm**: 保持不变（都是一维参数）
4. **谱归一化参数**: 跳过（推理时不需要）

### 可变形卷积

MLX 原生实现的可变形卷积：
- 输入: `x [B, H, W, C]`, `offset [B, H, W, 18]`（3x3 核）
- 使用双线性插值采样输入特征
- 应用卷积权重
- 输出: `y [B, H, W, C_out]`

### 调度器

支持两种采样器：
1. **DDPM**: 标准的扩散模型采样
2. **DPM-Solver**: 加速采样，步数更少

## 验证状态

### 已完成的测试

✅ 所有模块导入成功  
✅ ContentEncoder 创建和前向传播  
✅ StyleEncoder 创建和前向传播  
✅ UNet 创建和前向传播  
✅ DeformConv2d 创建和前向传播  
✅ 完整模型前向传播  
✅ DDPM 调度器创建和加噪测试  

### 待完成的测试

⏳ 权重转换验证（需要 PyTorch 权重）  
⏳ 推理结果验证（需要转换后的权重）  
⏳ 训练循环测试（需要数据集）  

## 性能

在 Apple M1/M2/M3 上的预期性能：
- 推理速度：相比 PyTorch CPU 快 2-3 倍
- 内存使用：优化后可支持更大批次
- 训练速度：利用统一内存架构，减少数据拷贝

## 后续工作

1. **获取 PyTorch 权重**: 从 FontDiffuser 官方仓库下载预训练权重
2. **转换并测试**: 运行权重转换并进行推理测试
3. **性能优化**: 针对 MLX 特性优化推理和训练性能
4. **数据集准备**: 准备中文书法数据集用于微调训练
5. **训练测试**: 在小规模数据集上验证训练流程

## 参考

- [FontDiffuser (PyTorch)](https://github.com/yeungchenwa/FontDiffuser)
- [MLX 文档](https://ml-explore.github.io/mlx/)
- [GOAL.md](./GOAL.md) - 项目目标详情
- [CLAUDE.md](./CLAUDE.md) - AI 助手指南

## 许可证

本项目遵循原始 FontDiffuser 项目的许可证。
