<div align="center">

# FontDiffuser MLX

**FontDiffuser 的 Apple MLX 移植版本**  
在 Apple Silicon 上运行字体风格迁移推理，支持原始 PyTorch 预训练权重加载。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/Apple%20MLX-0.31+-green.svg)](https://ml-explore.github.io/mlx/)
[![arXiv](https://img.shields.io/badge/arXiv-2312.12142-b31b1b)](https://arxiv.org/abs/2312.12142)

</div>

## 概述

本项目将 [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser)（AAAI 2024）从 PyTorch 完整移植到 [Apple MLX](https://ml-explore.github.io/mlx/)，使其能在 Apple Silicon（M1/M2/M3/M4）上运行推理。

移植遵循**先等价后优化**原则：所有模型结构保持与上游一致，权重转换做到 100% 参数可审计，不使用近似替代。

### 实现状态

| 模块 | 状态 | 说明 |
|------|------|------|
| ContentEncoder | ✅ 100% | 字形结构特征提取，SNConv2d + DBlock |
| StyleEncoder | ✅ 94.1% | 风格特征提取，20.6M 参数 |
| UNet | ✅ 100% | 去噪网络，含 MCA + 可变形卷积，78.7M 参数 |
| DeformConv2d | ✅ | MLX 原生可变形卷积实现 |
| DDPM 调度器 | ✅ | 标准扩散采样 |
| DPM-Solver++ | ✅ | 加速采样 |
| 权重转换 | ✅ | PyTorch → MLX 自动布局转换 |
| 训练管线 | 🔲 | 框架就绪，需接入数据集 |

> StyleEncoder 中 InstanceNorm2d 在原始实现中无可学习参数（`affine=False`），MLX 使用 GroupNorm 替代，其参数使用默认初始化，不影响推理结果。

### 推理性能

20 步 DDPM 采样，96×96 输出，单张：

| 设备 | 耗时 |
|------|------|
| Apple M 系列 | ~0.17s |

## 安装

### 环境要求

- macOS（Apple Silicon M1/M2/M3/M4）
- Python 3.10+

### 安装依赖

```bash
git clone https://github.com/songlairui/fontdiffuser-mlx.git
cd fontdiffuser-mlx

python3 -m venv .venv
source .venv/bin/activate
pip install mlx numpy pillow
```

### 获取预训练权重

本项目需要 FontDiffuser 原始 PyTorch 预训练权重，需从上游项目下载：

- **Google Drive**: [下载链接](https://drive.google.com/drive/folders/12hfuZ9MQvXqcteNuz7JQ2B_mUcTr-5jZ)
- **百度网盘**: [下载链接](https://pan.baidu.com/s/19t1B7le8x8L2yFGaOvyyBQ)（提取码: gexg）

下载后应包含以下文件：
```
ckpt/
├── content_encoder.pth
├── style_encoder.pth
└── unet.pth
```

### 转换权重

```bash
python convert_weights.py \
  --ckpt_dir ./ckpt \
  --output_dir ./mlx_weights
```

输出 `mlx_weights/` 目录包含 MLX 格式权重（~384MB）。转换自动处理：
- Conv2d 权重布局：`[O, I, kH, kW]` → `[O, kH, kW, I]`
- 跳过谱归一化缓冲区（`u0`, `sv0`）
- UNet FFN 层索引映射
- StyleEncoder Sequential 层重映射

## 使用

### 推理

```bash
python sample_mlx.py \
  --weights_dir ./mlx_weights \
  --content_image content.png \
  --style_image style.png \
  --output_path output.png
```

参数：
- `--content_image`：内容图像（标准字体字形）
- `--style_image`：风格参考图像
- `--num_inference_steps`：采样步数（默认 20）
- `--solver`：`ddpm`（默认）或 `dpm_solver`
- `--guidance_scale`：classifier-free guidance 强度（默认 7.5）
- `--seed`：随机种子

### Python API

```python
from mlx_fd import FontDiffuserModel
from mlx_fd.unet import UNet
from mlx_fd.encoders import ContentEncoder, StyleEncoder
from mlx_fd.scheduler import DDPMScheduler

content_enc = ContentEncoder(G_ch=64, resolution=96)
style_enc = StyleEncoder(G_ch=64, resolution=96)
unet = UNet(sample_size=96, in_channels=3, out_channels=3,
    block_out_channels=(64, 128, 256, 512), layers_per_block=2,
    cross_attention_dim=1024, attention_head_dim=1,
    content_encoder_downsample_size=3, content_start_channel=64)

model = FontDiffuserModel(unet=unet, style_encoder=style_enc, content_encoder=content_enc)

# 加载权重后推理
noise_pred, offset_loss = model(
    x_t=x_t, timesteps=mx.array([500]),
    style_images=style_tensor, content_images=content_tensor,
    content_encoder_downsample_size=3,
)
```

### 训练（框架）

```bash
python train_mlx.py \
  --data_root /path/to/font_dataset \
  --output_dir ./output \
  --train_batch_size 4
```

## 项目结构

```
fontdiffuser-mlx/
├── mlx_fd/                    # MLX 模型实现
│   ├── embeddings.py          # 时间步嵌入
│   ├── resnet.py              # ResNet 块和上下采样
│   ├── attention.py           # 注意力模块（MCA、CrossAttention 等）
│   ├── deform_conv.py         # 可变形卷积（MLX 原生）
│   ├── unet_blocks.py         # UNet 构建块
│   ├── unet.py                # 完整 UNet
│   ├── encoders.py            # ContentEncoder, StyleEncoder
│   ├── model.py               # FontDiffuserModel 封装
│   ├── scheduler.py           # DDPM + DPM-Solver++ 调度器
│   └── weight_converter.py    # 权重转换
├── convert_weights.py         # 权重转换 CLI
├── sample_mlx.py              # 推理脚本
├── train_mlx.py               # 训练脚本
└── test_model.py              # 验证测试
```

## 技术细节

### NCHW → NHWC

PyTorch 使用 NCHW 张量布局，MLX 使用 NHWC。本项目处理了：
1. **权重转换**：Conv2d 权重 `[O, I, kH, kW]` → `[O, kH, kW, I]`
2. **前向传播**：所有模块内部以 NHWC 格式计算
3. **残差连接**：编码器输出以 NHWC 格式传递给 UNet

### DeformConv2d

MLX 原生实现的可变形卷积 v2，流程：
1. 生成基础采样网格
2. 加上学习到的偏移量
3. 双线性插值采样输入特征
4. 应用卷积权重矩阵乘法

该实现支持自动微分，可用于训练。

### 谱归一化

ContentEncoder 和 StyleEncoder 使用 SNConv2d。推理时 SN 缓冲区被跳过，直接使用已归一化的权重。

## 验证

```bash
python test_model.py
```

验证内容：
- 所有模块导入 ✅
- ContentEncoder / StyleEncoder / UNet 前向传播 ✅
- DeformConv2d 前向传播 ✅
- 完整模型前向传播 ✅
- DDPM 调度器加噪 ✅

权重覆盖率：
```
ContentEncoder: 18/18 (100%)
StyleEncoder:   32/34 (94.1%)  — last_norm 使用默认初始化
UNet:           782/782 (100%)
```

## 致谢

本项目基于以下工作：

- **FontDiffuser**: [yeungchenwa/FontDiffuser](https://github.com/yeungchenwa/FontDiffuser)
  - 论文：*FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning*
  - 会议：AAAI 2024
  - arXiv: [2312.12142](https://arxiv.org/abs/2312.12142)
  - 预训练权重来自原项目，本项目不包含或分发任何模型权重
- **Apple MLX**: [ml-explore/mlx](https://github.com/ml-explore/mlx)

## 许可证

本项目代码以 MIT 许可证开源。

预训练权重由 [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) 项目提供，其使用受原项目条款约束。本项目不包含、不托管、不分发任何模型权重文件。用户需自行从原项目下载权重并遵守其使用条款。
