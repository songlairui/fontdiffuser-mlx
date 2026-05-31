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

### 重新生成 MLX 权重

`mlx_fd/weight_converter.py` 已支持对 `SNConv2d` 权重做谱归一化。若上游 PyTorch 权重更新或你首次本地转换，请使用：

```python
from mlx_fd.weight_converter import convert_and_save_weights

convert_and_save_weights(
    ckpt_dir="./ckpt",
    output_dir="./mlx_weights",
)
```

这会重新生成 `mlx_weights/content_encoder.npz`、`style_encoder.npz`、`unet.npz`，并自动处理：
- `SNConv2d` 的 `.sv0` 归一化
- Conv2d 权重布局从 PyTorch NCHW 转为 MLX NHWC
- SN buffer 的跳过与归一化后保存

## 使用

### 推理 demo

```bash
python sample_mlx.py \
  --weights_dir ./mlx_weights \
  --content_image test_content.png \
  --style_image test_style.png \
  --output_path output_mlx.png
```

参数：
- `--content_image`：内容图像（标准字体字形）
- `--style_image`：风格参考图像
- `--num_inference_steps`：采样步数（默认 20）
- `--solver`：`ddpm`（默认）或 `dpm_solver`
- `--guidance_scale`：classifier-free guidance 强度（默认 7.5）
- `--seed`：随机种子

### 训练 demo

```bash
python train_mlx.py \
  --data_root ./your_dataset \
  --output_dir ./output \
  --train_batch_size 2 \
  --max_train_steps 20 \
  --save_interval 20 \
  --checkpoint_format npz
```

数据目录结构应为：

```
your_dataset/
├── content/
├── style/
└── target/
```

要求：
- `content/`、`style/`、`target/` 中的文件名需要一一对应（例如 `0001.png`）
- 内容图通常是标准字体字形，风格图是目标书法/手写字体样本，目标图是该字形在目标风格下的 ground truth
- 图片会被 resize 到 96×96

如果 `--data_root` 下没有这三个目录，脚本会自动退化为 synthetic smoke-test，可用于快速验证训练链路是否跑通。

训练完成后，`--checkpoint_format npz` 会在 `--output_dir` 下生成：

```
output/
├── content_encoder.npz
├── style_encoder.npz
└── unet.npz
```

这三个文件可直接作为下游推理入口：

```bash
python sample_mlx.py \
  --weights_dir ./output \
  --content_image content.png \
  --style_image style.png \
  --output_path result.png
```

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
