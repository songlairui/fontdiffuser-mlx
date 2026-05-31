# FontDiffuser MLX

将 [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser)（PyTorch 扩散字体生成模型）移植到 Apple MLX，在 M4 Pro 上实现 ~10-18x 训练加速。

## 背景

FontDiffuser 是一个基于扩散模型的字体生成系统，能从少量样本学习手写风格并生成新字形。原版使用 PyTorch，但在 Apple Silicon 上训练时遇到 `deform_conv2d_backward` MPS 不支持的问题，CPU 训练速度约 1.78s/step。

MLX 是 Apple 的 ML 框架，统一内存架构消除了 MPS↔CPU 数据传输瓶颈，预计训练速度提升 10-18 倍。

## 项目结构

```
src/
  unet.py          # UNet 主模型（MLX, NHWC）
  blocks.py        # ResBlock, CrossAttention, StyleRSI, MCADown
  encoders.py      # ContentEncoder, StyleEncoder
  scheduler.py     # DDPM scheduler + DPM-Solver++
convert_weights.py # PyTorch → MLX 权重转换
weights.py         # 权重加载与映射
sample.py          # 推理脚本
train.py           # 训练循环（待实现）
```

## 关键差异（PyTorch → MLX）

| 方面 | PyTorch | MLX |
|------|---------|-----|
| 数据布局 | NCHW | NHWC |
| Conv2d 权重 | [O,I,H,W] | [O,H,W,I] |
| Module | forward() | __call__() |
| 激活函数 | nn.SiLU() | nn.silu(x) |
| 自动微分 | loss.backward() | nn.value_and_grad() |

## 设备要求

- Apple Silicon (M1/M2/M3/M4)
- macOS 14+
- Python 3.12+
