# FontDiffuser MLX

将 FontDiffuser（PyTorch 扩散字体生成模型）移植到 Apple MLX。

## 目标

在 M4 Pro 上实现 ~10-18x 训练加速，用于手写字体微调。

## 项目结构

```
src/
  unet.py          # UNet 主模型
  blocks.py        # ResBlock, Attention, StyleRSI, MCADown
  encoders.py      # ContentEncoder, StyleEncoder
  scheduler.py     # DDPM scheduler + DPM-Solver++
convert_weights.py # PyTorch → MLX 权重转换
weights.py         # 权重加载与映射
sample.py          # 推理脚本
train.py           # 训练循环
```

## 开发流程

1. 先跑通推理（加载 PyTorch 预训练权重，验证输出一致）
2. 实现训练循环（MLX nn.value_and_grad）
3. 用用户手写样本微调

## 关键约束

- 数据布局：NHWC（MLX 原生），非 NCHW（PyTorch）
- Conv2d 权重需 transpose：[O,I,H,W] → [O,H,W,I]
- deform_conv2d 需要自定义实现或用标准 Conv 近似
- GroupNorm 必须 pytorch_compatible=True
