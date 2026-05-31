# GOAL: FontDiffuser MLX

## 目标

将上游 FontDiffuser 等价移植到 Apple MLX，并同时支持推理和训练。最终 MLX 版应能加载原 PyTorch 预训练权重，复现原版推理链路，并在用户手写样本上进行微调。

## 当前判断

当前仓库里的 MLX 代码是废弃原型，不应继续修补作为主线。主要问题是模型结构不等价、权重匹配不完整、DeformConv2d 被近似替换、采样器与训练链路没有对齐上游。

后续实施应从干净克隆的上游 FontDiffuser 出发，逐步迁移到 MLX。

## 迁移原则

先正确性，后性能。不要用“能跑”代替“等价”。

权重加载必须是硬门槛：参数 key、shape、布局转换要完整闭环；任何未匹配权重都必须显式报错或被记录为有理由跳过。

DeformConv2d 是核心风险。优先实现 MLX 可微版本；若改用近似方案，必须把路线标记为模型改造，并重新验证质量，不能称为原模型移植。

## 实施阶段

1. 建立上游基线  
   固定 FontDiffuser 上游 commit、checkpoint、输入样例、PyTorch 输出和关键中间张量。

2. 移植模型结构  
   等价迁移 ContentEncoder、StyleEncoder、UNet、MCADownBlock2D、StyleRSIUpBlock2D、SpatialTransformer、ChannelAttnBlock、OffsetRefStrucInter、ResNet 和 timestep embedding。

3. 打通权重转换  
   完成 PyTorch NCHW 到 MLX NHWC 的布局转换，并做到 checkpoint 加载 100% 可解释。

4. 跑通推理  
   对齐原版预处理、DDPM schedule、DPM-Solver++、classifier-free guidance 和输出后处理。

5. 跑通训练  
   对齐 DDPM 加噪、classifier-free dropout、diffusion loss、offset loss、perceptual loss。SCR phase 2 可作为后续阶段。

6. 性能优化  
   在正确性闭环后再优化 batch size、mixed precision、MLX lazy eval、attention 和 deform conv 性能。

## 验收标准

- MLX 模型参数与上游 checkpoint 的匹配结果可审计。
- 固定输入下，关键模块输出和最终 `noise_pred` 与 PyTorch 基线接近。
- MLX 推理能生成与原版同分布的结果。
- MLX 训练能在小样本手写数据上完成微调，并保存/加载 checkpoint。
- 性能数据必须实测，不再保留未经验证的加速倍数承诺。

## 非目标

当前阶段不做 ControlNet、InstructPix2Pix、UI、服务化部署，也不追求一次性完成 SCR phase 2。

当前阶段不基于早期近似原型继续扩大功能面。
