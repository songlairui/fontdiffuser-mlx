# HAND_OFF

> 生成时间：2026-05-31 14:30

## 目标

将 FontDiffuser（AAAI 2024）从 PyTorch 完整移植到 Apple MLX，支持推理和训练。

## 当前状态

基础架构已完成，但存在关键 bug：SNConv2d 权重未做谱归一化，导致推理输出发散（288 倍）。需修复后才能正常使用。

## 已完成

- MLX 模型架构完整实现（mlx_fd/ 目录，11 个模块，~2800 行）
- ContentEncoder / StyleEncoder / UNet / DeformConv2d / DDPM / DPM-Solver++
- 权重转换器（convert_weights.py）+ 已转换权重（mlx_weights/）
- 推理脚本（sample_mlx.py）、训练脚本框架（train_mlx.py）、测试套件（test_model.py）
- PyTorch 基线对比测试脚本（test_against_pytorch_baseline.py）
- 上游 FontDiffuser 已克隆到 `/Users/larysong/repo/projects/fontdiffuser`（commit 7b28ce9）

## 待完成

- **P0 修复 SNConv2d 谱归一化 bug**（当前卡点）
- 重新转换权重并验证
- 提供可用的推理 demo（当前输出为纯噪点）
- 训练管线接入真实数据集
- 性能基准测试

## 关键决策

- **NHWC 格式**：MLX 原生 NHWC，权重转换时 Conv2d 从 `[O,I,kH,kW]` → `[O,kH,kW,I]`
- **DeformConv2d**：MLX 原生实现（双线性插值采样），不是近似替代
- **InstanceNorm**：StyleEncoder 最后一层用 GroupNorm 替代，参数使用默认初始化（weight=1, bias=0），不影响推理
- **UNet FFN 映射**：PyTorch 的 `ff.net.0.proj`(GEGLU) + `ff.net.2`(Linear) → MLX 的 `ff.net.0`(GEGLU) + `ff.net.1`(Linear)

## 踩坑记录

1. **SNConv2d 谱归一化**（最大坑）：PyTorch SNConv2d 保存的是未归一化的原始权重，前向时会除以 sigma。`weight_converter.py` 跳过了 `.sv0` 缓冲但未对权重做归一化，导致每层输出被放大 sigma 倍，3 层累积后发散 288 倍。
   - sigma 实测值：ContentEncoder conv1=0.618, conv2=1.961, conv_sc=0.456
   - 深层 sigma 更大（1.4~3.1），多层累积指数级放大
2. **GroupNorm 参数名**：MLX 的 GroupNorm 用 `dims` 而非 `num_channels`
3. **Upsample2D**：MLX 没有 `F.interpolate`，需要用 `mx.repeat` 实现最近邻上采样
4. **GEGLU 拆分**：PyTorch 的 `ff.net.0.proj` 输出 2x dim 再 split，MLX 的 GEGLU 内部实现相同逻辑，权重保持原样
5. **StyleEncoder blocks.5**：PyTorch 用 Sequential(InstanceNorm, ReLU, Conv)，MLX 拆成 `last_norm` + `last_conv`

## 下一步

1. 修复 `mlx_fd/weight_converter.py` 中的 `convert_content_encoder_weights()` 和 `convert_style_encoder_weights()`：对每个 SNConv2d 的 `.weight` 除以对应的 `.sv0`
2. 重新运行 `python convert_weights.py` 生成 `mlx_weights/`
3. 运行 `python test_against_pytorch_baseline.py` 验证 `noise_pred@t=999` 的 max_abs_diff < 0.1
4. 运行 `python test_model.py` 确认 7 项测试通过
5. 提交并推送

## 关键文件

- `mlx_fd/weight_converter.py`：权重转换器，当前 bug 所在
- `_inbox/tapped/20260531-1408-bug-snconv-weight-conversion.md`：bug 详细报告
- `_inbox/testdata/`：PyTorch 基线数据（.npy 文件，NCHW 格式）
- `test_against_pytorch_baseline.py`：基线对比测试脚本
- `mlx_weights/`：已转换的 MLX 权重（需重新生成）
