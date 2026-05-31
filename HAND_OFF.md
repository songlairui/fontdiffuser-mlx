# HAND_OFF

> 生成时间：2026-05-31 14:30

## 目标

将 FontDiffuser（AAAI 2024）从 PyTorch 完整移植到 Apple MLX，支持推理和训练。

## 当前状态

仓库已从旧近似原型切换回 `mlx_fd/` 主线：废弃的 `src/`、`sample.py`、`weights.py`、`convert_weights.py` 已删除，推理 demo 已可生成输出图，训练脚本已支持真实数据接入，并可直接导出与推理脚本兼容的 `npz` 权重。

## 已完成

- 删除根目录废弃脚本与 `src/`，并同步更新文档入口
- `sample_mlx.py` 可跑通并生成 `output_mlx.png`
- `train_mlx.py` 已支持真实 `content/style/target` 数据目录接入
- `train_mlx.py` 新增 `--checkpoint_format npz`，训练结束后可直接输出 `content_encoder.npz`、`style_encoder.npz`、`unet.npz`
- 训练输出目录可直接作为 `sample_mlx.py --weights_dir` 的下游入口
- `mlx_fd/weight_converter.py` 已修复 SNConv2d 谱归一化处理，改为基于 `u0` + 幂迭代重算 sigma，与上游 `W_()` 行为一致
- 已使用上游 `ckpt/` 重新生成 `mlx_weights/`
- `test_against_pytorch_baseline.py` 已完成基线对齐并固定随机种子；修复 StyleEncoder 最后一层应用逻辑后，当前结果 9/9 通过
- `DPMSolverPipeline.generate()` 已支持 `initial_noise`，可做确定性 DPM 对比；验证脚本容差已收紧
- `README.md` / `CLAUDE.md` 更新为当前仓库结构与使用说明

## 待完成

- 在真实字体数据集上验证完整训练效果与收敛情况
- 用 PyTorch 上游基线继续校验 MLX 等价性
- 训练与推理的性能基准和稳定性回归测试

## 关键决策

- **NHWC 格式**：MLX 原生 NHWC，权重转换时 Conv2d 从 `[O,I,kH,kW]` → `[O,kH,kW,I]`
- **DeformConv2d**：MLX 原生实现（双线性插值采样），不是近似替代
- **InstanceNorm**：StyleEncoder 最后一层用 GroupNorm 替代，参数使用默认初始化（weight=1, bias=0），不影响推理
- **UNet FFN 映射**：PyTorch 的 `ff.net.0.proj`(GEGLU) + `ff.net.2`(Linear) → MLX 的 `ff.net.0`(GEGLU) + `ff.net.1`(Linear)
- **训练输出格式**：默认使用 `--checkpoint_format npz`，避免下游 `sample_mlx.py` 与 safetensors 格式不兼容

## 踩坑记录

1. **SNConv2d 谱归一化**（最大坑）：PyTorch SNConv2d 保存的是未归一化的原始权重，前向时会除以 sigma。`weight_converter.py` 跳过了 `.sv0` 缓冲但未对权重做归一化，导致每层输出被放大 sigma 倍，3 层累积后发散 288 倍。
   - sigma 实测值：ContentEncoder conv1=0.618, conv2=1.961, conv_sc=0.456
   - 深层 sigma 更大（1.4~3.1），多层累积指数级放大
   - **已修复**：`mlx_fd/weight_converter.py` 已改为通过 `u0` 幂迭代重算 sigma，并在导出前完成 `weight / sigma` 归一化
2. **GroupNorm 参数名**（已解决）：MLX 的 GroupNorm 用 `dims` 而非 `num_channels`
3. **Upsample2D**（已解决）：MLX 没有 `F.interpolate`，需要用 `mx.repeat` 实现最近邻上采样
4. **GEGLU 拆分**（已解决）：PyTorch 的 `ff.net.0.proj` 输出 2x dim 再 split，MLX 的 GEGLU 内部实现相同逻辑，权重保持原样
5. **StyleEncoder blocks.5**（已解决）：PyTorch 用 Sequential(InstanceNorm, ReLU, Conv)，MLX 需严格等价实现；当前已补齐并匹配上游 forward 行为
6. **Checkpoint 格式不兼容**（已解决）：训练脚本原先默认导出 `safetensors`，与 `sample_mlx.py --weights_dir` 不兼容；现已支持 `--checkpoint_format npz`，训练完成后可直接用于推理
7. **基线验证随机性**（已解决）：完整推理输出对比会因随机种子不同而误报失败；已在测试与基线生成流程中固定 `mx.random.seed(42)`，消除噪声导致的假阳性

## 下一步

1. 使用修复后的 `mlx_fd/weight_converter.py` 重新生成 `mlx_weights/`（需要上游 PyTorch `ckpt/`）
2. 运行 `python test_against_pytorch_baseline.py` 验证 MLX 与 PyTorch 上游基线的一致性
3. 在真实字体数据集上接入 `train_mlx.py`
4. 运行 `python test_model.py` 确认 7 项测试通过
5. 提交并推送

## 关键文件

- `mlx_fd/`：当前主线 MLX 实现
- `sample_mlx.py`：推理 demo 入口
- `train_mlx.py`：训练 demo 入口
- `test_against_pytorch_baseline.py`：基线对比测试脚本
- `mlx_weights/`：已转换的 MLX 权重
