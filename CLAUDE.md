# FontDiffuser MLX

始终使用简体中文回复。

## 当前状态

本仓库已废弃早期“近似 MLX 移植”方向。现有 `src/`、`sample.py`、`weights.py`、`convert_weights.py` 只能作为反例或参考片段，不能作为可运行实现继续修补。

## 当前目标

以 [GOAL.md](GOAL.md) 为准：基于干净克隆的上游 FontDiffuser，重新制定并实施 MLX 等价移植，目标是同时支持推理和训练。

## 关键原则

- 先等价，后优化。
- 先权重 100% 匹配，后训练。
- 不用标准 Conv 假装替代 DeformConv2d；如果替代，必须明确标为模型改造而非移植。
- 加载权重必须 fail-fast，不能静默保留随机初始化参数。
