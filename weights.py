"""FontDiffuser MLX — 权重加载与映射。

将 PyTorch state_dict 映射到 MLX 模块参数。
"""

import mlx.core as mx
import numpy as np
import torch
from pathlib import Path


def load_pth(path: str | Path) -> dict[str, np.ndarray]:
    """加载 PyTorch .pth 文件为 numpy dict。"""
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k: v.cpu().numpy() for k, v in sd.items()}


def conv_weight(w: np.ndarray) -> np.ndarray:
    """[O,I,H,W] → [O,H,W,I] for MLX Conv2d。"""
    return w.transpose(0, 2, 3, 1) if w.ndim == 4 else w


def map_unet_weights(pt_sd: dict[str, np.ndarray]) -> dict[str, mx.array]:
    """映射 PyTorch UNet 权重到 MLX UNet 模块。

    映射关系（PyTorch → MLX）：
    - conv_in.weight → conv_in.weight (transpose)
    - time_embedding.linear_{1,2} → time_embedding.linear_{1,2}
    - down_blocks.{i}.resnets.{j}.* → down_blocks[i].resnets[j].*
    - down_blocks.{i}.attentions.{j}.transformer_blocks.{k}.* → down_blocks[i].attentions[j].transformer_blocks[k].*
    - mid_block.resnets.{i}.* → mid_block.resnet{i+1}.*
    - up_blocks.{i}.resnets.{j}.* → up_blocks[i].resnets[j].*
    - up_blocks.{i}.dcn_deforms.{j}.weight/bias → up_blocks[i].dcn_deforms[j].conv.weight/bias (transpose)
    - conv_norm_out.* → conv_norm_out.*
    - conv_out.* → conv_out.*
    """
    out = {}

    for k, v in pt_sd.items():
        # Skip spectral norm buffers
        if ".sn_" in k or (k.endswith(".u") and v.ndim == 1 and v.shape[0] <= 2):
            continue
        if k.endswith(".sv") and v.ndim == 1 and v.shape[0] <= 2:
            continue

        mk = k  # MLX key

        # Conv weights: transpose
        if v.ndim == 4:
            v = conv_weight(v)

        # 1x1 conv shortcut → same key but transpose already done
        # DeformConv2d → map to dcn_deforms[i].conv.*
        if "dcn_deforms." in mk:
            mk = mk.replace("dcn_deforms.", "dcn_deforms.").replace(
                "dcn_deforms.", "dcn_deforms."
            )
            # After dcn_deforms.{idx}.weight → dcn_deforms[{idx}].conv.weight
            # But the model uses a list, so we need to map dcn_deforms.0 → dcn_deforms.0
            # Actually in MLX ModuleList, the key is already correct

        # Mid block mapping: mid_block.resnets.0 → mid_block.resnet1, mid_block.resnets.1 → mid_block.resnet2
        if mk.startswith("mid_block.resnets.0."):
            mk = mk.replace("mid_block.resnets.0.", "mid_block.resnet1.", 1)
        elif mk.startswith("mid_block.resnets.1."):
            mk = mk.replace("mid_block.resnets.1.", "mid_block.resnet2.", 1)

        # Mid block attentions mapping
        if mk.startswith("mid_block.attentions.0."):
            mk = mk.replace("mid_block.attentions.0.", "mid_block.attn.", 1)

        # Mid block content_attentions mapping
        if mk.startswith("mid_block.content_attentions.0."):
            mk = mk.replace("mid_block.content_attentions.0.", "mid_block.content_attn.", 1)

        # GEGLU proj → split into linear1/linear2
        if "ff.net.0.proj.weight" in mk:
            k1 = mk.replace("ff.net.0.proj.weight", "ff.linear1.weight")
            k2 = mk.replace("ff.net.0.proj.weight", "ff.linear2.weight")
            v_split = np.split(v, 2, axis=0)
            out[k1] = mx.array(v_split[0].astype(np.float32))
            out[k2] = mx.array(v_split[1].astype(np.float32))
            continue
        if "ff.net.0.proj.bias" in mk:
            k1 = mk.replace("ff.net.0.proj.bias", "ff.linear1.bias")
            k2 = mk.replace("ff.net.0.proj.bias", "ff.linear2.bias")
            v_split = np.split(v, 2, axis=0)
            out[k1] = mx.array(v_split[0].astype(np.float32))
            out[k2] = mx.array(v_split[1].astype(np.float32))
            continue

        # FF output mapping
        if "ff.net.2." in mk:
            mk = mk.replace("ff.net.2.", "ff.out.")

        out[mk] = mx.array(v.astype(np.float32))

    return out


def map_encoder_weights(pt_sd: dict[str, np.ndarray]) -> dict[str, mx.array]:
    """映射 ContentEncoder / StyleEncoder 权重。

    SNConv2d 权重已经是 normalized 的，直接用。
    跳过 spectral norm 的 u/sv buffers。
    """
    out = {}
    for k, v in pt_sd.items():
        # Skip SN buffers
        if ".u" in k and v.ndim == 1 and v.shape[0] <= 2:
            continue
        if ".sv" in k and v.ndim == 1 and v.shape[0] <= 2:
            continue

        if v.ndim == 4:
            v = conv_weight(v)

        out[k] = mx.array(v.astype(np.float32))
    return out


def load_all_weights(ckpt_dir: str | Path) -> dict[str, dict[str, mx.array]]:
    """加载全部 FontDiffuser 权重并转换格式。

    Returns: {"unet": {...}, "style_encoder": {...}, "content_encoder": {...}}
    """
    ckpt_dir = Path(ckpt_dir)
    return {
        "unet": map_unet_weights(load_pth(ckpt_dir / "unet.pth")),
        "style_encoder": map_encoder_weights(load_pth(ckpt_dir / "style_encoder.pth")),
        "content_encoder": map_encoder_weights(load_pth(ckpt_dir / "content_encoder.pth")),
    }


def apply_weights(model, weights: dict[str, mx.array]):
    """将转换后的权重应用到 MLX 模型。

    使用 model.update() 递归更新参数。
    """
    # Flatten model parameters to match weight keys
    from mlx.utils import tree_flatten, tree_unflatten

    flat_params = tree_flatten(model.parameters(), prefix="")
    param_dict = dict(flat_params)

    matched = 0
    for k, v in weights.items():
        if k in param_dict:
            param_dict[k] = v
            matched += 1

    print(f"Matched {matched}/{len(weights)} weights → model")
    model.update(tree_unflatten(list(param_dict.items())))
    return model
