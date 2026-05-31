import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
from mlx_fd.encoders import StyleEncoder

import torch
from pathlib import Path

PT_CKPT = Path('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth')
TEST_STYLE = Path('_inbox/testdata/style_img_nchw.npy')

def nhwc_to_nchw(arr):
    return np.transpose(arr, (0, 3, 1, 2))

def main():
    # Load PT model with pre-normalized weights
    sys.path.insert(0, '/Users/larysong/repo/projects/fontdiffuser/src')
    import importlib
    mod = importlib.import_module('modules.style_encoder')
    PTStyleEncoder = mod.StyleEncoder

    pt_model = PTStyleEncoder(G_ch=64, resolution=96)
    state = torch.load(str(PT_CKPT), map_location='cpu', weights_only=True)
    # pre-normalize SNConv2d weights with sv0
    for k, v in list(state.items()):
        if k.endswith('.sv0'):
            wkey = k[:-len('.sv0')] + '.weight'
            if wkey in state:
                state[wkey] = state[wkey] / v[0]
    missing = pt_model.load_state_dict(state, strict=False)
    print('PT missing keys:', missing.missing_keys)
    print('PT unexpected keys:', missing.unexpected_keys[:10], 'count', len(missing.unexpected_keys))
    pt_model.eval()

    # Load input NCHW -> NHWC
    x_nchw = np.load(TEST_STYLE)
    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).astype(np.float32)

    with torch.no_grad():
        pt_in = torch.from_numpy(x_nchw)
        pt_out = pt_model(pt_in)
        pt_np = pt_out.cpu().numpy()

    # MLX model with converted weights
    mlx_model = StyleEncoder(G_ch=64, resolution=96)
    mlx_w = {k: mx.array(v) for k, v in np.load('mlx_weights/style_encoder.npz').items()}
    mlx_model.load_weights(mlx_w, strict=False)
    mlx_out, _, _ = mlx_model(mx.array(x_nhwc))
    mlx_np = np.array(mlx_out)
    if mlx_np.ndim == 4 and mlx_np.shape[-1] != pt_np.shape[1]:
        mlx_np = nhwc_to_nchw(mlx_np)

    diff = np.abs(pt_np - mlx_np)
    print('pt mean/std', pt_np.mean(), pt_np.std())
    print('mlx mean/std', mlx_np.mean(), mlx_np.std())
    print('diff max/mean', diff.max(), diff.mean())

if __name__ == '__main__':
    main()
