import numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_fd.encoders import StyleEncoder

enc = StyleEncoder(64, 96)
enc.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/style_encoder.npz').items()}, strict=False)
x = mx.array(np.transpose(np.load('_inbox/testdata/style_img_nchw.npy'), (0,2,3,1)).astype(np.float32))
h = x
for block in enc.blocks:
    h = block(h)
print('pre_last_norm', float(np.array(h).mean()), float(np.array(h).std()))
for eps in [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
    last_ch = enc.arch['out_channels'][-1]
    norm = nn.GroupNorm(num_groups=last_ch, dims=last_ch)
    hn = norm(h)
    print('eps',eps,'std',float(np.array(hn).std()))
